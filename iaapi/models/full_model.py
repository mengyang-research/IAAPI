"""
Full IA-API Model: End-to-end parameter inference model.

Combines MASE encoder, observation encoder, and ISP head into
a unified model for amortized parameter inference.
"""

from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("PyTorch is required")


class IAAPIModel(nn.Module):
    """
    End-to-end IA-API model for amortized parameter inference.

    Args:
        config: Configuration dictionary
    d_embed: Embedding dimension
        k_max: Maximum subspace dimension
    """

    def __init__(
        self,
        config: Dict[str, Any],
        d_embed: Optional[int] = None,
        k_max: Optional[int] = None,
    ):
        super().__init__()

        # Extract configuration
        model_config = config.get("model", {})
        self.d_embed = d_embed or model_config.get("d_embed", 256)
        self.k_max = k_max or model_config.get("k_max", 30)

        # Initialize modules
        self._init_mase_encoder(model_config.get("mase_encoder", {}))
        self._init_obs_encoder(model_config.get("obs_encoder", {}))
        self._init_isp_head(model_config.get("isp_head", {}))
        self._init_mean_regressor(model_config.get("mean_regressor", {}))

    def _init_mean_regressor(self, config: Dict[str, Any]):
        """Initialize independent mean regressor head (H5: C1 fix).

        Raw observations → MLP → theta_mean, bypassing ISP decomposition.
        The ISP pathway (obs_encoder invariant + ISP split) discards theta
        position info (regressor corr=0). This head recovers it directly from
        raw observation statistics. C3's rank head is untouched.

        Direction A (2026-06-30): the cross-model regressor (H6b) failed
        (corr≈0) despite a single-model raw-obs probe reaching corr=0.449,
        because the regressor input lacked *model identity* — the same obs
        statistics map to different theta across models. Adding z_M (the MASE
        model embedding) restores per-model conditioning, and amplitude
        features restore the magnitude information that z-score discards.
        ``use_model_cond`` toggles this conditioning on.
        """
        import torch.nn as nn
        self.use_mean_regressor = config.get("enabled", False)
        if not self.use_mean_regressor:
            self.mean_regressor = None
            return
        # Per-observable stats: mean/std/max/min/delta/log_max (6 per obs)
        # Padded to max_n_obs * n_stats, fed through MLP → k_max (theta_mean in U-space)
        n_stats = config.get("n_stats_per_obs", 17)  # 9 stats + 8 resampled timepoints
        max_n_obs = config.get("max_n_obs", 20)
        d_hidden = config.get("d_hidden", 512)
        n_layers = config.get("n_layers", 3)
        dropout = config.get("dropout", 0.0)  # v3: anti-overfitting (only 8 models)
        in_dim = max_n_obs * n_stats
        # Direction A: condition on model identity (z_M, d_embed) + amplitude
        # features (amp_feat_dim). These are concatenated to the raw obs stats
        # so the regressor can learn a model-specific obs→theta mapping.
        self._mr_use_model_cond = config.get("use_model_cond", False)
        self._mr_use_amp_cond = config.get("use_amp_cond", True)
        self._mr_linear_out = config.get("linear_out", False)  # v5: no LN+tanh
        # Direction A v2: regressor predicts theta (raw param space) instead of
        # alpha=U^T@theta. The U-rotation is model-specific and frozen, making
        # alpha a hard cross-model target. Theta is the model's natural coord
        # system, directly learnable from obs (per-model probe r=0.436).
        self._mr_theta_space = config.get("theta_space", True)
        extra_dim = 0
        if self._mr_use_model_cond:
            extra_dim += self.d_embed  # z_M
            if self._mr_use_amp_cond:
                extra_dim += config.get("amp_feat_dim", 16)
        in_dim += extra_dim
        # v3: dropout to fight overfitting (8 models → regressor memorizes
        # per-model mapping; dropout forces it to use obs features, not z_M shortcut)
        dropout = config.get("dropout", 0.3)
        layers = [nn.Linear(in_dim, d_hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout)]
        # Output: theta_space → max_n_params (padded), else k_max (U-space alpha)
        out_dim = config.get("max_n_params", 50) if self._mr_theta_space else self.k_max
        layers += [nn.Linear(d_hidden, out_dim)]
        self.mean_regressor = nn.Sequential(*layers)
        # LayerNorm + tanh for numerical stability (H6b config, validated).
        # tanh bounds output to [-10,10], LayerNorm normalizes across output dims.
        self.mean_regressor_norm = nn.LayerNorm(out_dim)
        self._mr_n_stats = n_stats
        self._mr_max_n_obs = max_n_obs

    def _init_mase_encoder(self, config: Dict[str, Any]):
        """Initialize MASE encoder."""
        from iaapi.models.mase_encoder import MASEEncoder

        self.mase = MASEEncoder(
            d_embed=self.d_embed,
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 8),
            d_model=config.get("d_model", self.d_embed),
            d_ff=config.get("d_ff", 1024),
            dropout=config.get("dropout", 0.1),
        )

    def _init_obs_encoder(self, config: Dict[str, Any]):
        """Initialize observation encoder."""
        from iaapi.models.obs_encoder import ObservationEncoder

        d_model = config.get("d_model", self.d_embed)
        self.obs_encoder = ObservationEncoder(
            d_embed=self.d_embed,
            n_heads=config.get("n_heads", 8),
            n_isab=config.get("n_isab", 4),
            n_inducing=config.get("n_inducing", 32),
            d_model=d_model,
            d_ff=config.get("d_ff", 4 * d_model),
            dropout=config.get("dropout", 0.1),
            time_scale=config.get("time_scale", 100.0),
            token_mode=config.get("token_mode", "cell"),
            max_n_obs=config.get("max_n_obs", 128),
            gradient_checkpoint=config.get("gradient_checkpoint", False),
            invariant=config.get("invariant", False),
        )

    def _init_isp_head(self, config: Dict[str, Any]):
        """Initialize ISP head."""
        from iaapi.models.isp_head import ISPHead

        self.isp_head = ISPHead(
            d_embed=self.d_embed,
            k_max=self.k_max,
            n_flow_layers=config.get("n_flow_layers", 8),
            flow_hidden_dim=config.get("flow_hidden_dim", 256),
            use_attn_subspace=config.get("use_attn_subspace", True),
            soft_gating=config.get("soft_gating", True),
            sloppy_prior_mode=config.get("sloppy_prior_mode", "mixture"),
            min_log_std=config.get("min_log_std", -7.0),
            max_log_std=config.get("max_log_std", 5.0),
            invariant_rank=config.get("invariant_rank", False),
            fim_only_rank=config.get("fim_only_rank", False),
            rank_mse_mode=config.get("rank_mse_mode", False),
            flow_type=config.get("flow_type", "diagonal"),
            use_amplitude_bypass=config.get("use_amplitude_bypass", False),
            amp_feat_dim=config.get("amp_feat_dim", 16),
        )

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - graph_data: Heterogeneous graph for model structure
                - observations: Dict with values, times, masks
                - theta_true: Ground truth parameters (optional, for training)
                - fim_eigenvalues: FIM eigenvalues (optional)
                - fim_eigenvectors: FIM eigenvectors (optional)
                - effective_rank: Effective rank (optional)

        Returns:
            Dictionary containing model outputs
        """
        # Encode model structure. For models without a reaction-network graph
        # (e.g. Hodgkin-Huxley), build a minimal parameter-node graph so MASE
        # has valid input — each parameter becomes an isolated node with no
        # edges, and the model embedding is pooled from parameter nodes.
        graph_data = batch["graph_data"]
        if graph_data is None or (isinstance(graph_data, list) and all(g is None for g in graph_data)):
            graph_data = self._build_param_only_graph(batch)
        z_M, z_theta = self.mase(graph_data)

        # Align z_theta dimension with PEtab estimated parameters. The SBML
        # graph may contain more parameter nodes than PEtab estimates (fixed/
        # derived parameters), so z_theta (graph params) can exceed theta_true
        # (PEtab n_params). Pad/truncate z_theta to the batch max n_params so
        # the ISP head sees a consistent dimension with theta_true.
        n_params = batch.get("n_params")
        if n_params is not None:
            max_n = int(n_params.max().item())
            if z_theta.shape[1] != max_n:
                if z_theta.shape[1] > max_n:
                    z_theta = z_theta[:, :max_n]
                else:
                    pad = torch.zeros(z_theta.shape[0], max_n - z_theta.shape[1], z_theta.shape[2],
                                      device=z_theta.device, dtype=z_theta.dtype)
                    z_theta = torch.cat([z_theta, pad], dim=1)

        # Encode observations
        z_D = self.obs_encoder(
            batch["observations"]["values"],
            batch["observations"]["times"],
            batch["observations"]["masks"],
        )

        # Amplitude bypass: extract per-observable magnitude statistics that the
        # invariant obs_encoder discards (z-score normalization removes magnitude).
        # These are needed by q_parallel to recover theta position information (C1).
        # C3's rank head is unaffected (uses fim_feats, not context).
        amp_feats = self._amplitude_features(batch["observations"])

        # H5: independent mean regressor from raw observation statistics.
        # Bypasses ISP decomposition to recover theta position info (C1).
        theta_mean_pred = None
        if self.use_mean_regressor and self.mean_regressor is not None:
            raw_feats = self._raw_obs_features(batch["observations"])
            # Direction A: condition on model identity (z_M) + amplitude so the
            # regressor learns a model-specific obs→theta mapping (the missing
            # ingredient in H6b, which lacked model identity → corr≈0).
            if self._mr_use_model_cond:
                cond_parts = [raw_feats, z_M]
                if getattr(self, "_mr_use_amp_cond", True):
                    cond_parts.append(amp_feats)
                raw_feats = torch.cat(cond_parts, dim=-1)
            theta_mean_pred = self.mean_regressor(raw_feats)  # (B, out_dim)
            # v5: raw linear output (no LayerNorm+tanh). The 10*tanh(LayerNorm(x))
            # compression caps the NN at r≈0.25 while an uncompressed Ridge reaches
            # r=0.534 — LayerNorm normalizes to unit variance (std≈1) but theta_true
            # has std≈3-4, so the NN literally cannot represent the target range.
            # A hard clamp at ±12 (slightly beyond the ±10 theta range) prevents
            # numerical explosion without compressing the learnable range.
            if self._mr_linear_out:
                theta_mean_pred = theta_mean_pred.clamp(-12.0, 12.0)
            else:
                theta_mean_pred = 10.0 * torch.tanh(self.mean_regressor_norm(theta_mean_pred))

        # Get ISP output (pass FIM eigenvalues for cross-model invariant rank features)
        isp_output = self.isp_head(z_theta, z_D, n_params, batch.get("fim_eigenvalues"), amp_feats)

        # Return results
        return {
            "z_M": z_M,
            "z_D": z_D,
            "z_theta": z_theta,
            "isp_output": isp_output,
            "theta_mean_pred": theta_mean_pred,
            "theta_space": self._mr_theta_space if self.use_mean_regressor else False,
        }

    def _amplitude_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-observable magnitude statistics that the invariant obs_encoder discards.

        obs_encoder z-score-normalizes values (removes magnitude) for cross-model
        invariance (C3). But theta position information is encoded in magnitude,
        so q_parallel needs it back (C1). Returns (batch, amp_feat_dim=16):
        for up to 4 observables × {mean, std, max, min}, padded/truncated to 16.
        """
        values = observations["values"]  # (B, n_obs, n_time)
        masks = observations.get("masks")
        if masks is not None:
            cnt = masks.sum(dim=2).clamp(min=1.0)  # (B, n_obs)
            v = values * masks if masks is not None else values
            mean_o = v.sum(dim=2) / cnt  # (B, n_obs)
            var_o = ((v - mean_o.unsqueeze(2)) ** 2 * masks).sum(dim=2) / cnt
            std_o = var_o.clamp(min=0).sqrt()
            max_o = (v + (1 - masks) * (-1e9)).max(dim=2).values  # masked max
            min_o = (v + (1 - masks) * 1e9).min(dim=2).values  # masked min
        else:
            mean_o = values.mean(dim=2)
            std_o = values.std(dim=2)
            max_o = values.max(dim=2).values
            min_o = values.min(dim=2).values
        # Log-scale for numerical stability (values span ~1e-14 to 1e14)
        eps = 1e-8
        stats = torch.stack([
            torch.log(mean_o.abs() + eps),
            torch.log(std_o + eps),
            torch.log(max_o.abs() + eps),
            torch.log((min_o.abs() + eps)),
        ], dim=-1)  # (B, n_obs, 4)
        B = stats.shape[0]
        flat = stats.reshape(B, -1)  # (B, n_obs*4)
        # Pad/truncate to amp_feat_dim=16
        amp_dim = 16
        if flat.shape[1] < amp_dim:
            flat = torch.cat([flat, torch.zeros(B, amp_dim - flat.shape[1], device=flat.device)], dim=1)
        else:
            flat = flat[:, :amp_dim]
        return flat

    def _raw_obs_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Raw-observation features for the mean regressor.

        n_stats_per_obs controls richness:
        - 6: basic stats (mean/std/max/min/delta/log_max) — H5/H6b
        - 17: + shape (slope/auc/peak_time) + 8 resampled timepoints — H7
        """
        values = observations["values"]  # (B, n_obs, n_time)
        masks = observations.get("masks")
        B, n_obs, n_time = values.shape
        if masks is not None:
            valid = masks.to(values.dtype)
            cnt = valid.sum(dim=-1).clamp(min=1)
            mean_o = (values * valid).sum(dim=-1) / cnt
            var_o = ((values - mean_o.unsqueeze(-1)) ** 2 * valid).sum(dim=-1) / cnt
            std_o = var_o.clamp(min=1e-12).sqrt()
            v_masked_max = values.masked_fill(valid == 0, float("-inf"))
            max_o = torch.where(torch.isfinite(v_masked_max.max(dim=-1).values),
                                v_masked_max.max(dim=-1).values, torch.zeros_like(mean_o))
            v_masked_min = values.masked_fill(valid == 0, float("inf"))
            min_o = torch.where(torch.isfinite(v_masked_min.min(dim=-1).values),
                                v_masked_min.min(dim=-1).values, torch.zeros_like(mean_o))
            vals_clean = values * valid
        else:
            mean_o = values.mean(dim=-1); std_o = values.std(dim=-1)
            max_o = values.max(dim=-1).values; min_o = values.min(dim=-1).values
            vals_clean = values
        delta_o = vals_clean[..., -1] - vals_clean[..., 0]
        log_max = torch.log(max_o.abs() + 1e-12)
        use_rich = self._mr_n_stats > 10
        if use_rich:
            slope_o = delta_o / max(n_time - 1, 1)
            auc_o = vals_clean.mean(dim=-1)
            peak_idx = vals_clean.argmax(dim=-1)
            peak_time = peak_idx.float() / max(n_time - 1, 1)
            n_resample = 8
            if n_time >= n_resample:
                idx = torch.linspace(0, n_time - 1, n_resample).long()
            else:
                idx = torch.arange(n_time); n_resample = n_time
            resampled = vals_clean[..., idx]
            stats = torch.stack([mean_o, std_o, max_o, min_o, delta_o, log_max,
                                 slope_o, auc_o, peak_time], dim=-1)
            all_feats = torch.cat([stats, resampled], dim=-1)
            flat = all_feats.reshape(B, -1)
            actual_stats = 9 + n_resample
            target_dim = self._mr_max_n_obs * max(actual_stats, self._mr_n_stats)
        else:
            stats = torch.stack([mean_o, std_o, max_o, min_o, delta_o, log_max], dim=-1)
            flat = stats.reshape(B, -1)
            target_dim = self._mr_max_n_obs * self._mr_n_stats
        if flat.shape[1] < target_dim:
            pad = torch.zeros(B, target_dim - flat.shape[1], device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[1] > target_dim:
            flat = flat[:, :target_dim]
        return torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_param_only_graph(self, batch: Dict[str, Any]):
        """Build a minimal HeteroData graph for models without an SBML network.

        Creates a 'parameter'-only graph (one node per parameter, no edges) so
        the MASE encoder has valid input. All other node types are empty.
        """
        from torch_geometric.data import HeteroData

        n_params = batch.get("n_params")
        if n_params is None:
            raise ValueError("n_params required to build param-only graph")
        batch_size = n_params.shape[0]

        graph_list = []
        for i in range(batch_size):
            n = int(n_params[i].item())
            g = HeteroData()
            g["parameter"].num_nodes = n
            g["species"].num_nodes = 0
            g["reaction"].num_nodes = 0
            g["compartment"].num_nodes = 0
            g["observable"].num_nodes = 0
            graph_list.append(g)
        return graph_list  # MASE encoder will Batch.from_data_list it

    def infer(
        self,
        graph_data: Any,
        observations: Dict[str, Any],
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run inference on a single problem.

        Args:
            graph_data: Model structure
            observations: Observation data
            n_samples: Number of posterior samples

        Returns:
            Dictionary containing:
                - posterior_samples: Posterior samples
                - identifiability_scores: Parameter identifiability scores
                - sloppy_subspace: Sloppy subspace information
        """
        with torch.no_grad():
            # Encodings
            z_M, z_theta = self.mase(graph_data)
            z_D = self.obs_encoder(
                observations["values"],
                observations["times"],
                observations["masks"],
            )

            # Get n_params
            n_params = torch.tensor(
                [graph_data["parameter"].num_nodes],
                dtype=torch.long,
                device=next(self.parameters()).device,
            )

            # Amplitude bypass for C1 (same as training forward)
            obs_dict = {"values": observations["values"], "masks": observations.get("masks")}
            amp_feats = self._amplitude_features(obs_dict)

            # H5: mean regressor (raw obs → theta_mean, bypasses ISP)
            theta_mean_pred = None
            if self.use_mean_regressor and self.mean_regressor is not None:
                raw_feats = self._raw_obs_features(obs_dict)
                # Direction A: condition on model identity + amplitude (same as training)
                if self._mr_use_model_cond:
                    cond_parts = [raw_feats, z_M]
                    if self._mr_use_amp_cond:
                        cond_parts.append(amp_feats)
                    raw_feats = torch.cat(cond_parts, dim=-1)
                theta_mean_pred = self.mean_regressor(raw_feats)
                # v5: match training output mode (linear vs LN+tanh)
                if self._mr_linear_out:
                    theta_mean_pred = theta_mean_pred.clamp(-12.0, 12.0)
                else:
                    theta_mean_pred = 10.0 * torch.tanh(self.mean_regressor_norm(theta_mean_pred))

            # Sample from posterior (with regressor mean override if available)
            posterior_samples = self.isp_head.sample(
                z_theta, z_D, n_params, n_samples, amp_feats=amp_feats, mean_override=theta_mean_pred
            )

            # Get identifiability scores
            isp_output = self.isp_head(z_theta, z_D, n_params, amp_feats=amp_feats)

            return {
                "posterior_samples": posterior_samples,
                "identifiability_scores": isp_output.identifiability_scores,
                "sloppy_subspace": isp_output.U,
                "effective_dimension": isp_output.k,
            }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "IAAPIModel":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded IAAPIModel
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config from checkpoint
        config = checkpoint.get("config", {})

        # Initialize model
        model = cls(config)

        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])

        return model
