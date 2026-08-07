"""
Observation Encoder (Module B) - Encodes irregular time series.

Uses Set Transformer architecture with continuous-time positional encoding
to handle irregularly sampled observations.
"""

from typing import Optional

try:
    import math

    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError("PyTorch is required") from exc


class ContinuousTimePositionalEncoding(nn.Module):
    """Continuous sinusoidal positional encoding for irregular time points."""

    def __init__(self, d_model: int, time_scale: float = 100.0):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")

        self.d_model = d_model
        self.time_scale = time_scale
        n_frequencies = (d_model + 1) // 2
        inv_freq = torch.exp(
            torch.arange(0, n_frequencies, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Encode time points.

        Args:
            times: Time points (batch, n_time)

        Returns:
            Positional encodings (batch, n_time, d_model)
        """
        if times.ndim != 2:
            raise ValueError(f"times must have shape (batch, n_time), got {tuple(times.shape)}")

        scaled_times = times.to(self.inv_freq.dtype) / self.time_scale
        angles = scaled_times.unsqueeze(-1) * self.inv_freq
        encodings = times.new_zeros((*times.shape, self.d_model), dtype=self.inv_freq.dtype)
        encodings[..., 0::2] = torch.sin(angles[..., : encodings[..., 0::2].shape[-1]])
        encodings[..., 1::2] = torch.cos(angles[..., : encodings[..., 1::2].shape[-1]])
        return encodings.to(times.dtype)


class MAB(nn.Module):
    """Multihead Attention Block used by Set Transformer layers."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention + feed-forward residual block."""
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(q + self.dropout(attn_out))
        ff_out = self.ff(h)
        return self.norm2(h + self.dropout(ff_out))


class ISABLayer(nn.Module):
    """Inducing Set Attention Block (ISAB), batch-first and mask-aware."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_inducing = n_inducing
        d_ff = d_ff or 4 * d_model

        self.inducing_points = nn.Parameter(torch.randn(n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, n_heads, d_ff, dropout)
        self.mab2 = MAB(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input set (batch, n_tokens, d_model)
            valid_mask: Boolean mask (batch, n_tokens), True for valid tokens

        Returns:
            Output set (batch, n_tokens, d_model)
        """
        batch_size = x.shape[0]
        inducing = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)
        h = self.mab1(inducing, x, x, key_padding_mask=~valid_mask)
        out = self.mab2(x, h, h)
        return out.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class ObservationEncoder(nn.Module):
    """
    Observation Encoder (Module B).

    Encodes irregularly sampled multi-variable time series using
    Set Transformer with continuous-time positional encoding.
    """

    def __init__(
        self,
        d_embed: int = 256,
        n_heads: int = 8,
        n_isab: int = 4,
        n_inducing: int = 32,
        d_model: int = 256,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        time_scale: float = 100.0,
        token_mode: str = "cell",
        max_n_obs: int = 128,
        gradient_checkpoint: bool = False,
        invariant: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if token_mode not in ("cell", "timepoint"):
            raise ValueError(f"token_mode must be 'cell' or 'timepoint', got {token_mode}")

        self.d_embed = d_embed
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.token_mode = token_mode
        self.max_n_obs = max_n_obs
        self.gradient_checkpoint = gradient_checkpoint
        # C3 fix (root cause 2): raw observation values span 7 orders of
        # magnitude across models (Boehm ~7, Fujita ~6e4, Rahman ~0.09), so a
        # plain Linear(1, d_model) learns model-specific numeric patterns that
        # cannot generalize (probe holdout r = -0.428). When ``invariant`` is
        # set, each sample is z-score normalized per-observable and augmented
        # with geometry/shape descriptors (curvature energy, log time span,
        # effective sampling density) that are scale-free, giving the encoder
        # cross-model-invariant identifiability anchors.
        self.invariant = invariant
        self.time_encoding = ContinuousTimePositionalEncoding(d_model, time_scale=time_scale)
        # Extra invariant feature width: 3 per-observable shape descriptors
        # (curvature energy, log-span, log-density) concatenated to the value.
        if token_mode == "cell":
            in_width = (1 + 3) if invariant else 1
            self.value_proj = nn.Linear(in_width, d_model)
        else:
            # token_mode == "timepoint": fold n_obs into feature dim. Pad/trim
            # the obs axis to max_n_obs so the linear has fixed input width.
            # Two channels per obs: value and validity flag.
            self.value_proj = nn.Linear(2 * max_n_obs, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.isab_layers = nn.ModuleList(
            [
                ISABLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_inducing=n_inducing,
                    d_ff=self.d_ff,
                    dropout=dropout,
                )
                for _ in range(n_isab)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_embed),
        )

    def _invariant_cell_features(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build cross-model-invariant per-token features for cell mode.

        Replaces the raw scalar value with (a) a per-observable z-scored value
        and (b) three scale-free shape descriptors per observable, all
        broadcast to the (batch, n_obs*n_time, 4) token grid:

          * z_value:    (v - mean_o) / std_o  per observable o (removes magnitude)
          * curvature:  mean |second difference| of the trajectory (shape)
          * log_span:   log(t_max / t_min)  (temporal coverage, scale-free)
          * log_density: log(n_valid / span)  (sampling density, scale-free)

        These are computed from masked statistics so missing observations do
        not contaminate the normalization. The descriptors are identical in
        spirit to the geometric quantities that determine identifiability
        (trajectory shape, temporal coverage, sampling) but are independent of
        the model's units, giving the encoder anchors that transfer across
        models — the missing ingredient behind the zero-shot probe r = -0.428.
        """
        batch_size, n_obs, n_time = values.shape
        eps = 1e-6
        # Compute statistics in fp32 even under bf16 autocast: raw observation
        # values span up to ~1e14 (Fujita), so bf16 mean/variance and the
        # squared deviation overflow / lose all precision. The z-scored output
        # is cast back to the input dtype afterward.
        v32 = torch.nan_to_num(values.float())
        valid = masks.to(dtype=v32.dtype)  # (B, n_obs, n_time)

        # Per-observable mean/std over valid timepoints.
        v_clean = v32
        cnt = valid.sum(dim=2).clamp_min(1.0)  # (B, n_obs)
        mean_o = (v_clean * valid).sum(dim=2) / cnt  # (B, n_obs)
        var_o = ((v_clean - mean_o.unsqueeze(2)) ** 2 * valid).sum(dim=2) / cnt
        std_o = var_o.clamp_min(eps).sqrt()  # (B, n_obs)
        z_value = (v_clean - mean_o.unsqueeze(2)) / std_o.unsqueeze(2)  # (B, n_obs, n_time)

        # Curvature: per-timepoint normalized second difference (scale-free).
        # Unlike the per-observable mean curvature, this keeps temporal
        # resolution so that samples of the same model with different
        # parameters — whose z-scored trajectories are nearly identical in
        # shape but differ in local curvature — remain distinguishable. This
        # is the channel that carries per-sample identifiability signal
        # (probe: invariant z_value alone has std ~4e-7 across same-model
        # samples; per-timepoint curvature restores sample-level variance).
        diff2 = torch.zeros_like(v_clean)
        diff2[:, :, 1:-1] = v_clean[:, :, :-2] - 2.0 * v_clean[:, :, 1:-1] + v_clean[:, :, 2:]
        curv_t = diff2 / (std_o.unsqueeze(2) + eps)  # (B, n_obs, n_time), scale-free
        curv_t = curv_t * valid  # zero out invalid timepoints

        # Temporal span and density per observable.
        t_valid = times.float().unsqueeze(1).expand(-1, n_obs, -1)  # (B, n_obs, n_time)
        big = torch.full_like(t_valid, float("-inf"))
        tmin = torch.where(valid > 0, t_valid, big).max(dim=2).values  # (B, n_obs)
        small = torch.full_like(t_valid, float("inf"))
        tmax = torch.where(valid > 0, t_valid, small).min(dim=2).values
        # Clamp span to a safe positive range so log() cannot overflow to inf
        # (happens for observables with degenerate/missing timepoints where
        # tmin/tmax collapse to +-inf). [eps, 1e12] keeps log in [-13.8, 27.6].
        span = (tmax - tmin).clamp(eps, 1.0e12)
        log_span = torch.log(span)  # (B, n_obs)
        log_density = torch.log(cnt / span + eps)  # (B, n_obs)
        # Guard against any residual non-finite values from degenerate rows.
        log_span = torch.nan_to_num(log_span, nan=0.0, posinf=27.0, neginf=-13.0)
        log_density = torch.nan_to_num(log_density, nan=0.0, posinf=27.0, neginf=-13.0)

        # Assemble per-token features: z_value + per-timepoint curvature
        # (both time-resolved, carry per-sample signal) + per-observable
        # span/density (cross-model invariant anchors).
        log_span_t = log_span.unsqueeze(2).expand(-1, -1, n_time)  # (B, n_obs, n_time)
        log_density_t = log_density.unsqueeze(2).expand(-1, -1, n_time)
        z_value = z_value.unsqueeze(-1)  # (B, n_obs, n_time, 1)
        curv_t = curv_t.unsqueeze(-1)    # (B, n_obs, n_time, 1)
        log_span_t = log_span_t.unsqueeze(-1)
        log_density_t = log_density_t.unsqueeze(-1)
        feats = torch.cat([z_value, curv_t, log_span_t, log_density_t], dim=-1)  # (B, n_obs, n_time, 4)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.reshape(batch_size, n_obs * n_time, 4).to(dtype=values.dtype)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            values: Observation values (batch, n_obs, n_time)
            times: Time points (batch, n_time)
            masks: Valid value masks (batch, n_obs, n_time)

        Returns:
            z_D: Observation embedding (batch, d_embed)
        """
        if values.ndim != 3:
            raise ValueError(f"values must have shape (batch, n_obs, n_time), got {tuple(values.shape)}")
        if times.ndim != 2:
            raise ValueError(f"times must have shape (batch, n_time), got {tuple(times.shape)}")
        if masks.shape != values.shape:
            raise ValueError(f"masks shape {tuple(masks.shape)} must match values shape {tuple(values.shape)}")

        batch_size, n_obs, n_time = values.shape
        if times.shape != (batch_size, n_time):
            raise ValueError(
                f"times shape {tuple(times.shape)} must match (batch, n_time)=({batch_size}, {n_time})"
            )

        if self.token_mode == "cell":
            valid_flat = masks.to(dtype=torch.bool).reshape(batch_size, n_obs * n_time)
            if not torch.all(valid_flat.any(dim=1)):
                raise ValueError("ObservationEncoder received a sample with no valid observations")

            values_flat = torch.nan_to_num(values).reshape(batch_size, n_obs * n_time, 1)
            time_encodings = self.time_encoding(times)
            time_encodings = time_encodings.unsqueeze(1).expand(-1, n_obs, -1, -1)
            time_encodings = time_encodings.reshape(batch_size, n_obs * n_time, self.d_model)

            if self.invariant:
                # Cross-model-invariant features (z-scored value + geometry
                # descriptors) instead of raw scalar magnitudes.
                token_feats = self._invariant_cell_features(values, times, masks)
            else:
                token_feats = values_flat
            x = self.value_proj(token_feats) + time_encodings
            x = self.input_norm(x)
            x = self.dropout(x)
            x = x.masked_fill(~valid_flat.unsqueeze(-1), 0.0)

            for isab_layer in self.isab_layers:
                if self.gradient_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        isab_layer, x, valid_flat, use_reentrant=False
                    )
                else:
                    x = isab_layer(x, valid_flat)

            valid = valid_flat.unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            return self.output_proj(pooled)

        # token_mode == "timepoint": one token per timepoint with n_obs folded
        # into the feature dim. Reduces N from n_obs*n_time to n_time, which
        # unblocks long-trajectory PEtab models like Lucarelli (65 obs * 1247 t).
        if n_obs > self.max_n_obs:
            raise ValueError(
                f"timepoint mode requires n_obs <= max_n_obs ({self.max_n_obs}), got {n_obs}"
            )
        # Pad obs axis to max_n_obs along feature dim
        pad = self.max_n_obs - n_obs
        # values: (B, n_obs, n_time) -> (B, n_time, n_obs)
        v = torch.nan_to_num(values).transpose(1, 2)
        m = masks.to(dtype=v.dtype).transpose(1, 2)
        if pad > 0:
            v = torch.nn.functional.pad(v, (0, pad))
            m = torch.nn.functional.pad(m, (0, pad))
        feat = torch.cat([v, m], dim=-1)  # (B, n_time, 2*max_n_obs)
        valid_time = (masks.any(dim=1)).to(dtype=torch.bool)  # (B, n_time)
        if not torch.all(valid_time.any(dim=1)):
            raise ValueError("ObservationEncoder received a sample with no valid observations")

        time_enc = self.time_encoding(times)  # (B, n_time, d_model)
        x = self.value_proj(feat) + time_enc
        x = self.input_norm(x)
        x = self.dropout(x)
        x = x.masked_fill(~valid_time.unsqueeze(-1), 0.0)

        for isab_layer in self.isab_layers:
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    isab_layer, x, valid_time, use_reentrant=False
                )
            else:
                x = isab_layer(x, valid_time)

        valid = valid_time.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.output_proj(pooled)
