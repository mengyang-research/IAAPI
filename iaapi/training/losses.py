"""
Loss functions for IA-API training.

Implements composite loss with likelihood, subspace, orthogonality, and rank components.
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError("PyTorch is required") from exc


class ISPLoss(nn.Module):
    """
    ISP Loss function for IA-API training.

    L = L_NLL + λ1 * L_subspace + λ2 * L_orth + λ3 * L_rank
    """

    def __init__(
        self,
        lambda_nll: float = 1.0,
        lambda_subspace: float = 1.0,
        lambda_orth: float = 0.1,
        lambda_rank: float = 0.5,
        lambda_calibration: float = 0.0,
        lambda_width: float = 0.0,
        lambda_mean: float = 0.0,
        lambda_mean_regressor: float = 0.0,
        calibration_n_samples: int = 64,
        width_min_log_std: float = -9.0,
        width_max_log_std: float = 6.0,
        rank_class_balanced: bool = False,
        rank_label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_subspace = lambda_subspace
        self.lambda_orth = lambda_orth
        self.lambda_rank = lambda_rank
        self.lambda_calibration = lambda_calibration
        self.lambda_width = lambda_width
        self.lambda_mean = lambda_mean
        self.lambda_mean_regressor = lambda_mean_regressor
        self.calibration_n_samples = calibration_n_samples
        # Reachable log_std range of q_parallel (must match the model's clamp).
        # The width target is clamped into this window; directions whose target
        # would exceed max_log_std are too sloppy to be represented and are masked.
        self.width_min_log_std = width_min_log_std
        self.width_max_log_std = width_max_log_std
        # C3 fix (root cause: rank gradient drowned by NLL): class-balanced CE
        # upweights rare ranks (k_true is long-tailed across models) and label
        # smoothing prevents the rank head from collapsing onto the modal rank.
        # Class weights are lazily computed from the running label histogram.
        self.rank_class_balanced = rank_class_balanced
        self.rank_label_smoothing = rank_label_smoothing
        self._rank_label_counts: dict[int, float] = {}
        self._rank_weight_cache: Optional[torch.Tensor] = None
        self._rank_weight_n_classes: int = 0

    def forward(self, outputs: dict, batch: dict) -> dict:
        """Compute composite ISP loss."""
        loss_nll = self._compute_nll_loss(outputs, batch)
        loss_subspace = self._compute_subspace_loss(outputs, batch)
        loss_orth = self._compute_orthogonality_loss(outputs, batch)
        loss_rank = self._compute_rank_loss(outputs, batch)
        if self.lambda_calibration > 0.0:
            loss_calibration = self._compute_calibration_loss(outputs, batch)
        else:
            loss_calibration = self._zero(outputs)
        if self.lambda_width > 0.0:
            loss_width = self._compute_width_loss(outputs, batch)
        else:
            loss_width = self._zero(outputs)
        if self.lambda_mean > 0.0:
            loss_mean = self._compute_mean_loss(outputs, batch)
        else:
            loss_mean = self._zero(outputs)
        if self.lambda_mean_regressor > 0.0 and outputs.get("theta_mean_pred") is not None:
            loss_mean_reg = self._compute_mean_regressor_loss(outputs, batch)
        else:
            loss_mean_reg = self._zero(outputs)
        total_loss = (
            self.lambda_nll * loss_nll
            + self.lambda_subspace * loss_subspace
            + self.lambda_orth * loss_orth
            + self.lambda_rank * loss_rank
            + self.lambda_calibration * loss_calibration
            + self.lambda_width * loss_width
            + self.lambda_mean * loss_mean
            + self.lambda_mean_regressor * loss_mean_reg
        )
        return {
            "nll": loss_nll,
            "subspace": loss_subspace,
            "orth": loss_orth,
            "rank": loss_rank,
            "calibration": loss_calibration,
            "width": loss_width,
            "mean": loss_mean,
            "mean_reg": loss_mean_reg,
            "total": total_loss,
        }

    def _compute_nll_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Compute negative log likelihood from ISP output distributions."""
        if "theta_true" not in batch:
            return self._zero(outputs)
        isp_output = outputs["isp_output"]
        theta = batch["theta_true"].to(device=isp_output.U.device, dtype=isp_output.U.dtype)
        z_theta = outputs["z_theta"]
        z_D = outputs["z_D"]
        n_params = batch.get("n_params")
        if n_params is not None:
            n_params = n_params.to(isp_output.U.device)
        param_mask = isp_output.param_mask
        context = isp_output.qp_context if isp_output.qp_context is not None else isp_output.context
        U = isp_output.U
        theta_masked = theta.masked_fill(~param_mask, 0.0)
        alpha = torch.einsum("bnk,bn->bk", U, theta_masked)
        theta_parallel = torch.einsum("bnk,bk->bn", U, alpha)
        residual = (theta_masked - theta_parallel).masked_fill(~param_mask, 0.0)
        log_prob = isp_output.q_parallel.log_prob(alpha, context, isp_output.column_mask)
        # Paper back-off formula: q_perp variance = b * sigmoid(m) + eps with b the
        # *projected* prior variance diag(U_perp^T Sigma_prior U_perp), computed in
        # ISPHead.forward when the batch provides prior_cov.
        log_prob = log_prob + isp_output.q_perp.log_prob(
            residual, isp_output.perp_features, param_mask,
            prior_var_perp=isp_output.prior_var_perp,
        )
        return -log_prob.mean()

    def _compute_subspace_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Subspace alignment loss via projection-matrix Frobenius norm.

        L = ||P_pred - P_true||_F^2 / k, where P = U @ U^T.

        A principal-angle (Grassmannian distance) formulation was tried (v4)
        but caused training divergence: SVD backpropagation becomes numerically
        unstable as singular values approach 1 (subspaces align), and the
        instability accumulates over steps. The Frobenius form is numerically
        robust (v1-v3 trained stably with it). Its weakness is loose coupling
        to the evaluation metric (Grassmannian distance), so grassmannian may
        plateau even as this loss decreases — this is a known limitation to
        address in future work (e.g. a clamped principal-angle variant).
        """
        isp_output = outputs["isp_output"]
        if "fim_eigenvectors" not in batch or "effective_rank" not in batch:
            return torch.tensor(0.0, device=isp_output.U.device)

        U_pred = isp_output.U
        U_true = batch["fim_eigenvectors"].to(device=U_pred.device, dtype=U_pred.dtype)
        k_true = batch["effective_rank"].to(device=U_pred.device, dtype=torch.long)
        n_params = batch.get("n_params")
        if n_params is None:
            n_params = torch.full((U_pred.shape[0],), U_pred.shape[1], device=U_pred.device, dtype=torch.long)
        else:
            n_params = n_params.to(device=U_pred.device, dtype=torch.long)

        losses = []
        for batch_idx in range(U_pred.shape[0]):
            n_i = int(n_params[batch_idx].clamp(0, U_pred.shape[1]).item())
            k_i = int(torch.minimum(k_true[batch_idx], torch.tensor(min(n_i, U_pred.shape[-1]), device=U_pred.device)).item())
            if n_i == 0 or k_i == 0:
                continue
            pred = U_pred[batch_idx, :n_i, :k_i]
            true = U_true[batch_idx, :n_i, :k_i]
            p_pred = pred @ pred.T
            p_true = true @ true.T
            losses.append(torch.sum((p_pred - p_true) ** 2) / max(k_i, 1))
        if not losses:
            return torch.tensor(0.0, device=U_pred.device)
        return torch.stack(losses).mean()

    def _compute_orthogonality_loss(self, outputs: dict, batch: dict | None = None) -> torch.Tensor:
        """Compute mask-aware orthogonality loss for feasible subspace columns."""
        U = outputs["isp_output"].U
        n_params = None if batch is None else batch.get("n_params")
        if n_params is None:
            n_params = torch.full((U.shape[0],), U.shape[1], device=U.device, dtype=torch.long)
        else:
            n_params = n_params.to(device=U.device, dtype=torch.long)

        losses = []
        for batch_idx in range(U.shape[0]):
            n_i = int(n_params[batch_idx].clamp(0, U.shape[1]).item())
            k_i = min(n_i, U.shape[-1])
            if n_i == 0 or k_i == 0:
                continue
            ui = U[batch_idx, :n_i, :k_i]
            eye = torch.eye(k_i, device=U.device, dtype=U.dtype)
            losses.append(torch.sum((ui.T @ ui - eye) ** 2) / k_i)
        if not losses:
            return torch.tensor(0.0, device=U.device)
        return torch.stack(losses).mean()

    def _compute_rank_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Compute rank prediction loss.

        Uses cross-entropy on the gate logits when available (encourages the
        network to place probability mass on the correct rank rather than
        hedging across ranks), falling back to MSE on the continuous k
        prediction otherwise. Cross-entropy is the preferred form: the soft
        MSE variant lets the network satisfy the loss by spreading probability
        uniformly so that the expected value matches, without ever committing
        to a discrete rank, which empirically yields a nearly constant k_pred.
        """
        isp_output = outputs["isp_output"]
        if "effective_rank" not in batch:
            return torch.tensor(0.0, device=isp_output.k.device)

        k_true = batch["effective_rank"].to(device=isp_output.k.device, dtype=torch.long)
        n_params = batch.get("n_params")
        if n_params is None:
            max_feasible = torch.full_like(k_true, isp_output.U.shape[-1])
        else:
            max_feasible = torch.minimum(
                n_params.to(device=isp_output.k.device, dtype=torch.long),
                torch.full_like(k_true, isp_output.U.shape[-1]),
            )
        k_true = torch.clamp(k_true, min=torch.zeros_like(max_feasible), max=max_feasible)

        gate_logits = getattr(isp_output, "gate_logits", None)
        if gate_logits is not None:
            # Cross-entropy on the (k_eff+1)-way rank classification. Targets
            # beyond the logits' column count are clamped to the last column.
            n_classes = gate_logits.shape[-1]
            target = torch.clamp(k_true, max=n_classes - 1)

            if self.rank_class_balanced and self.training:
                # Update the running label histogram from this batch so rare
                # ranks (small k_true counts) get upweighted. Weights follow the
                # effective-number reweighting w_c = (1-beta)/(1-beta^n_c).
                self._update_rank_counts(target, n_classes)
                weight = self._rank_class_weights(n_classes, device=gate_logits.device,
                                                  dtype=gate_logits.dtype)
            else:
                weight = None

            return torch.nn.functional.cross_entropy(
                gate_logits, target, weight=weight,
                label_smoothing=self.rank_label_smoothing,
            )

        # Fallback: MSE on the continuous k prediction.
        k_pred = isp_output.k
        k_true_f = k_true.to(dtype=k_pred.dtype)
        return torch.mean((k_pred - k_true_f) ** 2)

    @torch.no_grad()
    def _update_rank_counts(self, target: torch.Tensor, n_classes: int) -> None:
        """Maintain a running histogram of rank labels for class balancing."""
        target_cpu = target.detach().cpu().tolist()
        for t in target_cpu:
            self._rank_label_counts[int(t)] = self._rank_label_counts.get(int(t), 0.0) + 1.0
        self._rank_weight_n_classes = max(self._rank_weight_n_classes, n_classes)

    def _rank_class_weights(self, n_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Effective-number class weights (Cui et al. 2019) for the long-tailed
        rank distribution. Rare ranks get larger weight so the rank head does
        not collapse onto the modal k (the failure mode behind constant k_pred).
        """
        beta = 0.999
        counts = torch.ones(n_classes, dtype=torch.float64)
        for c in range(n_classes):
            counts[c] = self._rank_label_counts.get(c, 1.0)
        counts = counts.clamp_min(1.0)
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        w = w / w.mean()  # normalize so mean weight ~ 1
        return w.to(device=device, dtype=dtype)

    def _compute_calibration_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Multivariate energy score / CRPS calibration loss.

        ES(P, theta_true) = E[||theta_hat - theta_true||] - 0.5 * E[||theta_hat - theta_hat'||]

        Strictly proper scoring rule for multivariate distributions; pulls
        posterior toward correct mean AND scale. Mitigates the "ISP std >>
        MCMC std" calibration gap observed on Boehm (~72x). Samples are drawn
        from the parallel/perpendicular distributions and reconstructed via
        theta_hat = U @ alpha + residual.
        """
        if "theta_true" not in batch:
            return self._zero(outputs)
        isp_output = outputs["isp_output"]
        K = self.calibration_n_samples
        U = isp_output.U
        device = U.device
        dtype = U.dtype
        theta = batch["theta_true"].to(device=device, dtype=dtype)
        param_mask = isp_output.param_mask
        context = isp_output.qp_context if isp_output.qp_context is not None else isp_output.context
        B, N, k_eff = U.shape

        # q_parallel.sample(cond, n_samples) -> (B, K, k_max). Truncate to k_eff.
        alpha_full = isp_output.q_parallel.sample(context, n_samples=K)
        alpha = alpha_full[..., :k_eff]  # (B, K, k_eff)
        # Reconstruct theta_parallel = U @ alpha for each sample. (B, N, k_eff) @ (B, K, k_eff)^T
        theta_parallel = torch.einsum("bnk,bsk->bsn", U, alpha)  # (B, K, N)

        # q_perp.sample(perp_features, n_samples) -> (B, K, N) with per-parameter scale
        residual = isp_output.q_perp.sample(isp_output.perp_features, n_samples=K)

        theta_hat = theta_parallel + residual  # (B, K, N)
        mask = param_mask.unsqueeze(1).to(dtype=dtype)  # (B, 1, N)
        theta_hat = theta_hat * mask
        theta_target = (theta * param_mask).unsqueeze(1)  # (B, 1, N)

        # Term 1: E[||theta_hat - theta||]
        diff_target = theta_hat - theta_target  # (B, K, N)
        term1 = torch.linalg.norm(diff_target, dim=-1).mean(dim=-1)  # (B,)

        # Term 2: 0.5 * E[||theta_hat_i - theta_hat_j||] using halves of K
        K2 = K // 2
        if K2 < 1:
            term2 = torch.zeros(B, device=device, dtype=dtype)
        else:
            a = theta_hat[:, :K2, :]
            b = theta_hat[:, K2:2 * K2, :]
            term2 = 0.5 * torch.linalg.norm(a - b, dim=-1).mean(dim=-1)

        es = term1 - term2  # (B,)
        return es.mean()

    def _compute_mean_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Posterior-mean supervision loss (C1 H1 — anti mode-collapse).

        Diagnosis (2026-06-29): v10c's q_parallel collapses to a wide, roughly
        isotropic Gaussian whose mean barely tracks theta_true (corr ≈ 0 across
        5 samples per model). NLL alone permits this lazy solution — a wide
        posterior centered near the prior mean covers theta_true without the
        flow having to move its mean with the data. This term directly
        supervises the posterior mean toward theta_true, breaking the collapse.

        Samples K posterior draws (reusing the calibration sampling path) and
        minimizes MSE between the sample mean and theta_true, masked to valid
        parameters. The sample mean is differentiable w.r.t. the flow's
        location parameters, so gradients flow into q_parallel (and q_perp's
        mean head) to pull the posterior center onto the data.
        """
        if "theta_true" not in batch:
            return self._zero(outputs)
        isp_output = outputs["isp_output"]
        K = self.calibration_n_samples
        U = isp_output.U
        device = U.device
        dtype = U.dtype
        theta = batch["theta_true"].to(device=device, dtype=dtype)
        param_mask = isp_output.param_mask
        context = isp_output.qp_context if isp_output.qp_context is not None else isp_output.context
        B, N, k_eff = U.shape

        # Sample posterior draws (same reconstruction as calibration loss).
        alpha = isp_output.q_parallel.sample(context, n_samples=K)[..., :k_eff]
        theta_parallel = torch.einsum("bnk,bsk->bsn", U, alpha)
        residual = isp_output.q_perp.sample(isp_output.perp_features, n_samples=K)
        theta_hat = (theta_parallel + residual) * param_mask.unsqueeze(1).to(dtype=dtype)

        # Posterior sample mean -> (B, N), supervised against theta_true.
        post_mean = theta_hat.mean(dim=1)  # (B, N)
        sq_err = ((post_mean - theta) ** 2) * param_mask.to(dtype=dtype)
        # Per-sample mean over valid params (divide by n_params, not N).
        n_valid = param_mask.to(dtype=dtype).sum(dim=-1).clamp_min(1.0)  # (B,)
        return (sq_err.sum(dim=-1) / n_valid).mean()

    def _compute_mean_regressor_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """MSE between regressor output and ground truth.

        Direction A v2 (theta_space): the regressor predicts theta directly (raw
        param space), supervised by MSE against theta_true. This avoids the
        model-specific U-rotation that made U-space alpha a hard cross-model
        target (per-model probe r=0.436, cross-model regressor r≈0). The frozen
        q_parallel still provides the calibrated posterior *shape*; at inference
        the predicted theta is projected to U-space for recentering.

        Legacy (U-space): regressor predicts alpha = U^T @ theta, supervised
        against the projected ground truth. Kept for backward compatibility.
        """
        isp_output = outputs["isp_output"]
        theta_mean_pred = outputs["theta_mean_pred"]  # (B, out_dim)
        dtype = theta_mean_pred.dtype
        param_mask = isp_output.param_mask  # (B, N)
        theta = batch["theta_true"]

        # Direction A v2: theta-space MSE (model's natural coordinates).
        if getattr(outputs, "theta_space", False) or theta_mean_pred.shape[-1] == theta.shape[-1]:
            theta_target = theta.to(dtype=dtype)
            n_params = batch.get("n_params")
            N = theta_target.shape[-1]
            pred = theta_mean_pred[..., :N]
            sq_err = ((pred - theta_target) ** 2) * param_mask.to(dtype=dtype)
            n_valid = param_mask.to(dtype=dtype).sum(dim=-1).clamp_min(1.0)
            return (sq_err.sum(dim=-1) / n_valid).mean()

        # Legacy U-space: alpha = U^T @ theta
        U = isp_output.U  # (B, N, k_max)
        column_mask = isp_output.column_mask  # (B, k_max)
        theta_masked = theta * param_mask.to(dtype=dtype)
        alpha_true = torch.bmm(U.transpose(1, 2), theta_masked.unsqueeze(-1)).squeeze(-1)  # (B, k_max)
        k_eff = int(column_mask.sum(dim=-1).max().item())
        pred = theta_mean_pred[..., :k_eff]
        alpha_true = alpha_true[..., :k_eff]
        col_mask = column_mask[..., :k_eff].to(dtype=dtype)
        sq_err = ((pred - alpha_true) ** 2) * col_mask
        n_valid = col_mask.sum(dim=-1).clamp_min(1.0)
        return (sq_err.sum(dim=-1) / n_valid).mean()

    def _compute_width_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """FIM-derived absolute-scale width supervision for q_parallel.

        Supervises q_parallel's per-coordinate log_std against the Laplace
        approximation of the (corrected) FIM, projected onto the predicted
        subspace U:

            target_log_std_i = -0.5 * log(eig_i(U^T FIM U))

        Unlike the earlier v1/v3 formulation, this matches the ABSOLUTE scale,
        not just the relative pattern. v3 centered both log_stds, which by
        construction discarded the absolute scale and left std_ratio stuck at
        the baseline 72x. With the FIM now computed consistently with the PEtab
        noise model (per-observable sigma, chain-rule to log10, noise-parameter
        Fisher term), the Laplace target lands in the model's reachable range
        for identifiable directions.

        Two safeguards keep the loss learnable despite the FIM's large dynamic
        range (sloppy models span ~15 orders of magnitude in eigenvalue):

          * **target clamp**: identifiable (large-eigenvalue / tight) targets
            below ``min_log_std`` are clamped to ``min_log_std`` so the model is
            pushed to its tightest representable width rather than to an
            unreachable value.
          * **sloppy-direction mask**: targets above ``max_log_std`` correspond
            to directions the model cannot represent as wide enough (its
            log_std is clamped). Supervising these would fight the model's own
            clamp, so they are masked out of the loss entirely. The identifiable
            directions (which actually determine the per-parameter marginal
            width) remain supervised.
        """
        isp_output = outputs.get("isp_output")
        if isp_output is None:
            return self._zero(outputs)
        fim = batch.get("fim")
        if fim is None:
            return self._zero(outputs)

        U = isp_output.U  # (B, N, k_eff)
        context = isp_output.qp_context if isp_output.qp_context is not None else isp_output.context
        B, N, k_eff = U.shape
        device = U.device
        dtype = U.dtype

        fim = fim.to(device=device, dtype=dtype)
        # Project FIM onto the identifiable subspace: FIM_proj = U^T @ FIM @ U
        fim_proj = torch.einsum("bnk,bnm,bml->bkl", U, fim, U)  # (B, k_eff, k_eff)
        fim_proj_sym = 0.5 * (fim_proj + fim_proj.transpose(-1, -2))
        try:
            eigvals = torch.linalg.eigvalsh(fim_proj_sym)  # (B, k_eff), ascending
        except Exception:
            return self._zero(outputs)

        # Target: log_std = -0.5 * log(eig). Floor eigvals at a batch-tied
        # positive value so log() is finite; the clamp+mask below handles the
        # resulting very-wide (sloppy) targets.
        global_max = eigvals.max()
        pos_floor = global_max.clamp_min(1.0) * 1e-9
        eig_pos = eigvals.clamp_min(pos_floor)
        log_eig = torch.log(eig_pos)
        target_log_std = -0.5 * log_eig  # (B, k_eff)

        # Predicted log_std from q_parallel (already clamped to the model range
        # inside ConditionalDiagonalGaussian.params). A flow backend has no
        # closed-form per-dim log_std (its .params() is a noisy Monte-Carlo
        # estimate); supervising it is meaningless and the flow is trained
        # against MCMC-like targets only via the (disabled) width loss anyway,
        # so skip the q_parallel term for flows and keep only q_perp.
        from iaapi.models.isp_head import ConditionalMAF
        if isinstance(isp_output.q_parallel, ConditionalMAF):
            return self._compute_perp_width_loss(outputs, batch, fim, min_ls, max_ls)
        _, pred_log_std = isp_output.q_parallel.params(context)  # (B, k_max)
        pred_log_std = pred_log_std[:, :k_eff]  # (B, k_eff)

        # Clamp identifiable (tight) targets down to the model's min; mask out
        # sloppy targets the model cannot represent.
        min_ls = float(self.width_min_log_std)
        max_ls = float(self.width_max_log_std)
        target_clamped = target_log_std.clamp_min(min_ls)
        # mask = 1 for supervised directions (target <= max), 0 for masked (too sloppy)
        mask = (target_log_std <= max_ls).to(dtype=dtype)

        # Absolute-scale MSE on the supervised directions only.
        sq = (pred_log_std - target_clamped) ** 2
        sq = torch.nan_to_num(sq, nan=0.0, posinf=0.0, neginf=0.0)
        masked = sq * mask
        denom = mask.sum().clamp_min(1.0)
        width_loss = masked.sum() / denom

        # C1 fix: also supervise q_perp's per-parameter log_std against the
        # marginal FIM diagonal. The q_parallel term above only constrains the
        # identifiable subspace; the sloppy/residual width (q_perp, per-parameter
        # Gaussian) was unsupervised and stayed ~prior-wide on sloppy models
        # (Bertozzi/Rahman SBC bias stuck at ~0.40 even after q_parallel shrank).
        # The per-parameter Laplace target is log_std_i = -0.5 * log(F[i,i]):
        # a small F[i,i] (sloppy direction) => wide posterior, matching the
        # marginal unidentifiability. This is the per-parameter analogue of the
        # subspace target, computed on FIM directly (no U projection needed).
        perp_loss = self._compute_perp_width_loss(outputs, batch, fim, min_ls, max_ls)
        return width_loss + perp_loss

    def _compute_perp_width_loss(
        self,
        outputs: dict,
        batch: dict,
        fim: torch.Tensor,
        min_ls: float,
        max_ls: float,
    ) -> torch.Tensor:
        """FIM-diagonal Laplace width supervision for q_perp (per-parameter).

        q_perp predicts an independent (mean, log_std) for every parameter node.
        Its log_std should match the marginal identifiability -0.5*log(F[i,i]):
        sloppy parameters (small F[i,i]) get wide posteriors, identifiable ones
        get tight posteriors. This mirrors the q_parallel subspace target but on
        the FIM diagonal, and is the missing supervision for sloppy models where
        q_parallel (which only sees the identifiable subspace) cannot constrain
        the many sloppy directions.
        """
        isp_output = outputs.get("isp_output")
        if isp_output is None:
            return torch.zeros((), device=fim.device, dtype=fim.dtype)
        param_mask = isp_output.param_mask  # (B, N) bool
        perp_features = isp_output.perp_features  # (B, N, 2*d_embed)
        B, N = param_mask.shape
        device = fim.device
        dtype = fim.dtype

        # Per-parameter FIM diagonal: F[i,i]. Floor at a batch-tied positive
        # value so log() is finite (degenerate/sloppy params have ~0 F[i,i]).
        diag = torch.diagonal(fim, dim1=-2, dim2=-1)  # (B, N)
        global_max = diag.amax()
        pos_floor = global_max.clamp_min(1.0) * 1e-12
        diag_pos = diag.clamp_min(pos_floor)
        target_log_std = -0.5 * torch.log(diag_pos)  # (B, N)

        # Predicted per-parameter log_std from q_perp (per-node, already clamped
        # to the model range inside ConditionalResidualGaussian.params).
        _, pred_log_std = isp_output.q_perp.params(perp_features)  # (B, N)
        pred_log_std = pred_log_std.to(dtype=dtype)
        target_log_std = target_log_std.to(dtype=dtype)

        # Same safeguards as the q_parallel term: clamp tight targets to the
        # model min; mask out too-sloppy targets the model cannot represent.
        target_clamped = target_log_std.clamp_min(min_ls)
        mask = ((target_log_std <= max_ls) & param_mask).to(dtype=dtype)
        sq = (pred_log_std - target_clamped) ** 2
        sq = torch.nan_to_num(sq, nan=0.0, posinf=0.0, neginf=0.0)
        masked = sq * mask
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom

    def _zero(self, outputs: dict) -> torch.Tensor:
        isp_output = outputs.get("isp_output")
        device = isp_output.U.device if isp_output is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)


class NormalizingFlowLoss(nn.Module):
    """Negative log-likelihood loss for normalizing flows."""

    def forward(self, samples: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss."""
        return -torch.mean(log_prob)
