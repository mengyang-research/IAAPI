"""Variable-dimensional low-rank-plus-diagonal posterior baseline (NCS P3-03).

A stable posterior family working in PRIOR-WHITENED coordinates (prior ~ N(0, I))
with a per-parameter identifiability/confidence gate that backs SLOPPY directions
off toward the prior (mean offset -> 0, covariance -> prior) rather than an
arbitrary broad Gaussian.

Covariance:  Sigma = D + B B^T   (D = diagonal > 0, B low-rank), so PSD by
construction; a small jitter makes it PD. Prior backoff scales the learned offset
and concentration by a confidence c in [0,1]:
    mean_post = c * mean
    D_post    = c * D + (1-c) * prior_var
    B_post    = B * sqrt(c)            # low-rank also backs off
so c->0 (sloppy) => N(0, prior_var) = the prior.

Supports variable parameter dimension in one batch via a per-sample ``param_mask``
(pad to ``max_dim``; masked params do not contribute to log_prob / are not
sampled). log_prob uses the Woodbury identity (rank r << d) for O(d r^2) cost.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

SCHEMA_VERSION = "1.0"


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class LowRankDiagPosterior:
    """Low-rank + diagonal posterior in prior-whitened coords with prior backoff.

    Args:
        mean:    (B, D) raw mean offset (whitened coords).
        diag:    (B, D) diagonal variance (>= 0).
        B:       (B, D, r) low-rank factor.
        conf:    (B, D) identifiability confidence in [0,1] (per parameter).
        param_mask: (B, D) 1=valid parameter, 0=padding.
        prior_var: scalar prior variance in whitened coords (default 1.0).
        jitter:   PSD jitter added to the diagonal.
    """

    def __init__(self, mean: torch.Tensor, diag: torch.Tensor, B: torch.Tensor,
                 conf: torch.Tensor, param_mask: torch.Tensor,
                 prior_var: float = 1.0, jitter: float = 1e-4):
        self.mean = mean
        self.diag = diag.clamp_min(0.0)
        self.B = B
        self.conf = conf.clamp(0.0, 1.0)
        self.mask = param_mask
        self.prior_var = float(prior_var)
        self.jitter = float(jitter)
        # prior backoff
        c = self.conf
        self.mean_post = c * self.mean                       # (B, D)
        # floor keeps Dinv bounded (prevents Woodbury blow-up when the head emits
        # tiny variances in high-confidence directions); prior backoff keeps
        # D_post >= (1-c)*prior_var, so the floor only binds in pathological cases.
        _D = c * self.diag + (1.0 - c) * self.prior_var + self.jitter
        self.D_post = _D.clamp_min(0.05)                     # (B, D)
        # low-rank factor scales by conf (c^2 outer-product backoff; no sqrt => no
        # gradient singularity at c=0, and B_post is exactly 0 for masked/sloppy).
        self.B_post = self.B * c.unsqueeze(-1)               # (B, D, r)

    # ---- covariance ---------------------------------------------------- #
    def covariance(self) -> torch.Tensor:
        """Full (B, D, D) covariance (PSD; PD after jitter)."""
        D = self.D_post
        Bp = self.B_post
        cov = torch.diag_embed(D) + Bp @ Bp.transpose(-1, -2)
        # zero out padding rows/cols
        m = self.mask
        cov = cov * m.unsqueeze(-1) * m.unsqueeze(-2)
        return cov

    # ---- log_prob via Woodbury ---------------------------------------- #
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """log N(theta; mean_post, D_post + B_post B_post^T), summed over valid params.

        theta: (B, D) in whitened coords (masked params ignored).
        Returns: (B,) log-prob.
        """
        D = self.D_post.clamp_min(self.jitter)               # (B, D)
        Bp = self.B_post                                     # (B, D, r)
        m = self.mean_post                                  # (B, D)
        mask = self.mask                                    # (B, D)
        d = theta - m                                       # (B, D)
        d = d * mask
        Dinv = (1.0 / D) * mask                             # (B, D)
        # M = I_r + B^T Dinv B   (B, r, r): (B,D,r)^T @ (B,D,r) with Dinv scaling
        BD = Bp * Dinv.unsqueeze(-1)                        # (B, D, r)
        BTDinvB = torch.matmul(BD.transpose(-1, -2), Bp)    # (B, r, r)
        r = Bp.shape[-1]
        I = torch.eye(r, dtype=Bp.dtype, device=Bp.device).expand(Bp.shape[0], r, r)
        M = I + BTDinvB + self.jitter * I                   # jitter for PD stability
        # solve M z = B^T Dinv d (LU solve => stable gradients vs Cholesky)
        Dinv_d = Dinv * d                                   # (B, D)
        BtDinvd = torch.matmul(Bp.transpose(-1, -2), Dinv_d.unsqueeze(-1)).squeeze(-1)  # (B, r)
        z = torch.linalg.solve(M, BtDinvd.unsqueeze(-1)).squeeze(-1)   # (B, r)
        quad = (Dinv * (d * d)).sum(-1) - (BtDinvd * z).sum(-1)        # (B,)
        logdet = (torch.log(D) * mask).sum(-1) + torch.linalg.slogdet(M)[1]
        n_valid = mask.sum(-1).clamp_min(1.0)
        log2pi = math.log(2.0 * math.pi)
        lp = -0.5 * (n_valid * log2pi + logdet + quad)
        return lp

    # ---- sample -------------------------------------------------------- #
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples; returns (*sample_shape, B, D) in whitened coords, masked."""
        Bb, Dd = self.mean_post.shape
        r = self.B_post.shape[-1]
        shape = sample_shape + (Bb, Dd)
        eps1 = torch.randn(shape, dtype=self.mean_post.dtype, device=self.mean_post.device)
        eps2 = torch.randn(sample_shape + (Bb, r), dtype=self.mean_post.dtype, device=self.mean_post.device)
        std = self.D_post.clamp_min(self.jitter).sqrt()      # (B, D)
        # x = mean + std * eps1 + (B_post @ eps2)
        base = self.mean_post.unsqueeze(0) + std.unsqueeze(0) * eps1  # (S, B, D)
        low = torch.einsum("sbr,bdr->sbd", eps2, self.B_post)        # (S, B, D)
        x = base + low
        return x * self.mask.unsqueeze(0)


class LowRankPosteriorHead(nn.Module):
    """Maps a context vector -> posterior parameters (max_dim sized, masked).

    Outputs raw_mean, raw_diag (softplus), low-rank B, and confidence (sigmoid).
    ``param_mask`` selects the valid parameters per sample.
    """

    def __init__(self, d_context: int, max_dim: int, rank: int = 4, hidden: int = 128):
        super().__init__()
        self.max_dim = max_dim
        self.rank = rank
        self.mean_head = nn.Sequential(nn.Linear(d_context, hidden), nn.ReLU(), nn.Linear(hidden, max_dim))
        self.diag_head = nn.Sequential(nn.Linear(d_context, hidden), nn.ReLU(), nn.Linear(hidden, max_dim))
        self.B_head = nn.Sequential(nn.Linear(d_context, hidden), nn.ReLU(), nn.Linear(hidden, max_dim * rank))
        self.conf_head = nn.Sequential(nn.Linear(d_context, hidden), nn.ReLU(), nn.Linear(hidden, max_dim))

    def forward(self, context: torch.Tensor, param_mask: torch.Tensor) -> LowRankDiagPosterior:
        B = context.shape[0]
        mean = self.mean_head(context)                       # (B, max_dim)
        diag = torch.nn.functional.softplus(self.diag_head(context))  # >=0
        Blr = self.B_head(context).reshape(B, self.max_dim, self.rank)
        conf = torch.sigmoid(self.conf_head(context))        # (B, max_dim) in [0,1]
        return LowRankDiagPosterior(mean * param_mask, diag * param_mask,
                                    Blr * param_mask.unsqueeze(-1), conf * param_mask,
                                    param_mask)
