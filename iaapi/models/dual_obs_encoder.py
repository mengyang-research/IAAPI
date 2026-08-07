"""Dual observation encoder (NCS plan task P3-02).

Two observation branches producing SEPARATE contexts:

  * **shape branch** — per-observable z-score normalisation (amplitude-invariant)
    + scale-free shape descriptors (curvature, log-span, log-density). Produces
    ``geometry_context`` for posterior-geometry / identifiability prediction.
  * **scale branch** — preserves amplitude, time scale, condition and noise
    metadata (raw values + log-amplitude + supplied metadata). Produces
    ``scale_context`` for posterior-LOCATION (theta-mean) prediction.

The combined ``posterior_context`` = concat(geometry, scale) feeds the posterior
location head. Explicit ``valid_mask`` handles missing observations so they do
not contaminate the per-observable statistics.

Design rules (P3-02): shape invariance is ARCHITECTURAL (z-score over valid
obs only), so the shape context is invariant to amplitude scaling by
construction; the scale context is NOT. Separate contexts keep geometry and
posterior-location information disentangled.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

SCHEMA_VERSION = "1.0"


def _valid_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Mean of x over dim using only valid (mask>0) entries."""
    m = mask.to(x.dtype)
    s = (x * m).sum(dim=dim)
    cnt = m.sum(dim=dim).clamp_min(eps)
    return s / cnt


def _valid_std(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    mean = _valid_mean(x, mask, dim=dim, eps=eps)
    var = _valid_mean((x - mean.unsqueeze(dim)) ** 2, mask, dim=dim, eps=eps)
    return var.clamp_min(eps).sqrt()


class DualObservationEncoder(nn.Module):
    """Shape (invariant) + scale (amplitude-aware) dual observation encoder.

    Args:
        n_obs: number of observables (last dim of the observation tensor).
        d_context: width of each context vector.
        n_meta: number of scalar metadata features (amplitude/time/condition/noise)
            fed to the scale branch (0 => none).
        hidden: MLP hidden width.
    Inputs (forward):
        obs:    (B, T, n_obs) observation trajectories.
        times:  (B, T) time points.
        mask:   (B, T, n_obs) validity mask (1=valid, 0=missing).
        meta:   (B, n_meta) optional metadata for the scale branch.
    """

    def __init__(self, n_obs: int, d_context: int = 64, n_meta: int = 0, hidden: int = 128):
        super().__init__()
        self.n_obs = n_obs
        self.d_context = d_context
        self.n_meta = n_meta
        # Shape branch input: per-obs z-scored values (n_obs) + shape descriptors
        # (curvature energy, log-span, log-density => 3*n_obs) => 4*n_obs.
        shape_in = 4 * n_obs
        self.shape_mlp = nn.Sequential(
            nn.Linear(shape_in, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, d_context),
        )
        # Scale branch input: raw amplitude stats (n_obs: mean, max, log-max) +
        # metadata (n_meta). 3*n_obs + n_meta.
        scale_in = 3 * n_obs + n_meta
        self.scale_mlp = nn.Sequential(
            nn.Linear(scale_in, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, d_context),
        )

    def _shape_features(self, obs: torch.Tensor, times: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """Amplitude-invariant shape features (4*n_obs)."""
        # z-score per observable over valid time points: (B, n_obs)
        # obs: (B, T, n_obs); mask: (B, T, n_obs)
        mean = _valid_mean(obs, mask, dim=1)              # (B, n_obs)
        std = _valid_std(obs, mask, dim=1)                # (B, n_obs)
        z = (obs - mean.unsqueeze(1)) / std.unsqueeze(1)  # (B, T, n_obs)
        z = z * mask  # zero out missing
        # curvature energy: sum of squared second differences (scale-free after z)
        dz2 = torch.zeros_like(z)
        if z.shape[1] >= 3:
            dz2[:, 1:-1, :] = z[:, 2:, :] - 2 * z[:, 1:-1, :] + z[:, :-2, :]
        curvature = (dz2 ** 2 * mask).mean(dim=1)         # (B, n_obs)
        # log time span and log density per observable (scale-free)
        tspan = (times[:, -1] - times[:, 0]).clamp_min(1e-6)  # (B,)
        n_valid = mask.sum(dim=1).clamp_min(1.0)              # (B, n_obs)
        log_span = torch.log10(tspan + 1e-6).unsqueeze(1).expand(-1, self.n_obs)
        log_density = torch.log10(n_valid / tspan.unsqueeze(1).clamp_min(1e-6) + 1e-6)
        # z-scored mean amplitude shape (z of the mean) — still amplitude-invariant
        z_mean = (mean - mean.mean(dim=1, keepdim=True)) / (mean.std(dim=1, keepdim=True) + 1e-6)
        feats = torch.cat([z_mean, curvature, log_span, log_density], dim=1)  # (B, 4*n_obs)
        return feats

    def _scale_features(self, obs: torch.Tensor, mask: torch.Tensor,
                        meta: Optional[torch.Tensor]) -> torch.Tensor:
        """Amplitude-preserving scale features (3*n_obs + n_meta)."""
        mean = _valid_mean(obs, mask, dim=1)              # (B, n_obs)
        # max over valid time per observable; 0 where an observable is fully missing
        very_neg = torch.finfo(obs.dtype).min
        obs_masked = obs.masked_fill(mask < 0.5, very_neg)
        mx = obs_masked.max(dim=1).values                 # (B, n_obs)
        cnt = mask.sum(dim=1)                             # (B, n_obs)
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        log_mx = torch.sign(mx) * torch.log1p(mx.abs())   # signed log-amplitude
        feats = [mean, mx, log_mx]
        if meta is not None and self.n_meta > 0:
            feats.append(meta)
        return torch.cat(feats, dim=1)

    def forward(self, obs: torch.Tensor, times: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                meta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, n = obs.shape
        assert n == self.n_obs, f"obs last dim {n} != n_obs {self.n_obs}"
        if mask is None:
            mask = torch.ones_like(obs)
        if times is None:
            times = torch.arange(T, dtype=obs.dtype, device=obs.device).expand(B, T)
        shape_ctx = self.shape_mlp(self._shape_features(obs, times, mask))   # (B, d)
        scale_ctx = self.scale_mlp(self._scale_features(obs, mask, meta))     # (B, d)
        posterior_ctx = torch.cat([shape_ctx, scale_ctx], dim=1)              # (B, 2d)
        return {
            "geometry": shape_ctx,        # for identifiability/geometry head
            "scale": scale_ctx,           # amplitude-aware
            "posterior_location": posterior_ctx,  # combined for theta-mean
        }
