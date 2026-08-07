"""Geometry head + projector loss (NCS plan task P3-04).

Supervises PROJECTION MATRICES / subspaces (Grassmannian), not raw eigenvectors,
so the loss is invariant to eigenvector sign and to rotations within degenerate
subspaces. Adds a continuous effective-dimension prediction. FIM-derived labels
are auxiliary supervision only (source=FIM, NOT independent); primary claims are
evaluated against independent PL/MCMC targets elsewhere.

Projector (Grassmannian) loss:  ||P_pred P_pred^T - P_true P_true^T||_F^2  over
the valid parameter subspace, with P_pred orthonormalised via QR. Because the
loss depends only on the projector P P^T, it is invariant to:
  * sign:        (-P)(-P)^T = P P^T
  * rotations:   (P R)(P R)^T = P R R^T P^T = P P^T   for orthogonal R
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

SCHEMA_VERSION = "1.0"


def _orthonormalise(P: torch.Tensor) -> torch.Tensor:
    """Return Q with orthonormal columns spanning the column space of P (B, d, k).

    Uses QR; falls back to normalising columns if QR fails (e.g. rank-deficient).
    """
    try:
        Q, _ = torch.linalg.qr(P)  # Q: (B, d, k) with orthonormal cols
        return Q
    except Exception:
        norms = P.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return P / norms


def projector_loss(P_pred: torch.Tensor, P_true: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Grassmannian projector loss, sign/rotation invariant. Returns mean over batch.

    P_pred, P_true: (B, d, k). P_pred is orthonormalised internally; P_true is
    assumed to have orthonormal columns (top-k eigenvectors). ``mask`` (B, d)
    zeroes invalid parameter rows/cols of the projectors.
    """
    Q = _orthonormalise(P_pred)
    Pp = torch.matmul(Q, Q.transpose(-1, -2))        # (B, d, d) predicted projector
    Pt = torch.matmul(P_true, P_true.transpose(-1, -2))  # (B, d, d) true projector
    if mask is not None:
        m = mask
        Pp = Pp * m.unsqueeze(-1) * m.unsqueeze(-2)
        Pt = Pt * m.unsqueeze(-1) * m.unsqueeze(-2)
    diff = Pp - Pt
    # mean Frobenius^2 over the batch (sum over d,d, then mean over B)
    return (diff ** 2).sum(dim=(-1, -2)).mean()


def effective_dim_loss(eff_pred: torch.Tensor, eff_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true continuous effective dimension."""
    return ((eff_pred - eff_true) ** 2).mean()


class GeometryHead(nn.Module):
    """Predicts a low-rank projection matrix + continuous effective dimension.

    Args:
        d_context: width of the incoming geometry context.
        max_dim: max parameter dimension ( projector rows).
        k: number of subspace columns to predict.
        hidden: MLP hidden width.
    """

    def __init__(self, d_context: int, max_dim: int, k: int = 4, hidden: int = 128):
        super().__init__()
        self.max_dim = max_dim
        self.k = k
        self.proj_head = nn.Sequential(
            nn.Linear(d_context, hidden), nn.ReLU(),
            nn.Linear(hidden, max_dim * k),
        )
        self.eff_head = nn.Sequential(
            nn.Linear(d_context, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        B = context.shape[0]
        P = self.proj_head(context).reshape(B, self.max_dim, self.k)
        eff = self.eff_head(context).squeeze(-1).clamp_min(0.0)  # eff dim >= 0
        if mask is not None:
            P = P * mask.unsqueeze(-1)
        return {"projector": P, "effective_dim": eff}

    def loss(self, context: torch.Tensor, P_true: torch.Tensor, eff_true: torch.Tensor,
             mask: Optional[torch.Tensor] = None, lam_eff: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(context, mask)
        lp = projector_loss(out["projector"], P_true, mask)
        le = effective_dim_loss(out["effective_dim"], eff_true)
        total = lp + lam_eff * le
        return total, {"projector_loss": lp.detach(), "eff_dim_loss": le.detach(), "total": total.detach()}
