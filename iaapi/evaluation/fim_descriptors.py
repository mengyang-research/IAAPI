"""Frozen, prior-whitened, dimensionless FIM spectral descriptors (R0-03).

Replaces the defective 8-dim feature block (RISK-05) with a single, tested
module.  Every descriptor is:

  * **dimensionless** - no dependence on parameter units or noise scale;
  * **scale-invariant** - multiplying the FIM by a positive constant does not
    change any descriptor;
  * **prior/noise-aware** - computed on the prior-whitened FIM spectrum
    ``W = prior^{-1/2} FIM prior^{-1/2}``, so the fixed ``sigma=0.1`` defect
    is eliminated;
  * **finite** - robust to near-zero, near-singular and indefinite FIMs.

Quality flags (condition number, negative-eigenvalue mass) are returned
separately and must NOT be used as predictors; they are diagnostic only.

Frozen descriptor set (7 features, indices fixed):

| idx | name                      | formula                              |
|-----|---------------------------|--------------------------------------|
| 0   | spectral_entropy_norm     | H / log(n),  H = -sum p_i log p_i   |
| 1   | eff_rank_fraction         | (sum lam)^2 / (n sum lam^2)         |
| 2   | max_log_gap               | max(-Delta log10 lam)               |
| 3   | gap_position_norm         | argmax gap / (n-1)                  |
| 4   | spectral_curvature        | mean(Delta^2 log10 lam_norm)        |
| 5   | log_dynamic_range         | log10(lam_max / lam_min)            |
| 6   | top_r_information_fraction| sum_{i<=r} lam_i / sum lam, r=ceil(n/3) |

plus:

| 7   | above_threshold_fraction  | frac(lam > tau * lam_max), tau=1e-2 |

All 8 are returned as a vector; the caller selects which enter the Ridge via
``ProtocolConfig.primary_descriptor_idx``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Fixed threshold for above_threshold_fraction (prior-whitened eigenvalues).
DEFAULT_TAU = 1e-2

DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "spectral_entropy_norm",
    "eff_rank_fraction",
    "max_log_gap",
    "gap_position_norm",
    "spectral_curvature",
    "log_dynamic_range",
    "top_r_information_fraction",
    "above_threshold_fraction",
)


# --------------------------------------------------------------------------- #
# Prior whitening helpers (reuse the proven targets.py machinery)
# --------------------------------------------------------------------------- #
def _symmetrise(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _matrix_inv_sqrt(m: np.ndarray) -> np.ndarray:
    """Inverse symmetric square root via eigendecomposition (PSD input).

    Pseudo-inverse-safe for (near-)singular matrices.
    """
    m = _symmetrise(np.asarray(m, dtype=float))
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(vals > 1e-12, 1.0 / np.sqrt(vals), 0.0)
    return (vecs * inv_sqrt) @ vecs.T


def _prior_cov_from_bounds(param_bounds_log10: np.ndarray) -> np.ndarray:
    """Diagonal uniform-prior covariance in log10 space.

    For a uniform prior on [lower, upper], Var = (upper - lower)^2 / 12.
    """
    bounds = np.asarray(param_bounds_log10, dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    widths = np.where(widths > 1e-12, widths, 1.0)  # guard zero-width
    return np.diag(widths ** 2 / 12.0)


def _whitened_eigenvalues(
    fim: np.ndarray,
    prior_cov: Optional[np.ndarray] = None,
    param_bounds_log10: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Compute eigenvalues of the (optionally prior-whitened) FIM.

    Returns (eigenvalues_descending, whitened_flag, negative_eigenvalue_mass).
    """
    fim = _symmetrise(np.asarray(fim, dtype=float))
    whitened = False

    if prior_cov is not None:
        pinv_sqrt = _matrix_inv_sqrt(prior_cov)
        W = _symmetrise(pinv_sqrt @ fim @ pinv_sqrt)
        whitened = True
    elif param_bounds_log10 is not None:
        prior_cov = _prior_cov_from_bounds(param_bounds_log10)
        pinv_sqrt = _matrix_inv_sqrt(prior_cov)
        W = _symmetrise(pinv_sqrt @ fim @ pinv_sqrt)
        whitened = True
    else:
        W = fim

    raw_eigs = np.linalg.eigvalsh(W)
    pos_mask = raw_eigs > 0
    neg_mass = float(np.abs(raw_eigs[~pos_mask]).sum())
    pos_mass = float(raw_eigs[pos_mask].sum())
    neg_eig_mass = neg_mass / (neg_mass + pos_mass) if (neg_mass + pos_mass) > 0 else 0.0

    # Clip to non-negative for descriptor computation (negative eigenvalues are
    # numerical artefacts; their mass is reported as a quality flag).
    eigs = np.clip(raw_eigs, 0.0, None)
    eigs = np.sort(eigs)[::-1]  # descending
    return eigs, whitened, neg_eig_mass


# --------------------------------------------------------------------------- #
# Core descriptor computation
# --------------------------------------------------------------------------- #
def compute_fim_descriptors(
    fim: np.ndarray,
    prior_cov: Optional[np.ndarray] = None,
    param_bounds_log10: Optional[np.ndarray] = None,
    tau: float = DEFAULT_TAU,
) -> Dict[str, Any]:
    """Compute the frozen 8-dim prior-whitened FIM spectral descriptor vector.

    Parameters
    ----------
    fim : (d, d) array
        Fisher Information Matrix (symmetric).
    prior_cov : (d, d) array, optional
        Prior covariance matrix.  If provided, the FIM is whitened as
        ``W = prior^{-1/2} FIM prior^{-1/2}`` before descriptor extraction.
    param_bounds_log10 : (d, 2) array, optional
        Per-parameter (lower, upper) bounds in log10 space.  Used to build a
        diagonal uniform-prior covariance if ``prior_cov`` is None.
    tau : float
        Fixed threshold for ``above_threshold_fraction`` (default 1e-2).

    Returns
    -------
    dict with keys:
        ``descriptors`` : (8,) float array (the frozen feature vector)
        ``names`` : tuple of 8 descriptor names
        ``quality_flags`` : dict (condition_number, negative_eigenvalue_mass,
                          whitened, n_params) - diagnostic only, not predictors
        ``eigenvalues`` : (d,) array of (clipped) whitened eigenvalues descending
    """
    eigs, whitened, neg_mass = _whitened_eigenvalues(fim, prior_cov, param_bounds_log10)
    n = len(eigs)
    d = fim.shape[0]

    # Guard: all-zero eigenvalues (degenerate FIM)
    total = float(eigs.sum())
    if total <= 0 or n == 0:
        return {
            "descriptors": np.zeros(len(DESCRIPTOR_NAMES)),
            "names": DESCRIPTOR_NAMES,
            "quality_flags": {
                "condition_number": float("inf"),
                "negative_eigenvalue_mass": neg_mass,
                "whitened": whitened,
                "n_params": d,
            },
            "eigenvalues": eigs,
        }

    # ------------------------------------------------------------------ #
    # 0: Normalised spectral entropy  H / log(n)
    # ------------------------------------------------------------------ #
    p = eigs / total
    # Guard log(0) in p*log(p): p=0 contributes 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = float(-np.sum(p * log_p))
    entropy_norm = H / math.log(n) if n > 1 else 0.0

    # ------------------------------------------------------------------ #
    # 1: Effective rank fraction  (sum lam)^2 / (n * sum lam^2)
    # ------------------------------------------------------------------ #
    sum_sq = float((eigs ** 2).sum())
    eff_rank_frac = (total ** 2) / (n * sum_sq) if sum_sq > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 2-4: Log10 spectral gaps and curvature
    # ------------------------------------------------------------------ #
    log_eigs = np.log10(np.maximum(eigs, 1e-300))
    log_eigs_norm = log_eigs - log_eigs[0]  # normalise max to 0

    if n > 1:
        gaps = -np.diff(log_eigs)  # positive = drops (descending spectrum)
        max_log_gap = float(gaps.max())
        gap_pos = float(np.argmax(gaps)) / (n - 1)
    else:
        max_log_gap = 0.0
        gap_pos = 0.0

    if n > 2:
        # True second difference of the normalised log spectrum
        curvature = float(np.mean(np.diff(log_eigs_norm, n=2)))
    else:
        curvature = 0.0

    # ------------------------------------------------------------------ #
    # 5: Log dynamic range  log10(lam_max / lam_min)
    # ------------------------------------------------------------------ #
    lam_min = float(eigs[eigs > 0].min()) if np.any(eigs > 0) else 1e-300
    lam_max = float(eigs[0])
    log_range = math.log10(max(lam_max / max(lam_min, 1e-300), 1.0))

    # ------------------------------------------------------------------ #
    # 6: Top-r information fraction  (r = ceil(n/3))
    # ------------------------------------------------------------------ #
    r = max(1, math.ceil(n / 3))
    top_r_frac = float(eigs[:r].sum() / total)

    # ------------------------------------------------------------------ #
    # 7: Above-threshold fraction  (tau * max eigenvalue)
    # ------------------------------------------------------------------ #
    threshold = tau * lam_max
    above_frac = float(np.mean(eigs > threshold))

    descriptors = np.array([
        entropy_norm,
        eff_rank_frac,
        max_log_gap,
        gap_pos,
        curvature,
        log_range,
        top_r_frac,
        above_frac,
    ])

    # Quality flags (diagnostic only - must NOT enter the Ridge)
    cond = lam_max / max(lam_min, 1e-300) if lam_min > 0 else float("inf")
    quality_flags = {
        "condition_number": float(cond),
        "negative_eigenvalue_mass": neg_mass,
        "whitened": whitened,
        "n_params": d,
    }

    return {
        "descriptors": descriptors,
        "names": DESCRIPTOR_NAMES,
        "quality_flags": quality_flags,
        "eigenvalues": eigs,
    }
