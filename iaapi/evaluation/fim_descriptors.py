"""Versioned dimensionless FIM descriptors.

Version 2.0 uses the manuscript top-3 fraction and whole-spectrum decay slope.
For theta = mu + scale**(1/2) y, the information matrix in y coordinates is
W = scale**(1/2) FIM scale**(1/2). This is a coordinate normalization, not
a claim that the scale matrix equals an empirical-prior covariance.

Version 1.0 retains the old inverse-scale convention and old final two
features solely for explicit legacy checkpoint compatibility. Do not mix
versions in training and evaluation. Quality flags are not predictors.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Threshold used only by legacy version 1.0.
DEFAULT_TAU = 1e-2

FEATURE_VERSION = "2.0"

LEGACY_DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "spectral_entropy_norm",
    "eff_rank_fraction",
    "max_log_gap",
    "gap_position_norm",
    "spectral_curvature",
    "log_dynamic_range",
    "top_r_information_fraction",
    "above_threshold_fraction",
)

DESCRIPTOR_NAMES = LEGACY_DESCRIPTOR_NAMES[:6] + (
    "top_3_information_fraction", "decay_slope",
)

def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    matrix = _symmetrise(np.asarray(matrix, dtype=float))
    values, vectors = np.linalg.eigh(matrix)
    tolerance = 1e-12 * max(float(np.max(np.abs(values))), 1.0)
    if np.any(values < -tolerance):
        raise ValueError("Scale matrix must be positive semidefinite")
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T



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


def _prior_cov_from_bounds(param_bounds_log10: np.ndarray, *, legacy: bool = False) -> np.ndarray:
    """Diagonal uniform-prior covariance in log10 space.

    For a uniform prior on [lower, upper], Var = (upper - lower)^2 / 12.
    """
    bounds = np.asarray(param_bounds_log10, dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    if legacy:
        widths = np.where(widths > 1e-12, widths, 1.0)  # explicit old convention
    return np.diag(widths ** 2 / 12.0)


def _whitened_eigenvalues(
    fim: np.ndarray,
    prior_cov: Optional[np.ndarray] = None,
    param_bounds_log10: Optional[np.ndarray] = None,
    feature_version: str = FEATURE_VERSION,
) -> Tuple[np.ndarray, bool, float]:
    """Compute eigenvalues under the explicitly versioned FIM scale transform.

    Returns (eigenvalues_descending, whitened_flag, negative_eigenvalue_mass).
    """
    fim = _symmetrise(np.asarray(fim, dtype=float))
    whitened = False

    if prior_cov is not None:
        scale_transform = (_matrix_inv_sqrt(prior_cov) if feature_version == "1.0"
                     else _matrix_sqrt(prior_cov))
        W = _symmetrise(scale_transform @ fim @ scale_transform)
        whitened = True
    elif param_bounds_log10 is not None:
        prior_cov = _prior_cov_from_bounds(param_bounds_log10, legacy=feature_version == "1.0")
        scale_transform = _matrix_inv_sqrt(prior_cov)
        W = _symmetrise(scale_transform @ fim @ scale_transform)
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
    *, feature_version: str = FEATURE_VERSION,
) -> Dict[str, Any]:
    """Compute the versioned 8-dimensional FIM spectral descriptor vector.

    Parameters
    ----------
    fim : (d, d) array
        Fisher Information Matrix (symmetric).
    prior_cov : (d, d) array, optional
        Prior covariance matrix.  If provided, the FIM is whitened as
        ``W = scale^{1/2} FIM scale^{1/2}`` in version 2.0.
        The API name is retained for compatibility; this may be a bound scale.
    param_bounds_log10 : (d, 2) array, optional
        Per-parameter (lower, upper) bounds in log10 space.  Used to build a
        diagonal uniform-prior covariance if ``prior_cov`` is None.
    tau : float
        Threshold used only by the explicit legacy version 1.0.
    feature_version : str
        "2.0" (default) or "1.0" for legacy feature/checkpoint compatibility.
        Serialize this value with every fitted Ridge model.

    Returns
    -------
    dict with keys:
        ``descriptors`` : (8,) float array (the frozen feature vector)
        ``names`` : tuple of 8 descriptor names
        ``quality_flags`` : dict (condition_number, negative_eigenvalue_mass,
                          whitened, n_params) - diagnostic only, not predictors
        ``eigenvalues`` : (d,) array of (clipped) whitened eigenvalues descending
    """
    if feature_version not in {"1.0", "2.0"}:
        raise ValueError("Unknown FIM descriptor feature_version")
    names = LEGACY_DESCRIPTOR_NAMES if feature_version == "1.0" else DESCRIPTOR_NAMES
    fim = np.asarray(fim, dtype=float)
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1] or not np.all(np.isfinite(fim)):
        raise ValueError("FIM must be a finite square matrix")
    if prior_cov is not None:
        prior_cov = np.asarray(prior_cov, dtype=float)
        if prior_cov.shape != fim.shape or not np.all(np.isfinite(prior_cov)):
            raise ValueError("Scale matrix must be finite and match the FIM")
    if param_bounds_log10 is not None:
        bounds = np.asarray(param_bounds_log10, dtype=float)
        if bounds.shape != (len(fim), 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Bounds must be finite ordered intervals matching the FIM")
    eigs, whitened, neg_mass = _whitened_eigenvalues(fim, prior_cov, param_bounds_log10, feature_version)
    n = len(eigs)
    d = fim.shape[0]

    # Guard: all-zero eigenvalues (degenerate FIM)
    total = float(eigs.sum())
    if total <= 0 or n == 0:
        return {
            "descriptors": np.zeros(len(DESCRIPTOR_NAMES)),
            "names": names,
            "feature_version": feature_version,
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
    if feature_version == "2.0":
        # Relative clipping preserves scale invariance for zero eigenvalues.
        log_eigs = np.log10(np.maximum(eigs / eigs[0], 1e-300))
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
    log_range = max(math.log10(lam_max) - math.log10(max(lam_min, 1e-300)), 0.0)

    # ------------------------------------------------------------------ #
    # Final two features are selected by explicit feature version.
    # ------------------------------------------------------------------ #
    r = max(1, math.ceil(n / 3))
    information_fraction = float(eigs[:r].sum() / total)

    # ------------------------------------------------------------------ #
    # 7: Above-threshold fraction  (tau * max eigenvalue)
    # ------------------------------------------------------------------ #
    threshold = tau * lam_max
    final_feature = float(np.mean(eigs > threshold))
    if feature_version == "2.0":
        information_fraction = float(eigs[:min(3, n)].sum() / total)
        index = np.arange(1, n + 1, dtype=float)
        centered = index - index.mean()
        final_feature = float(-np.dot(centered, log_eigs) / np.dot(centered, centered)) if n > 1 else 0.0

    descriptors = np.array([
        entropy_norm,
        eff_rank_frac,
        max_log_gap,
        gap_pos,
        curvature,
        log_range,
        information_fraction,
        final_feature,
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
        "names": names,
        "feature_version": feature_version,
        "quality_flags": quality_flags,
        "eigenvalues": eigs,
    }
