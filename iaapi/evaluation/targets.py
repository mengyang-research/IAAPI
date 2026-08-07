"""Frozen identifiability target semantics (NCS plan task P2-01).

Replaces the ambiguous, threshold/gap-convention-dependent discrete ``k`` with a
set of auditable, frozen targets computed from a (prior-whitened) posterior
covariance plus per-parameter profile-likelihood classifications.

Targets:
  * prior-whitened posterior covariance  W = prior^{-1/2} post prior^{-1/2}
  * posterior contraction score          1 - (det(post)/det(prior))^{1/d}
  * continuous effective posterior dim   participation ratio of W's eigenvalues
  * optional discrete k                  count of whitened eigenvalues > tau
                                          (FIXED, predeclared threshold; not gap-based)
  * per-parameter PL identifiability     identifiable | practical_nonidentifiable
                                          | structural_nonidentifiable

Discipline (P2-01): FIM-derived labels are NOT independent. Every target record
carries a ``source`` ("PL"/"MCMC" independent, "FIM" circular) and ``independent``
flag; ``compute_targets`` refuses to mark an FIM source independent.

All continuous targets are invariant to congruence reparameterisation (parameter
units): whitened eigenvalues are the generalised eigenvalues of (post, prior),
and det(post)/det(prior) cancels the reparam determinant. Verified by tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"

# Frozen discrete-k threshold (relative to the largest whitened eigenvalue).
# Predeclared so the discrete k is reproducible and not gap-convention-dependent.
DEFAULT_TAU_REL = 1e-2
DEFAULT_WIDTH_THRESHOLD = 0.5  # CI width > this fraction of prior width => practical non-id

CAT_IDENTIFIABLE = "identifiable"
CAT_PRACTICAL = "practical_nonidentifiable"
CAT_STRUCTURAL = "structural_nonidentifiable"


@dataclass
class IdentifiabilityTargets:
    schema_version: str = SCHEMA_VERSION
    n_parameters: int = 0
    whitened_cov: Optional[np.ndarray] = None
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    # continuous
    contraction_score: float = 0.0
    effective_dimension: float = 0.0
    # discrete k (fixed threshold)
    k_threshold: int = 0
    threshold_tau: float = 0.0
    threshold_rule: str = "relative"
    # per-parameter PL classification
    param_classification: List[str] = field(default_factory=list)
    # provenance / independence
    source: str = "PL"          # "PL" | "MCMC" | "FIM"
    independent: bool = True    # False iff source == "FIM"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "n_parameters": self.n_parameters,
            "eigenvalues": list(self.eigenvalues),
            "contraction_score": self.contraction_score,
            "effective_dimension": self.effective_dimension,
            "k_threshold": self.k_threshold, "threshold_tau": self.threshold_tau,
            "threshold_rule": self.threshold_rule,
            "param_classification": list(self.param_classification),
            "source": self.source, "independent": self.independent,
        }


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #
def _symmetrise(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _matrix_inv_sqrt(m: np.ndarray) -> np.ndarray:
    """Inverse symmetric square root via eigendecomposition (PSD input).

    Uses pseudo-inverse for numerically-zero eigenvalues so a (near-)singular
    prior is handled gracefully rather than crashing.
    """
    m = _symmetrise(np.asarray(m, dtype=float))
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(vals > 1e-12, 1.0 / np.sqrt(vals), 0.0)
    return (vecs * inv_sqrt) @ vecs.T


def whiten_covariance(post_cov: np.ndarray, prior_cov: np.ndarray) -> np.ndarray:
    """Return W = prior^{-1/2} post prior^{-1/2} (prior-whitened posterior)."""
    post = _symmetrise(np.asarray(post_cov, dtype=float))
    pinv_sqrt = _matrix_inv_sqrt(prior_cov)
    return _symmetrise(pinv_sqrt @ post @ pinv_sqrt)


def _logdet_psd(m: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(_symmetrise(np.asarray(m, dtype=float)))
    vals = np.clip(vals, 0.0, None)
    return float(np.sum(np.log(vals + 1e-300)))


def contraction_score(post_cov: np.ndarray, prior_cov: np.ndarray) -> float:
    """1 - (det(post)/det(prior))^{1/d} in [0, 1].

    1 = posterior contracted to a point; 0 = no contraction vs prior.
    Unit-invariant: the reparam determinant cancels in the ratio.
    """
    post = np.asarray(post_cov, dtype=float)
    prior = np.asarray(prior_cov, dtype=float)
    d = post.shape[0]
    if d == 0:
        return 0.0
    log_ratio = _logdet_psd(post) - _logdet_psd(prior)
    ratio = math.exp(log_ratio / d)
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def effective_dimension(whitened_cov: np.ndarray) -> float:
    """Participation ratio (Σ λ)^2 / Σ λ^2 of whitened eigenvalues in [1, d]."""
    vals = np.linalg.eigvalsh(_symmetrise(np.asarray(whitened_cov, dtype=float)))
    vals = np.clip(vals, 0.0, None)
    s = vals.sum()
    if s <= 0:
        return 0.0
    ss = (vals ** 2).sum()
    if ss <= 0:
        return 0.0
    return float((s ** 2) / ss)


def k_from_threshold(
    whitened_eigenvalues: np.ndarray,
    tau: float = DEFAULT_TAU_REL,
    rule: str = "relative",
) -> Tuple[int, float]:
    """Count whitened eigenvalues above a FIXED threshold.

    rule="relative": count eig > tau * max(eig)  (default tau=1e-2).
    rule="absolute": count eig > tau.
    Returns (k, effective_tau_applied).
    """
    vals = np.asarray(whitened_eigenvalues, dtype=float)
    vals = np.clip(vals, 0.0, None)
    if vals.size == 0:
        return 0, float(tau)
    if rule == "relative":
        applied = float(tau * vals.max())
    elif rule == "absolute":
        applied = float(tau)
    else:
        raise ValueError(f"unknown rule: {rule}")
    k = int(np.sum(vals > applied))
    return k, applied


# --------------------------------------------------------------------------- #
# Per-parameter PL identifiability classification
# --------------------------------------------------------------------------- #
def classify_parameter(
    ci_lower: Optional[float],
    ci_upper: Optional[float],
    bound_lower: Optional[float],
    bound_upper: Optional[float],
    prior_width: Optional[float],
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
) -> str:
    """Classify a single parameter's identifiability from its PL confidence interval.

    * structural_nonidentifiable: CI is unbounded/infinite on >=1 side (flat PL).
    * practical_nonidentifiable: finite CI that hits a PEtab bound, or is wider
      than ``width_threshold`` * prior_width.
    * identifiable: finite CI strictly inside bounds and not overly wide.
    """
    inf = float("inf")
    lo = -inf if ci_lower is None else float(ci_lower)
    hi = inf if ci_upper is None else float(ci_upper)
    # structural: unbounded on either side
    if math.isinf(lo) or math.isinf(hi):
        return CAT_STRUCTURAL
    # practical: CI hits a finite bound
    if bound_lower is not None and lo <= float(bound_lower) + 0.0:
        if math.isclose(lo, float(bound_lower)):
            return CAT_PRACTICAL
    if bound_upper is not None and hi >= float(bound_upper) - 0.0:
        if math.isclose(hi, float(bound_upper)):
            return CAT_PRACTICAL
    # practical: too wide relative to prior
    if prior_width is not None and prior_width > 0:
        ci_width = hi - lo
        if ci_width > width_threshold * float(prior_width):
            return CAT_PRACTICAL
    return CAT_IDENTIFIABLE


def classify_parameters(
    cis: List[Tuple[Optional[float], Optional[float]]],
    bounds: List[Tuple[Optional[float], Optional[float]]],
    prior_widths: List[Optional[float]],
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
) -> List[str]:
    out = []
    for (lo, hi), (bl, bu), pw in zip(cis, bounds, prior_widths):
        out.append(classify_parameter(lo, hi, bl, bu, pw, width_threshold))
    return out


# --------------------------------------------------------------------------- #
# Top-level assembly
# --------------------------------------------------------------------------- #
def compute_targets(
    post_cov: np.ndarray,
    prior_cov: np.ndarray,
    source: str = "PL",
    pl_cis: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    prior_widths: Optional[List[Optional[float]]] = None,
    tau: float = DEFAULT_TAU_REL,
    rule: str = "relative",
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
) -> IdentifiabilityTargets:
    """Compute the frozen identifiability targets for one model.

    Args:
        post_cov: posterior covariance (d, d).
        prior_cov: prior covariance (d, d).
        source: "PL" | "MCMC" (independent) | "FIM" (circular, NOT independent).
        pl_cis/bounds/prior_widths: per-parameter PL CIs for classification
            (may be None => classification left empty).
    """
    if source not in ("PL", "MCMC", "FIM"):
        raise ValueError(f"unknown source: {source}")
    post = np.asarray(post_cov, dtype=float)
    prior = np.asarray(prior_cov, dtype=float)
    d = post.shape[0]
    W = whiten_covariance(post, prior)
    eig = np.linalg.eigvalsh(_symmetrise(W))[::-1]  # descending
    eff_dim = effective_dimension(W)
    contr = contraction_score(post, prior)
    k, applied_tau = k_from_threshold(eig, tau=tau, rule=rule)

    param_class: List[str] = []
    if pl_cis is not None and bounds is not None and prior_widths is not None:
        param_class = classify_parameters(pl_cis, bounds, prior_widths, width_threshold)

    independent = source != "FIM"
    return IdentifiabilityTargets(
        n_parameters=d, whitened_cov=W, eigenvalues=eig,
        contraction_score=round(contr, 8), effective_dimension=round(eff_dim, 8),
        k_threshold=k, threshold_tau=applied_tau, threshold_rule=rule,
        param_classification=param_class, source=source, independent=independent,
    )
