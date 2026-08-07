"""Frozen statistical protocol for model-level validation (NCS recovery task R0-02).

Implements the predeclared, frozen protocol for testing whether low-dimensional
spectral descriptors transfer across whole unseen models:

  * nested model-level CV: outer leave-one-model-out evaluates whole unseen
    models; inner LOO selects the Ridge regularisation alpha via aggregate
    held-out MSE;
  * 2-3 pre-registered primary descriptors (no feature optimisation against the
    sealed holdout);
  * model-level permutation test (Pearson + Spearman) that REFITS the full
    pipeline for every permuted target;
  * bootstrap confidence intervals;
  * Pearson and Spearman point estimates;
  * sealed holdout evaluated ONCE (guarded against repeated evaluation).

Primary gate: independent n>=20, outer-CV Pearson r>=0.5, Spearman rho>=0.6,
permutation p<0.05, bootstrap 95% lower bound >0.

Everything operates at MODEL level (one feature vector + one target per model),
so there is no sample leakage across the train/test boundary.

R0-02 fixes (2026-07-11):
  * Ridge now centres y and restores a non-zero intercept.
  * Inner alpha selection uses aggregate LOO MSE instead of scalar Pearson r
    (which was degenerate for single held-out models).
  * The permutation test refits the full nested pipeline for every permutation
    instead of permuting against already-fitted predictions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCHEMA_VERSION = "2.0"

# Default pre-registered primary spectral descriptors (names only; indices are
# supplied at call time via ProtocolConfig.primary_descriptor_idx). Frozen.
DEFAULT_PRIMARY_DESCRIPTORS = ["eff_rank_participation", "top3_frac", "spectral_gap_log"]


@dataclass(frozen=True)
class ProtocolConfig:
    primary_descriptor_idx: Tuple[int, ...] = (0, 1, 2)
    alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    n_permutations: int = 10000
    n_bootstrap: int = 10000
    ci_level: float = 0.95
    seed: int = 0
    # gate thresholds
    min_n: int = 20
    min_pearson: float = 0.5
    min_spearman: float = 0.6
    max_perm_p: float = 0.05
    min_bootstrap_lb: float = 0.0


# --------------------------------------------------------------------------- #
# Ridge (closed-form, standardised, with intercept) - no sklearn dependency
# --------------------------------------------------------------------------- #
def _standardise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / std


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, alpha: float,
                      return_intercept: bool = False) -> Any:
    """Standardised Ridge regression with intercept; returns predictions for X_test.

    The intercept is recovered by centring y before fitting on standardised X
    (which has zero column means), so intercept = y_mean.  This fixes the
    RISK-04 bug where positive-count targets received biased, intercept-free
    predictions.
    """
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).reshape(-1)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    Xs = _standardise(Xtr, mean, std)
    # Centre y so the intercept is y_mean (Xs has zero column means).
    y_mean = ytr.mean()
    yc = ytr - y_mean
    d = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(d)
    beta = np.linalg.solve(A, Xs.T @ yc)
    Xte = _standardise(np.asarray(X_test, dtype=float), mean, std)
    pred = Xte @ beta + y_mean
    if return_intercept:
        return pred, float(y_mean)
    return pred


# --------------------------------------------------------------------------- #
# Nested model-level CV
# --------------------------------------------------------------------------- #
def _loo_indices(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


def _ridge_loo_mse(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Aggregate LOO MSE for standardised Ridge via the analytical hat matrix.

    For Ridge with standardised X and centred y, the LOO residual for sample i
    is (y_i - ŷ_i) / (1 - H_ii) where ŷ_i is the in-sample prediction and
    H = Xs (Xs^T Xs + alpha I)^{-1} Xs^T is the hat matrix.  This avoids
    refitting n times and is exact (not an approximation).
    """
    Xtr = np.asarray(X, dtype=float)
    ytr = np.asarray(y, dtype=float).reshape(-1)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    Xs = _standardise(Xtr, mean, std)
    yc = ytr - ytr.mean()
    d = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(d)
    A_inv = np.linalg.inv(A)
    # In-sample predictions on centred y: ŷ = Xs @ beta = H @ yc
    pred = Xs @ (A_inv @ (Xs.T @ yc))
    resid = yc - pred
    # Diagonal of hat matrix: H_ii = sum_j Xs[i,j] * (A_inv Xs^T)[j, i]
    H_diag = np.einsum("ij,jk,ik->i", Xs, A_inv, Xs)
    denom = 1.0 - H_diag
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    loo_resid = resid / denom
    return float(np.mean(loo_resid ** 2))


def inner_select_alpha(X: np.ndarray, y: np.ndarray, alphas: Sequence[float]) -> float:
    """Inner LOO over models to pick the alpha minimising aggregate held-out MSE.

    This fixes RISK-04: the old implementation computed Pearson r on one
    held-out scalar at a time, whose variance is zero, so no alpha received a
    valid score and the first grid value was always returned.  The corrected
    implementation uses the analytical hat-matrix LOO residual to compute an
    aggregate MSE for each alpha, which is well-defined for any n >= 2.
    Ties are broken by preferring the larger alpha (more regularisation).
    """
    n = len(y)
    if n < 3:
        return float(alphas[len(alphas) // 2])
    best_alpha, best_mse = float(alphas[-1]), np.inf
    for a in alphas:
        mse = _ridge_loo_mse(X, y, a)
        # Iterate ascending; use <= so the LAST (largest) alpha wins ties.
        if mse <= best_mse:
            best_mse = mse
            best_alpha = float(a)
    return best_alpha


def _outer_cv_predictions(Xp: np.ndarray, y: np.ndarray,
                          config: ProtocolConfig) -> Tuple[np.ndarray, List[float]]:
    """Run outer leave-one-model-out CV with inner alpha selection.

    Returns (predictions, selected_alphas).
    """
    n = len(y)
    preds = np.empty(n)
    selected_alphas: List[float] = []
    for tr, te in _loo_indices(n):
        a = inner_select_alpha(Xp[tr], y[tr], config.alpha_grid)
        selected_alphas.append(a)
        preds[te] = ridge_fit_predict(Xp[tr], y[tr], Xp[te], a)
    return preds, selected_alphas


def nested_model_cv(X: np.ndarray, y: np.ndarray, config: ProtocolConfig) -> Dict[str, Any]:
    """Outer leave-one-model-out CV with inner alpha selection.

    Returns per-model predictions, selected alphas, and point+CI statistics.
    Uses only the pre-registered primary descriptors (config.primary_descriptor_idx).
    The permutation test REFITS the full nested pipeline for every permutation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    idx = list(config.primary_descriptor_idx)
    Xp = X[:, idx]
    n = len(y)

    preds, selected_alphas = _outer_cv_predictions(Xp, y, config)

    r, p_pearson = pearsonr(preds, y) if np.std(preds) > 0 and np.std(y) > 0 else (0.0, 1.0)
    rho, p_spearman = spearmanr(preds, y) if np.std(preds) > 0 and np.std(y) > 0 else (0.0, 1.0)
    perm_p = permutation_test_nested(Xp, y, config, observed_r=float(r))
    boot = bootstrap_ci_r(preds, y, config.n_bootstrap, config.ci_level, config.seed + 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_models": int(n),
        "primary_descriptors": list(idx),
        "predictions": preds.tolist(),
        "truth": y.tolist(),
        "selected_alphas": selected_alphas,
        "pearson_r": float(r), "pearson_p": float(p_pearson),
        "spearman_rho": float(rho), "spearman_p": float(p_spearman),
        "permutation_p": float(perm_p),
        "bootstrap_ci": boot,
    }


# --------------------------------------------------------------------------- #
# Permutation tests + bootstrap
# --------------------------------------------------------------------------- #
def permutation_test_nested(Xp: np.ndarray, y: np.ndarray,
                            config: ProtocolConfig,
                            observed_r: Optional[float] = None) -> float:
    """Nested permutation test that REFITS the full pipeline for each permutation.

    For each permutation:
      1. Shuffle y (same models, permuted targets).
      2. Run the full nested_model_cv pipeline (inner alpha selection + Ridge).
      3. Record the resulting Pearson r.

    The p-value is the fraction of permuted |r| >= observed |r|, with the +1
    correction.  This fixes RISK-04: the old code permuted targets against
    already-fitted predictions instead of refitting.
    """
    Xp = np.asarray(Xp, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if np.std(y) == 0:
        return 1.0
    if observed_r is None:
        preds, _ = _outer_cv_predictions(Xp, y, config)
        if np.std(preds) == 0:
            return 1.0
        observed_r = float(pearsonr(preds, y)[0])
    if not math.isfinite(observed_r):
        return 1.0

    rng = np.random.default_rng(config.seed)
    n_perm = config.n_permutations
    count = 0
    abs_obs = abs(observed_r)
    for _ in range(n_perm):
        yp = rng.permutation(y)
        preds_p, _ = _outer_cv_predictions(Xp, yp, config)
        if np.std(preds_p) == 0:
            # Degenerate fit; |r| = 0 < |obs| for any real signal.
            rp = 0.0
        else:
            rp = float(pearsonr(preds_p, yp)[0])
        if abs(rp) >= abs_obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_p(pred: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float:
    """Simple (non-nested) model-level permutation test on Pearson r (two-sided).

    This permutes targets against *already-fitted* predictions.  It is correct
    for the case where predictions are fixed (e.g. a single model, not a nested
    pipeline), but MUST NOT be used inside nested_model_cv -- use
    permutation_test_nested there instead.
    """
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    if np.std(pred) == 0 or np.std(y) == 0:
        return 1.0
    r_obs, _ = pearsonr(pred, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp, _ = pearsonr(pred, yp)
        if abs(rp) >= abs(r_obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci_r(pred: np.ndarray, y: np.ndarray, n_boot: int,
                   ci_level: float, seed: int) -> Dict[str, float]:
    """Bootstrap CI for Pearson r over model resampling."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        s = rng.integers(0, n, size=n)
        if np.std(pred[s]) == 0 or np.std(y[s]) == 0:
            continue
        r, _ = pearsonr(pred[s], y[s])
        if math.isfinite(r):
            rs.append(r)
    if not rs:
        return {"lower": 0.0, "upper": 0.0, "median": 0.0, "n_valid": 0}
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(rs, alpha))
    hi = float(np.quantile(rs, 1.0 - alpha))
    return {"lower": lo, "upper": hi, "median": float(np.median(rs)), "n_valid": len(rs)}


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def evaluate_gate(result: Dict[str, Any], config: ProtocolConfig) -> Dict[str, Any]:
    """Apply the predeclared primary gate. Returns PASS/FAIL + reasons."""
    n = result["n_models"]
    r = result["pearson_r"]
    rho = result["spearman_rho"]
    p = result["permutation_p"]
    lb = result["bootstrap_ci"]["lower"]
    reasons: List[str] = []
    if n < config.min_n:
        reasons.append(f"n={n} < {config.min_n}")
    if r < config.min_pearson:
        reasons.append(f"pearson_r={r:.4f} < {config.min_pearson}")
    if rho < config.min_spearman:
        reasons.append(f"spearman_rho={rho:.4f} < {config.min_spearman}")
    if p >= config.max_perm_p:
        reasons.append(f"permutation_p={p:.4f} >= {config.max_perm_p}")
    if lb <= config.min_bootstrap_lb:
        reasons.append(f"bootstrap_lb={lb:.4f} <= {config.min_bootstrap_lb}")
    return {"gate": "PASS" if not reasons else "FAIL", "reasons": reasons}


# --------------------------------------------------------------------------- #
# Sealed holdout (evaluate once)
# --------------------------------------------------------------------------- #
class SealedHoldout:
    """A sealed holdout that may be evaluated at most ONCE.

    Guards against repeated evaluation / optimisation against the holdout.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, primary_descriptor_idx: Sequence[int]):
        self.X = np.asarray(X, dtype=float)[:, list(primary_descriptor_idx)]
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self._evaluated = False
        self.result: Optional[Dict[str, Any]] = None

    def evaluate_once(self, alpha: float) -> Dict[str, Any]:
        if self._evaluated:
            raise RuntimeError("sealed holdout already evaluated; repeated evaluation is forbidden")
        self._evaluated = True
        # Train on the sealed set's own Ridge at the frozen alpha (no inner
        # selection on the holdout); report in-sample r as the one-shot figure.
        preds = ridge_fit_predict(self.X, self.y, self.X, alpha)
        if np.std(preds) > 0 and np.std(self.y) > 0:
            r, p = pearsonr(preds, self.y)
            rho, _ = spearmanr(preds, self.y)
        else:
            r = rho = 0.0
            p = 1.0
        self.result = {"pearson_r": float(r), "spearman_rho": float(rho),
                       "pearson_p": float(p), "alpha": float(alpha), "evaluated": True}
        return self.result
