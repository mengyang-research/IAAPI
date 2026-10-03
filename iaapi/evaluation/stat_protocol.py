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
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCHEMA_VERSION = "3.0"

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


@dataclass(frozen=True)
class FittedRidge:
    """Training-only scaler, intercept and coefficients frozen before testing."""
    mean: np.ndarray
    std: np.ndarray
    beta: np.ndarray
    intercept: float
    alpha: float
    primary_descriptor_idx: tuple[int, ...]
    training_model_ids: tuple[str, ...] = ()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite feature matrix")
        Xp = X[:, self.primary_descriptor_idx]
        return _standardise(Xp, self.mean, self.std) @ self.beta + self.intercept


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float,
              primary_descriptor_idx: Sequence[int] | None = None,
              training_model_ids: Sequence[str] | None = None) -> FittedRidge:
    """Fit using training data only; select alpha in training CV beforehand."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(y) < 2:
        raise ValueError("Need matching training features/targets with at least two rows")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Training data must be finite and alpha nonnegative")
    idx = tuple(range(X.shape[1])) if primary_descriptor_idx is None else tuple(primary_descriptor_idx)
    if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= X.shape[1] for i in idx):
        raise ValueError("Descriptor indices must be nonempty, unique, and in range")
    ids = () if training_model_ids is None else tuple(training_model_ids)
    if ids and (len(ids) != len(y) or len(set(ids)) != len(ids)):
        raise ValueError("Training model IDs must be unique and match the row count")
    Xp = X[:, idx]
    mean, std = Xp.mean(axis=0), Xp.std(axis=0)
    Xs = _standardise(Xp, mean, std)
    intercept = float(y.mean())
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(len(idx)),
                           Xs.T @ (y - intercept))
    for array in (mean, std, beta):
        array.setflags(write=False)
    return FittedRidge(mean, std, beta, intercept, float(alpha), idx, ids)


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, alpha: float,
                      return_intercept: bool = False) -> Any:
    fitted = fit_ridge(X_train, y_train, alpha)
    pred = fitted.predict(X_test)
    return (pred, fitted.intercept) if return_intercept else pred


# --------------------------------------------------------------------------- #
# Nested model-level CV
# --------------------------------------------------------------------------- #
def _loo_indices(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    return [(np.delete(idx, i), np.array([i])) for i in idx]


@lru_cache(maxsize=512)
def _loo_operator(shape: tuple[int, int], data: bytes, alpha: float) -> np.ndarray:
    """Linear prediction weights, with a separate scaler/intercept per fold."""
    X = np.frombuffer(data, dtype=np.float64).reshape(shape)
    n, d = shape
    operator = np.zeros((n, n))
    for train, test in _loo_indices(n):
        mean, std = X[train].mean(axis=0), X[train].std(axis=0)
        Xs = _standardise(X[train], mean, std)
        Xte = _standardise(X[test], mean, std)
        weights = Xte @ np.linalg.solve(Xs.T @ Xs + alpha * np.eye(d), Xs.T)
        # y centering and its training-only intercept are both in these weights.
        operator[test[0], train] = weights[0] - weights[0].sum() / len(train) + 1.0 / len(train)
    operator.setflags(write=False)
    return operator


def _ridge_loo_mse(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """LOO MSE with preprocessing fitted inside each training fold.

    Cache only X-dependent linear weights, never labels or full-data scalers.
    This is algebraically identical to refitting Ridge in every fold.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=float).reshape(-1)
    predictions = _loo_operator(X.shape, X.tobytes(), float(alpha)) @ y
    return float(np.mean((y - predictions) ** 2))


def inner_select_alpha(X: np.ndarray, y: np.ndarray, alphas: Sequence[float]) -> float:
    """Inner LOO over models to pick the alpha minimising aggregate held-out MSE.

    This fixes RISK-04: the old implementation computed Pearson r on one
    held-out scalar at a time, whose variance is zero, so no alpha received a
    valid score and the first grid value was always returned.  The corrected
    implementation refits scaling and regression in each fold to compute an
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
    """Evaluate an already fitted predictor once, without fitting on test labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 primary_descriptor_idx: Sequence[int] | None = None, *,
                 predictor: FittedRidge, model_ids: Sequence[str] | None = None):
        self.X = np.asarray(X, dtype=float).copy()
        self.y = np.asarray(y, dtype=float).reshape(-1).copy()
        if self.X.ndim != 2 or len(self.X) != len(self.y) or len(self.y) < 2:
            raise ValueError("Holdout needs matching features/targets and at least two models")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.y)):
            raise ValueError("Holdout data must be finite")
        if primary_descriptor_idx is not None and tuple(primary_descriptor_idx) != predictor.primary_descriptor_idx:
            raise ValueError("Holdout descriptors must match the frozen predictor")
        ids = () if model_ids is None else tuple(model_ids)
        if ids and (len(ids) != len(self.y) or len(set(ids)) != len(ids)):
            raise ValueError("Holdout model IDs must be unique and match the row count")
        if set(ids) & set(predictor.training_model_ids):
            raise ValueError("Training and holdout model IDs overlap")
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.predictor = predictor
        self.model_ids = ids
        self._evaluated = False
        self.result: Optional[Dict[str, Any]] = None

    def evaluate_once(self) -> Dict[str, Any]:
        if self._evaluated:
            raise RuntimeError("sealed holdout already evaluated; repeated evaluation is forbidden")
        # Predictions are produced by the frozen training-only model.
        preds = self.predictor.predict(self.X)
        self._evaluated = True
        if np.std(preds) > 0 and np.std(self.y) > 0:
            r, p = pearsonr(preds, self.y)
            rho, _ = spearmanr(preds, self.y)
        else:
            r = rho = 0.0
            p = 1.0
        self.result = {
            "schema_version": SCHEMA_VERSION,
            "pearson_r": float(r), "spearman_rho": float(rho),
            "pearson_p": float(p), "alpha": self.predictor.alpha,
            "predictions": preds.tolist(), "truth": self.y.tolist(),
            "model_ids": list(self.model_ids),
            "model_id_disjointness_checked": bool(self.model_ids and self.predictor.training_model_ids),
            "evaluated": True, "fit_on_holdout": False,
        }
        return self.result
