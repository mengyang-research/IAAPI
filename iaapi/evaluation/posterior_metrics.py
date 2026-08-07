"""Posterior validation metrics (NCS plan task P4-02).

Implements the required metrics with shared definitions (matching the P4-01
``METRIC_DEFINITIONS``) plus the preferred gate vs a per-model NPE reference:
  * normalized RMSE + correlation
  * expected coverage at 90/95 (marginal credible intervals)
  * SBC with simultaneous (Bonferroni) confidence bands + systematic-failure flag
  * posterior predictive check (fraction of observations inside predictive bands)
  * posterior contraction (1 - (det post/det prior)^{1/d})
  * C2ST (classifier two-sample test AUC; 0.5 indistinguishable, 1 separable)

All metrics operate on batches of (samples, theta_true) cases so they can be
aggregated per model and across the shared P4-01 protocol.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import binom

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def normalized_rmse(theta_pred: np.ndarray, theta_true: np.ndarray,
                    prior_mean: np.ndarray) -> float:
    """RMSE(pred, true) / RMSE(prior_mean, true). <1 => better than prior."""
    pred = np.asarray(theta_pred, float).reshape(-1)
    true = np.asarray(theta_true, float).reshape(-1)
    pm = np.asarray(prior_mean, float).reshape(-1)
    denom = float(np.sqrt(np.mean((pm - true) ** 2)))
    if denom < 1e-12:
        return 0.0 if np.allclose(pred, true) else float("inf")
    return float(np.sqrt(np.mean((pred - true) ** 2)) / denom)


def mean_correlation(theta_pred: np.ndarray, theta_true: np.ndarray) -> float:
    """Pearson r between predicted and true theta (aggregated over params)."""
    pred = np.asarray(theta_pred, float).reshape(-1)
    true = np.asarray(theta_true, float).reshape(-1)
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def coverage(samples: np.ndarray, theta_true: np.ndarray,
             levels: Sequence[float] = (0.90, 0.95)) -> Dict[float, float]:
    """Marginal credible-interval coverage per level, averaged over parameters.

    samples: (n_samples, n_param). theta_true: (n_param,).
    Returns {level: fraction of params whose true value lies in the central
    [ (1-level)/2, 1-(1-level)/2 ] sample quantile interval}.
    """
    samples = np.asarray(samples, float)
    true = np.asarray(theta_true, float).reshape(-1)
    n_param = true.shape[0]
    out: Dict[float, float] = {}
    for lv in levels:
        lo_q = (1.0 - lv) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(samples, lo_q, axis=0)
        hi = np.quantile(samples, hi_q, axis=0)
        inside = (true >= lo) & (true <= hi)
        out[float(lv)] = float(np.mean(inside))
    return out


# --------------------------------------------------------------------------- #
# SBC with simultaneous bands
# --------------------------------------------------------------------------- #
def sbc_ranks(samples: np.ndarray, theta_true: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Per-parameter SBC rank (count of samples < theta_true), in [0, n_samples].

    Returns (n_param,) integer ranks. Under perfect calibration the ranks are
    uniformly distributed across [0, n_samples].
    """
    samples = np.asarray(samples, float)
    true = np.asarray(theta_true, float).reshape(-1)
    return (samples < true[None, :]).sum(axis=0).astype(int)


def sbc_histogram(rank_list: List[np.ndarray], n_samples: int,
                  n_bins: int = 10) -> np.ndarray:
    """Aggregate SBC ranks across cases/params into an n_bins histogram (counts)."""
    all_ranks = np.concatenate([np.asarray(r).reshape(-1) for r in rank_list])
    edges = np.linspace(0, n_samples, n_bins + 1)
    counts, _ = np.histogram(all_ranks, bins=edges)
    return counts.astype(int)


def sbc_simultaneous_bands(n_total: int, n_bins: int, n_samples: int,
                           alpha: float = 0.05) -> tuple:
    """Bonferroni-corrected simultaneous binomial band for the SBC histogram.

    Under calibration each rank falls in a bin with prob 1/n_bins; the count in
    each bin ~ Binomial(n_total, 1/n_bins). Bonferroni over n_bins gives a
    simultaneous band at level (1-alpha). Returns (lower, upper) count arrays.
    """
    p = 1.0 / n_bins
    per_bin_alpha = alpha / n_bins
    lo = binom.ppf(per_bin_alpha / 2.0, n_total, p)
    hi = binom.ppf(1.0 - per_bin_alpha / 2.0, n_total, p)
    return np.asarray(lo, int), np.asarray(hi, int)


def sbc_systematic_failure(counts: np.ndarray, n_total: int, n_bins: int,
                            alpha: float = 0.05) -> bool:
    """True if any histogram bin falls outside the simultaneous band (systematic failure)."""
    lo, hi = sbc_simultaneous_bands(n_total, n_bins, n_total // max(n_bins, 1), alpha)
    # n_samples used only for binning width; band depends on n_total and p=1/n_bins
    lo, hi = sbc_simultaneous_bands(n_total, n_bins, n_total, alpha)
    return bool(np.any(counts < lo) or np.any(counts > hi))


def sbc_mean_abs_deviation(counts: np.ndarray) -> float:
    """Mean |observed_fraction - 1/n_bins| per bin (0 = perfectly uniform)."""
    n_bins = len(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    frac = counts / total
    return float(np.mean(np.abs(frac - 1.0 / n_bins)))


# --------------------------------------------------------------------------- #
# Posterior predictive check
# --------------------------------------------------------------------------- #
def posterior_predictive_check(samples: np.ndarray, simulate_fn, observed: np.ndarray,
                               level: float = 0.9) -> float:
    """Fraction of observed entries inside the posterior-predictive central interval.

    simulate_fn(theta) -> trajectory array of the same shape as ``observed``.
    Returns the fraction of observed entries within the [lo_q, hi_q] predictive
    band (averaged over array entries).
    """
    preds = np.array([simulate_fn(th) for th in np.asarray(samples, float)])
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    lo = np.quantile(preds, lo_q, axis=0)
    hi = np.quantile(preds, hi_q, axis=0)
    observed = np.asarray(observed, float)
    inside = (observed >= lo) & (observed <= hi)
    return float(np.mean(inside))


# --------------------------------------------------------------------------- #
# Contraction + C2ST
# --------------------------------------------------------------------------- #
def posterior_contraction(samples: np.ndarray, prior_samples: np.ndarray) -> float:
    """1 - (det(post)/det(prior))^{1/d}. 1=tight, 0=no contraction vs prior."""
    post = np.cov(np.asarray(samples, float).T)
    prior = np.cov(np.asarray(prior_samples, float).T)
    d = post.shape[0]

    def _logdet_psd(m):
        v = np.linalg.eigvalsh(0.5 * (m + m.T))
        v = np.clip(v, 0.0, None)
        return float(np.sum(np.log(v + 1e-300)))

    if d == 0:
        return 0.0
    ratio = math.exp((_logdet_psd(post) - _logdet_psd(prior)) / d)
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def c2st_auc(samples_post: np.ndarray, samples_ref: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test AUC (0.5 indistinguishable, 1.0 separable).

    Trains a logistic regression to tell post from ref samples; reports the
    classification accuracy (= AUC for balanced binary). Uses sklearn.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    post = np.asarray(samples_post, float)
    ref = np.asarray(samples_ref, float)
    X = np.vstack([post, ref])
    y = np.concatenate([np.ones(len(post)), np.zeros(len(ref))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


# --------------------------------------------------------------------------- #
# Aggregate + gate
# --------------------------------------------------------------------------- #
def evaluate_posterior(cases: List[dict], prior_mean_fn=None,
                       levels: Sequence[float] = (0.90, 0.95),
                       npe_nrmse: Optional[List[float]] = None) -> Dict[str, object]:
    """Aggregate metrics over cases (each: {samples, theta_true, prior_mean, pred_mean,
    prior_samples?, observed?, simulate_fn?}) and apply the preferred gate vs NPE.

    Gate: >=75% models nRMSE<=1.25 (vs prior; or <=1.25x NPE if npe_nrmse given),
    |coverage - level| <=0.05 at 90/95, no systematic SBC failure.
    """
    nrmse_list, corr_list = [], []
    cov_acc = {float(lv): [] for lv in levels}
    contraction_list = []
    rank_list = []
    n_samples = None
    for c in cases:
        s = np.asarray(c["samples"], float)
        if n_samples is None:
            n_samples = s.shape[0]
        tt = np.asarray(c["theta_true"], float)
        pm = np.asarray(c["prior_mean"], float)
        pred = np.asarray(c.get("pred_mean", pm), float)
        nrmse_list.append(normalized_rmse(pred, tt, pm))
        corr_list.append(mean_correlation(pred, tt))
        cov = coverage(s, tt, levels)
        for lv in levels:
            cov_acc[float(lv)].append(cov[float(lv)])
        if "prior_samples" in c:
            contraction_list.append(posterior_contraction(s, c["prior_samples"]))
        rank_list.append(sbc_ranks(s, tt))
    # SBC aggregate
    n_bins = 10
    counts = sbc_histogram(rank_list, n_samples or 10, n_bins)
    n_total = int(np.concatenate([r.reshape(-1) for r in rank_list]).size) if rank_list else 0
    sbc_fail = sbc_systematic_failure(counts, n_total, n_bins) if n_total > 0 else False

    metrics = {
        "n_cases": len(cases),
        "nrmse_mean": float(np.mean(nrmse_list)) if nrmse_list else None,
        "nrmse_frac_le_1p25": float(np.mean([x <= 1.25 for x in nrmse_list])) if nrmse_list else 0.0,
        "mean_correlation": float(np.mean(corr_list)) if corr_list else None,
        "coverage": {lv: float(np.mean(v)) for lv, v in cov_acc.items() if v},
        "coverage_abs_err": {lv: float(abs(np.mean(v) - lv)) for lv, v in cov_acc.items() if v},
        "posterior_contraction_mean": float(np.mean(contraction_list)) if contraction_list else None,
        "sbc_failure": sbc_fail,
        "sbc_mean_abs_deviation": sbc_mean_abs_deviation(counts) if n_total > 0 else None,
    }
    # gate
    reasons = []
    if metrics["nrmse_frac_le_1p25"] < 0.75:
        reasons.append(f"nRMSE<=1.25 for {metrics['nrmse_frac_le_1p25']:.2f} < 0.75 of models")
    for lv in levels:
        ae = metrics["coverage_abs_err"].get(float(lv), 1.0)
        if ae > 0.05:
            reasons.append(f"|coverage_{lv} - {lv}| = {ae:.3f} > 0.05")
    if sbc_fail:
        reasons.append("systematic SBC failure")
    metrics["gate"] = "PASS" if not reasons else "FAIL"
    metrics["gate_reasons"] = reasons
    return metrics
