"""Tests for the P4-02 posterior validation metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from iaapi.evaluation.posterior_metrics import (
    c2st_auc,
    coverage,
    evaluate_posterior,
    mean_correlation,
    normalized_rmse,
    posterior_contraction,
    posterior_predictive_check,
    sbc_histogram,
    sbc_mean_abs_deviation,
    sbc_ranks,
    sbc_systematic_failure,
)


def test_normalized_rmse():
    true = np.array([1.0, 2.0, 3.0])
    prior = np.array([0.0, 0.0, 0.0])
    assert math.isclose(normalized_rmse(true, true, prior), 0.0)
    assert math.isclose(normalized_rmse(prior, true, prior), 1.0)
    # halfway: pred = (true+prior)/2 -> RMSE half of prior -> nrmse 0.5
    pred = (true + prior) / 2
    assert math.isclose(normalized_rmse(pred, true, prior), 0.5)


def test_mean_correlation_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(mean_correlation(a, a), 1.0, abs_tol=1e-9)


def _calibrated(rng, true, sigma, n_samples):
    """Calibrated posterior: mean = theta_hat = true + N(0, sigma) (estimation
    error matching posterior std), samples ~ N(theta_hat, sigma)."""
    theta_hat = true + rng.normal(0.0, sigma, size=true.shape)
    samples = rng.normal(theta_hat, sigma, size=(n_samples,) + true.shape)
    return theta_hat, samples


def test_coverage_calibrated():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        true = rng.normal(size=3)
        _, samples = _calibrated(rng, true, 1.0, 2000)
        covs.append(coverage(samples, true, levels=(0.90, 0.95)))
    c90 = np.mean([c[0.90] for c in covs])
    c95 = np.mean([c[0.95] for c in covs])
    assert abs(c90 - 0.90) < 0.03
    assert abs(c95 - 0.95) < 0.03


def test_coverage_miscalibrated():
    rng = np.random.default_rng(0)
    true = np.array([0.0, 0.0, 0.0])
    samples = rng.normal(true + 5.0, 0.1, size=(5000, 3))  # badly biased -> ~0 coverage
    cov = coverage(samples, true, levels=(0.90,))
    assert cov[0.90] < 0.1


def test_sbc_uniform_no_failure():
    rng = np.random.default_rng(0)
    rank_list = []
    n_samples = 100
    for _ in range(400):
        true = rng.normal(size=3)
        _, samples = _calibrated(rng, true, 1.0, n_samples)  # calibrated => uniform ranks
        rank_list.append(sbc_ranks(samples, true))
    counts = sbc_histogram(rank_list, n_samples, n_bins=10)
    n_total = int(np.concatenate([r.reshape(-1) for r in rank_list]).size)
    assert not sbc_systematic_failure(counts, n_total, 10)
    assert sbc_mean_abs_deviation(counts) < 0.05


def test_sbc_biased_failure():
    rng = np.random.default_rng(0)
    rank_list = []
    n_samples = 100
    for _ in range(400):
        true = rng.normal(size=3)
        samples = rng.normal(true + 2.0, 0.3, size=(n_samples, 3))  # biased high -> ranks skewed to 0
        rank_list.append(sbc_ranks(samples, true))
    counts = sbc_histogram(rank_list, n_samples, n_bins=10)
    n_total = int(np.concatenate([r.reshape(-1) for r in rank_list]).size)
    assert sbc_systematic_failure(counts, n_total, 10)


def test_posterior_contraction():
    rng = np.random.default_rng(0)
    prior = rng.normal(size=(2000, 3)) * 5.0
    post = rng.normal(size=(2000, 3)) * 0.5  # tighter
    c = posterior_contraction(post, prior)
    assert 0.0 < c < 1.0
    # equal spread -> ~0 contraction
    c0 = posterior_contraction(prior, prior)
    assert abs(c0) < 0.05


def test_c2st_indistinguishable_vs_separable():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 3))
    b = rng.normal(size=(300, 3))  # same dist
    assert c2st_auc(a, b) < 0.6  # ~0.5
    c = rng.normal(size=(300, 3), loc=10.0)  # separated
    assert c2st_auc(a, c) > 0.9


def test_posterior_predictive_check_calibrated():
    rng = np.random.default_rng(0)
    true_theta = np.array([0.0])
    samples = rng.normal(true_theta, 1.0, size=(2000, 1))

    def sim(th):
        return np.array([th[0] + rng.normal(0, 0.5)])
    observed = np.array([0.0])
    frac = posterior_predictive_check(samples, sim, observed, level=0.9)
    assert 0.8 <= frac <= 1.0  # most observations inside the 90% predictive band


def test_evaluate_posterior_gate_pass_vs_fail():
    rng = np.random.default_rng(0)
    # well-calibrated: posterior mean = theta_hat ~ N(true, sigma) with tight sigma
    good = []
    for _ in range(120):
        true = rng.normal(size=3)
        theta_hat, samples = _calibrated(rng, true, 0.3, 500)
        good.append({"samples": samples, "theta_true": true, "prior_mean": np.zeros(3),
                     "pred_mean": theta_hat, "prior_samples": rng.normal(size=(100, 3)) * 5})
    mg = evaluate_posterior(good)
    assert mg["gate"] == "PASS", mg["gate_reasons"]
    assert mg["nrmse_frac_le_1p25"] >= 0.75
    # mis-calibrated: biased samples -> SBC fail + coverage off
    bad = []
    for _ in range(120):
        true = rng.normal(size=3)
        bad.append({"samples": rng.normal(true + 3.0, 0.2, size=(100, 3)),
                    "theta_true": true, "prior_mean": np.zeros(3), "pred_mean": true + 3.0})
    mb = evaluate_posterior(bad)
    assert mb["gate"] == "FAIL"
    assert mb["sbc_failure"] is True


def test_evaluate_posterior_coverage_gate():
    rng = np.random.default_rng(1)
    # over-confident (tiny sigma) -> coverage far below level -> gate fail on coverage
    cases = []
    for _ in range(60):
        true = rng.normal(size=3)
        cases.append({"samples": rng.normal(true, 0.01, size=(100, 3)),
                      "theta_true": true, "prior_mean": np.zeros(3), "pred_mean": true})
    m = evaluate_posterior(cases)
    assert m["gate"] == "FAIL"
    assert any("coverage" in r for r in m["gate_reasons"])
