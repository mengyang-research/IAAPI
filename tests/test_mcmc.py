"""Tests for the P2-03 standardized MCMC runner."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from iaapi.evaluation.mcmc import (
    CONVERGED,
    MCMCConfig,
    MCMCRunner,
    UNCONVERGED,
    effective_sample_size,
    split_rhat,
)


def _gauss_logpost(mu, sigma):
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    inv = 1.0 / sigma ** 2

    def lp(theta):
        d = np.asarray(theta, float) - mu
        return float(-0.5 * np.sum((d * inv) * d))
    return lp


def test_gaussian_converges_and_recovers_mean():
    mu = [0.3, -0.4]
    sigma = [0.25, 0.4]
    bounds = [[-3, 3], [-3, 3]]
    lp = _gauss_logpost(mu, sigma)
    cfg = MCMCConfig(n_walkers=32, n_steps=1000, n_burn=500, seed=1,
                     rhat_threshold=1.05, ess_threshold=100.0)
    r = MCMCRunner(lp, np.array(bounds), mu, ["a", "b"], cfg, model_id="test").run()
    assert r.convergence_status == CONVERGED
    assert r.label_allowed is True
    assert r.excluded_by_rule == ""
    assert math.isclose(r.posterior_mean[0], mu[0], abs_tol=0.08)
    assert math.isclose(r.posterior_mean[1], mu[1], abs_tol=0.08)
    assert max(r.rhat) < 1.05
    assert min(r.ess) > 100


def test_unconverged_low_ess_excluded():
    mu = [0.0, 0.0]
    lp = _gauss_logpost(mu, [0.3, 0.3])
    bounds = [[-3, 3], [-3, 3]]
    # Very few steps => ESS below threshold => UNCONVERGED, label disallowed.
    cfg = MCMCConfig(n_walkers=8, n_steps=10, n_burn=2, seed=0,
                     rhat_threshold=1.01, ess_threshold=1000.0)
    r = MCMCRunner(lp, np.array(bounds), mu, ["a", "b"], cfg).run()
    assert r.convergence_status == UNCONVERGED
    assert r.label_allowed is False
    assert r.excluded_by_rule != ""


def test_split_rhat_independent_near_one():
    rng = np.random.default_rng(0)
    chains = rng.standard_normal(size=(8, 1000, 3))  # independent
    rh = split_rhat(chains)
    assert np.all(rh < 1.02)


def test_ess_independent_near_total_and_correlated_lower():
    rng = np.random.default_rng(0)
    indep = rng.standard_normal(size=(8, 1000, 2))
    ess_ind = effective_sample_size(indep)
    assert np.all(ess_ind > 5000)  # near m*n=8000
    # AR(1) correlated chains: lower ESS
    phi = 0.9
    corr = np.zeros_like(indep)
    for c in range(8):
        corr[c, 0, :] = rng.standard_normal(2)
        for t in range(1, 1000):
            corr[c, t, :] = phi * corr[c, t - 1, :] + math.sqrt(1 - phi ** 2) * rng.standard_normal(2)
    ess_corr = effective_sample_size(corr)
    assert np.all(ess_corr < ess_ind)


def test_whitened_cov_present_when_prior_given():
    mu = [0.0, 0.0]
    lp = _gauss_logpost(mu, [0.2, 0.2])
    bounds = [[-3, 3], [-3, 3]]
    prior_cov = np.diag([1.0, 1.0])
    cfg = MCMCConfig(n_walkers=24, n_steps=300, n_burn=150, seed=2,
                     rhat_threshold=1.1, ess_threshold=50.0)
    r = MCMCRunner(lp, np.array(bounds), mu, ["a", "b"], cfg, prior_cov=prior_cov).run()
    assert r.whitened_cov is not None
    assert len(r.whitened_cov) == 2


def test_resume_skips_completed(tmp_path):
    mu = [0.0]
    lp = _gauss_logpost(mu, [0.3])
    bounds = [[-3, 3]]
    cfg = MCMCConfig(n_walkers=12, n_steps=100, n_burn=50, seed=3,
                     rhat_threshold=1.1, ess_threshold=20.0)
    runner = MCMCRunner(lp, np.array(bounds), mu, ["a"], cfg, model_id="m")
    out = tmp_path / "m"
    s1 = runner.run_versioned(out, resume=True)
    assert (out / "summary.json").is_file()
    # Second call resumes from disk (no recompute).
    import os
    mtime1 = os.path.getmtime(out / "summary.json")
    s2 = runner.run_versioned(out, resume=True)
    assert s2["model_id"] == s1["model_id"]
    assert os.path.getmtime(out / "summary.json") == mtime1


def test_no_label_without_convergence_status():
    # Every result carries convergence_status; label_allowed == (status==CONVERGED).
    mu = [0.0]
    lp = _gauss_logpost(mu, [0.3])
    bounds = [[-3, 3]]
    cfg = MCMCConfig(n_walkers=8, n_steps=5, n_burn=1, seed=0,
                     rhat_threshold=1.01, ess_threshold=1000.0)
    r = MCMCRunner(lp, np.array(bounds), mu, ["a"], cfg).run()
    assert r.convergence_status in (CONVERGED, UNCONVERGED)
    assert r.label_allowed == (r.convergence_status == CONVERGED)
