"""Tests for the real SBC evaluator (no fabricated random results).

Covers the three acceptance criteria from the review:
  1. calibrated model -> near-uniform ranks (small KS);
  2. deliberately biased model -> stably fails (large KS, small p);
  3. fixed seed -> exactly reproducible ranks.
"""

from __future__ import annotations

import numpy as np

from iaapi.evaluation.sbc import SBCEvaluator


def make_gaussian_problem(name="gauss", n_params=4, obs_dim=6):
    """A Gaussian-likelihood problem: y = theta @ A + N(0, sigma)."""
    rng = np.random.default_rng(123)
    A = rng.normal(size=(n_params, obs_dim))
    return {
        "name": name,
        "n_params": n_params,
        "obs_dim": obs_dim,
        "A": A,
        "prior_lo": np.full(n_params, -3.0),
        "prior_hi": np.full(n_params, 3.0),
    }


def prior_sampler(problem, n):
    lo, hi = problem["prior_lo"], problem["prior_hi"]
    rng = np.random.default_rng(problem.get("sbc_seed", 7))
    return rng.uniform(lo, hi, size=(n, problem["n_params"]))


def simulator(problem, theta):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    rng = np.random.default_rng(problem.get("sbc_seed", 7) + 100)
    noise = rng.normal(0.0, 0.5, size=problem["obs_dim"])
    return {"values": (theta @ problem["A"]) + noise}


class ExactGaussianModel:
    """Returns exact posterior samples for the Gaussian linear model.

    Posterior of theta given y under a uniform prior on [-3,3]^d and Gaussian
    noise is N((A^T A)^{-1} A^T y, sigma^2 (A^T A)^{-1}) truncated to the box;
    for the small-noise regime we sample the untruncated Gaussian (the box is
    wide enough that truncation is negligible).
    """

    def __init__(self, problem, n_samples=500):
        self.problem = problem
        self.n_samples = n_samples
        A = problem["A"]                       # (n_params, obs_dim)
        self.cov = 0.25 * np.linalg.inv(A @ A.T)   # (n_params, n_params)
        self.A = A

    def sample_posterior(self, problem, n_samples):
        rng = np.random.default_rng(problem.get("sbc_seed", 42))
        obs = np.asarray(problem["_last_obs"])     # (obs_dim,)
        # Posterior mean = (A A^T)^{-1} A y for y = theta @ A + noise.
        mu = np.linalg.solve(self.A @ self.A.T, self.A @ obs)
        return rng.multivariate_normal(mu, self.cov, size=n_samples)


class BiasedGaussianModel(ExactGaussianModel):
    """Posterior deliberately shifted by +1 in every coordinate."""

    def sample_posterior(self, problem, n_samples):
        rng = np.random.default_rng(problem.get("sbc_seed", 42))
        obs = np.asarray(problem["_last_obs"])
        mu = np.linalg.solve(self.A @ self.A.T, self.A @ obs)
        return rng.multivariate_normal(mu + 1.0, self.cov, size=n_samples)


class StoringModel(ExactGaussianModel):
    """Wraps a model and records the last observations for the analytic mean."""

    def __init__(self, inner):
        self.inner = inner

    def sample_posterior(self, problem, n_samples):
        return self.inner.sample_posterior(problem, n_samples)


def run_sbc(model, problem, n_samples=500, n_reps=30):
    """Run SBC, feeding each simulation's observations back to the model."""
    evaluator = SBCEvaluator(n_samples=n_samples, seed=0)
    simulator_probe = {}

    def sim_with_probe(problem, theta):
        obs = simulator(problem, theta)
        simulator_probe["obs"] = obs
        return obs

    class ObsAware:
        def __init__(self, inner):
            self.inner = inner

        def sample_posterior(self, problem, n):
            problem["_last_obs"] = simulator_probe["obs"]["values"]
            return self.inner.sample_posterior(problem, n)

    return evaluator.run(
        ObsAware(model), [problem], n_samples=n_samples,
        simulator=sim_with_probe, prior_sampler=prior_sampler,
        n_repetitions=n_reps, n_bootstrap=500,
    )


def test_calibrated_model_has_small_ks():
    problem = make_gaussian_problem()
    results = run_sbc(ExactGaussianModel(problem), problem, n_reps=30)
    # A well-calibrated posterior must not trip the screening threshold.
    assert results["model_level_median_ks"] < 0.15
    assert results["pooled_ks"] < 0.15


def test_biased_model_fails():
    problem = make_gaussian_problem()
    results = run_sbc(BiasedGaussianModel(problem), problem, n_reps=30)
    # A +1 shift in every coordinate must produce clearly non-uniform ranks.
    assert results["model_level_median_ks"] > 0.2
    assert results["pooled_ks"] > 0.2


def test_fixed_seed_reproducible():
    problem = make_gaussian_problem()
    r1 = run_sbc(ExactGaussianModel(problem), problem, n_reps=10)
    r2 = run_sbc(ExactGaussianModel(problem), problem, n_reps=10)
    assert r1["per_problem"]["gauss"]["ranks_per_param"] == \
        r2["per_problem"]["gauss"]["ranks_per_param"]


def test_raw_ranks_saved_and_shaped():
    problem = make_gaussian_problem(n_params=4)
    results = run_sbc(ExactGaussianModel(problem), problem, n_reps=8)
    ranks = results["per_problem"]["gauss"]["ranks_per_param"]
    assert np.array(ranks).shape == (8, 4)
    # ranks must be integer-valued within {0..K}
    arr = np.array(ranks)
    assert arr.min() >= 0 and arr.max() <= 500
    assert np.allclose(arr, arr.astype(int))


def test_placeholder_removed():
    """The old random placeholder path must be gone: missing simulator fails."""
    evaluator = SBCEvaluator(n_samples=10, seed=0)
    try:
        evaluator.run(model=object(), test_problems=[make_gaussian_problem()])
    except ValueError as exc:
        assert "simulator" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("run() without simulator should raise ValueError")
