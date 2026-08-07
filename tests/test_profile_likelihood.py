"""Tests for the P2-02 standardized profile-likelihood runner."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from iaapi.evaluation.profile_likelihood import (
    BOUNDARY_HIT,
    FINITE_CI,
    OPTIMIZER_FAILURE,
    PLConfig,
    ProfileLikelihoodRunner,
    SIMULATOR_FAILURE,
    SimulatorError,
    TIMEOUT,
)


def _quadratic_nll(mu, sigma):
    """nll(theta) = 0.5 * sum ((theta-mu)/sigma)^2  =>  CI = mu +/- 1.96 sigma."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)

    def nll(theta):
        return float(0.5 * np.sum(((np.asarray(theta, float) - mu) / sigma) ** 2))
    return nll


def _runner(nll, bounds, nominal, ids=None, **cfg):
    ids = ids or [f"p{i}" for i in range(len(nominal))]
    return ProfileLikelihoodRunner(nll, np.array(bounds), np.array(nominal), ids,
                                   PLConfig(n_starts=3, maxiter=100, profile_step=0.15,
                                            profile_max_steps=30, bisect_iters=20,
                                            timeout_per_param_s=60.0, **cfg))


def test_analytic_finite_ci():
    mu = [0.0, 1.0, -0.5]
    sigma = [0.3, 0.5, 0.4]
    nll = _quadratic_nll(mu, sigma)
    bounds = [[-5, 5]] * 3
    r = _runner(nll, bounds, mu)
    for i in range(3):
        pr = r.profile_parameter(i)
        assert pr.code == FINITE_CI, pr.detail
        assert math.isclose(pr.mle, mu[i], abs_tol=1e-3)
        half = 1.959964 * sigma[i]
        assert math.isclose(pr.ci_lower, mu[i] - half, rel_tol=2e-2), (i, pr.ci_lower, mu[i] - half)
        assert math.isclose(pr.ci_upper, mu[i] + half, rel_tol=2e-2), (i, pr.ci_upper, mu[i] + half)


def test_boundary_hit_when_ci_exceeds_bounds():
    # Very wide posterior (large sigma) => CI extends to bounds on at least one side.
    mu = [0.0]
    sigma = [3.0]  # 1.96*3 = 5.88 > bound half-width 1.0 => boundary hit
    nll = _quadratic_nll(mu, sigma)
    bounds = [[-1.0, 1.0]]
    r = _runner(nll, bounds, mu)
    pr = r.profile_parameter(0)
    assert pr.code == BOUNDARY_HIT
    # CI clamped to bounds
    assert pr.ci_lower == -1.0 and pr.ci_upper == 1.0


def test_simulator_failure_code():
    def nll(theta):
        raise SimulatorError("AMICI exploded")
    r = _runner(nll, [[-1, 1]] * 2, [0.0, 0.0])
    pr = r.profile_parameter(0)
    assert pr.code == SIMULATOR_FAILURE


def test_timeout_code():
    import time
    def nll(theta):
        time.sleep(0.5)  # each eval far exceeds the per-param budget
        return float(np.sum(np.asarray(theta) ** 2))
    r = ProfileLikelihoodRunner(nll, np.array([[-1.0, 1.0]] * 2), np.zeros(2),
                                ["a", "b"], PLConfig(n_starts=3, maxiter=100,
                                profile_step=0.2, profile_max_steps=50,
                                timeout_per_param_s=0.25))
    pr = r.profile_parameter(0)
    assert pr.code == TIMEOUT


def test_resume_at_next_unfinished_parameter(tmp_path):
    mu = [0.0, 0.0, 0.0]
    sigma = [0.3, 0.3, 0.3]
    nll = _quadratic_nll(mu, sigma)
    bounds = [[-3, 3]] * 3
    r = _runner(nll, bounds, mu)
    out = tmp_path / "pl"
    # Simulate an interrupted run: only param 0 completed & persisted.
    pr0 = r.profile_parameter(0)
    (out).mkdir(parents=True, exist_ok=True)
    from iaapi.evaluation.profile_likelihood import ParamResult
    (out / "p0.json").write_text(json.dumps(pr0.__dict__, indent=2) + "\n")
    # Resume: should skip p0, compute p1, p2.
    summary = r.run(out, resume=True)
    assert len(summary["results"]) == 3
    assert summary["results"][0]["parameter_id"] == "p0"  # reused cached
    assert summary["results"][1]["parameter_id"] == "p1"  # newly computed
    assert summary["results"][2]["code"] == FINITE_CI
    # progress file written
    assert (out / "progress.json").is_file()
    assert (out / "pl_results.json").is_file()


def test_output_distinguishes_codes(tmp_path):
    """param 0 has a clean quadratic (FINITE_CI); profiling param 1 leaves its
    viable region -> SIMULATOR_FAILURE."""
    def nll(theta):
        t = np.asarray(theta, float).reshape(-1)
        if abs(t[1]) > 0.5:
            raise SimulatorError("param1 out of viable region")
        return float(0.5 * (t[0] / 0.3) ** 2)  # independent of t[1] near 0

    r = _runner(nll, [[-3, 3], [-3, 3]], [0.0, 0.0])
    pr0 = r.profile_parameter(0)
    pr1 = r.profile_parameter(1)
    assert pr0.code == FINITE_CI, pr0.detail
    assert pr1.code == SIMULATOR_FAILURE, pr1.detail


def _fake_sim_for_petab_nll():
    """Fake sim + measurement_df for build_petab_data_nll (no AMICI)."""
    import numpy as np
    class _Pdf:
        def __init__(self): self.measurement_df = _mdf()
    class _PP:
        def __init__(self): self.measurement_df = _mdf()
    class _Sim:
        parameter_ids = ["k1", "sd_y"]
        observable_ids = ["y"]
        theta_nominal_log10 = np.array([0.0, 0.0])
        petab_problem_v2 = _PP()
        def simulate(self, theta, compute_sensitivities=False):
            # trajectory y(t) = 10*theta_phys_k1 at 2 timepoints [0, 1]
            k1 = 10.0 ** float(theta[0])
            return {"trajectories": np.array([[k1, k1]]), "time_points": np.array([0.0, 1.0]),
                    "status": 0}
    return _Sim()

def _mdf():
    import pandas as pd
    return pd.DataFrame({"observableId": ["y", "y"], "time": [0.0, 1.0],
                         "measurement": [5.0, 5.0], "noiseParameters": ["sd_y", "sd_y"]})

def test_build_petab_data_nll_mapping_and_finite():
    import sys; sys.path.insert(0, "scripts")
    import importlib.util
    spec = importlib.util.spec_from_file_location("rpl", "scripts/run_profile_likelihood.py")
    rpl = importlib.util.module_from_spec(spec); sys.modules["rpl"] = rpl
    spec.loader.exec_module(rpl)
    sim = _fake_sim_for_petab_nll()
    nll = rpl.build_petab_data_nll(sim)
    import numpy as np
    # k1=10^0=1 (theta=0) => y=1 vs meas=5, sd=1 => nll = 2*(0.5*(4)^2 + 0) = 16
    v0 = nll(np.array([0.0, 0.0]))
    assert np.isfinite(v0) and abs(v0 - 16.0) < 1e-6, v0
    # y=meas=5 => k1=5 (theta_k1=log10(5)); sd=1 => nll = 2*(0 + log1) = 0
    v_min = nll(np.array([np.log10(5.0), 0.0]))
    assert abs(v_min - 0.0) < 1e-6, v_min


# --------------------------------------------------------------------------- #
# R1-02 tests: convergence enforcement, per-start traces, coverage gate
# --------------------------------------------------------------------------- #
def test_per_start_traces():
    """ParamResult.start_traces records per-start optimizer results."""
    mu = [0.0, 1.0]
    sigma = [0.3, 0.5]
    nll = _quadratic_nll(mu, sigma)
    r = ProfileLikelihoodRunner(nll, np.array([[-5, 5]] * 2), np.array(mu),
                                ["p0", "p1"],
                                PLConfig(n_starts=4, maxiter=100, profile_step=0.15,
                                         profile_max_steps=30, bisect_iters=20,
                                         timeout_per_param_s=60.0))
    pr = r.profile_parameter(0)
    assert len(pr.start_traces) == 4
    # Each trace has the required fields
    for t in pr.start_traces:
        assert "start_idx" in t
        assert "nll" in t
        assert "converged" in t
        assert "n_evals" in t
    # At least one start should have converged for a clean quadratic
    assert any(t["converged"] for t in pr.start_traces)
    # Best nll should match the minimum across traces
    best_trace_nll = min(t["nll"] for t in pr.start_traces if t["nll"] is not None)
    assert math.isclose(best_trace_nll, pr.nll_min, abs_tol=1e-4)


def test_converged_mle_requirement():
    """require_converged_mle=True -> unconverged MLE produces OPTIMIZER_FAILURE."""
    # A steep nll with maxiter=1 => optimizer won't converge
    def steep_nll(theta):
        t = np.asarray(theta, float)
        return float(0.5 * np.sum((t * 100.0) ** 2))  # very steep gradient

    r = ProfileLikelihoodRunner(steep_nll, np.array([[-5, 5]] * 2),
                                np.array([3.0, -3.0]),  # far from minimum
                                ["a", "b"],
                                PLConfig(n_starts=2, maxiter=1, profile_step=0.15,
                                         profile_max_steps=5, bisect_iters=5,
                                         timeout_per_param_s=60.0,
                                         require_converged_mle=True))
    pr = r.profile_parameter(0)
    # With require_converged_mle=True, unconverged -> OPTIMIZER_FAILURE
    assert pr.code == OPTIMIZER_FAILURE
    assert "converge" in pr.detail.lower()


def test_converged_mle_disabled():
    """require_converged_mle=False -> unconverged MLE still produces a result."""
    def flat_nll(theta):
        return 42.0

    r = ProfileLikelihoodRunner(flat_nll, np.array([[-5, 5]] * 2), np.zeros(2),
                                ["a", "b"],
                                PLConfig(n_starts=3, maxiter=10, profile_step=0.15,
                                         profile_max_steps=30, bisect_iters=20,
                                         timeout_per_param_s=60.0,
                                         require_converged_mle=False))
    pr = r.profile_parameter(0)
    # Without the requirement, the result should not be OPTIMIZER_FAILURE
    # due to non-convergence (it may still fail for other reasons, but not
    # the "did not converge" check)
    assert pr.code != OPTIMIZER_FAILURE or "converge" not in pr.detail.lower()


def test_coverage_gate_admitted(tmp_path):
    """80%+ classified parameters -> label_admitted=True."""
    mu = [0.0, 1.0, -0.5, 2.0, 0.5]
    sigma = [0.3, 0.5, 0.4, 0.3, 0.5]
    nll = _quadratic_nll(mu, sigma)
    bounds = [[-5, 5]] * 5
    r = ProfileLikelihoodRunner(nll, np.array(bounds), np.array(mu),
                                [f"p{i}" for i in range(5)],
                                PLConfig(n_starts=3, maxiter=100, profile_step=0.15,
                                         profile_max_steps=30, bisect_iters=20,
                                         timeout_per_param_s=60.0,
                                         min_classified_coverage=0.80))
    summary = r.run(tmp_path / "pl", resume=False)
    assert summary["classified_coverage"] >= 0.80
    assert summary["label_admitted"] is True


def test_coverage_gate_denied(tmp_path):
    """<80% classified -> label_admitted=False."""
    # Simulator fails for params 0-3 (out of 5) -> only 1/5 = 20% classified
    def nll(theta):
        t = np.asarray(theta, float)
        if abs(t[0]) > 0.1 or abs(t[1]) > 0.1 or abs(t[2]) > 0.1 or abs(t[3]) > 0.1:
            raise SimulatorError("out of viable region")
        return float(0.5 * np.sum(t ** 2))

    r = ProfileLikelihoodRunner(nll, np.array([[-5, 5]] * 5), np.zeros(5),
                                [f"p{i}" for i in range(5)],
                                PLConfig(n_starts=2, maxiter=50, profile_step=0.15,
                                         profile_max_steps=30, bisect_iters=20,
                                         timeout_per_param_s=60.0,
                                         min_classified_coverage=0.80))
    summary = r.run(tmp_path / "pl_denied", resume=False)
    # Most params will fail -> coverage < 80%
    assert summary["classified_coverage"] < 0.80
    assert summary["label_admitted"] is False
