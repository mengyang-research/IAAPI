"""Unit tests for the P1-01 preflight scanner core (no AMICI required).

The PreflightRunner is exercised through an injectable ``simulate_fn`` so the
logic is tested deterministically: ADMIT on a healthy mock, REJECT with the
correct failure category on NaN / status / timeout mocks, and seed-deterministic
sampling.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from iaapi.data.preflight import (
    CAT_FORWARD_NAN,
    CAT_FORWARD_STATUS,
    CAT_TIMEOUT,
    PreflightConfig,
    PreflightRunner,
    evaluate_gate,
    sample_parameters,
)

N_PARAM = 3
BOUNDS = np.array([[-1.0, 1.0]] * N_PARAM)
NOMINAL = np.zeros(N_PARAM)


def _traj(theta):
    # (n_obs=2, n_time=4) finite trajectories depending on theta.
    return np.array([[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]]) + theta[0]


def _sens(theta):
    # (n_obs=2, n_time=4, n_param=3) finite sensitivities.
    base = np.arange(24, dtype=float).reshape(2, 4, 3)
    return base + theta.reshape(1, 1, 3)


def _make_sim(status=0, traj_fn=_traj, sens_fn=_sens, sleep_s=0.0, nan_traj=False):
    def simulate(theta, compute_sensitivities):
        if sleep_s:
            time.sleep(sleep_s)
        traj = traj_fn(theta)
        if nan_traj:
            traj = traj.copy()
            traj[0, 0] = np.nan
        out = {"status": status, "trajectories": traj}
        if compute_sensitivities and sens_fn is not None:
            out["sensitivities"] = sens_fn(theta)
        return out
    return simulate


def _runner(simulate_fn, config=None):
    cfg = config or PreflightConfig(n_samples=6, seed=0, per_simulate_timeout_s=10.0)
    return PreflightRunner("mock", simulate_fn, NOMINAL, BOUNDS, N_PARAM, cfg)


def test_healthy_mock_admits():
    r = _runner(_make_sim()).run()
    assert r.gate == "ADMIT"
    assert r.failure_categories == []
    assert r.nominal_forward["success"] is True
    assert r.nominal_sensitivity["success"] is True
    assert r.fim["finite"] is True
    assert r.pilot_forward["success_rate"] == 1.0
    assert r.pilot_forward["nan_rate"] == 0.0
    assert r.pilot_sensitivity["success_rate"] == 1.0
    assert r.fim["effective_rank_participation"] > 0


def test_nan_trajectories_reject_with_category():
    r = _runner(_make_sim(nan_traj=True)).run()
    assert r.gate == "REJECT"
    assert CAT_FORWARD_NAN in r.failure_categories
    assert r.pilot_forward["nan_rate"] > 0.0
    assert "pilot forward success" in " ".join(r.gate_reasons)


def test_nonzero_status_reject_with_category():
    r = _runner(_make_sim(status=1)).run()
    assert r.gate == "REJECT"
    assert CAT_FORWARD_STATUS in r.failure_categories
    assert "nominal forward failed" in r.gate_reasons


def test_timeout_reject_with_category():
    # Mock sleeps longer than the per-simulate timeout -> soft_timeout fires.
    cfg = PreflightConfig(n_samples=2, seed=0, per_simulate_timeout_s=0.4)
    r = _runner(_make_sim(sleep_s=2.0), config=cfg).run()
    assert r.gate == "REJECT"
    assert CAT_TIMEOUT in r.failure_categories


def test_low_pilot_success_rejects():
    # Every other sample returns NaN -> success rate ~0.5 < 0.9 threshold.
    calls = {"i": 0}
    def sim(theta, compute_sensitivities):
        calls["i"] += 1
        traj = _traj(theta)
        if calls["i"] % 2 == 0:
            traj = traj.copy(); traj[0, 0] = np.nan
        out = {"status": 0, "trajectories": traj}
        if compute_sensitivities:
            out["sensitivities"] = _sens(theta)
        return out
    r = _runner(sim).run()
    assert r.gate == "REJECT"
    assert any("pilot forward success" in x for x in r.gate_reasons)


def test_sampling_is_seed_deterministic():
    a = sample_parameters(BOUNDS, 10, seed=42)
    b = sample_parameters(BOUNDS, 10, seed=42)
    c = sample_parameters(BOUNDS, 10, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert a.shape == (10, N_PARAM)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_gate_directly():
    from iaapi.data.preflight import ModelPreflightResult
    cfg = PreflightConfig()
    ok = ModelPreflightResult(
        model_id="x", nominal_forward={"success": True}, nominal_sensitivity={"success": True},
        fim={"finite": True}, pilot_forward={"n": 10, "success_rate": 0.95, "runtime_p95_s": 1.0},
        pilot_sensitivity={"n_attempted": 10, "success_rate": 0.9},
    )
    assert evaluate_gate(ok, cfg)[0] == "ADMIT"
    bad = ModelPreflightResult(
        model_id="y", nominal_forward={"success": True}, nominal_sensitivity={"success": True},
        fim={"finite": True}, pilot_forward={"n": 10, "success_rate": 0.5, "runtime_p95_s": 1.0},
        pilot_sensitivity={"n_attempted": 10, "success_rate": 0.9},
    )
    gate, reasons = evaluate_gate(bad, cfg)
    assert gate == "REJECT" and any("pilot forward" in x for x in reasons)
