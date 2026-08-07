"""Unit tests for the P1-02 viability-aware sampler (no AMICI)."""
from __future__ import annotations

import numpy as np
import pytest

from iaapi.data.viability_sampler import (
    RoleWidths,
    ViabilitySampler,
    classify_role,
    compare_strategies,
    forward_success_rate,
)

PARAM_IDS = ["k1", "k2", "sd_noise", "x0_init", "scale_obs"]
BOUNDS = np.array([[-3.0, 3.0]] * 5)
NOMINAL = np.array([0.0, 0.0, -1.0, 0.5, 0.0])


def test_respects_bounds():
    s = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal", seed=1)
    th = s.sample(500)
    assert th.shape == (500, 5)
    assert np.all(th >= -3.0) and np.all(th <= 3.0)


def test_uniform_bounds_matches_legacy():
    s = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="uniform_bounds", seed=42)
    th = s.sample(200)
    assert np.all(th >= -3.0) and np.all(th <= 3.0)
    # Full-bound uniform should cover the whole range (min near -3, max near 3).
    assert th.min() < -2.0 and th.max() > 2.0


def test_seed_deterministic():
    a = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal", seed=7)
    b = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal", seed=7)
    c = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal", seed=8)
    np.testing.assert_array_equal(a.sample(50), b.sample(50))
    assert not np.allclose(a.sample(50), c.sample(50))


def test_role_specific_widths():
    rw = RoleWidths(kinetic=1.0, initial_state=0.3, noise=0.2, scale=0.1)
    roles = {pid: classify_role(pid) for pid in PARAM_IDS}
    assert roles == {"k1": "kinetic", "k2": "kinetic", "sd_noise": "noise",
                     "x0_init": "initial_state", "scale_obs": "scale"}
    s = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="uniform_window",
                         role_widths=rw, seed=0)
    # window half-width per param = role width
    th = s.sample(1000)
    # kinetic cols (0,1): +/-1.0 around 0.0 -> within [-1,1]
    assert th[:, 0].min() >= -1.0 - 1e-9 and th[:, 0].max() <= 1.0 + 1e-9
    # noise col 2: +/-0.2 around -1.0 -> within [-1.2, -0.8]
    assert th[:, 2].min() >= -1.2 - 1e-9 and th[:, 2].max() <= -0.8 + 1e-9
    # scale col 4: +/-0.1 around 0.0 -> within [-0.1, 0.1]
    assert th[:, 4].min() >= -0.1 - 1e-9 and th[:, 4].max() <= 0.1 + 1e-9


def test_provenance_param_ids_match():
    s = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal", seed=3)
    pv = s.provenance()
    assert pv.param_ids == PARAM_IDS
    assert pv.n_param == 5
    assert pv.strategy == "truncated_normal"
    assert pv.param_roles["sd_noise"] == "noise"


def test_precheck_keeps_only_viable():
    # forward_fn viable only when theta is within 0.8 of nominal per coord.
    def fwd(theta):
        return bool(np.all(np.abs(theta - NOMINAL) < 0.8))
    s = ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="truncated_normal",
                         role_widths=RoleWidths(kinetic=1.5, initial_state=1.5,
                                                noise=1.5, scale=1.5), seed=5)
    thetas, stats = s.sample_with_precheck(fwd, n_target=10, max_attempts=1000)
    assert thetas.shape[0] == 10
    assert all(fwd(th) for th in thetas)
    assert stats["n_viable"] == 10
    assert stats["n_attempted"] >= 10
    # The filter must have rejected some draws (proving it actually filters).
    assert stats["success_rate"] < 1.0


def test_compare_strategies_material_improvement():
    # Viability drops sharply away from nominal (full-bound uniform fails a lot).
    def fwd(theta):
        return bool(np.all(np.abs(theta - NOMINAL) < 1.0))
    cmp = compare_strategies(fwd, BOUNDS, NOMINAL, PARAM_IDS, n=300, seed=11,
                             new_strategy="truncated_normal",
                             role_widths=RoleWidths(default=1.0))
    assert cmp["old_forward_success_rate"] < cmp["new_forward_success_rate"]
    assert cmp["materially_exceeds"] is True
    assert cmp["improvement"] >= 0.10


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        ViabilitySampler(BOUNDS, NOMINAL, PARAM_IDS, strategy="bogus")
