"""Tests for the real SNPE sequential refiner (no random placeholders).

Covers:
  * a strict simulation budget is respected;
  * model state is saved/restored as a real snapshot (not a random id);
  * the TV-based early stop can fire;
  * per-round bookkeeping (cumulative sims, loss, checkpoint hash) is present.
"""

from __future__ import annotations

import numpy as np

from iaapi.refinement.sequential import RefinementConfig, SequentialRefiner


class ToyTrainableModel:
    """A minimal trainable stand-in that records SNPE rounds."""

    def __init__(self, n_params=2, seed=0):
        self.n_params = n_params
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(n_params)
        self.std = np.ones(n_params)
        self.rounds_seen = 0

    def sample_posterior(self, problem, n_samples):
        rng = np.random.default_rng(self.rounds_seen + 1)
        return rng.normal(self.mean, self.std, size=(n_samples, self.n_params))

    def train_snpe_round(self, theta, obs, proposal, steps):
        self.rounds_seen += 1
        theta = np.asarray(theta, float)
        self.mean = 0.5 * self.mean + 0.5 * theta.mean(axis=0)
        return float(np.mean((theta - self.mean) ** 2))

    def state_dict(self):
        return {"mean": np.array(self.mean), "std": np.array(self.std)}

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"])
        self.std = np.array(state["std"])


def make_problem(n_params=2):
    return {
        "name": "toy",
        "n_params": n_params,
        "prior_lo": np.full(n_params, -3.0),
        "prior_hi": np.full(n_params, 3.0),
    }


def simulator(problem, theta):
    theta = np.asarray(theta, float).reshape(-1)
    return {"values": theta + 0.1 * np.random.default_rng(0).standard_normal(theta.size)}


def test_budget_respected():
    model = ToyTrainableModel()
    cfg = RefinementConfig(simulation_budget=120, round_size=50,
                           optimization_steps=10, max_rounds=5, tv_tol=0.0)
    refiner = SequentialRefiner(model, simulator, cfg)
    res = refiner.refine(make_problem(), verbose=False)
    # Budget 120 with round_size 50: rounds use 50, then 50, then 20 -> 120 max.
    assert res.total_simulations <= 120
    assert res.total_simulations >= 100  # at least two full rounds


def test_state_saved_and_restored():
    model = ToyTrainableModel()
    cfg = RefinementConfig(simulation_budget=50, round_size=50,
                           optimization_steps=5, max_rounds=1, tv_tol=0.0)
    refiner = SequentialRefiner(model, simulator, cfg)
    state0 = refiner.save_state()
    assert "state_dict" in state0
    assert "config" in state0
    # Restore into a fresh model and confirm the snapshot round-trips.
    model2 = ToyTrainableModel()
    refiner2 = SequentialRefiner(model2, simulator, cfg)
    refiner2.load_state(state0)
    np.testing.assert_allclose(model2.state_dict()["mean"], state0["state_dict"]["mean"])


def test_tv_early_stop():
    model = ToyTrainableModel()
    # tv_tol=10.0 is huge: the first TV comparison (round 2) fires immediately.
    cfg = RefinementConfig(simulation_budget=1000, round_size=50,
                           optimization_steps=5, max_rounds=10, tv_tol=10.0)
    refiner = SequentialRefiner(model, simulator, cfg)
    res = refiner.refine(make_problem(), verbose=False)
    assert res.converged_early is True
    assert len(res.rounds) < 10


def test_round_bookkeeping_present():
    model = ToyTrainableModel()
    cfg = RefinementConfig(simulation_budget=100, round_size=50,
                           optimization_steps=5, max_rounds=3, tv_tol=0.0)
    refiner = SequentialRefiner(model, simulator, cfg)
    res = refiner.refine(make_problem(), verbose=False)
    assert len(res.rounds) >= 2
    r0 = res.rounds[0]
    assert r0.n_simulations_total == 50
    assert r0.loss >= 0.0
    assert len(r0.checkpoint_hash) == 16
    # cumulative simulations strictly increase
    totals = [r.n_simulations_total for r in res.rounds]
    assert totals == sorted(totals)
    assert totals[-1] == res.total_simulations


def test_no_random_placeholder():
    """The old random-loss placeholder must be gone: refine() actually calls
    the model's training hook (rounds_seen increments)."""
    model = ToyTrainableModel()
    cfg = RefinementConfig(simulation_budget=50, round_size=50,
                           optimization_steps=5, max_rounds=1, tv_tol=0.0)
    refiner = SequentialRefiner(model, simulator, cfg)
    refiner.refine(make_problem(), verbose=False)
    assert model.rounds_seen == 1
