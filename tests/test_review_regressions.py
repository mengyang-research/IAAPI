"""Regression checks for scientific-data handling; no manuscript data is altered."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from iaapi.evaluation import mcmc
from iaapi.evaluation.baselines import BaselineRunner
from iaapi.evaluation.coverage import CoverageEvaluator
from iaapi.evaluation.fim_descriptors import compute_fim_descriptors
from iaapi.evaluation.stat_protocol import (
    SealedHoldout, _ridge_loo_mse, fit_ridge, ridge_fit_predict,
)
from iaapi.oed.oed_module import OEDModule


def test_covariance_uses_every_walker_and_counts_draws(monkeypatch):
    """Old walker averaging loses the dominant variance in this fixture."""
    rng = np.random.default_rng(9)
    raw = rng.normal(0, 0.03, size=(4, 8, 100, 2))
    raw += np.linspace(-1.5, 1.5, 8)[None, :, None, None]

    class FakeSampler:
        next_ensemble = 0

        def __init__(self, *args):
            self.ensemble = FakeSampler.next_ensemble
            FakeSampler.next_ensemble += 1
            self.acceptance_fraction = np.full(8, 0.5)

        def run_mcmc(self, *args, **kwargs):
            pass

        def get_chain(self, **kwargs):
            return raw[self.ensemble].transpose(1, 0, 2)

    monkeypatch.setattr(mcmc.emcee, "EnsembleSampler", FakeSampler)
    runner = mcmc.MCMCRunner(
        lambda theta: -float(theta @ theta), np.array([[-5, 5], [-5, 5]]),
        np.zeros(2), ["a", "b"],
        mcmc.MCMCConfig(n_walkers=8, n_steps=100, n_burn=0, n_ensembles=4),
    )
    result = runner.run()
    expected = np.cov(raw.reshape(-1, 2), rowvar=False)
    np.testing.assert_allclose(result.posterior_cov, expected)
    assert result.n_samples == 4 * 8 * 100
    assert result.diagnostic_samples_per_walker == 4 * 100
    assert not result.label_allowed  # fixed walkers in distinct locations


def test_rhat_detects_different_chain_locations():
    rng = np.random.default_rng(7)
    chains = rng.normal(size=(4, 1000, 1))
    chains[2:] += 4.0
    assert mcmc.split_rhat(chains)[0] > 1.2


def test_folded_rhat_detects_different_scales():
    rng = np.random.default_rng(10)
    chains = rng.normal(size=(4, 1000, 1))
    chains[2:] *= 10.0
    assert mcmc.split_rhat(chains)[0] > 1.2


def test_constant_chains_cannot_pass_convergence():
    chains = np.zeros((4, 100, 1))
    assert not np.isfinite(mcmc.split_rhat(chains)[0])
    assert mcmc.effective_sample_size(chains)[0] == 0
    assert mcmc.tail_effective_sample_size(chains)[0] == 0


def test_resume_rejects_legacy_covariance_summaries(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"schema_version": "1.0"}))
    runner = mcmc.MCMCRunner(lambda x: -float(x @ x), np.array([[-2, 2]]),
                             np.zeros(1), ["a"])
    with pytest.raises(ValueError, match="Legacy"):
        runner.run_versioned(tmp_path)


def test_saved_chain_covariance_reproduces_summary(tmp_path):
    cfg = mcmc.MCMCConfig(n_walkers=8, n_steps=50, n_burn=10, n_ensembles=4, seed=6)
    runner = mcmc.MCMCRunner(lambda x: -float(x @ x), np.array([[-5, 5]]),
                             np.zeros(1), ["a"], cfg)
    summary = runner.run_versioned(tmp_path)
    with np.load(tmp_path / "chains.npz") as stored:
        raw = stored["chains"]
        assert raw.shape == (4, 8, 50, 1)
        expected = np.atleast_2d(np.cov(raw.reshape(-1, 1), rowvar=False))
    np.testing.assert_allclose(summary["posterior_cov"], expected)


def test_holdout_predictions_do_not_depend_on_holdout_labels():
    X = np.arange(30.0)[:, None]
    fitted = fit_ridge(X[:20], 2 * X[:20, 0] + 5, 1.0)
    a = SealedHoldout(X[20:], X[20:, 0], predictor=fitted).evaluate_once()
    b = SealedHoldout(X[20:], -100 * X[20:, 0], predictor=fitted).evaluate_once()
    np.testing.assert_array_equal(a["predictions"], b["predictions"])
    assert a["fit_on_holdout"] is False


def test_holdout_never_calls_a_training_function(monkeypatch):
    from iaapi.evaluation import stat_protocol
    X = np.arange(30.0)[:, None]
    fitted = fit_ridge(X[:20], X[:20, 0], 1.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting inside holdout evaluation is forbidden")

    monkeypatch.setattr(stat_protocol, "fit_ridge", forbidden)
    monkeypatch.setattr(stat_protocol, "ridge_fit_predict", forbidden)
    SealedHoldout(X[20:], X[20:, 0], predictor=fitted).evaluate_once()


def test_training_holdout_model_overlap_is_rejected():
    fitted = fit_ridge(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), 1.0,
                       training_model_ids=["A", "B"])
    with pytest.raises(ValueError, match="overlap"):
        SealedHoldout(np.array([[2.0], [3.0]]), np.array([2.0, 3.0]),
                      predictor=fitted, model_ids=["B", "C"])


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_inner_cv_matches_actual_train_only_refits(alpha):
    X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [100.0, 1.0]])
    y = np.array([1.0, 2.0, 4.0, 7.0])
    predictions = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        predictions.append(ridge_fit_predict(X[keep], y[keep], X[i:i + 1], alpha)[0])
    expected = np.mean((y - predictions) ** 2)
    assert _ridge_loo_mse(X, y, alpha) == pytest.approx(expected)


def test_decay_slope_matches_geometric_spectrum_and_version_is_recorded():
    result = compute_fim_descriptors(np.diag(10.0 ** -np.arange(6)))
    assert result["feature_version"] == "2.0"
    assert result["descriptors"][7] == pytest.approx(1.0)
    assert result["descriptors"][6] == pytest.approx(1.11 / 1.11111)


def test_fim_scale_transform_is_invariant_to_parameter_units():
    F = np.array([[10.0, 2.0], [2.0, 3.0]])
    C = np.array([[4.0, 0.5], [0.5, 1.0]])
    A = np.diag([100.0, 0.01])
    inv = np.linalg.inv(A)
    original = compute_fim_descriptors(F, prior_cov=C)
    converted = compute_fim_descriptors(inv.T @ F @ inv, prior_cov=A @ C @ A.T)
    np.testing.assert_allclose(original["eigenvalues"], converted["eigenvalues"], rtol=1e-8)
    np.testing.assert_allclose(original["descriptors"], converted["descriptors"], atol=1e-8)


def test_explicit_legacy_features_are_not_silently_renamed():
    legacy = compute_fim_descriptors(np.eye(12), feature_version="1.0")
    current = compute_fim_descriptors(np.eye(12))
    assert legacy["names"][6:] == ("top_r_information_fraction", "above_threshold_fraction")
    assert legacy["descriptors"][6] == pytest.approx(1 / 3)
    assert current["names"][6:] == ("top_3_information_fraction", "decay_slope")
    assert current["descriptors"][6] == pytest.approx(1 / 4)


def test_coverage_requires_real_samples_and_truth():
    with pytest.raises(NotImplementedError):
        CoverageEvaluator().run(None, [object()])


def test_coverage_uses_actual_coordinate_hits_and_denominators():
    samples = np.stack([np.arange(10.0), np.arange(10.0)], axis=1)
    provider = lambda model, problem, n: (samples, np.array([4.5, 100.0]))
    result = CoverageEvaluator([0.5, 0.95], posterior_provider=provider).run(None, [None])
    assert result["average_coverages"] == {0.5: 0.5, 0.95: 0.5}
    assert result["per_problem"][0]["n_coordinates"] == 2
    assert result["per_problem"][0]["coordinate_hits"][0.95] == [True, False]


def test_hpd_interval_window_has_correct_endpoints():
    evaluator = CoverageEvaluator()
    assert evaluator._hpd_interval(np.array([0, 1, 2, 100.0]), 0.5) == (0.0, 1.0)
    assert evaluator._hpd_interval(np.array([0, 1, 2, 100.0]), 1.0) == (0.0, 100.0)


@pytest.mark.parametrize("method", [
    "run_pypesto_optimization", "run_pypesto_mcmc", "run_pypesto_profile",
    "run_sbi_npe", "run_bayesflow",
])
def test_unimplemented_baselines_cannot_return_random_results(method):
    with pytest.raises(NotImplementedError):
        getattr(BaselineRunner(), method)("unused.yaml")


def test_oed_requires_actual_scoring_backend():
    output = SimpleNamespace(sloppy_subspace=np.eye(2))
    with pytest.raises(NotImplementedError):
        OEDModule().rank_experiments(output, [{"id": "A"}], np.zeros(2), None)


def test_oed_ranks_actual_utilities_and_rejects_nonfinite_values():
    output = SimpleNamespace(sloppy_subspace=np.eye(2))
    module = OEDModule(lambda basis, theta, candidate, sim: candidate["gain"])
    candidates = [{"gain": 2.0}, {"gain": 5.0}, {"gain": 1.0}]
    result = module.rank_experiments(output, candidates, np.zeros(2), None)
    assert result["information_gains"] == [5.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        module.rank_experiments(output, [{"gain": np.nan}], np.zeros(2), None)


def test_biological_problem_sampler_refuses_an_unspecified_mapping(tmp_path):
    # Load this standalone module without the unrelated graph/AMICI imports
    # of the legacy iaapi.data package initializer.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "iaapi" / "data" / "bio_priors.py"
    spec = importlib.util.spec_from_file_location("review_bio_priors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.BioPriorLoader(str(tmp_path))
    with pytest.raises(NotImplementedError, match="parameter-to-type"):
        loader.sample(SimpleNamespace(parameters=["a", "b"]), n_samples=5)
