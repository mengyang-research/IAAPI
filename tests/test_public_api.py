import json

import numpy as np

import iaapi
from iaapi.api import InferenceResult
from iaapi.cli import build_parser


def test_top_level_import_is_lightweight():
    assert iaapi.__version__ == "0.1.0"


def test_inference_result_round_trip_preserves_diagnostics(tmp_path):
    result = InferenceResult(
        posterior_samples=np.array([[1.0, 2.0], [3.0, 4.0]]),
        parameter_names=["a", "b"],
        identifiability_scores=np.array([0.9, 0.1]),
        sloppy_subspace_basis=np.array([[0.0], [1.0]]),
        sloppy_dimension=1,
        model_name="toy",
        model_description="toy model",
        log_probabilities=np.array([-1.0, -2.0]),
    )
    path = tmp_path / "result.json"
    result.save(path)
    loaded = InferenceResult.load(path)

    np.testing.assert_allclose(loaded.posterior_samples, result.posterior_samples)
    np.testing.assert_allclose(loaded.sloppy_subspace_basis, result.sloppy_subspace_basis)
    np.testing.assert_allclose(loaded.log_probabilities, result.log_probabilities)
    assert json.loads(path.read_text())["parameter_names"] == ["a", "b"]


def test_cli_parser_accepts_inspect_json():
    args = build_parser().parse_args(["inspect", "problem.yaml", "--json"])
    assert args.command == "inspect"
    assert args.json is True
