"""Tests for the P3-01 strict PEtab parameter-token alignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import petab
import pytest

from iaapi.data.param_alignment import (
    AlignmentError,
    LabeledFIM,
    LabeledPosteriorTokens,
    ParameterAlignment,
)

PETAB_ROOT = (Path(__file__).resolve().parents[2] / "Benchmark-Models-PEtab" / "Benchmark-Models").resolve()
FIVE_MODELS = ["Boehm_JProteomeRes2014", "Elowitz_Nature2000", "Fujita_SciSignal2010",
               "Raia_CancerResearch2011", "Schwen_PONE2014"]


def _free_ids(model: str) -> list:
    yaml = PETAB_ROOT / model / f"{model}.yaml"
    if not yaml.exists():
        candidates = sorted((PETAB_ROOT / model).glob("*.yaml"))
        if not candidates:
            pytest.skip("PEtab benchmark models are not bundled with the public package")
        yaml = candidates[0]
    p = petab.Problem.from_yaml(str(yaml))
    df = p.parameter_df
    ids = list(df["parameterId"].astype(str)) if "parameterId" in df.columns else [str(i) for i in df.index]
    if "estimate" in df.columns:
        mask = df["estimate"].astype(bool)
        ids = [pid for pid, m in zip(ids, mask) if bool(m)]
    return ids


def test_round_trip_five_heterogeneous_models():
    sizes = []
    for m in FIVE_MODELS:
        ids = _free_ids(m)
        assert len(ids) >= 3
        sizes.append(len(ids))
        al = ParameterAlignment(ids)
        v = np.random.default_rng(0).normal(size=al.n_parameters)
        assert al.round_trip(v) is True
        # FIM + bounds + tokens all align at this size
        fim = al.assert_fim(np.eye(al.n_parameters))
        assert isinstance(fim, LabeledFIM) and fim.param_ids == al.param_ids
        al.assert_bounds(np.column_stack([np.zeros(al.n_parameters), np.ones(al.n_parameters)]))
        toks = al.label_posterior_tokens(list(v), list(ids))
        al.assert_posterior_tokens(toks)
    # heterogeneous sizes confirmed
    assert len(set(sizes)) >= 3


def test_duplicate_ids_fail_loudly():
    with pytest.raises(AlignmentError, match="duplicate"):
        ParameterAlignment(["a", "b", "a"])


def test_empty_ids_fail():
    with pytest.raises(AlignmentError, match="empty"):
        ParameterAlignment([])


def test_align_vector_missing_id_fails():
    al = ParameterAlignment(["a", "b", "c"])
    with pytest.raises(AlignmentError, match="missing"):
        al.align_vector([1.0, 2.0], ["a", "b"])  # missing c


def test_align_vector_extra_id_fails():
    al = ParameterAlignment(["a", "b", "c"])
    with pytest.raises(AlignmentError, match="extra"):
        al.align_vector([1.0, 2.0, 3.0, 4.0], ["a", "b", "c", "z"])


def test_align_vector_length_mismatch_fails():
    al = ParameterAlignment(["a", "b", "c"])
    with pytest.raises(AlignmentError, match="length"):
        al.align_vector([1.0, 2.0, 3.0], ["a", "b"])  # 3 vec vs 2 ids


def test_align_vector_recovers_canonical_order():
    al = ParameterAlignment(["a", "b", "c"])
    # given in [c, a, b] order
    out = al.align_vector([30.0, 10.0, 20.0], ["c", "a", "b"])
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])


def test_fim_shape_mismatch_fails():
    al = ParameterAlignment(["a", "b", "c"])
    with pytest.raises(AlignmentError, match="FIM shape"):
        al.assert_fim(np.eye(2))


def test_fim_carries_param_ids():
    al = ParameterAlignment(["a", "b"])
    fim = al.label_fim(np.array([[1.0, 0.2], [0.2, 3.0]]))
    assert fim.param_ids == ("a", "b")
    assert fim.matrix.shape == (2, 2)


def test_posterior_tokens_missing_fails():
    al = ParameterAlignment(["a", "b", "c"])
    with pytest.raises(AlignmentError, match="missing"):
        al.label_posterior_tokens([1, 2], ["a", "b"])


def test_posterior_tokens_reordered_to_canonical():
    al = ParameterAlignment(["a", "b", "c"])
    toks = al.label_posterior_tokens([30, 10, 20], ["c", "a", "b"])
    assert toks.param_ids == ["a", "b", "c"]
    assert toks.tokens == [10, 20, 30]
    al.assert_posterior_tokens(toks)


def test_assert_vector_length():
    al = ParameterAlignment(["a", "b", "c"])
    al.assert_vector([1.0, 2.0, 3.0])
    with pytest.raises(AlignmentError, match="length"):
        al.assert_vector([1.0, 2.0])


def test_from_petab_problem_consistent_with_ids():
    # from_petab_problem on a real model gives the same IDs as _free_ids.
    ids = _free_ids("Boehm_JProteomeRes2014")
    p = petab.Problem.from_yaml(str(PETAB_ROOT / "Boehm_JProteomeRes2014" / "Boehm_JProteomeRes2014.yaml"))

    class _Wrapper:
        parameters = p.parameter_df
    al = ParameterAlignment.from_petab_problem(_Wrapper())
    assert list(al.param_ids) == ids
