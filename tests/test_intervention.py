"""Tests for the P5-01 candidate intervention interface."""
from __future__ import annotations

import math

import numpy as np
import pytest

from iaapi.oed.intervention import (
    PRIMARY_KINDS,
    CompartmentMeasurement,
    FixedParameterConstraint,
    InitialCondition,
    NewObservable,
    Perturbation,
    Reparameterisation,
    Stimulus,
    apply_intervention_chain,
    audit_intervention,
    geometry_delta,
    recompute_geometry,
    subspace_distance,
)


def _fim():
    # rank-2 FIM in 3 params: large eig on params 0,1; tiny on param 2.
    return np.diag([10.0, 1.0, 0.001])


def test_at_least_three_primary_types():
    types = [NewObservable, Perturbation, InitialCondition, Stimulus, CompartmentMeasurement]
    assert len(types) >= 3
    instances = [NewObservable(J=np.ones((1, 3))), Perturbation(new_fim=np.eye(3)),
                 InitialCondition(J_ic=np.ones((1, 3))), Stimulus(J_stim=np.ones((1, 3))),
                 CompartmentMeasurement(J_comp=np.ones((1, 3)))]
    for iv in instances:
        assert iv.kind in PRIMARY_KINDS
        assert iv.is_primary is True


def test_repetition_not_primary():
    assert "repetition" not in PRIMARY_KINDS
    assert "noise_scaling" not in PRIMARY_KINDS


def test_new_observable_increases_information():
    fim = _fim()
    J = np.array([[0.0, 0.0, 5.0]])  # informs the sloppy param 2
    iv = NewObservable(J=J, weight=1.0)
    fim2 = iv.apply(fim)
    gb = recompute_geometry(fim)
    ga = recompute_geometry(fim2)
    assert ga["effective_rank"] > gb["effective_rank"]
    assert ga["eigenvalues"][0] >= gb["eigenvalues"][0]


def test_perturbation_swaps_fim():
    fim = _fim()
    new = np.diag([1.0, 1.0, 1.0])
    fim2 = Perturbation(new_fim=new).apply(fim)
    np.testing.assert_allclose(fim2, new)


def test_initial_condition_stimulus_compartment_add_info():
    fim = _fim()
    for cls, attr in [(InitialCondition, "J_ic"), (Stimulus, "J_stim"),
                      (CompartmentMeasurement, "J_comp")]:
        iv = cls(**{attr: np.array([[3.0, 0.0, 0.0]])})
        fim2 = iv.apply(fim)
        assert recompute_geometry(fim2)["eigenvalues"][0] > recompute_geometry(fim)["eigenvalues"][0]


def test_reparameterisation_identity_unchanged():
    fim = _fim()
    iv = Reparameterisation(T=np.eye(3))
    fim2 = iv.apply(fim)
    np.testing.assert_allclose(fim2, fim, atol=1e-8)


def test_reparameterisation_scaling():
    fim = _fim()
    T = 2.0 * np.eye(3)  # theta' = theta/2 => FIM' = (1/4) FIM
    fim2 = Reparameterisation(T=T).apply(fim)
    np.testing.assert_allclose(fim2, fim / 4.0, atol=1e-6)


def test_fixed_parameter_constraint_submatrix():
    fim = _fim()
    iv = FixedParameterConstraint(free_idx=(0, 1))  # fix param 2
    fim2 = iv.apply(fim)
    assert fim2.shape == (2, 2)
    np.testing.assert_allclose(fim2, np.diag([10.0, 1.0]))


def test_subspace_distance_zero_for_same_and_positive_for_different():
    fim = _fim()
    assert subspace_distance(fim, fim, k=2) < 1e-6
    fim2 = np.diag([0.001, 10.0, 1.0])  # rotated eigenvalue order
    assert subspace_distance(fim, fim2, k=1) > 0.5


def test_geometry_delta_fields():
    fim = _fim()
    fim2 = NewObservable(J=np.array([[0.0, 0.0, 5.0]])).apply(fim)
    d = geometry_delta(fim, fim2, top_k=2)
    assert "effective_rank_before" in d and "effective_rank_after" in d
    assert d["effective_rank_delta"] > 0
    assert d["subspace_distance_topk"] is not None
    assert d["max_eigenvalue_ratio_after_before"] >= 1.0


def test_apply_intervention_chain_and_audit():
    fim = _fim()
    chain = [NewObservable(J=np.array([[0.0, 0.0, 5.0]])),
             Perturbation(new_fim=np.diag([10.0, 1.0, 5.0]))]
    fim_after = apply_intervention_chain(fim, chain)
    assert fim_after.shape == (3, 3)
    rep = audit_intervention(fim, chain, top_k=2)
    assert rep["n_primary"] == 2
    assert rep["intervention_kinds"] == ["new_observable", "perturbation"]
    assert rep["delta"]["effective_rank_after"] >= rep["delta"]["effective_rank_before"]


def test_audit_supporting_transforms_flagged():
    fim = _fim()
    rep = audit_intervention(fim, [Reparameterisation(T=np.eye(3)),
                                    FixedParameterConstraint(free_idx=(0, 1))])
    assert rep["n_primary"] == 0  # supporting transforms are not primary
    assert rep["delta"]["n_param_after"] == 2  # fixed-constraint reduced dim
