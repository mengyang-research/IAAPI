"""Candidate intervention interface (NCS plan task P5-01).

Interventions that alter the model's sensitivity directions, so the ISP geometry
can be re-evaluated for prospective experimental design. Each intervention takes
a base FIM (+ extra sensitivity context) and returns the post-intervention FIM;
the eigendirections/geometry are then recomputed and the before/after delta is
auditable.

Primary intervention types (>=3 required; simple repetition/noise scaling is a
CONTROL, not primary):
  * NewObservable          add new observable sensitivities  -> FIM += J^T J
  * Perturbation           re-evaluate FIM at a perturbed operating point
  * InitialCondition       add sensitivity w.r.t. an initial condition
  * Stimulus               add stimulus-driven observable sensitivities
  * CompartmentMeasurement add compartment-specific measurement sensitivities

Supporting transforms:
  * Reparameterisation     FIM' = T^{-T} FIM T^{-1}  (Jacobian dθ/dθ')
  * FixedParameterConstraint  marginal FIM over the free parameters
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"

PRIMARY_KINDS = {"new_observable", "perturbation", "initial_condition",
                 "stimulus", "compartment_measurement"}


@dataclass(frozen=True)
class Intervention:
    kind: str = ""
    label: str = ""
    is_primary: bool = True

    def apply(self, fim: np.ndarray, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError


def _sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


@dataclass(frozen=True)
class NewObservable(Intervention):
    """Add new observable(s) with sensitivity J (n_new, n_param). FIM += w J^T J."""
    J: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "kind", "new_observable")

    def apply(self, fim, ctx=None):
        J = np.atleast_2d(np.asarray(self.J, float))
        return _sym(np.asarray(fim, float) + self.weight * (J.T @ J))


@dataclass(frozen=True)
class Perturbation(Intervention):
    """Re-evaluate FIM at a perturbed operating point (new_fim provided)."""
    new_fim: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self):
        object.__setattr__(self, "kind", "perturbation")

    def apply(self, fim, ctx=None):
        return _sym(np.asarray(self.new_fim, float))


@dataclass(frozen=True)
class InitialCondition(Intervention):
    """Add sensitivity J_ic (n_obs, n_param) to an initial condition. FIM += J_ic^T J_ic."""
    J_ic: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self):
        object.__setattr__(self, "kind", "initial_condition")

    def apply(self, fim, ctx=None):
        J = np.atleast_2d(np.asarray(self.J_ic, float))
        return _sym(np.asarray(fim, float) + J.T @ J)


@dataclass(frozen=True)
class Stimulus(Intervention):
    """Add stimulus-driven observable sensitivities J_stim. FIM += J_stim^T J_stim."""
    J_stim: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self):
        object.__setattr__(self, "kind", "stimulus")

    def apply(self, fim, ctx=None):
        J = np.atleast_2d(np.asarray(self.J_stim, float))
        return _sym(np.asarray(fim, float) + J.T @ J)


@dataclass(frozen=True)
class CompartmentMeasurement(Intervention):
    """Add compartment-specific measurement sensitivities J_comp. FIM += J_comp^T J_comp."""
    J_comp: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self):
        object.__setattr__(self, "kind", "compartment_measurement")

    def apply(self, fim, ctx=None):
        J = np.atleast_2d(np.asarray(self.J_comp, float))
        return _sym(np.asarray(fim, float) + J.T @ J)


@dataclass(frozen=True)
class Reparameterisation(Intervention):
    """FIM' = T^{-T} FIM T^{-1} with Jacobian T = dθ/dθ' (n x n, invertible)."""
    T: np.ndarray = field(default_factory=lambda: np.eye(1))
    is_primary: bool = False

    def __post_init__(self):
        object.__setattr__(self, "kind", "reparameterisation")

    def apply(self, fim, ctx=None):
        T = np.asarray(self.T, float)
        Tinv = np.linalg.inv(T)
        return _sym(Tinv.T @ np.asarray(fim, float) @ Tinv)


@dataclass(frozen=True)
class FixedParameterConstraint(Intervention):
    """Marginal FIM over the free parameters (condition on the fixed ones)."""
    free_idx: Sequence[int] = field(default_factory=tuple)
    is_primary: bool = False

    def __post_init__(self):
        object.__setattr__(self, "kind", "fixed_parameter_constraint")

    def apply(self, fim, ctx=None):
        idx = list(self.free_idx)
        return _sym(np.asarray(fim, float)[np.ix_(idx, idx)])


# --------------------------------------------------------------------------- #
# Geometry recomputation + auditable delta
# --------------------------------------------------------------------------- #
def recompute_geometry(fim: np.ndarray) -> Dict[str, Any]:
    """Eigenvalues (descending) + effective rank (participation ratio) of FIM."""
    fim = _sym(np.asarray(fim, float))
    eig = np.linalg.eigvalsh(fim)
    eig = np.clip(eig, 0.0, None)
    s = eig.sum()
    ss = (eig ** 2).sum()
    eff = float((s ** 2) / ss) if ss > 0 else 0.0
    return {"eigenvalues": eig[::-1].tolist(), "effective_rank": eff, "n_param": fim.shape[0]}


def _topk_subspace(fim: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal top-k eigenvector matrix (d, k) of fim."""
    fim = _sym(np.asarray(fim, float))
    w, V = np.linalg.eigh(fim)
    order = np.argsort(w)[::-1][:k]
    Q = V[:, order]
    # orthonormalise (eigh already returns orthonormal V, but guard)
    Q, _ = np.linalg.qr(Q)
    return Q


def subspace_distance(fim_before: np.ndarray, fim_after: np.ndarray, k: int) -> float:
    """Grassmannian projector distance between top-k subspaces (0=identical, 2=max)."""
    d = np.asarray(fim_before, float).shape[0]
    k = min(k, d)
    Qb = _topk_subspace(fim_before, k)
    Qa = _topk_subspace(fim_after, k)
    Pb = Qb @ Qb.T
    Pa = Qa @ Qa.T
    return float(np.linalg.norm(Pb - Pa, "fro"))


def geometry_delta(fim_before: np.ndarray, fim_after: np.ndarray,
                   top_k: Optional[int] = None) -> Dict[str, Any]:
    """Auditable before/after ISP-geometry delta."""
    gb = recompute_geometry(fim_before)
    ga = recompute_geometry(fim_after)
    k = top_k or min(gb["n_param"], ga["n_param"])
    try:
        sd = subspace_distance(fim_before, fim_after, k) if gb["n_param"] == ga["n_param"] else None
    except Exception:
        sd = None
    eb = np.asarray(gb["eigenvalues"])
    ea = np.asarray(ga["eigenvalues"])
    return {
        "effective_rank_before": gb["effective_rank"],
        "effective_rank_after": ga["effective_rank"],
        "effective_rank_delta": ga["effective_rank"] - gb["effective_rank"],
        "subspace_distance_topk": sd,
        "eigenvalues_before": gb["eigenvalues"],
        "eigenvalues_after": ga["eigenvalues"],
        "max_eigenvalue_ratio_after_before": float(ea.max() / max(eb.max(), 1e-12)),
        "n_param_before": gb["n_param"],
        "n_param_after": ga["n_param"],
    }


def apply_intervention_chain(fim: np.ndarray, interventions: Sequence[Intervention],
                             ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply a sequence of interventions; returns the final FIM."""
    cur = np.asarray(fim, float)
    for iv in interventions:
        cur = iv.apply(cur, ctx)
    return cur


def audit_intervention(fim_before: np.ndarray, interventions: Sequence[Intervention],
                        top_k: Optional[int] = None) -> Dict[str, Any]:
    """Apply interventions and return the auditable geometry delta + per-step kinds."""
    fim_after = apply_intervention_chain(fim_before, interventions)
    return {
        "schema_version": SCHEMA_VERSION,
        "intervention_kinds": [iv.kind for iv in interventions],
        "n_primary": sum(1 for iv in interventions if iv.is_primary and iv.kind in PRIMARY_KINDS),
        "delta": geometry_delta(fim_before, fim_after, top_k),
        "fim_after": fim_after.tolist(),
    }
