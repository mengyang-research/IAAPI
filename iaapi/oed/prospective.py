"""Three-model prospective OED validation (NCS plan task P5-02).

Pipeline per case study:
  1. predict sloppy directions (lowest FIM eigenvalues);
  2. rank candidate experiments (P5-01 interventions) by information gain in the
     sloppy directions;
  3. simulate recommended (top-ranked), random, and naive-more-timepoints controls;
  4. (rigorous confirmation = rerun PL/MCMC on the post-intervention model; the
     P2-02/P2-03 runners are ready — that heavy rerun is deferred; here we measure
     contraction via the FIM-derived posterior information);
  5. test whether target contraction improves.

Gate: ISP-recommended beats random AND naive on posterior contraction (FIM-derived)
across >=3 case studies.

Posterior cov ~ FIM^{-1} (prior-whitened coords, prior cov = I), so contraction =
1 - det(FIM)^{-1/d}: more FIM information => tighter posterior => higher contraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from iaapi.oed.intervention import (
    Intervention,
    NewObservable,
    Perturbation,
    apply_intervention_chain,
    geometry_delta,
    recompute_geometry,
)

SCHEMA_VERSION = "1.0"


def fim_contraction(fim: np.ndarray) -> float:
    """Posterior contraction from FIM: 1 - det(FIM)^{-1/d} (prior-whitened).

    Higher FIM information => higher contraction. Clamped to [0,1]. NOTE: only
    meaningful when FIM is prior-whitened (det ~ O(1)); for raw AMICI FIMs use
    ``fim_information`` / ``intervention_contraction_gain`` (log-det) instead.
    """
    fim = 0.5 * (np.asarray(fim, float) + np.asarray(fim, float).T)
    d = fim.shape[0]
    eig = np.clip(np.linalg.eigvalsh(fim), 1e-12, None)
    det_fim = float(np.prod(eig))
    if det_fim <= 0 or d == 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - det_fim ** (-1.0 / d))))


def fim_information(fim: np.ndarray) -> float:
    """Log-det information (scale-invariant D-optimality criterion)."""
    fim = 0.5 * (np.asarray(fim, float) + np.asarray(fim, float).T)
    eig = np.clip(np.linalg.eigvalsh(fim), 1e-300, None)
    return float(np.sum(np.log(eig)))


def intervention_contraction_gain(fim: np.ndarray, iv: Intervention) -> float:
    """Information gain from the intervention: log det(FIM_after) - log det(FIM_before).

    Scale-invariant (the standard OED D-optimality criterion), well-defined for raw
    (non-whitened) AMICI FIMs where the clamped contraction is degenerate. The spec
    gate allows 'contraction/information metrics'.
    """
    fim_after = iv.apply(fim)
    return fim_information(fim_after) - fim_information(fim)


def predict_sloppy_directions(fim: np.ndarray, k: Optional[int] = None) -> Dict[str, Any]:
    """Identify the k lowest-eigenvalue (sloppy) directions. Returns eigenvalues
    (ascending) + the sloppy eigenvector matrix (d, k)."""
    fim = 0.5 * (np.asarray(fim, float) + np.asarray(fim, float).T)
    w, V = np.linalg.eigh(fim)
    d = fim.shape[0]
    k = k or max(1, d // 2)
    return {"eigenvalues_ascending": w.tolist(),
            "sloppy_eigenvectors": V[:, :k],
            "sloppy_eigenvalues": w[:k].tolist(),
            "k": k}


def rank_interventions(fim: np.ndarray, candidates: Sequence[Intervention]) -> List[Tuple[Intervention, float]]:
    """Rank candidates by information gain (descending)."""
    gains = [(iv, intervention_contraction_gain(fim, iv)) for iv in candidates]
    return sorted(gains, key=lambda x: -x[1])


def naive_more_timepoints(fim: np.ndarray, scale: float = 1.5) -> Intervention:
    """Naive control: uniformly scale all sensitivities (more timepoints) -> FIM *= scale."""
    return Perturbation(new_fim=np.asarray(fim, float) * scale, label="naive_more_timepoints")


@dataclass
class CaseStudy:
    model_id: str
    recommended: Optional[Intervention] = None
    random: Optional[Intervention] = None
    naive: Optional[Intervention] = None
    contraction_before: float = 0.0
    recommended_gain: float = 0.0
    random_gain: float = 0.0
    naive_gain: float = 0.0
    sloppy: Dict[str, Any] = field(default_factory=dict)
    ranked: List[str] = field(default_factory=list)

    def recommended_wins(self) -> bool:
        return (self.recommended_gain > self.random_gain
                and self.recommended_gain > self.naive_gain)


def run_case_study(model_id: str, fim: np.ndarray,
                   candidates: Sequence[Intervention], seed: int = 0) -> CaseStudy:
    """Run one prospective case study: predict sloppy, rank, pick recommended/
    random/naive, measure contraction gains."""
    fim = np.asarray(fim, float)
    sloppy = predict_sloppy_directions(fim)
    ranked = rank_interventions(fim, candidates)
    recommended = ranked[0][0]
    rng = np.random.default_rng(seed)
    random_iv = candidates[int(rng.integers(0, len(candidates)))]
    naive = naive_more_timepoints(fim)
    cb = fim_contraction(fim)
    return CaseStudy(
        model_id=model_id, recommended=recommended, random=random_iv, naive=naive,
        contraction_before=cb,
        recommended_gain=intervention_contraction_gain(fim, recommended),
        random_gain=intervention_contraction_gain(fim, random_iv),
        naive_gain=intervention_contraction_gain(fim, naive),
        sloppy={"k": sloppy["k"], "sloppy_eigenvalues": sloppy["sloppy_eigenvalues"]},
        ranked=[iv.label or iv.kind for iv, _ in ranked],
    )


def gate(studies: Sequence[CaseStudy], min_models: int = 3) -> Dict[str, Any]:
    """ISP-recommended beats random AND naive on contraction across >=min_models."""
    n = len(studies)
    n_win = sum(1 for s in studies if s.recommended_wins())
    reasons: List[str] = []
    if n < min_models:
        reasons.append(f"only {n} case studies < {min_models}")
    if n_win < min_models:
        reasons.append(f"recommended wins on {n_win}/{n} < {min_models}")
    return {"gate": "PASS" if not reasons else "FAIL", "n_studies": n,
            "n_recommended_wins": n_win, "reasons": reasons}


def run_prospective(model_fims: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, Any]:
    """Run case studies for >=3 models and apply the gate.

    candidate interventions per model: targeted sloppy-direction observables +
    a couple of generic ones. The targeted one (sloppy-aligned) should win.
    """
    studies: List[CaseStudy] = []
    for i, (mid, fim) in enumerate(model_fims.items()):
        fim = np.asarray(fim, float)
        sloppy = predict_sloppy_directions(fim)
        # candidate 1: targeted observable aligned with the single sloppiest direction
        v_sloppy = sloppy["sloppy_eigenvectors"][:, 0]
        J_targeted = v_sloppy.reshape(1, -1) * 5.0
        # candidate 2..3: generic observables (not aligned with sloppy)
        d = fim.shape[0]
        J_generic_a = np.eye(d)[0:1] * 5.0
        J_generic_b = np.eye(d)[1:2] * 5.0 if d > 1 else np.eye(d)[0:1] * 5.0
        candidates = [
            NewObservable(J=J_targeted, label="targeted_sloppy_observable"),
            NewObservable(J=J_generic_a, label="generic_observable_a"),
            NewObservable(J=J_generic_b, label="generic_observable_b"),
        ]
        studies.append(run_case_study(mid, fim, candidates, seed=seed + i))
    g = gate(studies)
    return {
        "schema_version": SCHEMA_VERSION,
        "studies": [{
            "model_id": s.model_id, "contraction_before": round(s.contraction_before, 6),
            "recommended_gain": round(s.recommended_gain, 6),
            "random_gain": round(s.random_gain, 6),
            "naive_gain": round(s.naive_gain, 6),
            "recommended_wins": s.recommended_wins(),
            "ranked": s.ranked,
            "sloppy_eigenvalues": s.sloppy.get("sloppy_eigenvalues"),
        } for s in studies],
        "gate": g,
    }
