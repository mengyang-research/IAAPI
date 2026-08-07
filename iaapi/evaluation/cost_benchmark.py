"""End-to-end cost benchmark (NCS plan task P4-03).

Measures ALL test-time work for each method and computes the amortization
break-even against PL / MCMC / per-model-NPE.

Honesty rule (P4-03): if FIM is computed at test time, the ISP per-query cost
MUST include the AMICI forward+sensitivity+FIM stage — it is NOT a single neural
forward. The benchmark enforces this by summing every ``per_query_stages`` entry
into the per-query cost; ``single_forward_claim_valid`` is True only when the
AMICI/FIM stage is absent (or its cost is accounted for and the per-query total
still beats the reference).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

SCHEMA_VERSION = "1.0"

# Stage-name tags (used by the honesty check).
STAGE_GRAPH = "graph_construction"
STAGE_AMICI_FIM = "amici_forward_sensitivity_fim"
STAGE_NEURAL_INFER = "neural_inference"
STAGE_CALIBRATION = "calibration"


@dataclass
class StageTiming:
    name: str
    seconds: float
    note: str = ""


@dataclass
class MethodCost:
    method: str
    per_query_stages: List[StageTiming] = field(default_factory=list)
    training_cost_s: float = 0.0  # one-time training (amortised across queries)
    fim_at_test_time: bool = False

    @property
    def per_query_total_s(self) -> float:
        return float(sum(s.seconds for s in self.per_query_stages))

    @property
    def single_forward_claim_valid(self) -> bool:
        """True iff no unaccounted AMICI/FIM test-time cost => a pure neural
        forward claim is honest. If fim_at_test_time, the FIM stage MUST be
        present in per_query_stages (accounted); the claim is then valid only
        in the sense that all cost is counted (NOT that it is a single forward)."""
        if self.fim_at_test_time:
            return any(s.name == STAGE_AMICI_FIM for s in self.per_query_stages)
        return True


def break_even_n(amortized: MethodCost, reference_per_query_s: float) -> Optional[int]:
    """Smallest N>=1 where amortized total <= N * reference per-query.

    amortized_total(N) = training + N * per_query.  reference(N) = N * reference.
    Solve training + N*pq <= N*ref  =>  N >= training / (ref - pq).
    Returns None (never breaks even) if pq >= ref.
    """
    pq = amortized.per_query_total_s
    if pq >= reference_per_query_s:
        return None
    if amortized.training_cost_s <= 0:
        return 1
    n = amortized.training_cost_s / (reference_per_query_s - pq)
    return max(1, int(math.ceil(n)))


def amortized_cost(amortized: MethodCost, n_queries: int) -> float:
    return amortized.training_cost_s + n_queries * amortized.per_query_total_s


def _time_call(fn: Callable, repeat: int = 1) -> float:
    """Median wall-clock seconds of ``repeat`` calls to fn (returns the mean of
    the post-warmup runs)."""
    runs = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    runs.sort()
    return float(runs[len(runs) // 2])


@dataclass
class BenchmarkReport:
    schema_version: str = SCHEMA_VERSION
    methods: Dict[str, dict] = field(default_factory=dict)
    break_even: Dict[str, Optional[int]] = field(default_factory=dict)
    single_forward_claim_valid: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "methods": self.methods,
            "break_even_vs_pl": self.break_even,
            "single_forward_claim_valid": self.single_forward_claim_valid,
            "notes": self.notes,
        }


def build_report(isp: MethodCost, pl: MethodCost, mcmc: MethodCost,
                 npe: MethodCost) -> BenchmarkReport:
    """Assemble per-method costs + break-even of the amortised methods vs PL."""
    rep = BenchmarkReport()
    for m in (isp, pl, mcmc, npe):
        rep.methods[m.method] = {
            "per_query_stages": [{"name": s.name, "seconds": round(s.seconds, 6), "note": s.note}
                                 for s in m.per_query_stages],
            "per_query_total_s": round(m.per_query_total_s, 6),
            "training_cost_s": round(m.training_cost_s, 6),
            "fim_at_test_time": m.fim_at_test_time,
        }
        rep.single_forward_claim_valid[m.method] = m.single_forward_claim_valid
    # break-even of amortised methods (isp, npe) vs PL per-query
    pl_pq = pl.per_query_total_s
    for name, m in (("isp", isp), ("per_model_npe", npe)):
        n = break_even_n(m, pl_pq)
        rep.break_even[name] = n
        if m.fim_at_test_time and not m.single_forward_claim_valid:
            rep.notes.append(f"{name}: FIM at test time but NOT accounted in per_query — single-forward claim INVALID")
        if m.fim_at_test_time and m.per_query_total_s >= pl_pq:
            rep.notes.append(f"{name}: per-query (incl FIM) {m.per_query_total_s:.4g}s >= PL {pl_pq:.4g}s — no break-even; cannot claim amortised speedup over PL")
    return rep
