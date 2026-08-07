"""Tests for the P4-03 cost benchmark."""
from __future__ import annotations

import math

import pytest

from iaapi.evaluation.cost_benchmark import (
    STAGE_AMICI_FIM,
    STAGE_NEURAL_INFER,
    MethodCost,
    StageTiming,
    _time_call,
    amortized_cost,
    break_even_n,
    build_report,
)


def _method(name, stages, training=0.0, fim=False):
    return MethodCost(method=name, per_query_stages=[StageTiming(n, s) for n, s in stages],
                      training_cost_s=training, fim_at_test_time=fim)


def test_per_query_total_sums_stages():
    m = _method("isp", [("a", 0.1), ("b", 0.2), ("c", 0.05)])
    assert math.isclose(m.per_query_total_s, 0.35)


def test_break_even_known():
    # training=100, per_query=0.1, reference=1.0 -> N >= 100/(0.9) = 111.1 -> 112
    m = _method("isp", [("infer", 0.1)], training=100.0)
    assert break_even_n(m, 1.0) == 112


def test_break_even_never_if_slower():
    m = _method("isp", [("fim", 2.0)], training=10.0)  # per-query 2.0 >= ref 1.0
    assert break_even_n(m, 1.0) is None


def test_break_even_no_training_is_one():
    m = _method("isp", [("infer", 0.1)], training=0.0)
    assert break_even_n(m, 1.0) == 1


def test_amortized_cost_formula():
    m = _method("isp", [("infer", 0.5)], training=10.0)
    assert math.isclose(amortized_cost(m, 4), 10.0 + 4 * 0.5)


def test_single_forward_claim_invalid_when_fim_unaccounted():
    # FIM at test time but no FIM stage in per_query -> claim invalid
    m = _method("isp", [(STAGE_NEURAL_INFER, 0.01)], training=5.0, fim=True)
    assert m.single_forward_claim_valid is False
    # add the FIM stage -> accounted -> claim "valid" (all cost counted)
    m2 = _method("isp", [(STAGE_AMICI_FIM, 0.5), (STAGE_NEURAL_INFER, 0.01)], training=5.0, fim=True)
    assert m2.single_forward_claim_valid is True


def test_build_report_assembles_methods_and_honesty_note():
    # ISP with FIM at test time, per-query (incl FIM) >= PL per-query -> no break-even + honesty note
    isp = _method("isp", [(STAGE_AMICI_FIM, 2.0), (STAGE_NEURAL_INFER, 0.01)], training=10.0, fim=True)
    pl = _method("pl", [("profile_one_param", 1.0)])
    mcmc = _method("mcmc", [("sample", 5.0)])
    npe = _method("per_model_npe", [(STAGE_NEURAL_INFER, 0.02)], training=100.0)
    rep = build_report(isp, pl, mcmc, npe)
    assert set(rep.methods) == {"isp", "pl", "mcmc", "per_model_npe"}
    assert rep.break_even["isp"] is None  # isp per-query 2.01 >= pl 1.0
    assert rep.break_even["per_model_npe"] is not None  # 0.02 < 1.0
    assert rep.single_forward_claim_valid["isp"] is True  # FIM accounted
    assert any("no break-even" in n for n in rep.notes)


def test_build_report_isp_breaks_even_when_fim_cheaper_than_pl():
    isp = _method("isp", [(STAGE_AMICI_FIM, 0.2), (STAGE_NEURAL_INFER, 0.01)], training=50.0, fim=True)
    pl = _method("pl", [("profile_one_param", 1.0)])
    mcmc = _method("mcmc", [("sample", 5.0)])
    npe = _method("per_model_npe", [(STAGE_NEURAL_INFER, 0.02)], training=100.0)
    rep = build_report(isp, pl, mcmc, npe)
    # N >= 50 / (1.0 - 0.21) = 63.3 -> 64
    assert rep.break_even["isp"] == 64


def test_time_call_measures_positive():
    def f():
        s = 0.0
        for _ in range(1000):
            s += 1.0
    t = _time_call(f, repeat=3)
    assert t >= 0.0
