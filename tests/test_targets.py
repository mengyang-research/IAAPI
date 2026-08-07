"""Tests for the P2-01 frozen identifiability target semantics.

Analytic covariance fixtures with known answers + unit-invariance checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from iaapi.evaluation.targets import (
    CAT_IDENTIFIABLE,
    CAT_PRACTICAL,
    CAT_STRUCTURAL,
    classify_parameter,
    compute_targets,
    k_from_threshold,
)


def test_isotropic_posterior_analytic():
    d, sigma2 = 4, 0.1
    post = sigma2 * np.eye(d)
    prior = np.eye(d)
    t = compute_targets(post, prior, source="PL")
    assert t.n_parameters == d
    assert t.independent is True
    np.testing.assert_allclose(np.sort(t.eigenvalues), [sigma2] * d, rtol=1e-9)
    assert math.isclose(t.effective_dimension, d, rel_tol=1e-9)
    assert math.isclose(t.contraction_score, 1 - sigma2, rel_tol=1e-9)
    assert t.k_threshold == d  # all eig (0.1) > 1e-2*0.1


def test_rank_r_posterior_analytic():
    eps = 1e-6
    post = np.diag([1.0, 1.0, eps, eps])
    prior = np.eye(4)
    t = compute_targets(post, prior, source="MCMC")
    assert t.independent is True
    assert math.isclose(t.effective_dimension, 2.0, rel_tol=1e-3)
    assert t.k_threshold == 2
    assert math.isclose(t.contraction_score, 1 - math.sqrt(eps), rel_tol=1e-4)


def test_unit_invariance_diagonal_prior():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    post = A @ A.T + 0.1 * np.eye(4)
    prior = np.diag([1.0, 2.0, 0.5, 3.0])
    D = np.diag([10.0, 0.1, 5.0, 2.0])
    t1 = compute_targets(post, prior, source="PL")
    t2 = compute_targets(D @ post @ D, D @ prior @ D, source="PL")
    np.testing.assert_allclose(np.sort(t1.eigenvalues), np.sort(t2.eigenvalues), rtol=1e-8)
    assert math.isclose(t1.contraction_score, t2.contraction_score, rel_tol=1e-8)
    assert math.isclose(t1.effective_dimension, t2.effective_dimension, rel_tol=1e-8)
    assert t1.k_threshold == t2.k_threshold


def test_unit_invariance_full_prior():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(3, 3)); prior = P @ P.T + np.eye(3)
    Q = rng.normal(size=(3, 3)); post = Q @ Q.T + 0.2 * np.eye(3)
    D = np.diag([7.0, 0.3, 4.0])
    t1 = compute_targets(post, prior, source="PL")
    t2 = compute_targets(D @ post @ D, D @ prior @ D, source="PL")
    np.testing.assert_allclose(np.sort(t1.eigenvalues), np.sort(t2.eigenvalues), rtol=1e-7)


def test_classify_parameter_all_categories():
    pw = 2.0
    assert classify_parameter(0.5, 1.5, 0.0, 2.0, pw) == CAT_IDENTIFIABLE
    assert classify_parameter(0.0, 1.5, 0.0, 2.0, pw) == CAT_PRACTICAL  # hits lower bound
    assert classify_parameter(0.1, 1.9, 0.0, 2.0, pw) == CAT_PRACTICAL  # too wide (1.8 > 1.0)
    assert classify_parameter(0.5, None, 0.0, 2.0, pw) == CAT_STRUCTURAL  # infinite upper
    assert classify_parameter(None, 1.5, 0.0, 2.0, pw) == CAT_STRUCTURAL  # infinite lower


def test_fim_source_is_not_independent():
    t = compute_targets(0.1 * np.eye(3), np.eye(3), source="FIM")
    assert t.independent is False
    assert t.source == "FIM"
    t2 = compute_targets(0.1 * np.eye(3), np.eye(3), source="PL")
    assert t2.independent is True


def test_k_threshold_absolute_rule():
    eig = np.array([1.0, 0.5, 0.02, 1e-4])
    k_rel, _ = k_from_threshold(eig, tau=1e-2, rule="relative")  # applied=0.01 => >0.01 = 3
    assert k_rel == 3
    k_abs, _ = k_from_threshold(eig, tau=0.05, rule="absolute")  # >0.05 = 2
    assert k_abs == 2


def test_per_parameter_classification_in_compute():
    post = 0.1 * np.eye(2)
    prior = np.eye(2)
    cis = [(0.5, 1.5), (None, 1.0)]
    bounds = [(0.0, 2.0), (0.0, 2.0)]
    pw = [2.0, 2.0]
    t = compute_targets(post, prior, source="PL", pl_cis=cis, bounds=bounds, prior_widths=pw)
    assert t.param_classification == [CAT_IDENTIFIABLE, CAT_STRUCTURAL]


def test_singular_prior_handled():
    post = 0.1 * np.eye(3)
    prior = np.diag([1.0, 1.0, 0.0])  # singular
    t = compute_targets(post, prior, source="PL")
    assert t.n_parameters == 3
    assert t.effective_dimension >= 0.0
