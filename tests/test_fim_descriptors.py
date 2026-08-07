"""Tests for the R0-03 frozen prior-whitened FIM spectral descriptors.

Covers analytic spectra, scale invariance, unit (prior-whitening) invariance,
finite outputs for extreme ranges, prior-whitening sensitivity, negative-
eigenvalue quality flags, normalized r, threshold fraction, and a real-model
smoke test on pilot HDF5 data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from iaapi.evaluation.fim_descriptors import (
    DESCRIPTOR_NAMES,
    DEFAULT_TAU,
    compute_fim_descriptors,
)


# --------------------------------------------------------------------------- #
# Analytic spectra
# --------------------------------------------------------------------------- #
def test_isotropic_spectrum():
    """FIM = alpha * I -> all eigenvalues equal.

    Entropy should be 1 (maximal), eff_rank 1, gaps 0, top_r_frac = r/n.
    """
    d = 6
    fim = 42.0 * np.eye(d)  # arbitrary scale
    res = compute_fim_descriptors(fim)
    desc = res["descriptors"]
    r = max(1, math.ceil(d / 3))

    assert math.isclose(desc[0], 1.0, abs_tol=1e-10), f"entropy_norm {desc[0]}"
    assert math.isclose(desc[1], 1.0, abs_tol=1e-10), f"eff_rank {desc[1]}"
    assert math.isclose(desc[2], 0.0, abs_tol=1e-10), f"max_gap {desc[2]}"
    assert math.isclose(desc[4], 0.0, abs_tol=1e-10), f"curvature {desc[4]}"
    assert math.isclose(desc[5], 0.0, abs_tol=1e-10), f"log_range {desc[5]}"
    assert math.isclose(desc[6], r / d, abs_tol=1e-10), f"top_r_frac {desc[6]}"
    assert math.isclose(desc[7], 1.0, abs_tol=1e-10), f"above_thresh {desc[7]}"


def test_rank1_spectrum():
    """FIM = diag([1, 0, ..., 0]) -> entropy 0, eff_rank 1/n, large max_gap."""
    d = 5
    fim = np.diag([1.0] + [0.0] * (d - 1))
    res = compute_fim_descriptors(fim)
    desc = res["descriptors"]

    assert math.isclose(desc[0], 0.0, abs_tol=1e-10), f"entropy {desc[0]}"
    assert math.isclose(desc[1], 1.0 / d, abs_tol=1e-10), f"eff_rank {desc[1]}"
    # max_log_gap should be large (drop from 1 to ~0)
    assert desc[2] > 1.0, f"max_log_gap should be large {desc[2]}"
    assert math.isclose(desc[6], 1.0, abs_tol=1e-10), f"top_r_frac {desc[6]}"


def test_two_level_spectrum():
    """FIM with k large + (n-k) small eigenvalues."""
    d = 6
    eigs = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01])
    fim = np.diag(eigs)
    res = compute_fim_descriptors(fim)
    desc = res["descriptors"]

    # 3 identifiable, 3 sloppy -> eff_rank ~ (sum)^2 / (n * sum_sq)
    total = eigs.sum()
    expected_eff = total ** 2 / (d * (eigs ** 2).sum())
    assert math.isclose(desc[1], expected_eff, abs_tol=1e-8)
    # Big gap between eigenvalue 3 and 4 (0-indexed: gap at index 2->3)
    assert desc[2] > 1.0, f"max_log_gap should be large {desc[2]}"
    # Gap position: argmax(gaps)=2 (0-indexed before the gap), / (d-1) = 2/5
    assert math.isclose(desc[3], 2.0 / 5, abs_tol=1e-8)


# --------------------------------------------------------------------------- #
# Scale invariance
# --------------------------------------------------------------------------- #
def test_scale_invariance():
    """Multiplying FIM by positive constant must not change any descriptor."""
    rng = np.random.default_rng(0)
    d = 5
    A = rng.normal(size=(d, d))
    fim = A @ A.T + 0.1 * np.eye(d)  # PSD

    res1 = compute_fim_descriptors(fim)
    res2 = compute_fim_descriptors(1e10 * fim)
    res3 = compute_fim_descriptors(1e-10 * fim)

    for i, name in enumerate(DESCRIPTOR_NAMES):
        assert math.isclose(res1["descriptors"][i], res2["descriptors"][i],
                            abs_tol=1e-8), f"{name} not scale-invariant (x1e10)"
        assert math.isclose(res1["descriptors"][i], res3["descriptors"][i],
                            abs_tol=1e-8), f"{name} not scale-invariant (x1e-10)"


# --------------------------------------------------------------------------- #
# Unit invariance via prior whitening
# --------------------------------------------------------------------------- #
def test_whitening_absorbs_prior_scale():
    """Whitened descriptors are invariant to SCALING the prior by a constant.

    W = (cP)^{-1/2} F (cP)^{-1/2} = c^{-1} P^{-1/2} F P^{-1/2} = c^{-1} W.
    The eigenvalues scale by c^{-1}, but all descriptors are scale-invariant
    (log10-normalised, entropy-normalised), so descriptors are unchanged.
    This is the correct "unit" invariance: scaling all parameter units by
    the same factor scales the prior uniformly, and the whitened descriptors
    are unaffected.
    """
    rng = np.random.default_rng(42)
    d = 4
    A = rng.normal(size=(d, d))
    fim = A @ A.T + np.eye(d)
    prior = np.diag(rng.uniform(0.5, 3.0, d) ** 2)

    res1 = compute_fim_descriptors(fim, prior_cov=prior)
    res2 = compute_fim_descriptors(fim, prior_cov=100.0 * prior)
    res3 = compute_fim_descriptors(fim, prior_cov=0.01 * prior)

    for i, name in enumerate(DESCRIPTOR_NAMES):
        assert math.isclose(res1["descriptors"][i], res2["descriptors"][i],
                            abs_tol=1e-8), f"{name} not prior-scale-invariant (x100)"
        assert math.isclose(res1["descriptors"][i], res3["descriptors"][i],
                            abs_tol=1e-8), f"{name} not prior-scale-invariant (x0.01)"


def test_prior_whitening_changes_results():
    """Same FIM, different (non-uniform) prior -> different descriptors."""
    rng = np.random.default_rng(1)
    d = 4
    A = rng.normal(size=(d, d))
    fim = A @ A.T + np.eye(d)

    # Non-uniform priors: different parameters get different prior widths
    prior_a = np.diag([0.01, 1.0, 100.0, 0.1])  # anisotropic
    prior_b = np.diag([100.0, 0.01, 0.1, 1.0])  # different anisotropy

    res_a = compute_fim_descriptors(fim, prior_cov=prior_a)
    res_b = compute_fim_descriptors(fim, prior_cov=prior_b)
    res_raw = compute_fim_descriptors(fim)  # no whitening

    assert res_a["quality_flags"]["whitened"] is True
    assert res_b["quality_flags"]["whitened"] is True
    assert res_raw["quality_flags"]["whitened"] is False

    # Different anisotropic priors should produce different whitened spectra
    assert not np.allclose(res_a["eigenvalues"], res_b["eigenvalues"])
    # And therefore different descriptors
    assert not np.allclose(res_a["descriptors"], res_b["descriptors"])


def test_bounds_based_whitening():
    """param_bounds_log10 builds diagonal uniform-prior covariance."""
    d = 4
    fim = np.diag([10.0, 5.0, 1.0, 0.1])
    bounds = np.array([[-2, 2]] * d)  # width 4 in log10

    res = compute_fim_descriptors(fim, param_bounds_log10=bounds)
    assert res["quality_flags"]["whitened"] is True
    assert all(math.isfinite(x) for x in res["descriptors"])


# --------------------------------------------------------------------------- #
# Finite outputs for extreme eigenvalue ranges
# --------------------------------------------------------------------------- #
def test_extreme_eigenvalue_range_finite():
    """FIM with eigenvalues spanning 1e-20 to 1e20 -> no NaN/Inf."""
    eigs = np.logspace(-20, 20, 8)
    fim = np.diag(eigs)
    res = compute_fim_descriptors(fim)
    desc = res["descriptors"]
    assert all(math.isfinite(x) for x in desc), f"non-finite: {desc}"
    # log_dynamic_range should be ~40 (log10(1e20/1e-20))
    assert desc[5] > 30, f"log_range too small: {desc[5]}"


def test_near_singular_fim_finite():
    """Near-singular FIM -> finite descriptors."""
    fim = np.eye(5)
    fim[4, 4] = 1e-30
    res = compute_fim_descriptors(fim)
    assert all(math.isfinite(x) for x in res["descriptors"])


def test_zero_fim():
    """All-zero FIM -> zero descriptors (degenerate but finite)."""
    fim = np.zeros((4, 4))
    res = compute_fim_descriptors(fim)
    assert all(x == 0.0 for x in res["descriptors"])
    assert math.isinf(res["quality_flags"]["condition_number"])


# --------------------------------------------------------------------------- #
# Negative eigenvalue quality flag
# --------------------------------------------------------------------------- #
def test_negative_eigenvalue_mass_flag():
    """Indefinite matrix -> negative_eigenvalue_mass > 0, descriptors finite."""
    fim = np.diag([10.0, -1.0, 5.0, 0.1])  # one negative eigenvalue
    res = compute_fim_descriptors(fim)
    assert res["quality_flags"]["negative_eigenvalue_mass"] > 0
    assert all(math.isfinite(x) for x in res["descriptors"])


# --------------------------------------------------------------------------- #
# Normalized r and threshold fraction
# --------------------------------------------------------------------------- #
def test_top_r_uses_normalized_r():
    """r = ceil(n/3), not fixed 3."""
    for d in [3, 6, 9, 12, 15]:
        fim = np.eye(d)
        res = compute_fim_descriptors(fim)
        r = max(1, math.ceil(d / 3))
        expected_top_r = r / d  # all eigenvalues equal
        assert math.isclose(res["descriptors"][6], expected_top_r,
                            abs_tol=1e-10), f"d={d}: {res['descriptors'][6]} != {expected_top_r}"


def test_above_threshold_fraction():
    """Known spectrum -> exact count above tau * max."""
    eigs = np.array([10.0, 5.0, 0.5, 0.05, 0.001])
    fim = np.diag(eigs)
    res = compute_fim_descriptors(fim, tau=DEFAULT_TAU)
    # tau=0.01, threshold = 0.01 * 10 = 0.1
    # eigs > 0.1: [10, 5, 0.5] -> 3/5
    assert math.isclose(res["descriptors"][7], 3.0 / 5, abs_tol=1e-10)


# --------------------------------------------------------------------------- #
# Descriptor names and structure
# --------------------------------------------------------------------------- #
def test_descriptor_names_and_count():
    """8 frozen descriptors with canonical names."""
    assert len(DESCRIPTOR_NAMES) == 8
    assert DESCRIPTOR_NAMES[0] == "spectral_entropy_norm"
    assert DESCRIPTOR_NAMES[7] == "above_threshold_fraction"
    res = compute_fim_descriptors(np.eye(4))
    assert res["names"] == DESCRIPTOR_NAMES
    assert len(res["descriptors"]) == 8


def test_quality_flags_present():
    """Quality flags are diagnostic-only, not in the descriptor vector."""
    res = compute_fim_descriptors(np.eye(4))
    qf = res["quality_flags"]
    assert "condition_number" in qf
    assert "negative_eigenvalue_mass" in qf
    assert "whitened" in qf
    assert "n_params" in qf
    assert qf["n_params"] == 4


# --------------------------------------------------------------------------- #
# Real-model smoke test on pilot HDF5 shard
# --------------------------------------------------------------------------- #
PILOT_DIR = Path(__file__).resolve().parents[1] / "runs" / "petab_traindata_pilot" / "pilot_v1" / "shards"


def test_real_hdf5_shard_smoke():
    """Load a real FIM from a pilot HDF5 shard -> finite descriptors."""
    shard = PILOT_DIR / "Boehm_JProteomeRes2014" / "shard_00000.h5"
    if not shard.exists():
        pytest.skip("pilot shard not available")

    import h5py
    with h5py.File(shard, "r") as f:
        # Load metadata for parameter IDs
        meta = dict(f["metadata"].attrs)
        param_ids = json.loads(meta["parameter_ids"]) if "parameter_ids" in meta else None
        # Load first sample's FIM
        g = f["samples"]
        s = sorted(k for k in g.keys() if k.startswith("sample_"))[0]
        fim = np.asarray(g[f"{s}/fim"], dtype=float)

    d = fim.shape[0]
    assert d == 9  # Boehm has 9 parameters

    # Without whitening (raw FIM)
    res_raw = compute_fim_descriptors(fim)
    assert all(math.isfinite(x) for x in res_raw["descriptors"]), "raw descriptors non-finite"

    # With bounds-based whitening (simulated log10 bounds)
    bounds = np.array([[-4, 4]] * d)  # uniform 8-decade log10 bounds
    res_whitened = compute_fim_descriptors(fim, param_bounds_log10=bounds)
    assert res_whitened["quality_flags"]["whitened"] is True
    assert all(math.isfinite(x) for x in res_whitened["descriptors"]), "whitened descriptors non-finite"

    # Multiple samples should give a distribution (not all identical)
    descs = []
    with h5py.File(shard, "r") as f:
        g = f["samples"]
        samples = sorted(k for k in g.keys() if k.startswith("sample_"))[:20]
        for s in samples:
            fim_i = np.asarray(g[f"{s}/fim"], dtype=float)
            descs.append(compute_fim_descriptors(fim_i, param_bounds_log10=bounds)["descriptors"])
    descs = np.array(descs)
    # At least one descriptor should vary across samples
    assert np.any(np.std(descs, axis=0) > 1e-6), "all descriptors identical across 20 samples"
