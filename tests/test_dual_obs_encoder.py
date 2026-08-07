"""Tests for the P3-02 dual observation encoder mechanics."""
from __future__ import annotations

import torch

from iaapi.models.dual_obs_encoder import DualObservationEncoder


def _batch(B=4, T=8, n_obs=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, T, n_obs, generator=g) * 5 + 10  # positive-ish amplitudes
    times = torch.linspace(0, 100, T).expand(B, T)
    mask = torch.ones(B, T, n_obs)
    return obs, times, mask


def test_forward_shapes():
    enc = DualObservationEncoder(n_obs=3, d_context=32, n_meta=2)
    obs, times, mask = _batch()
    meta = torch.randn(4, 2)
    out = enc(obs, times, mask, meta)
    assert out["geometry"].shape == (4, 32)
    assert out["scale"].shape == (4, 32)
    assert out["posterior_location"].shape == (4, 64)


def test_shape_branch_is_amplitude_invariant():
    enc = DualObservationEncoder(n_obs=3, d_context=32)
    enc.eval()
    obs, times, mask = _batch()
    c = 123.45  # large amplitude scaling
    with torch.no_grad():
        g1 = enc(obs, times, mask)["geometry"]
        g2 = enc(obs * c, times, mask)["geometry"]
    # z-score removes amplitude => geometry context must be (near) identical
    assert torch.allclose(g1, g2, atol=1e-4), (g1 - g2).abs().max()


def test_scale_branch_is_amplitude_aware():
    enc = DualObservationEncoder(n_obs=3, d_context=32)
    enc.eval()
    obs, times, mask = _batch()
    c = 10.0
    with torch.no_grad():
        s1 = enc(obs, times, mask)["scale"]
        s2 = enc(obs * c, times, mask)["scale"]
    assert not torch.allclose(s1, s2, atol=1e-3)


def test_geometry_and_scale_contexts_differ():
    enc = DualObservationEncoder(n_obs=3, d_context=32)
    enc.eval()
    obs, times, mask = _batch()
    with torch.no_grad():
        out = enc(obs, times, mask)
    assert not torch.allclose(out["geometry"], out["scale"], atol=1e-4)


def test_mask_handling_changes_geometry_and_no_crash():
    enc = DualObservationEncoder(n_obs=3, d_context=32)
    enc.eval()
    obs, times, mask = _batch()
    mask_half = mask.clone()
    mask_half[:, ::2, :] = 0  # zero out every other time point
    with torch.no_grad():
        g_full = enc(obs, times, mask)["geometry"]
        g_half = enc(obs, times, mask_half)["geometry"]
    # masking changes the per-observable statistics => geometry differs
    assert not torch.allclose(g_full, g_half, atol=1e-4)
    # also: masked obs values must not influence geometry (set them to garbage)
    obs_garbage = obs.clone()
    obs_garbage[mask_half < 0.5] = 1e9
    with torch.no_grad():
        g_garbage = enc(obs_garbage, times, mask_half)["geometry"]
    assert torch.allclose(g_half, g_garbage, atol=1e-4)


def test_missing_whole_observable_handled():
    enc = DualObservationEncoder(n_obs=3, d_context=32)
    enc.eval()
    obs, times, mask = _batch()
    mask[:, :, 1] = 0  # observable 1 entirely missing
    with torch.no_grad():
        out = enc(obs, times, mask)
    assert torch.isfinite(out["geometry"]).all()
    assert torch.isfinite(out["scale"]).all()


def test_meta_used_by_scale_branch():
    enc = DualObservationEncoder(n_obs=3, d_context=32, n_meta=2)
    enc.eval()
    obs, times, mask = _batch()
    m1 = torch.zeros(4, 2)
    m2 = torch.ones(4, 2) * 5
    with torch.no_grad():
        s1 = enc(obs, times, mask, m1)["scale"]
        s2 = enc(obs, times, mask, m2)["scale"]
    assert not torch.allclose(s1, s2, atol=1e-3)
    # geometry must NOT depend on meta
    g1 = enc(obs, times, mask, m1)["geometry"]
    g2 = enc(obs, times, mask, m2)["geometry"]
    assert torch.allclose(g1, g2, atol=1e-5)


def test_gradient_flows():
    enc = DualObservationEncoder(n_obs=3, d_context=16)
    obs, times, mask = _batch()
    out = enc(obs, times, mask)
    loss = out["geometry"].sum() + out["scale"].sum()
    loss.backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert n_grad > 0
