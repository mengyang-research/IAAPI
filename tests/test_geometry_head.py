"""Tests for the P3-04 geometry head + projector loss."""
from __future__ import annotations

import torch

from iaapi.models.geometry_head import (
    GeometryHead,
    effective_dim_loss,
    projector_loss,
)


def _ortho(d, k, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(B, d, k, generator=g)
    Q, _ = torch.linalg.qr(P)  # orthonormal columns
    return Q


def _rand_orthogonal(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(k, k, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


def test_sign_invariance():
    d, k = 6, 3
    P = _ortho(d, k)
    Q = _ortho(d, k, seed=99)  # different subspace
    lp = projector_loss(P, Q)
    lp_neg = projector_loss(-P, Q)
    assert torch.allclose(lp, lp_neg, atol=1e-6)


def test_rotation_invariance_within_subspace():
    d, k = 6, 3
    P = _ortho(d, k)
    Q = _ortho(d, k, seed=99)
    R = _rand_orthogonal(k, seed=7)  # k x k orthogonal
    PR = torch.matmul(P, R)          # rotate columns within the subspace
    lp = projector_loss(P, Q)
    lp_rot = projector_loss(PR, Q)
    assert torch.allclose(lp, lp_rot, atol=1e-5)


def test_loss_zero_for_same_subspace():
    d, k = 6, 3
    P = _ortho(d, k)
    lp = projector_loss(P, P)
    assert lp.item() < 1e-8


def test_loss_positive_for_different_subspace():
    d, k = 6, 3
    P = _ortho(d, k, seed=1)
    Q = _ortho(d, k, seed=2)
    lp = projector_loss(P, Q)
    assert lp.item() > 1e-3


def test_mask_zeros_invalid_params():
    d, k = 5, 2
    P = _ortho(d, k)
    Q = _ortho(d, k, seed=5)
    mask = torch.tensor([[1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]], dtype=torch.float)
    lp = projector_loss(P, Q, mask)
    assert torch.isfinite(lp)


def test_effective_dim_loss_mse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    true = torch.tensor([1.5, 2.5, 3.5])
    # MSE = mean((0.5)^2*3) = 0.25
    assert torch.isclose(effective_dim_loss(pred, true), torch.tensor(0.25))


def test_geometry_head_forward_shapes_and_mask():
    head = GeometryHead(d_context=16, max_dim=5, k=3)
    ctx = torch.randn(2, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1]], dtype=torch.float)
    out = head(ctx, mask)
    assert out["projector"].shape == (2, 5, 3)
    assert out["effective_dim"].shape == (2,)
    assert (out["effective_dim"] >= 0).all()
    # masked rows zeroed
    assert torch.all(out["projector"][0, 3:, :] == 0)


def test_geometry_head_loss_and_gradient():
    head = GeometryHead(d_context=16, max_dim=5, k=3)
    ctx = torch.randn(2, 16, requires_grad=True)
    mask = torch.ones(2, 5)
    P_true = _ortho(5, 3, B=2, seed=3)
    eff_true = torch.tensor([2.0, 3.0])
    total, parts = head.loss(ctx, P_true, eff_true, mask, lam_eff=0.5)
    assert torch.isfinite(total)
    assert "projector_loss" in parts and "eff_dim_loss" in parts
    total.backward()
    assert ctx.grad is not None and ctx.grad.abs().sum() > 0


def test_loss_invariant_under_head_input_perturbation_only_changes_value():
    # Sanity: loss changes when the true subspace changes (not trivially constant).
    d, k = 6, 3
    P = _ortho(d, k, seed=1)
    Q1 = _ortho(d, k, seed=2)
    Q2 = _ortho(d, k, seed=3)
    lp1 = projector_loss(P, Q1)
    lp2 = projector_loss(P, Q2)
    assert not torch.allclose(lp1, lp2, atol=1e-3)
