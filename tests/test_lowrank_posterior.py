"""Tests for the P3-03 low-rank+diagonal posterior baseline."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from iaapi.models.lowrank_posterior import LowRankDiagPosterior, LowRankPosteriorHead


def _post(mean=None, diag=None, B=None, conf=None, mask=None, max_dim=5, rank=2, Bb=2):
    torch.manual_seed(0)
    if mask is None:
        mask = torch.ones(Bb, max_dim)
    if mean is None:
        mean = torch.randn(Bb, max_dim) * 0.3
    if diag is None:
        diag = torch.rand(Bb, max_dim).abs() + 0.1
    if B is None:
        B = torch.randn(Bb, max_dim, rank) * 0.2
    if conf is None:
        conf = torch.rand(Bb, max_dim) * 0.5 + 0.5
    return LowRankDiagPosterior(mean * mask, diag * mask, B * mask.unsqueeze(-1),
                                conf * mask, mask)


def test_variable_dim_in_one_batch():
    # sample 0: 3 valid params; sample 1: 5 valid params (max_dim=5).
    mask = torch.tensor([[1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1]], dtype=torch.float)
    post = _post(mask=mask)
    theta = torch.randn(2, 5) * 0.2
    lp = post.log_prob(theta)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()
    s = post.rsample((8,))
    assert s.shape == (8, 2, 5)
    # masked params are zero in samples; valid params vary
    assert torch.all(s[:, 0, 3:] == 0)
    assert s[:, 0, :3].std() > 0


def test_log_prob_and_sample_shapes_mask_agreement():
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    post = _post(mask=mask, Bb=1)
    theta = torch.randn(1, 5)
    assert post.log_prob(theta).shape == (1,)
    s = post.rsample((16,))
    assert s.shape == (16, 1, 5)
    # masked params exactly zero
    assert torch.all(s[:, :, 3:] == 0)
    assert torch.all(s[:, :, :3].abs().mean() > 0)


def test_covariance_pd_on_valid_subspace():
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    post = _post(mask=mask, Bb=1)
    cov = post.covariance()  # (1,5,5)
    # extract 3x3 valid block
    sub = cov[0, :3, :3]
    eig = torch.linalg.eigvalsh(sub)
    assert (eig > 0).all(), eig  # positive definite on valid subspace


def test_posterior_mean_changes_with_observation():
    torch.manual_seed(1)
    head = LowRankPosteriorHead(d_context=16, max_dim=5, rank=2)
    mask = torch.ones(1, 5)
    ctx_a = torch.randn(1, 16)
    ctx_b = torch.randn(1, 16)
    post_a = head(ctx_a, mask)
    post_b = head(ctx_b, mask)
    assert not torch.allclose(post_a.mean_post, post_b.mean_post, atol=1e-5)


def test_prior_backoff_low_confidence_approaches_prior():
    # confidence -> 0 => mean -> 0, cov diag -> prior_var, sample ~ prior.
    mask = torch.ones(1, 4)
    mean = torch.randn(1, 4) * 5.0       # large offset
    diag = torch.rand(1, 4).abs() + 2.0  # large learned variance
    B = torch.randn(1, 4, 2) * 3.0
    conf = torch.zeros(1, 4)             # fully sloppy
    post = LowRankDiagPosterior(mean, diag, B, conf, mask, prior_var=1.0)
    # mean backed off to 0
    assert torch.allclose(post.mean_post, torch.zeros(1, 4), atol=1e-6)
    # diagonal backed off to prior_var (1.0)
    assert torch.allclose(post.D_post, torch.ones(1, 4), atol=1e-4)
    # low-rank backed off to 0
    assert torch.allclose(post.B_post, torch.zeros(1, 4, 2), atol=1e-6)
    # log_prob at theta=0 ~= prior log_prob (N(0,1) per param): -0.5*log(2pi)*4
    lp0 = post.log_prob(torch.zeros(1, 4)).item()
    expected = -0.5 * math.log(2 * math.pi) * 4
    assert math.isclose(lp0, expected, rel_tol=1e-4)


def test_high_confidence_uses_learned_posterior():
    mask = torch.ones(1, 4)
    mean = torch.randn(1, 4) * 0.5
    diag = torch.rand(1, 4).abs() + 0.3
    B = torch.randn(1, 4, 2) * 0.4
    conf = torch.ones(1, 4)  # fully identifiable
    post = LowRankDiagPosterior(mean, diag, B, conf, mask, prior_var=1.0)
    assert torch.allclose(post.mean_post, mean, atol=1e-6)
    assert torch.allclose(post.D_post, diag + post.jitter, atol=1e-6)
    assert torch.allclose(post.B_post, B, atol=1e-6)


def test_sample_matches_log_prob_moments():
    torch.manual_seed(2)
    mask = torch.ones(1, 4)
    mean = torch.randn(1, 4) * 0.4
    diag = torch.rand(1, 4).abs() + 0.2
    B = torch.randn(1, 4, 2) * 0.3
    conf = torch.ones(1, 4)
    post = LowRankDiagPosterior(mean, diag, B, conf, mask, prior_var=1.0)
    s = post.rsample((20000,))  # (20000, 1, 4)
    emp_mean = s.mean(dim=0)
    emp_cov = torch.cov(s[:, 0, :].T)
    assert torch.allclose(emp_mean, post.mean_post, atol=2e-2)
    assert torch.allclose(emp_cov, post.covariance()[0], atol=3e-2)


def test_log_prob_finite_and_decreases_away_from_mean():
    mask = torch.ones(1, 5)
    post = _post(mask=mask, Bb=1)
    at_mean = post.log_prob(post.mean_post).item()
    far = post.log_prob(post.mean_post + 5.0).item()
    assert math.isfinite(at_mean) and math.isfinite(far)
    assert at_mean > far


def test_head_gradient_flows_and_masks():
    head = LowRankPosteriorHead(d_context=8, max_dim=5, rank=2)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    ctx = torch.randn(1, 8, requires_grad=True)
    post = head(ctx, mask)
    loss = post.log_prob(torch.randn(1, 5)).sum()
    loss.backward()
    assert ctx.grad is not None and ctx.grad.abs().sum() > 0
