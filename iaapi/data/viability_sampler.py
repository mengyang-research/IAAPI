"""Viability-aware parameter sampler (NCS plan task P1-02).

Replaces full-bound uniform sampling for model admission and training data on
models that P1-01 flagged as numerically unstable (Crauste / Elowitz / Fujita).
Full-bound uniform sampling places most draws outside numerically viable
regions (the documented DATA-HO / DATA-X6 root cause). This module samples
*nominal-centered* draws within configurable per-role log10 windows, and can
filter candidate draws with a cheap forward pre-check before the expensive
sensitivity/FIM step.

Strategies:
  * ``uniform_bounds``  — legacy full-bound uniform (baseline for comparison).
  * ``uniform_window``  — nominal-centered uniform window [nominal +/- width].
  * ``truncated_normal``— nominal-centered truncated normal in [lo, hi].

Discipline: this sampler represents the *training-data sampling distribution*
and must not be conflated with the Bayesian *prior*. Sampling provenance (seed,
strategy, widths, bounds) is recorded with every batch so draws are auditable.
Stdlib-only truncated normal (via statistics.NormalDist) so no scipy needed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"
_ND = NormalDist()
_SQRT2 = math.sqrt(2.0)

_NOISE_HINTS = ("sd", "sigma", "noise")
_INIT_HINTS = ("x0", "init")
_SCALE_HINTS = ("scale", "offset")


@dataclass(frozen=True)
class RoleWidths:
    """Per-role half-width in log10 units (window = nominal +/- width)."""
    kinetic: float = 1.0
    initial_state: float = 1.0
    noise: float = 0.5
    scale: float = 0.5
    default: float = 1.0

    def width_for(self, role: str) -> float:
        return float(getattr(self, role, self.default))


def classify_role(param_id: str) -> str:
    pid = param_id.lower()
    if any(h in pid for h in _NOISE_HINTS):
        return "noise"
    if any(h in pid for h in _INIT_HINTS):
        return "initial_state"
    if any(h in pid for h in _SCALE_HINTS):
        return "scale"
    return "kinetic"


@dataclass
class SampleProvenance:
    schema_version: str = SCHEMA_VERSION
    strategy: str = "truncated_normal"
    n_param: int = 0
    param_ids: List[str] = field(default_factory=list)
    param_roles: Dict[str, str] = field(default_factory=dict)
    role_widths: Dict[str, float] = field(default_factory=dict)
    seed: int = 0
    bounds_theta: List[List[float]] = field(default_factory=list)
    theta_nominal: List[float] = field(default_factory=list)


class ViabilitySampler:
    """Nominal-centered, bounds-respecting sampler with optional viability filter.

    Args:
        bounds_theta: (n_param, 2) sampling bounds in theta (log10) space.
        theta_nominal: (n_param,) nominal theta.
        param_ids: PEtab parameter IDs in order (provenance).
        strategy: uniform_bounds | uniform_window | truncated_normal.
        role_widths: per-role log10 half-widths.
        param_roles: optional {param_id: role}; defaults via classify_role.
        seed: RNG seed.
    """

    def __init__(
        self,
        bounds_theta: np.ndarray,
        theta_nominal: np.ndarray,
        param_ids: List[str],
        strategy: str = "truncated_normal",
        role_widths: Optional[RoleWidths] = None,
        param_roles: Optional[Dict[str, str]] = None,
        seed: int = 0,
    ):
        if strategy not in ("uniform_bounds", "uniform_window", "truncated_normal"):
            raise ValueError(f"unknown strategy: {strategy}")
        self.bounds = np.asarray(bounds_theta, dtype=float)
        self.nominal = np.asarray(theta_nominal, dtype=float).reshape(-1)
        self.n_param = self.bounds.shape[0]
        if self.nominal.shape[0] != self.n_param:
            raise ValueError("theta_nominal length must match bounds")
        self.param_ids = list(param_ids)
        if len(self.param_ids) != self.n_param:
            raise ValueError("param_ids length must match bounds")
        self.strategy = strategy
        self.role_widths = role_widths or RoleWidths()
        self.param_roles = {pid: (param_roles or {}).get(pid, classify_role(pid))
                            for pid in self.param_ids}
        self.seed = int(seed)
        self._widths = np.array(
            [self.role_widths.width_for(self.param_roles[pid]) for pid in self.param_ids],
            dtype=float,
        )

    # ---- core sampling -------------------------------------------------- #
    def _rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def _clip_bounds(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.bounds[:, 0], self.bounds[:, 1])

    def sample(self, n: int, seed_offset: int = 0) -> np.ndarray:
        """Return (n, n_param) theta draws within bounds."""
        rng = self._rng(seed_offset)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        if self.strategy == "uniform_bounds":
            return rng.uniform(lo, hi, size=(n, self.n_param))
        if self.strategy == "uniform_window":
            w = self._widths
            a = self.nominal - w
            b = self.nominal + w
            a = np.maximum(a, lo)
            b = np.minimum(b, hi)
            return rng.uniform(a, b, size=(n, self.n_param))
        # truncated_normal
        return self._trunc_normal(rng, n)

    def _trunc_normal(self, rng: np.random.Generator, n: int) -> np.ndarray:
        loc = self.nominal
        scale = np.maximum(self._widths / 2.0, 1e-6)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        out = np.empty((n, self.n_param), dtype=float)
        for j in range(self.n_param):
            a_std = (lo[j] - loc[j]) / scale[j]
            b_std = (hi[j] - loc[j]) / scale[j]
            phi_a = _norm_cdf(a_std)
            phi_b = _norm_cdf(b_std)
            if phi_b - phi_a < 1e-12:
                out[:, j] = loc[j]
                continue
            u = rng.uniform(0.0, 1.0, size=n)
            p = phi_a + u * (phi_b - phi_a)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            z = np.array([_norm_ppf(pi) for pi in p])
            out[:, j] = loc[j] + scale[j] * z
        return out

    # ---- viability pre-check ------------------------------------------- #
    def sample_with_precheck(
        self,
        forward_fn: Callable[[np.ndarray], bool],
        n_target: int,
        max_attempts: Optional[int] = None,
        seed_offset: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample theta, keep only those passing a cheap forward pre-check.

        forward_fn(theta) -> True if the forward simulation is viable (status 0,
        finite). Used to reject non-viable draws *before* the expensive
        sensitivity/FIM step. Returns (viable_thetas, stats).
        """
        max_attempts = max_attempts or max(n_target * 20, 200)
        rng = self._rng(seed_offset)
        viable: List[np.ndarray] = []
        attempted = 0
        n_viable = 0
        # Sample in batches; re-seed per batch deterministically.
        batch = max(n_target, 16)
        while attempted < max_attempts and len(viable) < n_target:
            thetas = self._sample_batch(rng, batch)
            for th in thetas:
                if attempted >= max_attempts or len(viable) >= n_target:
                    break
                attempted += 1
                ok = bool(forward_fn(th))
                if ok:
                    viable.append(th)
                    n_viable += 1
        if not viable:
            return np.empty((0, self.n_param)), self._stats(attempted, n_viable, n_target)
        return np.array(viable), self._stats(attempted, n_viable, n_target)

    def _sample_batch(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # Deterministic per-batch draws that do not depend on forward_fn.
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        if self.strategy == "uniform_bounds":
            return rng.uniform(lo, hi, size=(n, self.n_param))
        if self.strategy == "uniform_window":
            w = self._widths
            a = np.maximum(self.nominal - w, lo)
            b = np.minimum(self.nominal + w, hi)
            return rng.uniform(a, b, size=(n, self.n_param))
        return self._trunc_normal(rng, n)

    def _stats(self, attempted: int, viable: int, target: int) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_target": target,
            "n_attempted": attempted,
            "n_viable": viable,
            "success_rate": round(viable / attempted, 4) if attempted else 0.0,
        }

    def provenance(self) -> SampleProvenance:
        return SampleProvenance(
            strategy=self.strategy, n_param=self.n_param, param_ids=list(self.param_ids),
            param_roles=dict(self.param_roles),
            role_widths={r: self.role_widths.width_for(r) for r in
                         ("kinetic", "initial_state", "noise", "scale", "default")},
            seed=self.seed, bounds_theta=self.bounds.tolist(),
            theta_nominal=self.nominal.tolist(),
        )


# --------------------------------------------------------------------------- #
# Comparison helper (old vs new success rates)
# --------------------------------------------------------------------------- #
def forward_success_rate(
    forward_fn: Callable[[np.ndarray], bool],
    sampler: ViabilitySampler,
    n: int,
    seed_offset: int = 0,
) -> float:
    """Fraction of n draws that pass the forward pre-check."""
    thetas = sampler.sample(n, seed_offset=seed_offset)
    ok = sum(1 for th in thetas if bool(forward_fn(th)))
    return round(ok / n, 4) if n else 0.0


def compare_strategies(
    forward_fn: Callable[[np.ndarray], bool],
    bounds_theta: np.ndarray,
    theta_nominal: np.ndarray,
    param_ids: List[str],
    n: int,
    seed: int = 0,
    new_strategy: str = "truncated_normal",
    role_widths: Optional[RoleWidths] = None,
) -> Dict[str, Any]:
    """Report old (full-bound uniform) vs new (viability-aware) forward success."""
    old = ViabilitySampler(bounds_theta, theta_nominal, param_ids,
                           strategy="uniform_bounds", seed=seed)
    new = ViabilitySampler(bounds_theta, theta_nominal, param_ids,
                           strategy=new_strategy, role_widths=role_widths, seed=seed)
    old_rate = forward_success_rate(forward_fn, old, n, seed_offset=0)
    new_rate = forward_success_rate(forward_fn, new, n, seed_offset=1)
    return {
        "n": n, "seed": seed,
        "old_strategy": "uniform_bounds", "old_forward_success_rate": old_rate,
        "new_strategy": new_strategy, "new_forward_success_rate": new_rate,
        "improvement": round(new_rate - old_rate, 4),
        "materially_exceeds": new_rate - old_rate >= 0.10,
        "new_provenance": new.provenance().__dict__,
    }


# --------------------------------------------------------------------------- #
# Stdlib normal CDF / inverse-CDF (no scipy)
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_ppf(p: float) -> float:
    return _ND.inv_cdf(p)
