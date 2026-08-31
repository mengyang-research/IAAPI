"""Standardized MCMC runner (NCS plan task P2-03).

Multi-chain MCMC (emcee affine-invariant ensemble) with diagnostics: split-R-hat,
effective sample size (Geyer initial-positive-sequence), divergence/failure
count, posterior covariance in prior-whitened coordinates (reuses P2-01), and a
convergence gate.

Convergence discipline (P2-03 acceptance):
  * every model result carries a ``convergence_status`` (CONVERGED | UNCONVERGED);
  * a ground-truth label is allowed ONLY when CONVERGED (``label_allowed``);
  * unconverged models remain in the output (visible) but are flagged
    ``excluded_by_rule`` per a predeclared rule (R-hat >= threshold OR ESS < threshold).

The log-posterior ``logpost_fn(theta) -> float`` is INJECTABLE, so the runner is
unit-testable with analytic targets and can be paired with a real-PEtab-data
likelihood in production. emcee is not HMC, so ``divergences`` is reported as
None (honest) and a ``failure_rate`` (out-of-bounds / -inf evaluations) is recorded.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import emcee
except ImportError as e:  # pragma: no cover
    raise ImportError("P2-03 MCMC runner requires emcee") from e

from iaapi.evaluation.targets import whiten_covariance

SCHEMA_VERSION = "1.0"
CONVERGED = "CONVERGED"
UNCONVERGED = "UNCONVERGED"


@dataclass(frozen=True)
class MCMCConfig:
    n_walkers: int = 32
    n_steps: int = 2000
    n_burn: int = 1000
    n_ensembles: int = 4
    seed: int = 0
    timeout_s: float = 600.0
    rhat_threshold: float = 1.01
    ess_threshold: float = 100.0
    tau_coverage: float = 20.0  # total chain length must cover this many autocorr times
    ball_scale: float = 0.05  # init ball around nominal (fraction of bound width)


@dataclass
class MCMCResult:
    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    n_parameters: int = 0
    n_walkers: int = 0
    n_ensembles: int = 0
    n_samples: int = 0
    posterior_mean: List[float] = field(default_factory=list)
    posterior_cov: List[List[float]] = field(default_factory=list)
    whitened_cov: Optional[List[List[float]]] = None
    rhat: List[float] = field(default_factory=list)
    ess_bulk: List[float] = field(default_factory=list)
    ess_tail: List[float] = field(default_factory=list)
    autocorr_time: List[float] = field(default_factory=list)
    acceptance_fraction: float = 0.0
    divergences: Optional[int] = None  # None for emcee (not HMC)
    failure_rate: float = 0.0
    convergence_status: str = UNCONVERGED
    label_allowed: bool = False
    excluded_by_rule: str = ""
    elapsed_s: float = 0.0


LogPostFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------- #
# Diagnostics (rank-normalized, per Vehtari et al. 2021)
# --------------------------------------------------------------------------- #
def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize each chain: u = (rank + U(0,1)) / (n+1), then z = Phi^-1(u).

    chains: (n_chains, n_steps, n_param) -> same shape, marginally Gaussian.
    """
    chains = np.asarray(chains, dtype=float)
    from scipy import stats as _stats

    n_chains, n_steps, d = chains.shape
    out = np.empty_like(chains)
    for c in range(n_chains):
        for k in range(d):
            x = chains[c, :, k]
            ranks = _stats.rankdata(x, method="average")  # ties -> average rank
            u = (ranks - 0.5) / n_steps
            u = np.clip(u, 1e-12, 1 - 1e-12)
            out[c, :, k] = _stats.norm.ppf(u)
    return out


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Rank-normalized split-R-hat per parameter on *independent ensembles*.

    chains: (n_chains, n_steps, n_param) where n_chains are independent
    ensembles (walkers within one ensemble are NOT treated as independent).
    """
    from scipy import stats as _stats

    chains = np.asarray(chains, dtype=float)
    n_c, n_s, d = chains.shape
    if n_c < 4:
        return np.full(d, np.nan)
    half = n_s // 2
    if half < 2:
        return np.full(d, np.nan)
    # split each chain into first/second half -> (2*n_c, half, d)
    split = np.concatenate([chains[:, :half, :], chains[:, half:2 * half, :]], axis=0)
    m, n = split.shape[0], half
    z = _rank_normalize(split)
    means = z.mean(axis=1)  # (m, d)
    B = n * means.var(axis=0, ddof=1)
    W = z.var(axis=1, ddof=1).mean(axis=0)
    W = np.where(W > 0, W, 1e-300)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation of a 1D series via FFT."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if np.allclose(x, 0) or n < 2:
        return np.zeros(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf


def _geyer_tau(rho: np.ndarray) -> float:
    """Geyer initial-positive-sequence autocorrelation time from an ACF."""
    n = len(rho)
    tau = 1.0
    t = 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
        t += 2
    return max(tau, 1.0)


def _ess_from_chains(chains: np.ndarray, frac: float = 1.0) -> np.ndarray:
    """ESS per parameter from (n_chains, n_steps, n_param) using the last
    ``frac`` fraction of each chain (frac=1 bulk, frac=0.5 tail)."""
    chains = np.asarray(chains, dtype=float)
    m, n, d = chains.shape
    keep = max(1, int(n * frac))
    sub = chains[:, -keep:, :]
    ess = np.empty(d)
    for k in range(d):
        acfs = np.array([_autocorr(sub[c, :, k]) for c in range(m)])
        rho = acfs.mean(axis=0)
        tau = _geyer_tau(rho)
        ess[k] = m * keep / tau
    return ess


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS per parameter (Geyer initial-positive-sequence)."""
    return _ess_from_chains(chains, frac=1.0)


def tail_effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Tail ESS per parameter: ESS of the last 50% of each chain."""
    return _ess_from_chains(chains, frac=0.5)


def autocorrelation_time(chains: np.ndarray) -> np.ndarray:
    """Geyer integrated autocorrelation time per parameter (bulk)."""
    chains = np.asarray(chains, dtype=float)
    m, n, d = chains.shape
    tau = np.empty(d)
    for k in range(d):
        acfs = np.array([_autocorr(chains[c, :, k]) for c in range(m)])
        rho = acfs.mean(axis=0)
        tau[k] = _geyer_tau(rho)
    return tau


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class MCMCRunner:
    def __init__(self, logpost_fn: LogPostFn, bounds_theta: np.ndarray,
                 theta_nominal: np.ndarray, param_ids: List[str],
                 config: MCMCConfig = MCMCConfig(), prior_cov: Optional[np.ndarray] = None,
                 model_id: str = ""):
        self.logpost_fn = logpost_fn
        self.bounds = np.asarray(bounds_theta, dtype=float)
        self.nominal = np.asarray(theta_nominal, dtype=float).reshape(-1)
        self.param_ids = list(param_ids)
        self.n_param = self.bounds.shape[0]
        self.config = config
        self.prior_cov = None if prior_cov is None else np.asarray(prior_cov, dtype=float)
        self.model_id = model_id
        self._n_eval = 0
        self._n_fail = 0

    def _logpost_wrapped(self, theta: np.ndarray) -> float:
        self._n_eval += 1
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < self.bounds[:, 0]) or np.any(theta > self.bounds[:, 1]):
            self._n_fail += 1
            return -np.inf
        try:
            lp = float(self.logpost_fn(theta))
        except Exception:
            self._n_fail += 1
            return -np.inf
        if not np.isfinite(lp):
            self._n_fail += 1
        return lp

    def _init_walkers(self, rng: np.random.Generator, n_walkers: int) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        ball = self.config.ball_scale * (hi - lo)
        p0 = self.nominal + ball * rng.standard_normal(size=(n_walkers, self.n_param))
        return np.clip(p0, lo, hi)

    def run(self) -> MCMCResult:
        t0 = time.perf_counter()
        self._n_eval = 0
        self._n_fail = 0
        ndim = self.n_param

        # emcee's default StretchMove requires n_walkers >= 2 * ndim; enforce it.
        n_walkers = max(self.config.n_walkers, 2 * ndim)
        if n_walkers > self.config.n_walkers:
            import warnings

            warnings.warn(
                f"n_walkers raised from {self.config.n_walkers} to {n_walkers} "
                f"to satisfy emcee's n_walkers >= 2*ndim requirement (d={ndim})."
            )
        n_ensembles = self.config.n_ensembles

        # Run several *independent* ensembles (each with its own RNG and initial
        # positions); walkers within one ensemble are not treated as chains.
        chains_list = []
        accept_list = []
        for e in range(n_ensembles):
            rng = np.random.default_rng(self.config.seed + e)
            p0 = self._init_walkers(rng, n_walkers)
            sampler = emcee.EnsembleSampler(n_walkers, ndim, self._logpost_wrapped)
            total = self.config.n_burn + self.config.n_steps
            sampler.run_mcmc(p0, total, progress=False)
            chain = sampler.get_chain(discard=self.config.n_burn, flat=False)
            chains_list.append(np.transpose(chain, (1, 0, 2)))  # (n_walkers, n_steps, ndim)
            accept_list.append(float(np.mean(sampler.acceptance_fraction)))

        # Combine ensembles: (n_ensembles, n_steps, ndim) by averaging walkers.
        chains = np.stack(
            [c.mean(axis=0) for c in chains_list], axis=0
        )  # (n_ensembles, n_steps, ndim)
        n_samples = int(chains.shape[0] * chains.shape[1])

        post_cov = np.cov(chains.reshape(-1, ndim).T)
        whitened = None
        if self.prior_cov is not None:
            try:
                whitened = whiten_covariance(post_cov, self.prior_cov)
            except Exception:
                whitened = None
        rhat = split_rhat(chains)
        ess_bulk = effective_sample_size(chains)
        ess_tail = tail_effective_sample_size(chains)
        tau = autocorrelation_time(chains)
        acceptance = float(np.mean(accept_list))

        rhat_max = float(np.nanmax(rhat)) if rhat.size else np.inf
        ess_min = float(np.nanmin(ess_bulk)) if ess_bulk.size else 0.0
        ess_tail_min = float(np.nanmin(ess_tail)) if ess_tail.size else 0.0
        tau_max = float(np.nanmax(tau)) if tau.size else np.inf
        chain_len = chains.shape[1]
        tau_covered = tau_max < chain_len / self.config.tau_coverage

        converged = (
            rhat_max < self.config.rhat_threshold
            and ess_min > self.config.ess_threshold
            and ess_tail_min > self.config.ess_threshold
            and tau_covered
            and np.all(np.isfinite(post_cov))
        )
        failure_rate = self._n_fail / self._n_eval if self._n_eval else 0.0

        reasons = []
        if not (rhat_max < self.config.rhat_threshold):
            reasons.append(f"rhat_max={rhat_max:.4f}>={self.config.rhat_threshold}")
        if not (ess_min > self.config.ess_threshold):
            reasons.append(f"ess_bulk_min={ess_min:.1f}<={self.config.ess_threshold}")
        if not (ess_tail_min > self.config.ess_threshold):
            reasons.append(f"ess_tail_min={ess_tail_min:.1f}<={self.config.ess_threshold}")
        if not tau_covered:
            reasons.append(f"tau_max={tau_max:.1f} not covered by chain_len={chain_len}")

        return MCMCResult(
            model_id=self.model_id, n_parameters=ndim,
            n_walkers=n_walkers, n_ensembles=n_ensembles, n_samples=n_samples,
            posterior_mean=chains.reshape(-1, ndim).mean(axis=0).tolist(),
            posterior_cov=post_cov.tolist(),
            whitened_cov=whitened.tolist() if whitened is not None else None,
            rhat=np.where(np.isfinite(rhat), rhat, np.nan).tolist(),
            ess_bulk=ess_bulk.tolist(),
            ess_tail=ess_tail.tolist(),
            autocorr_time=tau.tolist(),
            acceptance_fraction=round(acceptance, 4),
            divergences=None, failure_rate=round(failure_rate, 4),
            convergence_status=CONVERGED if converged else UNCONVERGED,
            label_allowed=bool(converged),
            excluded_by_rule="; ".join(reasons) if reasons else "",
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    def run_versioned(self, out_dir: Path, resume: bool = True) -> Dict[str, Any]:
        """Resumable: skip if summary.json exists; else run + write samples + summary."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        if resume and summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        res = self.run()
        summary = asdict(res)
        summary["param_ids"] = self.param_ids
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        return summary
