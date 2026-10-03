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

SCHEMA_VERSION = "2.0"
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
    diagnostic_method: str = "worst_walker_across_independent_ensembles"
    diagnostic_samples_per_walker: int = 0


LogPostFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------- #
# Diagnostics (rank-normalized, per Vehtari et al. 2021)
# --------------------------------------------------------------------------- #
def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Pool ranks across chains; separate per-chain ranks hide location shifts."""
    from scipy import stats

    chains = np.asarray(chains, dtype=float)
    out = np.empty_like(chains)
    for k in range(chains.shape[2]):
        ranks = stats.rankdata(chains[:, :, k].reshape(-1), method="average")
        u = (ranks - 0.375) / (ranks.size + 0.25)
        out[:, :, k] = stats.norm.ppf(u).reshape(chains.shape[:2])
    return out


def _split_chains(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or not np.all(np.isfinite(chains)):
        raise ValueError("chains must be finite (independent_chains, steps, parameters)")
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)


def _basic_rhat(chains: np.ndarray) -> np.ndarray:
    m, n, d = chains.shape
    if m < 2 or n < 2:
        return np.full(d, np.nan)
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = n * chains.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n
    result = np.full(d, np.inf)
    np.sqrt(np.divide(var_plus, within, out=np.full(d, np.inf),
                      where=within > 0), out=result)
    return result


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Maximum of rank-normalized and folded split R-hat (Vehtari et al.).

    Input chains must be independent. Walkers in one emcee ensemble are
    dependent and must not be stacked as independent chains.
    """
    chains = np.asarray(chains, dtype=float)
    split = _split_chains(chains)
    if chains.shape[0] < 2 or split.shape[1] < 2:
        return np.full(chains.shape[2], np.nan)
    location = _basic_rhat(_rank_normalize(split))
    medians = np.median(split, axis=(0, 1))
    scale = _basic_rhat(_rank_normalize(np.abs(split - medians)))
    return np.maximum(location, scale)


def _autocov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    centered = x - x.mean()
    f = np.fft.fft(centered, n=2 * n)
    return np.fft.ifft(f * np.conjugate(f))[:n].real / n


def _autocorr(x: np.ndarray) -> np.ndarray:
    acov = _autocov(x)
    return acov / acov[0] if acov[0] > 0 else np.full(len(x), np.nan)


def _geyer_tau(rho: np.ndarray) -> float:
    """Initial positive, monotone sequence; conservative cap at tau >= 1."""
    pairs = []
    for t in range(0, len(rho) - 1, 2):
        pair = float(rho[t] + rho[t + 1])
        if not np.isfinite(pair):
            return np.inf
        if pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    return max(-1.0 + 2.0 * sum(pairs), 1.0)


def _ess_from_chains(chains: np.ndarray) -> np.ndarray:
    """Split-chain ESS including between-chain variance."""
    split = _split_chains(chains)
    m, n, d = split.shape
    if m < 2 or n < 2:
        return np.zeros(d)
    ess = np.zeros(d)
    for k in range(d):
        x = split[:, :, k]
        within = x.var(axis=1, ddof=1).mean()
        var_plus = (n - 1) / n * within + x.mean(axis=1).var(ddof=1)
        if var_plus <= 0 or not np.isfinite(var_plus):
            continue
        mean_acov = np.mean([_autocov(row) for row in x], axis=0)
        rho = 1.0 - (within - mean_acov) / var_plus
        rho[0] = 1.0
        ess[k] = min(m * n / _geyer_tau(rho), float(m * n))
    return ess


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS of pooled rank-normalized draws."""
    return _ess_from_chains(_rank_normalize(np.asarray(chains, dtype=float)))


def tail_effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Minimum ESS of the lower 5% and upper 95% quantile indicators."""
    chains = np.asarray(chains, dtype=float)
    lower, upper = np.quantile(chains, [0.05, 0.95], axis=(0, 1))
    return np.minimum(
        _ess_from_chains((chains <= lower).astype(float)),
        _ess_from_chains((chains >= upper).astype(float)),
    )


def autocorrelation_time(chains: np.ndarray) -> np.ndarray:
    """Geyer integrated autocorrelation time in each original coordinate."""
    chains = np.asarray(chains, dtype=float)
    _split_chains(chains)  # validate without treating walkers as independent
    tau = np.empty(chains.shape[2])
    for k in range(chains.shape[2]):
        rho = np.mean([_autocorr(row[:, k]) for row in chains], axis=0)
        tau[k] = _geyer_tau(rho)
    return tau


def ensemble_diagnostics(raw_chains: np.ndarray) -> tuple[np.ndarray, ...]:
    """Diagnose each walker index across independently seeded ensembles.

    For a fixed walker index, ensembles are independent chains. Aggregate
    worst R-hat/tau and minimum ESS across walker indices, without multiplying
    ESS by the number of mutually dependent walkers.
    """
    raw = np.asarray(raw_chains, dtype=float)
    if raw.ndim != 4 or not np.all(np.isfinite(raw)):
        raise ValueError("raw_chains must be finite (ensembles, walkers, steps, parameters)")
    results = []
    for w in range(raw.shape[1]):
        chains = raw[:, w]
        results.append((split_rhat(chains), effective_sample_size(chains),
                        tail_effective_sample_size(chains), autocorrelation_time(chains)))
    rhat, bulk, tail, tau = (np.stack([r[i] for r in results]) for i in range(4))
    return rhat.max(axis=0), bulk.min(axis=0), tail.min(axis=0), tau.max(axis=0)


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
            sampler.random_state = np.random.RandomState(self.config.seed + e).get_state()
            total = self.config.n_burn + self.config.n_steps
            sampler.run_mcmc(p0, total, progress=False)
            chain = sampler.get_chain(discard=self.config.n_burn, flat=False)
            chains_list.append(np.transpose(chain, (1, 0, 2)))  # (n_walkers, n_steps, ndim)
            accept_list.append(float(np.mean(sampler.acceptance_fraction)))

        # Keep every post-burn-in state; never compute posterior width from means.
        raw_chains = np.stack(chains_list, axis=0)
        self.last_chains = raw_chains
        samples = raw_chains.reshape(-1, ndim)
        n_samples = len(samples)
        post_cov = np.atleast_2d(np.cov(samples, rowvar=False))
        whitened = None
        if self.prior_cov is not None:
            try:
                whitened = whiten_covariance(post_cov, self.prior_cov)
            except Exception:
                whitened = None
        rhat, ess_bulk, ess_tail, tau = ensemble_diagnostics(raw_chains)
        acceptance = float(np.mean(accept_list))

        rhat_max = float(np.max(rhat)) if rhat.size and np.all(np.isfinite(rhat)) else np.inf
        ess_min = float(np.min(ess_bulk)) if ess_bulk.size and np.all(np.isfinite(ess_bulk)) else 0.0
        ess_tail_min = float(np.min(ess_tail)) if ess_tail.size and np.all(np.isfinite(ess_tail)) else 0.0
        tau_max = float(np.max(tau)) if tau.size and np.all(np.isfinite(tau)) else np.inf
        chain_len = raw_chains.shape[2]
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
            posterior_mean=samples.mean(axis=0).tolist(),
            diagnostic_samples_per_walker=n_ensembles * chain_len,
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
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != SCHEMA_VERSION or not (out_dir / "chains.npz").is_file():
                raise ValueError("Legacy or incomplete MCMC result: use a new output directory "
                                 "or resume=False after archiving old results.")
            if summary.get("config") != asdict(self.config) or summary.get("param_ids") != self.param_ids:
                raise ValueError("MCMC configuration or parameter order changed; use a new output directory.")
            return summary
        res = self.run()
        np.savez_compressed(out_dir / "chains.npz", chains=self.last_chains,
                            param_ids=np.asarray(self.param_ids),
                            axis_order="ensemble,walker,step,parameter",
                            schema_version=SCHEMA_VERSION)
        summary = asdict(res)
        summary["param_ids"] = self.param_ids
        summary["config"] = asdict(self.config)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        return summary
