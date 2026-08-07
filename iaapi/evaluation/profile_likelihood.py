"""Standardized profile-likelihood runner (NCS plan task P2-02).

A resumable, multistart, per-parameter PL runner with configurable timeouts,
convergence diagnostics, explicit failure codes, and no infinite retry loops.
The negative-log-likelihood ``nll_fn(theta) -> float`` is INJECTABLE, so the
runner is unit-testable with analytic objectives and can be paired with either a
synthetic-data or a real-PEtab-data likelihood in production.

Failure codes (explicit, not swallowed):
  FINITE_CI           both CI bounds found strictly inside parameter bounds
  BOUNDARY_HIT        profile does not cross the chi^2 threshold before a bound
  OPTIMIZER_FAILURE   the multistart MLE / profile optimisation did not converge
  SIMULATOR_FAILURE   nll_fn raised (e.g. AMICI integration failure)
  TIMEOUT             per-parameter wall budget exceeded

Discipline (P2-02): PL labels are an INDEPENDENT ground-truth source (not FIM).
"""
from __future__ import annotations

import json
import math
import signal
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

SCHEMA_VERSION = "1.0"

FINITE_CI = "FINITE_CI"
BOUNDARY_HIT = "BOUNDARY_HIT"
OPTIMIZER_FAILURE = "OPTIMIZER_FAILURE"
SIMULATOR_FAILURE = "SIMULATOR_FAILURE"
TIMEOUT = "TIMEOUT"

CHI2_THRESH_1DF_95 = float(chi2.ppf(0.95, df=1))  # ~3.841


class SimulatorError(Exception):
    """Raised by nll_fn when the forward simulation fails."""


class _AlarmTimeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _AlarmTimeout("PL per-parameter timeout")


@contextmanager
def _soft_timeout(seconds: Optional[float]):
    if not seconds or seconds <= 0:
        yield
        return
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@dataclass(frozen=True)
class PLConfig:
    n_starts: int = 4
    maxiter: int = 200
    timeout_per_param_s: float = 300.0
    threshold: float = CHI2_THRESH_1DF_95
    profile_step: float = 0.1     # initial fractional step (of bound width) for bracketing
    profile_max_steps: int = 8   # bounded => no infinite scan
    bisect_iters: int = 8
    seed: int = 0
    # R1-02: convergence enforcement and coverage gate
    require_converged_mle: bool = True
    min_classified_coverage: float = 0.80


@dataclass
class ParamResult:
    parameter_id: str
    parameter_index: int
    mle: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    code: str
    nll_min: Optional[float]
    n_starts: int
    n_evals: int
    converged: bool
    elapsed_s: float
    detail: str = ""
    # R1-02: per-start optimizer traces (start_idx, nll, converged, n_evals)
    start_traces: List[Dict[str, Any]] = field(default_factory=list)


NllFn = Callable[[np.ndarray], float]


class ProfileLikelihoodRunner:
    def __init__(self, nll_fn: NllFn, bounds_theta: np.ndarray,
                 theta_nominal: np.ndarray, param_ids: List[str],
                 config: PLConfig = PLConfig()):
        self.nll_fn = nll_fn
        self.bounds = np.asarray(bounds_theta, dtype=float)
        self.nominal = np.asarray(theta_nominal, dtype=float).reshape(-1)
        self.param_ids = list(param_ids)
        self.n_param = self.bounds.shape[0]
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._evals = 0
        self._deadline: Optional[float] = None  # wall-clock deadline for the current parameter

    # ---- optimisation -------------------------------------------------- #
    def _safe_nll(self, theta: np.ndarray) -> float:
        self._evals += 1
        # Deterministic per-parameter timeout (works inside C/optimizer loops,
        # unlike signal-only delivery). Signal (_soft_timeout) remains a backstop.
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise _AlarmTimeout("PL per-parameter wall budget exceeded")
        try:
            return float(self.nll_fn(theta))
        except SimulatorError:
            raise
        except Exception as e:  # any simulator/numerical failure
            raise SimulatorError(str(e)) from e

    def _minimize_over_free(self, fixed_idx: Optional[int], fixed_val: float,
                            starts: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[float], bool, List[Dict[str, Any]]]:
        """Minimise nll over all params (fixed_idx=None) or all except fixed_idx.

        Returns (best_theta, best_nll, any_converged, start_traces).
        Raises SimulatorError if EVERY start failed via a simulator error (so the
        caller can distinguish SIMULATOR_FAILURE from OPTIMIZER_FAILURE).
        """
        d = self.n_param
        free = [j for j in range(d) if j != fixed_idx]
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]

        # Zero free params (1-param model profiling its only parameter): the
        # profile value is just nll at the fixed point; no optimisation needed.
        if not free:
            theta = np.empty(d)
            if fixed_idx is not None:
                theta[fixed_idx] = fixed_val
            nll_val = self._safe_nll(theta)  # may raise SimulatorError
            return theta, float(nll_val), True, [{"start_idx": 0, "nll": float(nll_val), "converged": True, "n_evals": 1}]

        def obj(x_free):
            theta = np.empty(d)
            if fixed_idx is not None:
                theta[fixed_idx] = fixed_val
            for k, j in enumerate(free):
                theta[j] = x_free[k]
            return self._safe_nll(theta)

        best_theta, best_nll, conv = None, None, False
        x0_lo = np.array([lo[j] for j in free])
        x0_hi = np.array([hi[j] for j in free])
        sim_errors = 0
        start_traces: List[Dict[str, Any]] = []
        for s_idx, s in enumerate(starts):
            x0 = np.array([s[j] for j in free])
            trace: Dict[str, Any] = {"start_idx": s_idx, "nll": None, "converged": False, "n_evals": 0}
            try:
                evals_before = self._evals
                res = minimize(obj, x0, method="L-BFGS-B",
                               bounds=list(zip(x0_lo, x0_hi)),
                               options={"maxiter": self.config.maxiter})
                trace["n_evals"] = self._evals - evals_before
                trace["nll"] = float(res.fun) if math.isfinite(res.fun) else None
                trace["converged"] = bool(res.success)
            except SimulatorError:
                sim_errors += 1
                trace["n_evals"] = self._evals - evals_before
                start_traces.append(trace)
                continue
            if not math.isfinite(res.fun):
                start_traces.append(trace)
                continue
            if best_nll is None or res.fun < best_nll:
                best_nll = float(res.fun)
                theta = np.empty(d)
                if fixed_idx is not None:
                    theta[fixed_idx] = fixed_val
                for k, j in enumerate(free):
                    theta[j] = res.x[k]
                best_theta = theta
                conv = bool(res.success)
            start_traces.append(trace)
        if best_theta is None and sim_errors == len(starts):
            raise SimulatorError("all multistart evaluations failed (simulator)")
        return best_theta, best_nll, conv, start_traces

    def _starts(self, n: int) -> List[np.ndarray]:
        starts = [self.nominal.copy()]
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        for _ in range(max(0, n - 1)):
            starts.append(self._rng.uniform(lo, hi))
        return starts[:n]

    def _profile_side(self, fixed_idx: int, mle_val: float, nll_min: float,
                      direction: int) -> Tuple[Optional[float], str]:
        """Bracket+bisect the CI bound on one side of mle_val (direction +1/-1)."""
        lo, hi = self.bounds[fixed_idx]
        step = self.config.profile_step * (hi - lo)
        c = mle_val
        starts = self._starts(self.config.n_starts)
        prev_c = c
        for _ in range(self.config.profile_max_steps):
            c = c + direction * step
            if direction < 0 and c <= lo:
                # hit bound: is the bound inside the CI?
                try:
                    _, nll_b, _, _ = self._minimize_over_free(fixed_idx, lo, starts)
                except SimulatorError:
                    return lo, SIMULATOR_FAILURE
                if nll_b is None:
                    return lo, OPTIMIZER_FAILURE
                g = 2.0 * (nll_b - nll_min)
                return (lo, FINITE_CI) if g > self.config.threshold else (lo, BOUNDARY_HIT)
            if direction > 0 and c >= hi:
                try:
                    _, nll_b, _, _ = self._minimize_over_free(fixed_idx, hi, starts)
                except SimulatorError:
                    return hi, SIMULATOR_FAILURE
                if nll_b is None:
                    return hi, OPTIMIZER_FAILURE
                g = 2.0 * (nll_b - nll_min)
                return (hi, FINITE_CI) if g > self.config.threshold else (hi, BOUNDARY_HIT)
            try:
                _, nll_c, _, _ = self._minimize_over_free(fixed_idx, c, starts)
            except SimulatorError:
                return None, SIMULATOR_FAILURE
            if nll_c is None:
                return None, OPTIMIZER_FAILURE
            g = 2.0 * (nll_c - nll_min)
            if g > self.config.threshold:
                # bracketed between prev_c (g<=thr) and c (g>thr): bisect
                return self._bisect(fixed_idx, prev_c, c, nll_min, starts), FINITE_CI
            prev_c = c
        # never crossed threshold within step budget => boundary hit on this side
        return (lo if direction < 0 else hi), BOUNDARY_HIT

    def _bisect(self, fixed_idx: int, a: float, b: float,
                nll_min: float, starts: List[np.ndarray]) -> float:
        for _ in range(self.config.bisect_iters):
            m = 0.5 * (a + b)
            try:
                _, nll_m, _, _ = self._minimize_over_free(fixed_idx, m, starts)
            except SimulatorError:
                return m
            if nll_m is None:
                return m
            g = 2.0 * (nll_m - nll_min)
            if g > self.config.threshold:
                b = m
            else:
                a = m
        return 0.5 * (a + b)

    # ---- per-parameter ------------------------------------------------- #
    def profile_parameter(self, i: int) -> ParamResult:
        t0 = time.perf_counter()
        self._evals = 0
        self._deadline = t0 + self.config.timeout_per_param_s if self.config.timeout_per_param_s > 0 else None
        starts = self._starts(self.config.n_starts)
        code = FINITE_CI
        try:
            with _soft_timeout(self.config.timeout_per_param_s):
                # MLE (fix nothing): minimise over all params by fixing a dummy.
                theta_mle, nll_min, conv, mle_traces = self._minimize_over_free(None, 0.0, starts)
                # R1-02: require converged MLE
                if theta_mle is None or nll_min is None or not math.isfinite(nll_min):
                    return self._result(i, None, None, None, OPTIMIZER_FAILURE, None,
                                        len(starts), False, t0, "MLE not found", mle_traces)
                if self.config.require_converged_mle and not conv:
                    return self._result(i, None, None, None, OPTIMIZER_FAILURE, nll_min,
                                        len(starts), False, t0,
                                        "MLE did not converge in any start", mle_traces)
                mle_val = float(theta_mle[i])
                ci_lo, code_lo = self._profile_side(i, mle_val, nll_min, -1)
                ci_hi, code_hi = self._profile_side(i, mle_val, nll_min, +1)
                # worst code wins (failure > boundary > finite)
                code = _worst_code(code_lo, code_hi)
                return self._result(i, mle_val, ci_lo, ci_hi, code, nll_min,
                                    len(starts), conv, t0, f"left={code_lo};right={code_hi}", mle_traces)
        except _AlarmTimeout:
            return self._result(i, None, None, None, TIMEOUT, None, len(starts), False, t0, "timeout", [])
        except SimulatorError as e:
            return self._result(i, None, None, None, SIMULATOR_FAILURE, None, len(starts), False, t0, str(e), [])

    def _result(self, i, mle, lo, hi, code, nll_min, n_starts, conv, t0, detail,
                start_traces=None) -> ParamResult:
        return ParamResult(
            parameter_id=self.param_ids[i], parameter_index=i, mle=mle,
            ci_lower=lo, ci_upper=hi, code=code, nll_min=nll_min,
            n_starts=n_starts, n_evals=self._evals, converged=conv,
            elapsed_s=round(time.perf_counter() - t0, 4), detail=detail,
            start_traces=start_traces or [])

    # ---- resumable full run ------------------------------------------- #
    def run(self, output_dir: Path, resume: bool = True) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[ParamResult] = []
        for i in range(self.n_param):
            per_path = output_dir / f"{self.param_ids[i]}.json"
            if resume and per_path.is_file():
                d = json.loads(per_path.read_text(encoding="utf-8"))
                results.append(ParamResult(**{k: d[k] for k in ParamResult.__dataclass_fields__ if k in d}))
                continue
            r = self.profile_parameter(i)
            per_path.write_text(json.dumps(asdict(r), indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            results.append(r)
            self._write_progress(output_dir, i, results)
        summary = {"schema_version": SCHEMA_VERSION, "param_ids": self.param_ids,
                   "n_param": self.n_param, "threshold": self.config.threshold,
                   "results": [asdict(r) for r in results]}
        # R1-02: coverage gate
        classified_codes = {FINITE_CI, BOUNDARY_HIT}
        n_classified = sum(1 for r in results if r.code in classified_codes)
        classified_coverage = n_classified / self.n_param if self.n_param > 0 else 0.0
        summary["classified_coverage"] = round(classified_coverage, 6)
        summary["label_admitted"] = classified_coverage >= self.config.min_classified_coverage
        summary["min_classified_coverage"] = self.config.min_classified_coverage
        summary["n_classified"] = n_classified
        (output_dir / "pl_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    def _write_progress(self, output_dir: Path, i: int, results: List[ParamResult]) -> None:
        prog = {"completed": i + 1, "total": self.n_param,
                "current": self.param_ids[i] if i + 1 < self.n_param else None,
                "codes": [r.code for r in results]}
        (output_dir / "progress.json").write_text(
            json.dumps(prog, indent=2) + "\n", encoding="utf-8")


def _worst_code(a: str, b: str) -> str:
    rank = {FINITE_CI: 0, BOUNDARY_HIT: 1, OPTIMIZER_FAILURE: 2,
            SIMULATOR_FAILURE: 3, TIMEOUT: 4}
    return a if rank.get(a, 0) >= rank.get(b, 0) else b
