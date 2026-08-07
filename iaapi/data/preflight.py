"""PEtab model preflight scanner (NCS plan task P1-01).

Measures numerical viability of a candidate PEtab model *before* committing to
expensive training-data generation, so unstable models are rejected up-front
instead of burning a generation run (the documented DATA-HO / DATA-X6 failure
mode caused by full-bound uniform sampling).

For every candidate model it measures the 8 quantities required by the plan:
  1. nominal forward success + runtime
  2. nominal sensitivity / FIM success
  3. N-sample pilot forward success rate
  4. pilot sensitivity success rate
  5. NaN/Inf rate
  6. median and p95 runtime
  7. FIM finite / effective-rank statistics
  8. failure categories

The core ``PreflightRunner`` depends only on an injectable ``simulate_fn`` and a
theta-space bounds array, so it is unit-testable without AMICI. The real AMICI
glue lives in :func:`run_model_from_yaml`.

Scientific discipline: this scanner reports failures by category; it never
silently drops a model. A REJECT is a valid preflight outcome, not an error.
"""
from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"

# Failure category tags (stable strings; do not rename without updating tests).
CAT_COMPILE = "compile_error"
CAT_FORWARD_STATUS = "forward_status_nonzero"
CAT_FORWARD_NAN = "forward_nan_inf"
CAT_SENS_NAN = "sensitivity_nan_inf"
CAT_FIM_NONFINITE = "fim_non_finite"
CAT_TIMEOUT = "timeout"
CAT_EXCEPTION = "exception"
CAT_UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Soft per-call timeout (main thread only). AMICI is a C call; the alarm fires
# when control returns to Python. The CLI adds a harder per-model wall budget.
# --------------------------------------------------------------------------- #
class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _AlarmTimeout("simulate timed out")


@contextmanager
def soft_timeout(seconds: Optional[float]):
    """Best-effort wall-clock timeout around a block (POSIX, main thread)."""
    if not seconds or seconds <= 0:
        yield
        return
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# --------------------------------------------------------------------------- #
# Config + results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreflightConfig:
    n_samples: int = 50
    seed: int = 0
    noise_level: float = 0.1
    forward_success_threshold: float = 0.90
    sensitivity_success_threshold: float = 0.80
    runtime_p95_threshold_s: float = 300.0
    per_simulate_timeout_s: float = 120.0
    pilot_sensitivity_subset: int = 20


@dataclass
class ModelPreflightResult:
    model_id: str
    schema_version: str = SCHEMA_VERSION
    n_parameters: int = 0
    n_samples: int = 0
    nominal_forward: Dict[str, Any] = field(default_factory=dict)
    nominal_sensitivity: Dict[str, Any] = field(default_factory=dict)
    fim: Dict[str, Any] = field(default_factory=dict)
    pilot_forward: Dict[str, Any] = field(default_factory=dict)
    pilot_sensitivity: Dict[str, Any] = field(default_factory=dict)
    failure_categories: List[str] = field(default_factory=list)
    gate: str = "PENDING"
    gate_reasons: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_parameters(bounds_theta: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Uniformly sample ``n`` theta vectors within ``bounds_theta`` (n_param, 2)."""
    rng = np.random.default_rng(seed)
    lo = bounds_theta[:, 0]
    hi = bounds_theta[:, 1]
    return rng.uniform(lo, hi, size=(n, len(lo)))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
SimulateFn = Callable[[np.ndarray, bool], Dict[str, Any]]


class PreflightRunner:
    """Measure one model's viability via an injectable ``simulate_fn``.

    Args:
        model_id: identifier for the report.
        simulate_fn: ``theta, compute_sensitivities -> dict`` with keys
            ``status`` (int; 0 = AMICI success) and ``trajectories`` /
            ``sensitivities`` arrays (may be None).
        theta_nominal: nominal theta vector (in the simulator's scale).
        bounds_theta: (n_param, 2) sampling bounds in the simulator's scale.
        n_parameters: number of free parameters.
        config: :class:`PreflightConfig`.
        fim_computer: optional object with ``compute_fim(sensitivities, sigma)``
            (e.g. :class:`iaapi.data.fim_compute.FIMComputer`). If None, FIM is
            computed inline.
    """

    def __init__(
        self,
        model_id: str,
        simulate_fn: SimulateFn,
        theta_nominal: np.ndarray,
        bounds_theta: np.ndarray,
        n_parameters: int,
        config: PreflightConfig,
        fim_computer: Any = None,
    ):
        self.model_id = model_id
        self.simulate_fn = simulate_fn
        self.theta_nominal = np.asarray(theta_nominal, dtype=float).reshape(-1)
        self.bounds_theta = np.asarray(bounds_theta, dtype=float)
        self.n_parameters = int(n_parameters)
        self.config = config
        self.fim_computer = fim_computer
        self._categories: List[str] = []

    def _simulate(self, theta: np.ndarray, compute_sens: bool) -> Tuple[bool, Optional[Dict[str, Any]], str, float]:
        """Run one simulate call. Returns (ok, result, category, runtime_s)."""
        t0 = time.perf_counter()
        try:
            with soft_timeout(self.config.per_simulate_timeout_s):
                res = self.simulate_fn(theta, compute_sens)
        except _AlarmTimeout:
            return False, None, CAT_TIMEOUT, time.perf_counter() - t0
        except Exception:  # noqa: BLE001 - categorize, do not swallow silently
            return False, None, CAT_EXCEPTION, time.perf_counter() - t0
        runtime = time.perf_counter() - t0
        category = _classify_result(res, compute_sens)
        ok = category == ""
        return ok, res, category, runtime

    def run(self) -> ModelPreflightResult:
        t_start = time.perf_counter()
        result = ModelPreflightResult(
            model_id=self.model_id,
            n_parameters=self.n_parameters,
            n_samples=self.config.n_samples,
        )

        # 1+2. Nominal forward + sensitivity / FIM.
        ok_fwd, res_fwd, cat_fwd, rt_fwd = self._simulate(self.theta_nominal, compute_sens=False)
        result.nominal_forward = _summarize_forward(res_fwd, ok_fwd, cat_fwd, rt_fwd)
        if cat_fwd:
            self._categories.append(cat_fwd)

        ok_sens, res_sens, cat_sens, rt_sens = self._simulate(self.theta_nominal, compute_sens=True)
        result.nominal_sensitivity = _summarize_sensitivity(res_sens, ok_sens, cat_sens, rt_sens)
        if cat_sens:
            self._categories.append(cat_sens)

        # 7. FIM eigen-statistics (inline, transparent; reuses FIMComputer matrix if provided).
        result.fim = _fim_stats(res_sens, self.fim_computer, self.config.noise_level, self._categories)

        # 3+4+5+6. Pilot forward + sensitivity.
        samples = sample_parameters(self.bounds_theta, self.config.n_samples, self.config.seed)
        fwd_ok_flags: List[bool] = []
        fwd_runtimes: List[float] = []
        nan_count = 0
        successful_thetas: List[np.ndarray] = []
        for theta in samples:
            ok, res, cat, rt = self._simulate(theta, compute_sens=False)
            fwd_ok_flags.append(ok)
            fwd_runtimes.append(rt)
            if cat == CAT_FORWARD_NAN:
                nan_count += 1
            if cat:
                self._categories.append(cat)
            if ok and len(successful_thetas) < self.config.pilot_sensitivity_subset:
                successful_thetas.append(theta)

        result.pilot_forward = _summarize_pilot_forward(fwd_ok_flags, fwd_runtimes, nan_count)

        sens_ok_flags: List[bool] = []
        for theta in successful_thetas:
            ok, _res, cat, _rt = self._simulate(theta, compute_sens=True)
            sens_ok_flags.append(ok)
            if cat:
                self._categories.append(cat)
        result.pilot_sensitivity = _summarize_pilot_sensitivity(sens_ok_flags)

        result.failure_categories = sorted(set(self._categories))
        result.gate, result.gate_reasons = evaluate_gate(result, self.config)
        result.elapsed_s = round(time.perf_counter() - t_start, 4)
        return result


# --------------------------------------------------------------------------- #
# Summarizers + classifiers
# --------------------------------------------------------------------------- #
def _has_nan_inf(arr: Optional[np.ndarray]) -> bool:
    if arr is None:
        return False
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return False
    return bool(not np.all(np.isfinite(a)))


def _classify_result(res: Optional[Dict[str, Any]], compute_sens: bool) -> str:
    if res is None:
        return CAT_UNKNOWN
    status = res.get("status")
    if status is not None and int(status) != 0:
        return CAT_FORWARD_STATUS
    if _has_nan_inf(res.get("trajectories")):
        return CAT_FORWARD_NAN
    if compute_sens and _has_nan_inf(res.get("sensitivities")):
        return CAT_SENS_NAN
    return ""


def _summarize_forward(res, ok, cat, rt) -> Dict[str, Any]:
    return {"success": bool(ok), "runtime_s": round(rt, 4),
            "status": (res.get("status") if res else None), "failure_category": cat}


def _summarize_sensitivity(res, ok, cat, rt) -> Dict[str, Any]:
    return {"success": bool(ok), "runtime_s": round(rt, 4), "failure_category": cat}


def _summarize_pilot_forward(ok_flags, runtimes, nan_count) -> Dict[str, Any]:
    n = len(ok_flags)
    runtimes = sorted(runtimes)
    return {
        "n": n,
        "success_rate": round(sum(ok_flags) / n, 4) if n else 0.0,
        "nan_rate": round(nan_count / n, 4) if n else 0.0,
        "runtime_median_s": round(_percentile(runtimes, 50), 4) if runtimes else 0.0,
        "runtime_p95_s": round(_percentile(runtimes, 95), 4) if runtimes else 0.0,
    }


def _summarize_pilot_sensitivity(ok_flags) -> Dict[str, Any]:
    n = len(ok_flags)
    return {"n_attempted": n, "success_rate": round(sum(ok_flags) / n, 4) if n else 0.0}


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    return float(np.percentile(sorted_vals, q))


def _fim_stats(res_sens, fim_computer, noise_level, categories) -> Dict[str, Any]:
    sens = (res_sens or {}).get("sensitivities")
    if sens is None:
        return {"finite": False, "reason": "no sensitivities"}
    sens = np.asarray(sens, dtype=float)
    if sens.ndim != 3 or not np.all(np.isfinite(sens)):
        return {"finite": False, "reason": "non-finite sensitivities"}
    try:
        if fim_computer is not None:
            fim = fim_computer.compute_fim(sens, sigma=None)
        else:
            j = sens.reshape(-1, sens.shape[-1])
            j = np.nan_to_num(j)
            fim = j.T @ j * (1.0 / (noise_level ** 2))
            fim = 0.5 * (fim + fim.T)
    except Exception:  # noqa: BLE001
        categories.append(CAT_FIM_NONFINITE)
        return {"finite": False, "reason": "fim computation failed"}
    if not np.all(np.isfinite(fim)):
        categories.append(CAT_FIM_NONFINITE)
        return {"finite": False, "reason": "non-finite fim"}
    eig = np.linalg.eigvalsh(fim)
    eig = np.where(eig < 0, 0.0, eig)
    pos = eig[eig > 0]
    eff_rank = float((pos.sum() ** 2) / (pos ** 2).sum()) if pos.size else 0.0
    max_e = float(eig.max())
    min_e = float(eig.min())
    cond = float(max_e / min_e) if min_e > 0 else float("inf")
    return {
        "finite": True,
        "n_finite_eigenvalues": int(np.sum(np.isfinite(eig))),
        "effective_rank_participation": round(eff_rank, 6),
        "max_eigenvalue": max_e,
        "min_eigenvalue": min_e,
        "condition_number": cond,
    }


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def evaluate_gate(result: ModelPreflightResult, config: PreflightConfig) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if not result.nominal_forward.get("success"):
        reasons.append("nominal forward failed")
    if not result.nominal_sensitivity.get("success"):
        reasons.append("nominal sensitivity failed")
    if not result.fim.get("finite"):
        reasons.append("FIM not finite")

    pf = result.pilot_forward
    if pf.get("n") and pf["success_rate"] < config.forward_success_threshold:
        reasons.append(f"pilot forward success {pf['success_rate']} < {config.forward_success_threshold}")
    if pf.get("runtime_p95_s", 0.0) > config.runtime_p95_threshold_s:
        reasons.append(f"p95 runtime {pf['runtime_p95_s']}s > {config.runtime_p95_threshold_s}s")

    ps = result.pilot_sensitivity
    if ps.get("n_attempted") and ps["success_rate"] < config.sensitivity_success_threshold:
        reasons.append(f"pilot sensitivity success {ps['success_rate']} < {config.sensitivity_success_threshold}")

    return ("ADMIT" if not reasons else "REJECT"), reasons


# --------------------------------------------------------------------------- #
# AMICI glue
# --------------------------------------------------------------------------- #
def run_model_from_yaml(
    yaml_path: str,
    model_id: str,
    config: PreflightConfig,
    cache_dir: Optional[str] = None,
    compile_model: bool = True,
) -> ModelPreflightResult:
    """Load a PEtab model with AMICI and run the preflight. Real-repo glue."""
    from iaapi.data.fim_compute import FIMComputer
    from iaapi.data.forward_sim import AMICISimulator
    from iaapi.data.petab_loader import PEtabLoader

    loader = PEtabLoader()
    problem = loader.load(str(yaml_path), validate=False)
    sim = AMICISimulator(
        problem, compile_model=compile_model, cache_dir=cache_dir,
        theta_scale="log10", validate=False,
    )

    pdf = sim.petab_problem_v2.parameter_df
    if "parameterId" in pdf.columns:
        pdf = pdf.set_index("parameterId")
    free = list(sim.parameter_ids)
    lower = pdf.loc[free, "lowerBound"].astype(float).to_numpy()
    upper = pdf.loc[free, "upperBound"].astype(float).to_numpy()
    bounds_theta = np.column_stack([sim._to_log10(lower), sim._to_log10(upper)])

    def simulate_fn(theta, compute_sensitivities):
        return sim.simulate(theta, compute_sensitivities=compute_sensitivities)

    runner = PreflightRunner(
        model_id=model_id,
        simulate_fn=simulate_fn,
        theta_nominal=sim.theta_nominal_log10,
        bounds_theta=bounds_theta,
        n_parameters=sim.n_parameters,
        config=config,
        fim_computer=FIMComputer(noise_level=config.noise_level),
    )
    return runner.run()
