"""Coverage from actual posterior samples and known generating parameters.

A provider is required because PEtab real observations do not have known
parameter truth. Per-problem summaries average coordinate indicators; they
are not independent simulation-replicate confidence intervals.
"""
from collections.abc import Callable
from typing import Any

import numpy as np

PosteriorProvider = Callable[[Any, Any, int], tuple[np.ndarray, np.ndarray]]


class CoverageEvaluator:
    def __init__(self, levels: list[float] | None = None, *,
                 posterior_provider: PosteriorProvider | None = None):
        self.levels = [0.5, 0.68, 0.95] if levels is None else list(levels)
        if not self.levels or any(not np.isfinite(x) or not 0 < x <= 1 for x in self.levels):
            raise ValueError("Coverage levels must be in (0, 1]")
        self.posterior_provider = posterior_provider

    def run(self, model: Any, test_problems: list, n_samples: int = 1000) -> dict:
        if self.posterior_provider is None:
            raise NotImplementedError(
                "Supply posterior_provider(model, problem, n_samples) -> (samples, theta_true). "
                "Coverage is unavailable without real posterior samples and synthetic truth."
            )
        if not test_problems or n_samples < 2:
            raise ValueError("Need nonempty synthetic problems and at least two samples")
        per_problem = [self._run_coverage_for_problem(model, p, n_samples)
                       for p in test_problems]
        coverages = {level: [r["coverages"][level] for r in per_problem]
                     for level in self.levels}
        return {
            "coverages": coverages,
            "biases": [r["bias"] for r in per_problem],
            "n_problems": len(test_problems),
            "average_coverages": {level: float(np.mean(values))
                                  for level, values in coverages.items()},
            "per_problem": per_problem,
            "interval_type": "equal_tailed",
            "aggregation": "mean of per-problem coordinate coverages",
        }

    def _run_coverage_for_problem(self, model: Any, problem: Any, n_samples: int) -> dict:
        if self.posterior_provider is None:
            raise NotImplementedError("A real posterior_provider is required")
        samples, truth = self.posterior_provider(model, problem, n_samples)
        samples = np.asarray(samples, dtype=float)
        truth = np.asarray(truth, dtype=float).reshape(-1)
        if samples.ndim == 1 and truth.size == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] != truth.size or truth.size == 0 or len(samples) < 2:
            raise ValueError("samples must have shape (draws, parameters) matching theta_true")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
            raise ValueError("Posterior samples and synthetic truth must be finite")
        hits = {}
        for level in self.levels:
            alpha = (1 - level) / 2
            lower, upper = np.quantile(samples, [alpha, 1 - alpha], axis=0)
            hits[level] = ((truth >= lower) & (truth <= upper)).tolist()
        return {
            "coverages": {level: float(np.mean(values)) for level, values in hits.items()},
            "coordinate_hits": hits,
            "n_coordinates": int(truth.size),
            "n_posterior_draws": int(len(samples)),
            "bias": float(np.mean(samples.mean(axis=0) - truth)),
        }

    @staticmethod
    def _validate_interval(samples: np.ndarray, level: float) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1 or len(samples) < 2 or not np.all(np.isfinite(samples)):
            raise ValueError("Need at least two finite univariate draws")
        if not np.isfinite(level) or not 0 < level <= 1:
            raise ValueError("level must be in (0, 1]")
        return samples

    def _hpd_interval(self, samples: np.ndarray, level: float) -> tuple[float, float]:
        ordered = np.sort(self._validate_interval(samples, level))
        # A window of count samples has endpoint index start + count - 1.
        count = max(2, int(np.ceil(level * len(ordered))))
        widths = ordered[count - 1:] - ordered[:len(ordered) - count + 1]
        start = int(np.argmin(widths))
        return float(ordered[start]), float(ordered[start + count - 1])

    def compute_hpd_coverage(self, posterior_samples: np.ndarray,
                             true_value: float, level: float) -> float:
        if not np.isfinite(true_value):
            raise ValueError("true_value must be finite")
        lower, upper = self._hpd_interval(posterior_samples, level)
        return float(lower <= true_value <= upper)

    def compute_equal_tailed_coverage(self, posterior_samples: np.ndarray,
                                     true_value: float, level: float) -> float:
        samples = self._validate_interval(posterior_samples, level)
        if not np.isfinite(true_value):
            raise ValueError("true_value must be finite")
        alpha = (1 - level) / 2
        lower, upper = np.quantile(samples, [alpha, 1 - alpha])
        return float(lower <= true_value <= upper)

    def plot_coverage_diagram(self, observed_coverages: dict):
        import matplotlib.pyplot as plt

        nominal = sorted(observed_coverages)
        observed = [np.mean(observed_coverages[level]) for level in nominal]
        fig, ax = plt.subplots()
        ax.plot(nominal, observed, "o-")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="Nominal coverage", ylabel="Observed coverage", xlim=(0, 1), ylim=(0, 1))
        return fig
