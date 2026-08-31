"""
Simulation-Based Calibration (SBC) evaluation.

SBC is a diagnostic for checking if a posterior estimator is well-calibrated.

Protocol (aligned with the paper's Methods):
  1. For each test problem, draw ``theta_true`` from the declared prior;
  2. simulate observations ``y = simulator(problem, theta_true)``;
  3. run the posterior estimator to obtain ``K`` posterior samples;
  4. compute the per-parameter SBC rank
        rank_i = sum_j 1[theta_sample[j, i] < theta_true[i]],   rank_i in {0..K};
  5. save the raw per-model, per-sample, per-parameter ranks (not only the
     final KS), and summarize at three levels:
       - per-parameter SBC KS;
       - per-model pooled KS (pooling the randomized ranks of all parameters
         of one model);
       - model-level median of the per-model pooled KS (the number quoted in
         the main text).

Because SBC ranks are discrete and parameters of the same model are
correlated, the KS distance is computed on *randomized* ranks
``u = (rank + Uniform(0,1))/(K+1)``; the ordinary continuous-KS p-value of a
cross-parameter pooled histogram is reported for transparency only and is not
treated as a significance test. Confidence intervals are bootstrapped at the
model level.

The evaluator is dependency-injected so it works with any simulator/prior
implementation (PyTorch models or pure NumPy stand-ins):

  * ``simulator(problem, theta) -> observations``
  * ``prior_sampler(problem, n) -> (n, n_params)``
  * ``model.infer(graph_data, observations, n_samples)`` returning a dict with
    ``posterior_samples`` of shape ``(n_samples, n_params)``, or
  * ``model.sample_posterior(problem, n_samples) -> (n_samples, n_params)``

All random draws are seeded from ``seed`` plus the problem index so a fixed
seed reproduces the exact ranks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class SBCEvaluator:
    """
    Simulation-Based Calibration evaluator.

    Checks if posterior samples are properly calibrated by computing
    rank statistics and testing for uniformity.
    """

    def __init__(self, n_samples: int = 500, seed: int = 0):
        """
        Initialize SBC evaluator.

        Args:
            n_samples: Number of posterior samples per test problem (K).
            seed: Base random seed; the per-problem seed is ``seed + index``.
        """
        self.n_samples = n_samples
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        model: Any,
        test_problems: list,
        n_samples: Optional[int] = None,
        simulator: Optional[Callable] = None,
        prior_sampler: Optional[Callable] = None,
        n_repetitions: int = 1,
        n_bootstrap: int = 10000,
    ) -> Dict[str, Any]:
        """
        Run SBC evaluation.

        Args:
            model: Trained model. Must provide either
                ``model.infer(graph_data, observations, n_samples)`` returning
                ``{"posterior_samples": (n_samples, n_params)}`` or
                ``model.sample_posterior(problem, n_samples)``.
            test_problems: List of problem dicts. Each dict must provide
                ``name`` and ``n_params``; ``graph_data`` and ``bounds`` are
                required when using ``model.infer`` and the prior sampler.
            n_samples: Override default n_samples (K).
            simulator: Callable ``simulator(problem, theta) -> observations``.
            prior_sampler: Callable ``prior_sampler(problem, n) -> (n, n_params)``.
            n_repetitions: Number of prior draws per problem.
            n_bootstrap: Bootstrap resamples for the model-level CI.

        Returns:
            Nested dict with raw ranks and three aggregation levels.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if simulator is None or prior_sampler is None:
            raise ValueError(
                "SBC requires a simulator and a prior_sampler; the public "
                "placeholder implementation that fabricates random ranks has "
                "been removed."
            )

        per_problem: Dict[str, Any] = {}
        all_ranks: List[np.ndarray] = []  # per-model pooled randomized ranks

        for pi, problem in enumerate(test_problems):
            name = str(problem.get("name", f"problem_{pi}"))
            n_params = int(problem["n_params"])
            rng = np.random.default_rng(self.seed + pi)

            # (n_reps, n_params) integer ranks, each in {0..n_samples}
            ranks = np.zeros((n_repetitions, n_params), dtype=np.int64)
            for rep in range(n_repetitions):
                # Per-repetition seed so prior draws, simulations and posterior
                # samples differ across repetitions while a fixed evaluator
                # seed remains fully reproducible.
                problem["sbc_seed"] = self.seed + pi * 10000 + rep
                theta_true = prior_sampler(problem, 1)[0]          # (n_params,)
                theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
                obs = simulator(problem, theta_true)               # observations
                post = self._draw_posterior(model, problem, obs, n_samples, rng)
                post = np.asarray(post, dtype=float)
                if post.ndim != 2 or post.shape[0] != n_samples:
                    raise ValueError(
                        f"posterior_samples must have shape ({n_samples}, {n_params}), "
                        f"got {post.shape}"
                    )
                if post.shape[1] != n_params:
                    raise ValueError(
                        f"posterior_samples width {post.shape[1]} != n_params {n_params}"
                    )
                ranks[rep] = (post < theta_true).sum(axis=0)       # (n_params,)

            # Per-parameter KS on randomized ranks.
            ks_per_param, p_per_param = self._ks_per_parameter(ranks, n_samples, rng)
            # Per-model pooled KS (all parameters of this model combined).
            pooled = self.randomized_ranks(ranks.reshape(-1), n_samples, rng)
            ks_model, p_model = self._ks_on_randomized(pooled)

            per_problem[name] = {
                "n_params": n_params,
                "n_repetitions": n_repetitions,
                "ranks_per_param": ranks.tolist(),        # raw integer ranks
                "ks_per_param": ks_per_param.tolist(),
                "ks_per_param_pvalue": p_per_param.tolist(),
                "ks_model_pooled": float(ks_model),
                "ks_model_pooled_pvalue": float(p_model),
            }
            all_ranks.append(pooled)

        # Pooled KS across all models/parameters (reported for transparency).
        pooled_all = np.concatenate(all_ranks) if all_ranks else np.zeros(0)
        ks_pooled, p_pooled = self._ks_on_randomized(pooled_all)

        # Model-level median of per-model pooled KS + bootstrap CI.
        model_ks = np.array([per_problem[n]["ks_model_pooled"] for n in per_problem])
        med = float(np.median(model_ks)) if model_ks.size else float("nan")
        ci = self._bootstrap_median_ci(model_ks, n_bootstrap, self.seed)

        return {
            "per_problem": per_problem,
            "pooled_ks": float(ks_pooled),
            "pooled_ks_pvalue": float(p_pooled),
            "model_level_median_ks": med,
            "model_level_median_ks_ci95": ci,
            "n_problems": len(test_problems),
            "n_samples_per_posterior": n_samples,
            "n_repetitions": n_repetitions,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _draw_posterior(
        self,
        model: Any,
        problem: Dict[str, Any],
        observations: Any,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw K posterior samples from the estimator (dependency-injected)."""
        if hasattr(model, "sample_posterior"):
            return np.asarray(model.sample_posterior(problem, n_samples))
        if hasattr(model, "infer"):
            graph_data = problem.get("graph_data")
            if graph_data is None:
                raise ValueError("problem['graph_data'] required for model.infer")
            out = model.infer(graph_data, observations, n_samples=n_samples)
            return np.asarray(out["posterior_samples"])
        raise ValueError(
            "model must provide sample_posterior(problem, n_samples) or "
            "infer(graph_data, observations, n_samples)"
        )

    @staticmethod
    def randomized_ranks(
        ranks: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Randomized ranks u = (rank + U(0,1)) / (K+1) in (0,1)."""
        ranks = np.asarray(ranks, dtype=float)
        u = (ranks + rng.uniform(0.0, 1.0, size=ranks.shape)) / (n_samples + 1)
        return u

    @staticmethod
    def _ks_on_randomized(u: np.ndarray) -> tuple[float, float]:
        """Two-sided KS distance of randomized ranks against Uniform(0,1)."""
        from scipy import stats

        if u.size == 0:
            return float("nan"), float("nan")
        ks, p = stats.kstest(u, stats.uniform().cdf)
        return float(ks), float(p)

    def _ks_per_parameter(
        self,
        ranks: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-parameter KS on randomized ranks. ranks: (n_reps, n_params)."""
        ks = np.empty(ranks.shape[1])
        p = np.empty(ranks.shape[1])
        for i in range(ranks.shape[1]):
            u = self.randomized_ranks(ranks[:, i], n_samples, rng)
            ks[i], p[i] = self._ks_on_randomized(u)
        return ks, p

    @staticmethod
    def _bootstrap_median_ci(
        values: np.ndarray, n_bootstrap: int, seed: int
    ) -> List[float]:
        """Model-level bootstrap 95% CI of the median (resample models)."""
        if values.size == 0:
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(seed + 999)
        n = values.size
        medians = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            medians[b] = np.median(values[idx])
        return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]

    # ------------------------------------------------------------------ #
    # Backwards-compatible helpers
    # ------------------------------------------------------------------ #
    def compute_ks_distance(
        self,
        rank_statistics: np.ndarray,
        n_samples: int,
    ) -> tuple[float, float]:
        """
        Compute KS distance and p-value for rank statistics.

        Uses randomized ranks to account for discreteness of SBC ranks.
        """
        rng = np.random.default_rng(self.seed)
        u = self.randomized_ranks(rank_statistics, n_samples, rng)
        return self._ks_on_randomized(u)

    def plot_rank_histogram(
        self,
        rank_statistics: np.ndarray,
        n_samples: int,
        bins: int = 30,
    ):
        """
        Plot rank histogram for SBC visualization.

        Args:
            rank_statistics: Array of rank values
            n_samples: Number of posterior samples
            bins: Number of histogram bins
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(rank_statistics, bins=bins, density=True, alpha=0.7,
                label="Observed")
        ax.axhline(1.0 / n_samples, color="r", linestyle="--", label="Expected")

        # Binomial 95% confidence band for the null histogram.
        n = np.asarray(rank_statistics).size
        if n > 0:
            p_hat = 1.0 / bins
            half = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n)
            ax.axhspan(p_hat - half, p_hat + half, color="r", alpha=0.1,
                       label="95% null band")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")
        ax.set_title("SBC Rank Histogram")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
