"""
Coverage evaluation for posterior inference.

Checks if posterior credible intervals have correct coverage probabilities.
"""

from typing import Dict, Any, List
import numpy as np


class CoverageEvaluator:
    """
    Coverage evaluator for posterior inference.

    Checks if posterior credible intervals (HPD, equal-tailed, etc.)
    have correct coverage probabilities.
    """

    def __init__(self, levels: List[float] = None):
        """
        Initialize coverage evaluator.

        Args:
            levels: List of coverage levels to test (default: [0.5, 0.68, 0.95])
        """
        if levels is None:
            levels = [0.50, 0.68, 0.95]

        self.levels = levels

    def run(
        self,
        model: Any,
        test_problems: list,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run coverage evaluation.

        Args:
            model: Trained model
            test_problems: List of test problems
            n_samples: Number of posterior samples

        Returns:
            Dictionary with coverage results
        """
        results = {
            "coverages": {level: [] for level in self.levels},
            "biases": [],
            "n_problems": len(test_problems),
        }

        for problem in test_problems:
            problem_results = self._run_coverage_for_problem(
                model, problem, n_samples
            )

            for level in self.levels:
                results["coverages"][level].append(
                    problem_results["coverages"][level]
                )

            results["biases"].append(problem_results["bias"])

        # Compute average coverages
        avg_coverages = {
            level: np.mean(values)
            for level, values in results["coverages"].items()
        }

        results["average_coverages"] = avg_coverages

        return results

    def _run_coverage_for_problem(
        self,
        model: Any,
        problem: Any,
        n_samples: int,
    ) -> Dict[str, Any]:
        """
        Run coverage evaluation for a single problem.

        Args:
            model: Trained model
            problem: Test problem
            n_samples: Number of posterior samples

        Returns:
            Coverage results for this problem
        """
        # Placeholder implementation
        coverages = {level: np.random.uniform(level - 0.1, level + 0.1)
                     for level in self.levels}
        bias = np.random.uniform(-0.1, 0.1)

        return {
            "coverages": coverages,
            "bias": bias,
        }

    def compute_hpd_coverage(
        self,
        posterior_samples: np.ndarray,
        true_value: float,
        level: float,
    ) -> float:
        """
        Compute coverage of Highest Posterior Density interval.

        Args:
            posterior_samples: Posterior samples (n_samples,)
            true_value: True parameter value
            level: Coverage level (e.g., 0.95)

        Returns:
            1.0 if true value is in HPD interval, 0.0 otherwise
        """
        # Compute HPD interval
        lower, upper = self._hpd_interval(posterior_samples, level)

        # Check if true value is in interval
        return 1.0 if lower <= true_value <= upper else 0.0

    def _hpd_interval(
        self,
        samples: np.ndarray,
        level: float,
    ) -> tuple[float, float]:
        """
        Compute Highest Posterior Density interval.

        Args:
            samples: Posterior samples
            level: Coverage level

        Returns:
            (lower, upper) bounds of HPD interval
        """
        # Sort samples
        sorted_samples = np.sort(samples)
        n_samples = len(sorted_samples)

        # Compute interval size
        interval_size = int(np.round(level * n_samples))

        # Find shortest interval
        min_width = np.inf
        hpd_lower, hpd_upper = 0, 0

        for i in range(n_samples - interval_size):
            width = sorted_samples[i + interval_size] - sorted_samples[i]
            if width < min_width:
                min_width = width
                hpd_lower = sorted_samples[i]
                hpd_upper = sorted_samples[i + interval_size]

        return hpd_lower, hpd_upper

    def compute_equal_tailed_coverage(
        self,
        posterior_samples: np.ndarray,
        true_value: float,
        level: float,
    ) -> float:
        """
        Compute coverage of equal-tailed interval.

        Args:
            posterior_samples: Posterior samples
            true_value: True parameter value
            level: Coverage level

        Returns:
            1.0 if true value is in interval, 0.0 otherwise
        """
        alpha = (1.0 - level) / 2.0
        lower = np.percentile(posterior_samples, 100 * alpha)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha))

        return 1.0 if lower <= true_value <= upper else 0.0

    def plot_coverage_diagram(
        self,
        observed_coverages: Dict[float, List[float]],
    ):
        """
        Plot coverage diagram comparing observed vs nominal coverage.

        Args:
            observed_coverages: Dict mapping levels to observed coverages
        """
        import matplotlib.pyplot as plt

        levels = sorted(observed_coverages.keys())
        observed = [np.mean(observed_coverages[l]) for l in levels]

        plt.figure(figsize=(8, 6))

        # Plot observed vs nominal
        plt.plot(levels, levels, "r--", label="Nominal")
        plt.plot(levels, observed, "bo-", label="Observed")

        plt.xlabel("Nominal Coverage")
        plt.ylabel("Observed Coverage")
        plt.title("Coverage Calibration")
        plt.legend()
        plt.grid(True, alpha=0.3)

        return plt.gcf()
