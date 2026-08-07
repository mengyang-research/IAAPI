"""
Optimal Experimental Design (OED) module.

Suggests experiments that maximize information gain about
poorly constrained parameter directions.
"""

from typing import Dict, Any, List
import numpy as np


class OEDModule:
    """
    Optimal Experimental Design module.

    Recommends experiments that maximize information gain in
    the sloppy (poorly constrained) subspace.
    """

    def __init__(self):
        """Initialize OED module."""
        pass

    def rank_experiments(
        self,
        isp_output: Any,
        candidate_experiments: List[Dict[str, Any]],
        current_theta: np.ndarray,
        simulator: Any,
    ) -> Dict[str, Any]:
        """
        Rank candidate experiments by expected information gain.

        Args:
            isp_output: ISP head output containing sloppy subspace
            candidate_experiments: List of candidate experiments
            current_theta: Current parameter estimate
            simulator: AMICI simulator

        Returns:
            Dictionary with ranked experiments and information gains
        """
        # Get sloppy subspace
        U_perp = isp_output.sloppy_subspace  # (n_params, n_perp)

        # Compute information gain for each candidate
        information_gains = []

        for experiment in candidate_experiments:
            gain = self._compute_information_gain(
                U_perp,
                current_theta,
                experiment,
                simulator,
            )
            information_gains.append(gain)

        # Sort experiments by information gain
        sorted_indices = np.argsort(information_gains)[::-1]
        ranked_experiments = [candidate_experiments[i] for i in sorted_indices]
        sorted_gains = [information_gains[i] for i in sorted_indices]

        return {
            "ranked_experiments": ranked_experiments,
            "information_gains": sorted_gains,
        }

    def _compute_information_gain(
        self,
        U_perp: np.ndarray,
        current_theta: np.ndarray,
        experiment: Dict[str, Any],
        simulator: Any,
    ) -> float:
        """
        Compute expected information gain in sloppy subspace.

        Args:
            U_perp: Sloppy subspace basis
            current_theta: Current parameter estimate
            experiment: Experiment configuration
            simulator: AMICI simulator

        Returns:
            Information gain in sloppy subspace
        """
        # Simulate with new experiment
        # Compute FIM for new experiment
        # Project onto sloppy subspace
        # Compute trace as information gain

        # Placeholder implementation
        return np.random.uniform(0, 1)

    def suggest_next_experiment(
        self,
        isp_output: Any,
        current_theta: np.ndarray,
        simulator: Any,
        n_candidates: int = 100,
        strategy: str = "random",
    ) -> Dict[str, Any]:
        """
        Generate and suggest the next experiment.

        Args:
            isp_output: ISP head output
            current_theta: Current parameter estimate
            simulator: AMICI simulator
            n_candidates: Number of candidate experiments to generate
            strategy: Experiment generation strategy

        Returns:
            Suggested experiment with rationale
        """
        # Generate candidate experiments
        candidates = self._generate_candidates(
            current_theta,
            n_candidates,
            strategy,
        )

        # Rank candidates
        results = self.rank_experiments(
            isp_output,
            candidates,
            current_theta,
            simulator,
        )

        # Return best experiment
        return {
            "recommended_experiment": results["ranked_experiments"][0],
            "information_gain": results["information_gains"][0],
            "rationale": "Maximizes information gain in sloppy subspace",
        }

    def _generate_candidates(
        self,
        current_theta: np.ndarray,
        n_candidates: int,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate experiments.

        Args:
            current_theta: Current parameter estimate
            n_candidates: Number of candidates
            strategy: Generation strategy

        Returns:
            List of candidate experiments
        """
        candidates = []

        if strategy == "random":
            # Random time points and observable combinations
            for i in range(n_candidates):
                candidates.append({
                    "time_points": np.sort(np.random.uniform(0, 100, 10)),
                    "observables": np.random.choice(
                        list(range(current_theta.shape[0])),
                        size=5,
                        replace=False,
                    ),
                    "conditions": {},
                })
        else:
            # Placeholder for other strategies
            pass

        return candidates
