"""
Optimal Experimental Design (OED) module.

Suggests experiments that maximize information gain about
poorly constrained parameter directions.
"""

from typing import Dict, Any, List, Callable, Optional
import numpy as np


class OEDModule:
    """
    Optimal Experimental Design module.

    Recommends experiments that maximize information gain in
    the sloppy (poorly constrained) subspace.
    """

    def __init__(self, information_gain_fn: Optional[Callable] = None, *,
                 observable_ids: Optional[List[Any]] = None):
        """Inject a validated, higher-is-better utility; no random scoring fallback."""
        self.information_gain_fn = information_gain_fn
        self.observable_ids = None if observable_ids is None else list(observable_ids)

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
        if not candidate_experiments:
            raise ValueError("Provide a nonempty, protocol-defined candidate set")
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
        if self.information_gain_fn is None:
            raise NotImplementedError(
                "Supply a validated information_gain_fn(basis, theta, experiment, simulator); "
                "random OED scores are disabled. For expected posterior variance, use its "
                "negative as a higher-is-better utility."
            )
        gain = float(self.information_gain_fn(U_perp, current_theta, experiment, simulator))
        if not np.isfinite(gain):
            raise ValueError("Design utility must be finite")
        return gain

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
        if strategy != "random":
            raise NotImplementedError("Provide explicit candidates for this design strategy")
        if n_candidates < 1 or not self.observable_ids:
            raise ValueError("Random candidate generation requires n_candidates >= 1 and explicit observable_ids")
        candidates = []

        if strategy == "random":
            # Random time points and observable combinations
            for i in range(n_candidates):
                candidates.append({
                    "time_points": np.sort(np.random.uniform(0, 100, 10)),
                    "observables": np.random.choice(
                        self.observable_ids,
                        size=min(5, len(self.observable_ids)),
                        replace=False,
                    ),
                    "conditions": {},
                })
        else:
            # Placeholder for other strategies
            pass

        return candidates
