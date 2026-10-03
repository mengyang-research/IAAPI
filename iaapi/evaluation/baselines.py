"""
Baseline method runners for comparison.

Implements baseline parameter inference methods for comparison.
"""

from typing import Dict, Any, List


class BaselineRunner:
    """
    Runner for baseline inference methods.

    Supports:
        - pyPESTO optimization (multi-start)
        - pyPESTO MCMC (emcee)
        - pyPESTO profile likelihood
        - SBI NPE-C
        - BayesFlow
    """

    def __init__(self):
        """Initialize baseline runner."""
        pass

    def run_all(
        self,
        model: Any,
        test_problems: list,
    ) -> Dict[str, Any]:
        """
        Run all baseline methods.

        Args:
            model: IA-API model (for comparison)
            test_problems: List of test problems

        Returns:
            Dictionary with results from all baselines
        """
        results = {}

        results["pypesto_optimization"] = self.run_pypesto_optimization_batch(
            test_problems
        )
        results["pypesto_mcmc"] = self.run_pypesto_mcmc_batch(
            test_problems
        )
        results["pypesto_profile"] = self.run_pypesto_profile_batch(
            test_problems
        )
        results["sbi_npe"] = self.run_sbi_npe_batch(
            test_problems
        )
        results["bayesflow"] = self.run_bayesflow_batch(
            test_problems
        )

        return results

    def run_pypesto_optimization(
        self,
        petab_yaml: str,
        n_starts: int = 100,
    ) -> Dict[str, Any]:
        """
        Run pyPESTO multi-start optimization.

        Args:
            petab_yaml: Path to PEtab YAML file
            n_starts: Number of optimization starts

        Returns:
            Optimization results
        """
        raise NotImplementedError(
            "run_pypesto_optimization has no production adapter in this public preview. "
            "Use a validated backend and record its configuration; random outputs are disabled."
        )

    def run_pypesto_optimization_batch(
        self,
        problems: list,
    ) -> List[Dict[str, Any]]:
        """Run pyPESTO optimization on multiple problems."""
        results = []
        for problem in problems:
            results.append(self.run_pypesto_optimization(problem))
        return results

    def run_pypesto_mcmc(
        self,
        petab_yaml: str,
        n_chains: int = 32,
        n_steps: int = 100000,
    ) -> Dict[str, Any]:
        """
        Run pyPESTO MCMC with emcee.

        Args:
            petab_yaml: Path to PEtab YAML file
            n_chains: Number of MCMC chains
            n_steps: Number of MCMC steps

        Returns:
            MCMC results
        """
        raise NotImplementedError(
            "run_pypesto_mcmc has no production adapter in this public preview. "
            "Use a validated backend and record its configuration; random outputs are disabled."
        )

    def run_pypesto_mcmc_batch(
        self,
        problems: list,
    ) -> List[Dict[str, Any]]:
        """Run pyPESTO MCMC on multiple problems."""
        results = []
        for problem in problems:
            results.append(self.run_pypesto_mcmc(problem))
        return results

    def run_pypesto_profile(
        self,
        petab_yaml: str,
    ) -> Dict[str, Any]:
        """
        Run pyPESTO profile likelihood analysis.

        Args:
            petab_yaml: Path to PEtab YAML file

        Returns:
            Profile likelihood results
        """
        raise NotImplementedError(
            "run_pypesto_profile has no production adapter in this public preview. "
            "Use a validated backend and record its configuration; random outputs are disabled."
        )

    def run_pypesto_profile_batch(
        self,
        problems: list,
    ) -> List[Dict[str, Any]]:
        """Run pyPESTO profile likelihood on multiple problems."""
        results = []
        for problem in problems:
            results.append(self.run_pypesto_profile(problem))
        return results

    def run_sbi_npe(
        self,
        petab_yaml: str,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run SBI NPE-C (neural posterior estimation).

        Args:
            petab_yaml: Path to PEtab YAML file
            n_samples: Number of posterior samples

        Returns:
            NPE results
        """
        raise NotImplementedError(
            "run_sbi_npe has no production adapter in this public preview. "
            "Use a validated backend and record its configuration; random outputs are disabled."
        )

    def run_sbi_npe_batch(
        self,
        problems: list,
    ) -> List[Dict[str, Any]]:
        """Run SBI NPE on multiple problems."""
        results = []
        for problem in problems:
            results.append(self.run_sbi_npe(problem))
        return results

    def run_bayesflow(
        self,
        petab_yaml: str,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run BayesFlow inference.

        Args:
            petab_yaml: Path to PEtab YAML file
            n_samples: Number of posterior samples

        Returns:
            BayesFlow results
        """
        raise NotImplementedError(
            "run_bayesflow has no production adapter in this public preview. "
            "Use a validated backend and record its configuration; random outputs are disabled."
        )

    def run_bayesflow_batch(
        self,
        problems: list,
    ) -> List[Dict[str, Any]]:
        """Run BayesFlow on multiple problems."""
        results = []
        for problem in problems:
            results.append(self.run_bayesflow(problem))
        return results
