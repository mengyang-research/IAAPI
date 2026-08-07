"""
Biological Priors Loader: Load empirical distributions for biological parameters.

This module provides access to empirical distributions for common biological
parameters (Km, kcat, Kd, etc.) based on data from BRENDA, SABIO-RK,
BioNumbers, and other sources.
"""

from typing import Dict, Any, Optional
import numpy as np
import json
from pathlib import Path


class BioPriorLoader:
    """
    Load and sample from biological parameter priors.

    Supports loading pre-processed empirical distributions for various
    parameter types (Km, kcat, Kd, etc.).
    """

    PARAM_TYPES = [
        "Km",
        "kcat",
        "Kd",
        "k_deg",
        "k_trans",
        "k_transl",
        "hill_coeff",
        "protein_concentration",
        "mRNA_concentration",
        "k_phos",
    ]

    def __init__(self, priors_path: str):
        """
        Initialize bio prior loader.

        Args:
            priors_path: Path to directory containing prior JSON files
        """
        self.priors_path = Path(priors_path)
        self.priors = {}

        # Load all available priors
        self._load_priors()

    def _load_priors(self):
        """Load all available prior distributions."""
        for param_type in self.PARAM_TYPES:
            prior_file = self.priors_path / f"{param_type.lower()}_prior.json"
            if prior_file.exists():
                with open(prior_file, "r") as f:
                    self.priors[param_type] = json.load(f)

    def get_prior(self, param_type: str) -> Optional[Dict[str, Any]]:
        """
        Get prior distribution for a parameter type.

        Args:
            param_type: Parameter type (e.g., "Km", "kcat")

        Returns:
            Prior distribution dictionary or None if not available
        """
        return self.priors.get(param_type)

    def sample(self, petab_problem: Any, n_samples: int = 1) -> np.ndarray:
        """
        Sample parameters for a PEtab problem from biological priors.

        Args:
            petab_problem: PEtab problem object
            n_samples: Number of samples to generate

        Returns:
            Sampled parameter vectors (n_samples, n_params)
        """
        # This is a simplified version
        # In a full implementation, would match parameter names to types
        # and sample from appropriate distributions

        n_params = len(petab_problem.parameters)
        samples = np.random.randn(n_samples, n_params)

        return samples

    def sample_from_distribution(
        self,
        param_type: str,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Sample from a specific parameter type distribution.

        Args:
            param_type: Parameter type
            n_samples: Number of samples

        Returns:
            Sampled values
        """
        prior = self.get_prior(param_type)
        if prior is None:
            raise ValueError(f"Prior for {param_type} not available")

        dist_type = prior.get("distribution")

        if dist_type == "log_normal":
            log_mean = prior.get("log10_mean", 0.0)
            log_std = prior.get("log10_std", 1.0)

            # Sample in log10 space and convert
            log_samples = np.random.normal(log_mean, log_std, n_samples)
            samples = 10**log_samples

        elif dist_type == "normal":
            mean = prior.get("mean", 0.0)
            std = prior.get("std", 1.0)
            samples = np.random.normal(mean, std, n_samples)

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return samples

    def list_available_priors(self) -> list[str]:
        """List available parameter type priors."""
        return list(self.priors.keys())


# Default priors for testing (would be loaded from data in production)
DEFAULT_PRIORS = {
    "Km": {
        "distribution": "log_normal",
        "log10_mean": -4.2,
        "log10_std": 1.8,
        "unit": "M",
        "n_samples": 12500,
    },
    "kcat": {
        "distribution": "log_normal",
        "log10_mean": 0.0,
        "log10_std": 1.5,
        "unit": "s^-1",
        "n_samples": 8000,
    },
    "Kd": {
        "distribution": "log_normal",
        "log10_mean": -6.0,
        "log10_std": 2.0,
        "unit": "M",
        "n_samples": 5000,
    },
}
