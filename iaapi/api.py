# High-level API for IA-API

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from iaapi.models.full_model import IAAPIModel
from iaapi.data.petab_loader import PEtabLoader


@dataclass
class InferenceResult:
    """Container for inference results with identifiability analysis."""

    # Posterior samples

    posterior_samples: np.ndarray  # Shape: (n_samples, n_params)
    parameter_names: List[str]

    # Identifiability analysis
    identifiability_scores: np.ndarray  # Shape: (n_params,)
    sloppy_subspace_basis: Optional[np.ndarray]  # Shape: (n_params, sloppy_dim)
    sloppy_dimension: int

    # Model info
    model_name: str
    model_description: str

    # Diagnostics
    log_probabilities: Optional[np.ndarray] = None  # Shape: (n_samples,)
    warnings: List[str] = field(default_factory=list)

    def get_identifiable_params(self, threshold: float = 0.5) -> List[str]:
        """Get parameters with identifiability score above threshold."""
        mask = self.identifiability_scores >= threshold
        return [name for name, is_id in zip(self.parameter_names, mask) if is_id]

    def get_sloppy_params(self, threshold: float = 0.5) -> List[str]:
        """Get parameters with identifiability score below threshold."""
        mask = self.identifiability_scores < threshold
        return [name for name, is_sloppy in zip(self.parameter_names, mask) if is_sloppy]

    def get_posterior_stats(self) -> Dict[str, Dict[str, float]]:
        """Get posterior statistics (mean, std, percentiles)."""
        stats = {}
        for i, name in enumerate(self.parameter_names):
            samples = self.posterior_samples[:, i]
            stats[name] = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'median': float(np.median(samples)),
                'q5': float(np.percentile(samples, 5)),
                'q95': float(np.percentile(samples, 95)),
            }
        return stats

    def get_sloppy_subspace_projection(self, samples: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project samples onto sloppy subspace.

        Args:
            samples: Samples to project (default: self.posterior_samples)

        Returns:
            Projected samples in sloppy subspace
        """
        if self.sloppy_subspace_basis is None:
            raise ValueError("No sloppy subspace basis available")

        if samples is None:
            samples = self.posterior_samples

        return samples @ self.sloppy_subspace_basis

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'posterior_samples': self.posterior_samples.tolist(),
            'parameter_names': self.parameter_names,
            'identifiability_scores': self.identifiability_scores.tolist(),
            'sloppy_dimension': self.sloppy_dimension,
            'sloppy_subspace_basis': (
                None
                if self.sloppy_subspace_basis is None
                else self.sloppy_subspace_basis.tolist()
            ),
            'log_probabilities': (
                None
                if self.log_probabilities is None
                else self.log_probabilities.tolist()
            ),
            'identifiable_params': self.get_identifiable_params(),
            'sloppy_params': self.get_sloppy_params(),
            'posterior_stats': self.get_posterior_stats(),
            'model_name': self.model_name,
            'model_description': self.model_description,
            'warnings': self.warnings,
        }

    def save(self, path: Union[str, Path]):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InferenceResult":
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            posterior_samples=np.array(data['posterior_samples']),
            parameter_names=data['parameter_names'],
            identifiability_scores=np.array(data['identifiability_scores']),
            sloppy_subspace_basis=(
                None
                if data.get('sloppy_subspace_basis') is None
                else np.array(data['sloppy_subspace_basis'])
            ),
            sloppy_dimension=data['sloppy_dimension'],
            model_name=data['model_name'],
            model_description=data['model_description'],
            log_probabilities=(
                None
                if data.get('log_probabilities') is None
                else np.array(data['log_probabilities'])
            ),
            warnings=data.get('warnings', []),
        )


class InferenceModel:
    """
    High-level interface for parameter inference with identifiability analysis.

    This is main user-facing API for IA-API. It provides:
    - Easy model loading from local checkpoints
    - Simple inference interface
    - Automatic identifiability analysis
    - Cross-domain validation support

    Example:
        >>> model = InferenceModel.from_checkpoint('checkpoints/isp.pt')
        >>> result = model.infer(petab_problem='problem.yaml', observations=data)
        >>> print(result.get_identifiable_params())
    """

    def __init__(
        self,
        model: IAAPIModel,
        device: str = 'auto',
    ):
        """
        Initialize InferenceModel.

        Args:
            model: IAAPIModel instance
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model = model

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # Cache for loaded models
        self._petab_cache: Dict[str, Any] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = 'auto',
    ) -> "InferenceModel":
        """
        Load model from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pt or .pth)
            device: Device to run on

        Returns:
            InferenceModel instance
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = IAAPIModel.from_pretrained(str(checkpoint_path))
        return cls(model, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = 'iaapi-base',
        device: str = 'auto',
    ) -> "InferenceModel":
        """
        Load a model from a local checkpoint.

        Args:
            model_name: Local checkpoint path
            device: Device to run on

        Returns:
            InferenceModel instance

        Pretrained weights are not bundled with IAAPI 0.1.0.
        """
        if Path(model_name).exists():
            return cls.from_checkpoint(model_name, device=device)
        raise FileNotFoundError(
            f"Checkpoint not found: {model_name}. IAAPI 0.1.0 does not bundle "
            "pretrained weights; pass a local checkpoint created by the training code."
        )

    def infer(
        self,
        petab_problem: Union[str, Path],
        observations: Optional[Dict[str, np.ndarray]] = None,
        n_samples: int = 1000,
        return_diagnosis: bool = True,
        batch_size: int = 64,
    ) -> InferenceResult:
        """
        Run amortized inference on a PEtab problem.

        Args:
            petab_problem: Path to PEtab YAML file
            observations: Optional observation data (if None, loads from PEtab)
            n_samples: Number of posterior samples to draw
            return_diagnosis: Whether to return identifiability diagnosis
            batch_size: Batch size for inference

        Returns:
            InferenceResult with posterior samples and identifiability analysis

        This method requires a compatible trained checkpoint.

        Example:
            >>> model = InferenceModel.from_checkpoint('path/to/checkpoint.pt')
            >>> result = model.infer(petab_problem='problem.yaml',
            >>>                        observations={'time': t, 'values': y, 'mask': mask})
            >>> print(result.get_identifiable_params())
        """
        # Load PEtab problem
        try:
            petab_problem_obj = PEtabLoader().load(str(petab_problem))
        except Exception as e:
            raise ValueError(f"Failed to load PEtab problem: {e}")

        # Get model info
        model_name = Path(petab_problem_obj.yaml_path).stem
        model_description = f"PEtab problem loaded from {petab_problem_obj.yaml_path}"

        # Process observations
        if observations is None:
            # Load from PEtab problem
            # TODO: Implement observation loading from PEtab
            raise NotImplementedError(
                "Observation loading from PEtab not yet implemented. "
                "Please provide observations directly."
            )

        # Convert observations to tensor format
        try:
            time_points = torch.tensor(
                observations.get('time', np.array([0.0])),
                dtype=torch.float32,
            ).to(self.device)

            obs_values = torch.tensor(
                observations.get('values', np.array([[0.0]])),
                dtype=torch.float32,
            ).to(self.device)

            obs_masks = torch.tensor(
                observations.get('mask', np.array([[1.0]])),
                dtype=torch.float32,
            ).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to process observations: {e}")

        if time_points.ndim != 1:
            raise ValueError("observations['time'] must have shape (n_times,)")
        if obs_values.ndim != 2:
            raise ValueError(
                "observations['values'] must have shape (n_observables, n_times)"
            )
        if tuple(obs_masks.shape) != tuple(obs_values.shape):
            raise ValueError("observations['mask'] must have the same shape as values")
        if obs_values.shape[1] != time_points.shape[0]:
            raise ValueError("the final values dimension must match the number of times")

        # Run real inference via the underlying IAAPIModel.
        # graph_data (PyG HeteroData) was built by PEtabLoader from the SBML
        # structure; observations are the user-provided (or PEtab-loaded)
        # measurement data. The ISP head samples the full posterior and the
        # ISP output provides identifiability diagnostics in one forward pass.
        n_params_actual = len(petab_problem_obj.parameters)
        if "parameterId" in petab_problem_obj.parameters.columns:
            parameter_names = petab_problem_obj.parameters["parameterId"].astype(str).tolist()
        else:
            parameter_names = petab_problem_obj.parameters.index.astype(str).tolist()
        graph_data = petab_problem_obj.graph_data
        if hasattr(graph_data, "to"):
            graph_data = graph_data.to(self.device)

        with torch.no_grad():
            self.model.eval()
            # IAAPIModel.infer expects batched obs dict with 'values'/'times'/'masks'.
            infer_obs = {
                "values": obs_values.unsqueeze(0),   # (1, n_obs, n_time)
                "times": time_points.unsqueeze(0),    # (1, n_time)
                "masks": obs_masks.unsqueeze(0),      # (1, n_obs, n_time)
            }
            infer_out = self.model.infer(graph_data, infer_obs, n_samples=n_samples)
            posterior_samples = infer_out["posterior_samples"][0].cpu().numpy()  # (n_samples, n_params)
            identifiability_scores = infer_out["identifiability_scores"][0].cpu().numpy()
            sloppy_basis = infer_out["sloppy_subspace"][0].cpu().numpy()
            k_eff = float(infer_out["effective_dimension"][0].cpu())
            sloppy_dim = max(0, n_params_actual - int(round(k_eff)))
            sloppy_basis = sloppy_basis[:, :max(1, sloppy_dim)] if sloppy_dim > 0 else None

        # Create result
        result = InferenceResult(
            posterior_samples=posterior_samples[:, :n_params_actual],
            parameter_names=parameter_names,
            identifiability_scores=identifiability_scores[:n_params_actual],
            sloppy_subspace_basis=sloppy_basis,
            sloppy_dimension=sloppy_dim,
            model_name=model_name,
            model_description=model_description,
        )

        return result

    def validate_cross_domain(
        self,
        models_config: Union[str, Path, List[Dict]],
        n_samples: int = 1000,
    ) -> Dict[str, InferenceResult]:
        """
        Run cross-domain validation on multiple models.

        Args:
            models_config: Path to models JSON or list of model configs
            n_samples: Number of samples per model

        Returns:
            Dictionary of model_name -> InferenceResult

        Each model configuration must contain `petab_path` and an
        `observations` mapping accepted by :meth:`infer`.
        """
        # Load models config
        if isinstance(models_config, (str, Path)):
            with open(models_config, 'r') as f:
                models = json.load(f)
        else:
            models = models_config

        results = {}
        for model_config in models:
            model_name = model_config.get('name', 'unknown')
            petab_path = model_config.get('petab_path')

            observations = model_config.get('observations')
            if petab_path and observations is not None:
                result = self.infer(
                    petab_problem=petab_path,
                    observations={
                        key: np.asarray(value) for key, value in observations.items()
                    },
                    n_samples=n_samples,
                )
                results[model_name] = result

        return results

    def diagnose_model(self, result: InferenceResult) -> Dict[str, Any]:
        """
        Generate detailed identifiability diagnosis.

        Args:
            result: InferenceResult to analyze

        Returns:
            Dictionary with diagnosis information
        """
        diagnosis = {
            'model_name': result.model_name,
            'n_identifiable': len(result.get_identifiable_params()),
            'n_sloppy': len(result.get_sloppy_params()),
            'sloppy_fraction': result.sloppy_dimension / len(result.parameter_names),
            'identifiable_params': result.get_identifiable_params(),
            'sloppy_params': result.get_sloppy_params(),
            'parameter_scores': dict(
                zip(result.parameter_names, result.identifiability_scores)
            ),
            'recommendations': self._generate_recommendations(result),
        }

        return diagnosis

    def _generate_recommendations(self, result: InferenceResult) -> List[str]:
        """Generate improvement recommendations based on identifiability analysis."""
        recommendations = []

        sloppy_fraction = result.sloppy_dimension / len(result.parameter_names)

        if sloppy_fraction > 0.5:
            recommendations.append(
                "More than 50% of parameters are sloppy. "
                "Consider additional experimental measurements "
                "or different observable combinations."
            )

        if sloppy_fraction > 0.3:
            recommendations.append(
                "Significant sloppy subspace detected. "
                "Use OED module to design informative experiments."
            )

        if len(result.get_sloppy_params()) > 0:
            recommendations.append(
                f"Focus experimental design on improving "
                f"identifiability of {len(result.get_sloppy_params())} sloppy parameters."
            )

        return recommendations


def infer(
    petab_problem: Union[str, Path],
    observations: Dict[str, np.ndarray],
    model_name: str,
    n_samples: int = 1000,
    **kwargs,
) -> InferenceResult:
    """
    Convenience function for single-shot inference.

    Args:
        petab_problem: Path to PEtab YAML file
        observations: Observation data
        model_name: Pre-trained model to use
        n_samples: Number of posterior samples
        **kwargs: Additional arguments for InferenceModel.infer()

    Returns:
        InferenceResult

    Example:
        >>> result = infer('problem.yaml', observations={'time': t, 'values': y})
        >>> print(result.get_identifiable_params())

    Note:
        This is a function-based API alternative to the class-based API.
        It automatically loads the model and performs inference.
    """
    model = InferenceModel.from_pretrained(model_name)
    return model.infer(
        petab_problem=petab_problem,
        observations=observations,
        n_samples=n_samples,
        **kwargs,
    )
