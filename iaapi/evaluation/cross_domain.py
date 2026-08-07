# Cross-Domain Validation Module
# Support for neuroscience (Hodgkin-Huxley) and epidemiology (SIR/SEIR) models

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class CrossDomainModel:
    """Base class for cross-domain models."""
    name: str
    domain: str  # "neurodynamics" or "epidemiology"
    model_type: str
    n_params: int
    n_vars: int
    time_range: Tuple[float, float]
    description: str = ""
    base_model: Optional[str] = None
    variant_info: Optional[str] = None
    parameters: Dict[str, Dict] = field(default_factory=dict)

    def get_observation_mask(self, obs_names: List[str]) -> np.ndarray:
        """Get observation mask for specified observables."""
        # Default implementation - override in subclasses
        return np.ones(self.n_vars, dtype=bool)


@dataclass
class NeurodynamicsModel(CrossDomainModel):
    """Neuroscience model (Hodgkin-Huxley, FitzHugh-Nagumo, etc.)."""

    def __post_init__(self):
        if not self.domain:
            self.domain = "neurodynamics"

    def get_voltage_trace_params(self) -> List[str]:
        """Return parameters that affect voltage trace identifiability."""
        # Based on Lueckmann 2017, Boelts 2022 benchmarks
        voltage_params = {
            "hodgkin_huxley": ["g_Na", "g_K", "g_L", "E_Na", "E_K", "E_L"],
            "fitz_hugh_nagumo": ["a", "b", "c", "d"],
            "morris_lecar": ["g_Ca", "g_K", "g_L"],
            "hindmarsh_rose": ["a", "b", "c", "d", "r", "s"],
            "izhikevich": ["a", "b", "c", "d"],
        }
        return voltage_params.get(self.model_type.lower(), [])

    def get_unrecoverable_params(self) -> List[str]:
        """Parameters known to be unrecoverable from voltage traces."""
        # Based on literature benchmarks
        unrecoverable = {
            "hodgkin_huxley": ["C_m"],  # Membrane capacitance often unidentifiable
            "fitz_hugh_nagumo": [],
            "morris_lecar": ["V_Ca", "V_K"],  # Reversal potentials can be unidentifiable
            "hindmarsh_rose": ["s"],
            "izhikevich": [],
        }
        return unrecoverable.get(self.model_type.lower(), [])


@dataclass
class EpidemiologyModel(CrossDomainModel):
    """Epidemiology model (SIR, SEIR, etc.)."""

    def __post_init__(self):
        if not self.domain:
            self.domain = "epidemiology"

    def get_r0_identifiability(self) -> Tuple[bool, str]:
        """
        Check if R0 is identifiable given model structure.

        Returns:
            (is_identifiable, reasoning)
        """
        identifiability = {
            "sir": (True, "R0 = beta/gamma is identifiable from prevalence data"),
            "seir": (True, "R0 = beta/gamma identifiable if E state observed"),
            "seirs": (True, "R0 identifiable with sufficient immunity data"),
            "seiahr": (False, "R0 requires additional asymptomatic data"),
            "covid19": (False, "R0 highly uncertain due to asymptomatic cases"),
        }
        return identifiability.get(self.model_type.lower(), (False, "Unknown model type"))

    def get_incubation_period_identifiability(self) -> Tuple[bool, str]:
        """
        Check if incubation period is identifiable.

        Returns:
            (is_identifiable, reasoning)
        """
        identifiability = {
            "sir": (False, "No exposed (E) compartment"),
            "seir": (True, "sigma directly gives incubation period"),
            "seirs": (True, "sigma gives incubation period"),
            "seiahr": (True, "incubation trackable in A compartment"),
            "covid19": (True, "can estimate from reported case delays"),
        }
        return identifiability.get(self.model_type.lower(), (False, "Unknown model type"))


class CrossDomainLoader:
    """Loader for cross-domain validation models."""

    def __init__(self, data_path: str):
        """
        Initialize cross-domain loader.

        Args:
            data_path: Path to cross_domain_models directory
        """
        self.data_path = Path(data_path)
        self.models_summary_path = self.data_path / "models_summary_expanded.json"
        self.neurodynamics_path = self.data_path / "neurodynamics"
        self.epidemiology_path = self.data_path / "epidemiology"

        self._models_cache: Optional[List[CrossDomainModel]] = None

    def load_model_summary(self) -> List[Dict]:
        """Load models_summary_expanded.json."""
        if not self.models_summary_path.exists():
            raise FileNotFoundError(
                f"Models summary not found at {self.models_summary_path}"
            )

        with open(self.models_summary_path, 'r') as f:
            return json.load(f)

    def _create_model(self, model_data: Dict) -> CrossDomainModel:
        """Create CrossDomainModel from dictionary."""
        domain = model_data.get('domain', '')

        if domain == 'neurodynamics':
            return NeurodynamicsModel(
                name=model_data['name'],
                domain=domain,  # Explicitly pass domain
                model_type=model_data.get('model_type', ''),
                n_params=model_data.get('n_params', 0),
                n_vars=model_data.get('n_vars', 0),
                time_range=tuple(model_data.get('time_range', [0, 100])),
                description=model_data.get('description', ''),
                base_model=model_data.get('base_model'),
                variant_info=model_data.get('variant_info'),
            )
        elif domain == 'epidemiology':
            return EpidemiologyModel(
                name=model_data['name'],
                domain=domain,  # Explicitly pass domain
                model_type=model_data.get('model_type', ''),
                n_params=model_data.get('n_params', 0),
                n_vars=model_data.get('n_vars', 0),
                time_range=tuple(model_data.get('time_range', [0, 100])),
                description=model_data.get('description', ''),
                base_model=model_data.get('base_model'),
                variant_info=model_data.get('variant_info'),
            )
        else:
            return CrossDomainModel(
                name=model_data['name'],
                domain=domain,
                model_type=model_data.get('model_type', ''),
                n_params=model_data.get('n_params', 0),
                n_vars=model_data.get('n_vars', 0),
                time_range=tuple(model_data.get('time_range', [0, 100])),
                description=model_data.get('description', ''),
                base_model=model_data.get('base_model'),
                variant_info=model_data.get('variant_info'),
            )

    def get_models(
        self,
        domain: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> List[CrossDomainModel]:
        """
        Get all models, optionally filtered by domain and type.

        Args:
            domain: Filter by domain ("neurodynamics" or "epidemiology")
            model_type: Filter by model_type

        Returns:
            List of CrossDomainModel objects
        """
        if self._models_cache is None:
            self._models_cache = []
            summary = self.load_model_summary()
            for model_data in summary:
                model = self._create_model(model_data)
                self._models_cache.append(model)

        models = self._models_cache

        if domain:
            models = [m for m in models if m.domain == domain]

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        return models

    def get_neurodynamics_models(self) -> List[NeurodynamicsModel]:
        """Get all neurodynamics models."""
        models = self.get_models(domain='neurodynamics')
        return [m for m in models if isinstance(m, NeurodynamicsModel)]

    def get_epidemiology_models(self) -> List[EpidemiologyModel]:
        """Get all epidemiology models."""
        models = self.get_models(domain='epidemiology')
        return [m for m in models if isinstance(m, EpidemiologyModel)]

    def get_model_by_name(self, name: str) -> Optional[CrossDomainModel]:
        """Get model by name."""
        models = self.get_models()
        for model in models:
            if model.name == name:
                return model
        return None

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded models."""
        models = self.get_models()
        neuro_models = self.get_neurodynamics_models()
        epi_models = self.get_epidemiology_models()

        return {
            'total': len(models),
            'neurodynamics': len(neuro_models),
            'epidemiology': len(epi_models),
            'unique_neuro_types': len(set(m.model_type for m in neuro_models)),
            'unique_epi_types': len(set(m.model_type for m in epi_models)),
        }


class CrossDomainValidator:
    """Validator for cross-domain models using IA results."""

    def __init__(
        self,
        loader: CrossDomainLoader,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize cross-domain validator.

        Args:
            loader: CrossDomainLoader instance
            checkpoint_path: Path to pre-trained IA checkpoint
        """
        self.loader = loader
        self.checkpoint_path = checkpoint_path
        self.results: Dict[str, Dict] = {}

    def validate_neurodynamics_model(
        self,
        model: NeurodynamicsModel,
        n_samples: int = 1000,
    ) -> Dict:
        """
        Validate a single neurodynamics model.

        Args:
            model: NeurodynamicsModel instance
            n_samples: Number of inference samples

        Returns:
            Validation results dictionary
        """
        results = {
            'model_name': model.name,
            'model_type': model.model_type,
            'domain': model.domain,
            'n_params': model.n_params,
            'n_vars': model.n_vars,
            'voltage_params': model.get_voltage_trace_params(),
            'unrecoverable_params': model.get_unrecoverable_params(),
            'identifiability': {},
            'isp_analysis': {},
            'baselines': {},
        }

        # Get expected identifiability from literature
        voltage_params = model.get_voltage_trace_params()
        unrecoverable = model.get_unrecoverable_params()

        for param in unrecoverable:
            results['identifiability'][param] = {
                'expected': False,
                'reasoning': 'Known unrecoverable from voltage traces',
                'literature_reference': 'Lueckmann 2017, Boelts 2022',
            }

        for param in voltage_params:
            if param not in unrecoverable:
                results['identifiability'][param] = {
                    'expected': True,
                    'reasoning': 'Expected to be recoverable from voltage traces',
                    'literature_reference': 'Lueckmann 2017, Boelts 2022',
                }

        # Placeholder for ISP analysis (to be filled with actual inference)
        results['isp_analysis'] = {
            'n_samples': n_samples,
            'sloppy_subspace_dim': None,
            'identifiable_params': [],
            'unidentifiable_params': [],
            'warnings': [],
        }

        return results

    def validate_epidemiology_model(
        self,
        model: EpidemiologyModel,
        n_samples: int = 1000,
    ) -> Dict:
        """
        Validate a single epidemiology model.

        Args:
            model: EpidemiologyModel instance
            n_samples: Number of inference samples

        Returns:
            Validation results dictionary
        """
        results = {
            'model_name': model.name,
            'model_type': model.model_type,
            'domain': model.domain,
            'n_params': model.n_params,
            'n_vars': model.n_vars,
            'identifiability': {},
            'isp_analysis': {},
            'baselines': {},
        }

        # Check R0 identifiability
        r0_id, r0_reason = model.get_r0_identifiability()
        results['identifiability']['R0'] = {
            'expected': r0_id,
            'reasoning': r0_reason,
            'literature_reference': 'Pinotti 2025',
        }

        # Check incubation period identifiability
        inc_id, inc_reason = model.get_incubation_period_identifiability()
        results['identifiability']['incubation_period'] = {
            'expected': inc_id,
            'reasoning': inc_reason,
            'literature_reference': 'Pinotti 2025',
        }

        # Placeholder for ISP analysis
        results['isp_analysis'] = {
            'n_samples': n_samples,
            'sloppy_subspace_dim': None,
            'identifiable_params': [],
            'unidentifiable_params': [],
            'warnings': [],
        }

        return results

    def validate_all_models(
        self,
        domain: Optional[str] = None,
        n_samples: int = 1000,
    ) -> Dict[str, Dict]:
        """
        Validate all models (or filtered by domain).

        Args:
            domain: Filter by domain
            n_samples: Number of inference samples per model

        Returns:
            Dictionary of validation results
        """
        models = self.loader.get_models(domain=domain)

        for model in models:
            if model.domain == 'neurodynamics':
                result = self.validate_neurodynamics_model(
                    model, n_samples=n_samples
                )
            elif model.domain == 'epidemiology':
                result = self.validate_epidemiology_model(
                    model, n_samples=n_samples
                )
            else:
                continue

            self.results[model.name] = result

        return self.results

    def generate_report(self) -> str:
        """Generate markdown report of validation results."""
        report = "# Cross-Domain Validation Report\n\n"
        report += f"Generated: {np.datetime64('now')}\n\n"

        stats = self.loader.get_statistics()
        report += "## Statistics\n\n"
        report += f"- Total models: {stats['total']}\n"
        report += f"- Neurodynamics: {stats['neurodynamics']}\n"
        report += f"- Epidemiology: {stats['epidemiology']}\n"
        report += f"- Unique neuro types: {stats['unique_neuro_types']}\n"
        report += f"- Unique epi types: {stats['unique_epi_types']}\n\n"

        # Neurodynamics section
        report += "## Neurodynamics Models\n\n"
        neuro_models = self.loader.get_neurodynamics_models()
        for model in neuro_models[:5]:  # Show first 5
            if model.name in self.results:
                result = self.results[model.name]
                report += f"### {model.name}\n\n"
                report += f"- Type: {model.model_type}\n"
                report += f"- Parameters: {model.n_params}\n"
                report += f"- Voltage trace params: {', '.join(result['voltage_params'])}\n"
                report += f"- Unrecoverable params: {', '.join(result['unrecoverable_params'])}\n\n"

        # Epidemiology section
        report += "## Epidemiology Models\n\n"
        epi_models = self.loader.get_epidemiology_models()
        for model in epi_models[:5]:  # Show first 5
            if model.name in self.results:
                result = self.results[model.name]
                report += f"### {model.name}\n\n"
                report += f"- Type: {model.model_type}\n"
                report += f"- Parameters: {model.n_params}\n"
                r0_id = result['identifiability']['R0']['expected']
                inc_id = result['identifiability']['incubation_period']['expected']
                report += f"- R0 identifiable: {r0_id}\n"
                report += f"- Incubation period identifiable: {inc_id}\n\n"

        return report


def create_cross_domain_loader(data_path: str = "../data/cross_domain_models") -> CrossDomainLoader:
    """Factory function to create CrossDomainLoader."""
    return CrossDomainLoader(data_path)


def create_cross_domain_validator(
    loader: Optional[CrossDomainLoader] = None,
    data_path: str = "../data/cross_domain_models",
    checkpoint_path: Optional[str] = None,
) -> CrossDomainValidator:
    """Factory function to create CrossDomainValidator."""
    if loader is None:
        loader = create_cross_domain_loader(data_path)
    return CrossDomainValidator(loader, checkpoint_path)
