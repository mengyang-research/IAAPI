"""IAAPI public package.

The top-level package uses lazy imports so array-only diagnostics can be used
without importing every optional simulation or training dependency.
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "InferenceModel": ("iaapi.api", "InferenceModel"),
    "InferenceResult": ("iaapi.api", "InferenceResult"),
    "infer": ("iaapi.api", "infer"),
    "IAAPIModel": ("iaapi.models.full_model", "IAAPIModel"),
    "ISPHead": ("iaapi.models.isp_head", "ISPHead"),
    "ISPOutput": ("iaapi.models.isp_head", "ISPOutput"),
    "MASEEncoder": ("iaapi.models.mase_encoder", "MASEEncoder"),
    "ObservationEncoder": ("iaapi.models.obs_encoder", "ObservationEncoder"),
    "SBMLParser": ("iaapi.data.sbml_parser", "SBMLParser"),
    "PEtabLoader": ("iaapi.data.petab_loader", "PEtabLoader"),
    "PEtabProblem": ("iaapi.data.petab_loader", "PEtabProblem"),
    "AMICISimulator": ("iaapi.data.forward_sim", "AMICISimulator"),
    "FIMComputer": ("iaapi.data.fim_compute", "FIMComputer"),
    "TrainingDataGenerator": ("iaapi.data.data_generator", "TrainingDataGenerator"),
    "BioPriorLoader": ("iaapi.data.bio_priors", "BioPriorLoader"),
    "ISPTrainer": ("iaapi.training.trainer", "ISPTrainer"),
    "ISPLoss": ("iaapi.training.losses", "ISPLoss"),
    "OEDModule": ("iaapi.oed.oed_module", "OEDModule"),
    "SequentialRefiner": ("iaapi.refinement.sequential", "SequentialRefiner"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load public objects on first access."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'iaapi' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
