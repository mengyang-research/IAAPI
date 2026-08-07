# IAAPI

IAAPI is an open-source Python toolkit for identifiability-aware parameter inference in mechanistic ordinary differential equation (ODE) models. It connects PEtab/SBML model definitions, neural posterior models, Fisher-information diagnostics, posterior validation, and experimental-design utilities in one reusable package.

The project is intended for researchers who use mechanistic ODEs in systems biology, neuroscience, epidemiology, pharmacology, and other quantitative fields. PEtab/SBML is the most complete input path today; the model and diagnostic modules can also be used with custom simulators.

> **Release status:** research preview (`0.1.0`). The model architecture, training code, inference API, diagnostics, and examples are public. Pretrained weights are not bundled with this release, so neural inference requires a compatible checkpoint trained by the user.

## What is included

- PEtab/SBML loading and graph construction.
- Canonical parameter alignment across model tables, FIMs, and neural outputs.
- Mechanism-aware model encoding and variable-dimensional ISP posterior heads.
- Identifiability targets, FIM descriptors, profile-likelihood and MCMC helpers.
- Posterior calibration and validation metrics.
- Experimental-design primitives for observables, perturbations, stimuli, and initial conditions.
- A checkpoint-based high-level inference API.

## Installation

Clone the repository and install the editable package:

```bash
git clone https://github.com/mengyang-research/IAAPI.git
cd IAAPI
python -m pip install -e .
```

Install optional capabilities only when needed:

```bash
# AMICI simulation and training-data generation
python -m pip install -e ".[simulation]"

# Neural training and validation helpers
python -m pip install -e ".[training,validation]"

# Everything
python -m pip install -e ".[all]"
```

Python 3.10 or newer is required. AMICI-based workflows may also require a compiler toolchain appropriate for your platform.

## Quick start: inspect a PEtab model

The command-line interface validates and summarizes a PEtab problem:

```bash
iaapi inspect path/to/problem.yaml
iaapi inspect path/to/problem.yaml --json
```

The equivalent Python API is:

```python
from iaapi.data.petab_loader import PEtabLoader

problem = PEtabLoader().load("path/to/problem.yaml")

print(problem.n_parameters)
print(problem.n_observables)
print(problem.n_conditions)
print(list(problem.parameters["parameterId"]))
```

## Quick start: identifiability descriptors

The diagnostic modules accept arrays directly, so they can be used outside systems biology:

```python
import numpy as np

from iaapi.evaluation.fim_descriptors import compute_fim_descriptors

fim = np.diag([100.0, 10.0, 0.1, 0.001])
result = compute_fim_descriptors(fim)

print(dict(zip(result["names"], result["descriptors"])))
print(result["quality_flags"])
```

See [`examples/fim_diagnostics.py`](examples/fim_diagnostics.py) for a runnable example.

## Quick start: inference from a checkpoint

```python
import numpy as np

from iaapi.api import InferenceModel

model = InferenceModel.from_checkpoint("checkpoints/isp.pt", device="cpu")
result = model.infer(
    petab_problem="path/to/problem.yaml",
    observations={
        "time": np.array([0.0, 1.0, 2.0]),
        "values": np.array([[1.0, 0.8, 0.6]]),
        "mask": np.ones((1, 3)),
    },
    n_samples=1000,
)

print(result.get_posterior_stats())
print(result.get_sloppy_params())
result.save("result.json")
```

Checkpoint architecture settings are read from the checkpoint's `config` field. Observation arrays use `(n_observables, n_times)` for `values` and `mask`, and `(n_times,)` for `time`.

## Training workflow

The repository includes reusable data-generation and training entry points. Copy `configs/default.yaml`, update the PEtab benchmark path and model list, then run:

```bash
python scripts/generate_training_data.py \
  --config configs/default.yaml \
  --output training_data \
  --dry-run

# Remove --dry-run after checking the resolved models and shard plan.
python scripts/generate_training_data.py \
  --config configs/default.yaml \
  --output training_data

python scripts/train.py --config configs/default.yaml
```

For a new model collection, run the numerical preflight before large data generation:

```bash
python scripts/preflight_petab_models.py --config configs/ncs_preflight.yaml
```

Generated datasets, checkpoints, and run logs are intentionally ignored by Git.

## Main package layout

```text
iaapi/
  api.py             high-level checkpoint inference API
  cli.py             command-line model inspection
  data/              PEtab/SBML loading, simulation, FIM and data generation
  models/            MASE, observation encoders and ISP posterior heads
  training/          losses and training loop
  evaluation/        identifiability, PL/MCMC and calibration utilities
  oed/               experimental-design primitives
  refinement/        sequential refinement helpers
examples/             small runnable examples
tests/                unit and smoke tests
```

## Scope and scientific status

IAAPI is a research toolkit, not a clinical or safety-critical inference system. Cross-model generalization and prospective experimental-design performance remain active research topics. The public package deliberately does not present incomplete manuscript experiments as established results.

For a new domain, begin with the array-based diagnostic and posterior components. PEtab/SBML users can additionally use the full ingestion and graph path. Contributions that add simulator adapters or domain-specific examples are welcome.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

Some integration tests require AMICI and external PEtab benchmark models; those tests are marked `slow`.

## License

Released under the [MIT License](LICENSE).
