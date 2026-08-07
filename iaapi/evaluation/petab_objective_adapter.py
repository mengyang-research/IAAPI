"""Official PEtab objective adapter (R1-01).

Wraps ``pypesto.petab.PetabImporter.from_yaml(yaml).create_objective()`` into
the ``nll_fn(theta_log10) -> float`` signature that ``ProfileLikelihoodRunner``
already consumes.  The official pyPESTO/AMICI objective correctly handles all
PEtab conditions, observable formulas, transformations, noise models and
parameter overrides -- replacing the simplified ``build_petab_data_nll``
(RISK-02) that collapsed multi-condition models onto one time grid.

Scale bridging
--------------
IA-API optimizes in **log10 of physical values** for all free parameters.
pyPESTO's ``AmiciObjective`` operates on **all** parameters (free + fixed at
nominal) in PEtab-declared scale (``log10`` / ``log`` / ``lin`` per parameter).

Conversion from IA-API log10-theta to PEtab-scaled x, per parameter:

    parameterScale = 'log10':  x = theta_log10           (identity)
    parameterScale = 'log':    x = theta_log10 * ln(10)  (natural log)
    parameterScale = 'lin':    x = 10 ** theta_log10     (physical value)

The gradient chain rule inverts this:

    d_nll / d_theta_log10 = (d_nll / d_x) * (d_x / d_theta_log10)

where d_x / d_theta_log10 is 1, ln(10), or ln(10)*10**theta_log10 respectively.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LN10 = math.log(10.0)


class SimulatorError(Exception):
    """Raised when the AMICI forward simulation fails (mirrors PL runner convention)."""


def _log10_to_petab_scaled(theta_log10: np.ndarray, scale: str) -> float:
    """Convert one log10-space parameter value to PEtab-scaled space."""
    if scale == "log10":
        return float(theta_log10)
    elif scale == "log":
        return float(theta_log10) * LN10
    elif scale in ("lin", "linear"):
        return float(10.0 ** theta_log10)
    else:
        raise ValueError(f"Unsupported parameterScale '{scale}'")


def _scale_jacobian(theta_log10: np.ndarray, scale: str) -> float:
    """d(pypesto_x) / d(theta_log10) for the scale conversion."""
    if scale == "log10":
        return 1.0
    elif scale == "log":
        return LN10
    elif scale in ("lin", "linear"):
        return LN10 * float(10.0 ** theta_log10)
    else:
        raise ValueError(f"Unsupported parameterScale '{scale}'")


class PetabObjectiveAdapter:
    """Wraps the official pyPESTO/AMICI PEtab objective as an injectable nll_fn.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the PEtab problem YAML file.
    force_compile : bool
        If True, force AMICI model recompilation.
    cache_dir : str, optional
        AMICI model cache directory (unused by pyPESTO directly, but kept for
        interface compatibility with AMICISimulator).
    """

    def __init__(
        self,
        yaml_path: str | Path,
        force_compile: bool = False,
        cache_dir: Optional[str] = None,
    ):
        import petab.v1 as petab
        from pypesto.petab import PetabImporter

        self.yaml_path = str(yaml_path)
        self.force_compile = force_compile
        self.cache_dir = cache_dir

        # Load PEtab problem for parameter metadata
        self.petab_problem = petab.Problem.from_yaml(self.yaml_path)
        self.x_ids: List[str] = list(self.petab_problem.x_ids)
        self.x_free_indices: List[int] = list(self.petab_problem.x_free_indices)
        self.free_ids: List[str] = [self.x_ids[i] for i in self.x_free_indices]
        self.n_free: int = len(self.x_free_indices)
        self.n_total: int = len(self.x_ids)

        # Parameter scales (per-parameter)
        scales_raw = self.petab_problem.parameter_df.get("parameterScale", "lin")
        self.x_scales: List[str] = [
            scales_raw.get(pid, "lin") if hasattr(scales_raw, "get") else scales_raw
            for pid in self.x_ids
        ]
        # Ensure list of strings
        self.x_scales = [str(s) for s in self.x_scales]

        # Free-parameter scales (in free-param order)
        self.free_scales: List[str] = [self.x_scales[i] for i in self.x_free_indices]

        # Nominal scaled values for ALL parameters (fixed params stay at these)
        self.x_nominal_scaled: np.ndarray = np.asarray(
            self.petab_problem.x_nominal_scaled, dtype=float
        )
        # Nominal log10 values for free params (IA-API convention)
        self.theta_nominal_log10: np.ndarray = np.asarray(
            [self._petab_scaled_to_log10(self.x_nominal_scaled[i], self.x_scales[i])
             for i in self.x_free_indices],
            dtype=float,
        )

        # Build the official pyPESTO objective
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.importer = PetabImporter.from_yaml(self.yaml_path)
            self.objective = self.importer.create_objective(
                force_compile=force_compile
            )

    @staticmethod
    def _petab_scaled_to_log10(x_scaled: float, scale: str) -> float:
        """Inverse of _log10_to_petab_scaled."""
        if scale == "log10":
            return float(x_scaled)
        elif scale == "log":
            return float(x_scaled) / LN10
        elif scale in ("lin", "linear"):
            return math.log10(max(x_scaled, 1e-300))
        else:
            raise ValueError(f"Unsupported parameterScale '{scale}'")

    def _theta_log10_to_full_x(self, theta_log10: np.ndarray) -> np.ndarray:
        """Convert free-param log10 vector to full PEtab-scaled x vector."""
        theta = np.asarray(theta_log10, dtype=float).reshape(-1)
        if len(theta) != self.n_free:
            raise ValueError(
                f"theta_log10 has {len(theta)} values, expected {self.n_free} free params"
            )
        x_full = self.x_nominal_scaled.copy()
        for j, idx in enumerate(self.x_free_indices):
            x_full[idx] = _log10_to_petab_scaled(theta[j], self.free_scales[j])
        return x_full

    def nll(self, theta_log10: np.ndarray) -> float:
        """Negative log-likelihood at theta_log10 (free params, log10 scale).

        This is the injectable ``nll_fn`` for ``ProfileLikelihoodRunner``.
        Raises ``SimulatorError`` on AMICI failure.
        """
        try:
            x_full = self._theta_log10_to_full_x(theta_log10)
            val = self.objective(x_full, sensi_orders=(0,), mode="mode_fun")
            if not math.isfinite(float(val)):
                raise SimulatorError(f"non-finite nll value: {val}")
            return float(val)
        except SimulatorError:
            raise
        except Exception as e:
            raise SimulatorError(f"AMICI objective evaluation failed: {e}") from e

    def nll_with_grad(
        self, theta_log10: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """NLL and gradient w.r.t. theta_log10.

        Returns (nll, grad) where grad has shape (n_free,).
        """
        try:
            x_full = self._theta_log10_to_full_x(theta_log10)
            val, grad_full = self.objective(
                x_full, sensi_orders=(0, 1), mode="mode_fun"
            )
            val = float(val)
            if not math.isfinite(val):
                raise SimulatorError(f"non-finite nll value: {val}")
            # Extract gradient for free params and apply chain rule
            grad_full = np.asarray(grad_full, dtype=float).reshape(-1)
            if len(grad_full) != self.n_total:
                raise SimulatorError(
                    f"gradient length {len(grad_full)} != n_total {self.n_total}"
                )
            grad_free = np.array([
                grad_full[idx] * _scale_jacobian(theta_log10[j], self.free_scales[j])
                for j, idx in enumerate(self.x_free_indices)
            ])
            if not np.all(np.isfinite(grad_free)):
                raise SimulatorError("non-finite gradient")
            return val, grad_free
        except SimulatorError:
            raise
        except Exception as e:
            raise SimulatorError(f"AMICI gradient evaluation failed: {e}") from e

    def evaluate_direct(self, theta_log10: np.ndarray) -> float:
        """Direct pyPESTO objective call (for equivalence testing).

        Returns the raw pyPESTO NLL without the adapter's float-cast / error
        handling, so the smoke test can verify the adapter wraps correctly.
        """
        x_full = self._theta_log10_to_full_x(theta_log10)
        return float(self.objective(x_full))
