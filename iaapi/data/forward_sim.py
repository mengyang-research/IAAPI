"""
AMICI Simulator: ODE forward simulation based on AMICI.

This module handles forward simulation of SBML/PEtab models using AMICI,
including sensitivity analysis for Fisher Information Matrix computation.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import amici
    import amici.sim.sundials as amici_sundials
    import petab.v1 as petab_v1
    import petab.v2 as petab_v2
    from amici.importers.petab import PetabImporter
except ImportError as exc:
    raise ImportError("AMICI and PEtab are required") from exc


class AMICISimulator:
    """
    AMICI-based ODE forward simulator.

    IA-API uses log10 physical parameter values as the default theta convention.
    Internally, PEtab simulations receive physical parameter values keyed by the
    PEtab free parameter IDs.
    """

    def __init__(
        self,
        petab_problem: Any,
        compile_model: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        force_compile: bool = False,
        module_name: Optional[str] = None,
        theta_scale: str = "log10",
        validate: bool = False,
        time_points: Optional[np.ndarray] = None,
        **solver_options,
    ):
        """
        Initialize AMICI simulator from a PEtab problem or IA-API PEtabProblem.

        Args:
            petab_problem: Raw PEtab problem or IA-API PEtabProblem wrapper.
            compile_model: Whether to compile/load the AMICI model.
            cache_dir: Directory for compiled AMICI model caches.
            force_compile: Whether to force AMICI model re-import/recompile.
            module_name: Optional compiled module name.
            theta_scale: Scale of theta accepted by simulate(). Defaults to log10.
            validate: Whether AMICI/PEtab should validate during import.
            time_points: Optional default time points for direct simulations.
            solver_options: Additional solver options, mapped to set_<option>().
        """
        self.petab_problem = petab_problem
        self.raw_petab_problem = getattr(petab_problem, "raw_problem", petab_problem)
        self.yaml_path = getattr(petab_problem, "yaml_path", None)
        self.theta_scale = theta_scale
        self.force_compile = force_compile
        self.solver_options = solver_options
        self.default_time_points = np.asarray(time_points, dtype=float) if time_points is not None else None

        self.petab_problem_v2 = self._load_petab_v2_problem(validate=validate)
        self.parameter_ids = self._get_free_parameter_ids()
        self.observable_ids = self._get_observable_ids()
        self.n_parameters = len(self.parameter_ids)
        self.theta_nominal = self._get_nominal_parameters()
        self.theta_nominal_log10 = self._to_log10(self.theta_nominal)

        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else Path("~/.cache/iaapi/amici").expanduser()
        self.module_name = module_name or self._default_module_name()
        self.output_dir = self.cache_dir / self.module_name

        if compile_model:
            self._ensure_compiler_environment()
            self.importer = PetabImporter(
                self.petab_problem_v2,
                compile_=True,
                validate=validate,
                output_dir=self.output_dir,
                module_name=self.module_name,
                verbose=False,
            )
            self.simulator = self.importer.create_simulator(force_import=force_compile)
            self.ami_model = self.simulator.model
            self.solver = self.simulator.solver
            self._configure_solver()
        else:
            self.importer = None
            self.simulator = None
            self.ami_model = None
            self.solver = None

    def _load_petab_v2_problem(self, validate: bool = False) -> Any:
        """Return a PEtab v2 problem, upgrading v1 YAML-backed problems when needed."""
        if isinstance(self.raw_petab_problem, petab_v2.Problem):
            return self.raw_petab_problem

        if self.yaml_path is not None:
            return petab_v2.Problem.from_yaml(str(self.yaml_path))

        if isinstance(self.raw_petab_problem, petab_v1.Problem):
            raise ValueError(
                "AMICI 1.x requires a PEtab v2 problem; provide an IA-API PEtabProblem "
                "with yaml_path so the v1 problem can be upgraded from file."
            )

        return self.raw_petab_problem

    def _get_free_parameter_ids(self) -> List[str]:
        """Get estimated/free PEtab parameter IDs in simulation order."""
        x_free_ids = getattr(self.petab_problem_v2, "x_free_ids", None)
        if x_free_ids is not None:
            return list(x_free_ids)

        param_df = getattr(self.petab_problem, "parameters", None)
        if param_df is None:
            param_df = getattr(self.petab_problem_v2, "parameter_df", None)
        if param_df is None:
            return []

        if "parameterId" in param_df.columns:
            ids = param_df["parameterId"].astype(str)
        else:
            ids = param_df.index.astype(str)

        if "estimate" in param_df.columns:
            mask = param_df["estimate"].astype(bool)
            ids = ids[mask]
        return list(ids)

    def _get_observable_ids(self) -> List[str]:
        """Get PEtab observable IDs when available."""
        observable_df = getattr(self.petab_problem_v2, "observable_df", None)
        if observable_df is None:
            observable_df = getattr(self.petab_problem, "observables", None)
        if observable_df is None:
            return []
        if isinstance(observable_df, dict):
            return list(observable_df.keys())
        if "observableId" in observable_df.columns:
            return list(observable_df["observableId"].astype(str))
        return list(observable_df.index.astype(str))

    def _get_nominal_parameters(self) -> np.ndarray:
        """Get nominal physical parameter values in free parameter order."""
        param_df = getattr(self.petab_problem_v2, "parameter_df", None)
        if param_df is None:
            param_df = getattr(self.petab_problem, "parameters", None)
        if param_df is None or len(self.parameter_ids) == 0:
            return np.ones(len(self.parameter_ids), dtype=float)

        values = []
        for param_id in self.parameter_ids:
            if param_id in param_df.index:
                row = param_df.loc[param_id]
            elif "parameterId" in param_df.columns:
                matches = param_df[param_df["parameterId"].astype(str) == param_id]
                row = matches.iloc[0] if not matches.empty else None
            else:
                row = None
            if row is not None and "nominalValue" in row:
                values.append(float(row["nominalValue"]))
            else:
                values.append(1.0)
        return np.asarray(values, dtype=float)

    def _default_module_name(self) -> str:
        """Build a deterministic AMICI module/cache name for this problem."""
        model_id = getattr(self.petab_problem_v2, "model_id", None)
        if model_id is None:
            model = getattr(self.petab_problem_v2, "model", None)
            model_id = getattr(model, "model_id", None) or getattr(model, "id", None) or "petab_model"
        safe_model_id = re.sub(r"\W+", "_", str(model_id)).strip("_") or "petab_model"

        yaml_part = str(Path(self.yaml_path).resolve()) if self.yaml_path else safe_model_id
        mtime = ""
        if self.yaml_path and Path(self.yaml_path).exists():
            mtime = str(Path(self.yaml_path).stat().st_mtime_ns)
        key = f"{yaml_part}|{mtime}|{getattr(amici, '__version__', '')}|{self.theta_scale}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return f"iaapi_{safe_model_id}_{digest}"

    def _ensure_compiler_environment(self) -> None:
        """Prefer GCC-11 for AMICI compilation while respecting user overrides."""
        if "CC" not in os.environ:
            gcc = shutil.which("gcc-11") or ("/usr/bin/gcc-11" if Path("/usr/bin/gcc-11").exists() else None)
            if gcc:
                os.environ["CC"] = gcc
        if "CXX" not in os.environ:
            gxx = shutil.which("g++-11") or ("/usr/bin/g++-11" if Path("/usr/bin/g++-11").exists() else None)
            if gxx:
                os.environ["CXX"] = gxx

    def _configure_solver(self) -> None:
        """Configure AMICI solver defaults and user-specified options."""
        if self.solver is None:
            return
        self.solver.set_sensitivity_order(amici_sundials.SensitivityOrder_first)
        if hasattr(self.solver, "set_sensitivity_method"):
            self.solver.set_sensitivity_method(amici_sundials.SensitivityMethod_forward)

        for key, value in self.solver_options.items():
            setter = getattr(self.solver, f"set_{key}", None)
            if setter is None:
                setter = getattr(self.solver, f"set{key[:1].upper()}{key[1:]}", None)
            if setter is not None:
                setter(value)

    def _theta_to_array(self, theta: Union[np.ndarray, Dict[str, float]]) -> np.ndarray:
        """Return theta as an array ordered by self.parameter_ids."""
        if isinstance(theta, dict):
            missing = [param_id for param_id in self.parameter_ids if param_id not in theta]
            if missing:
                raise ValueError(f"Missing parameter values for: {missing}")
            return np.asarray([theta[param_id] for param_id in self.parameter_ids], dtype=float)

        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        if len(theta_array) != self.n_parameters:
            raise ValueError(
                f"Parameter dimension mismatch: expected {self.n_parameters}, got {len(theta_array)}"
            )
        return theta_array

    def _theta_to_physical_dict(self, theta: Union[np.ndarray, Dict[str, float]]) -> Dict[str, float]:
        """Convert theta in IA-API scale to physical PEtab parameter values."""
        theta_array = self._theta_to_array(theta)
        if self.theta_scale == "log10":
            physical = np.power(10.0, theta_array)
        elif self.theta_scale in {"lin", "linear"}:
            physical = theta_array
        else:
            raise ValueError(f"Unsupported theta_scale: {self.theta_scale}")
        return {param_id: float(value) for param_id, value in zip(self.parameter_ids, physical)}

    def _to_log10(self, values: np.ndarray) -> np.ndarray:
        """Convert positive physical values to log10, clipping invalid values to epsilon."""
        values = np.asarray(values, dtype=float)
        return np.log10(np.clip(values, 1e-300, None))

    def simulate(
        self,
        theta: Union[np.ndarray, Dict[str, float]],
        time_points: Optional[np.ndarray] = None,
        compute_sensitivities: bool = True,
    ) -> Dict[str, Any]:
        """
        Run forward simulation.

        Args:
            theta: Parameter vector or dict in ``self.theta_scale``.
            time_points: Optional custom time points. When provided, they are set
                on the AMICI model before running the PEtab simulator.
            compute_sensitivities: Whether to compute first-order sensitivities.

        Returns:
            Dictionary containing trajectories ``(n_obs, n_time)`` and
            sensitivities ``(n_obs, n_time, n_param)`` when requested.
        """
        if self.simulator is None or self.ami_model is None or self.solver is None:
            raise RuntimeError("Model not compiled")

        if time_points is not None:
            self.ami_model.set_timepoints(np.asarray(time_points, dtype=float))
        elif self.default_time_points is not None:
            self.ami_model.set_timepoints(self.default_time_points)

        if compute_sensitivities:
            self.solver.set_sensitivity_order(amici_sundials.SensitivityOrder_first)
            if hasattr(self.solver, "set_sensitivity_method"):
                self.solver.set_sensitivity_method(amici_sundials.SensitivityMethod_forward)
        else:
            self.solver.set_sensitivity_order(amici_sundials.SensitivityOrder_none)

        physical_parameters = self._theta_to_physical_dict(theta)
        result = self.simulator.simulate(problem_parameters=physical_parameters)
        rdatas = list(result.rdatas)
        return self._normalize_simulation_result(rdatas, compute_sensitivities=compute_sensitivities)

    def _normalize_simulation_result(
        self,
        rdatas: List[Any],
        compute_sensitivities: bool = True,
    ) -> Dict[str, Any]:
        """Normalize AMICI return data to IA-API array conventions."""
        trajectories = []
        sensitivities = []
        states = []
        times = []
        statuses = []

        for rdata in rdatas:
            y = self._rdata_array(rdata, "y")
            if y is not None:
                trajectories.append(y.T)  # AMICI: (time, obs) -> IA-API: (obs, time)

            sy = self._rdata_array(rdata, "sy") if compute_sensitivities else None
            if sy is not None and sy.size:
                sensitivities.append(self._normalize_sensitivity_shape(sy, y))

            x = self._rdata_array(rdata, "x")
            if x is not None:
                states.append(x.T)

            t = self._rdata_array(rdata, "t")
            if t is not None:
                times.append(t.reshape(-1))

            status = self._rdata_array(rdata, "status")
            if status is not None:
                statuses.append(int(np.asarray(status).reshape(())))

        output = {
            "trajectories": self._concat_time_axis(trajectories),
            "sensitivities": self._concat_time_axis(sensitivities) if sensitivities else None,
            "states": self._concat_time_axis(states),
            "time_points": np.concatenate(times) if times else None,
            "status": max(statuses) if statuses else None,
            "parameter_ids": list(self.parameter_ids),
            "observable_ids": list(self.observable_ids),
        }
        return output

    def _rdata_array(self, rdata: Any, key: str) -> Optional[np.ndarray]:
        """Best-effort array extraction from AMICI ReturnDataView."""
        try:
            value = rdata[key]
        except Exception:
            return None
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.dtype == object and arr.size == 0:
            return None
        return arr

    def _normalize_sensitivity_shape(self, sy: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        """Convert AMICI sensitivity arrays to (n_obs, n_time, n_param)."""
        sy = np.asarray(sy, dtype=float)
        if sy.ndim != 3:
            raise ValueError(f"Expected 3D AMICI sensitivities, got shape {sy.shape}")

        n_time = y.shape[0] if y is not None else None
        n_obs = y.shape[1] if y is not None else len(self.observable_ids)
        n_param = self.n_parameters

        # AMICI 1.x PEtab path commonly returns (n_time, n_param, n_obs).
        if n_time is not None and sy.shape == (n_time, n_param, n_obs):
            return np.transpose(sy, (2, 0, 1))
        # Direct AMICI simulations may return (n_time, n_obs, n_param).
        if n_time is not None and sy.shape == (n_time, n_obs, n_param):
            return np.transpose(sy, (1, 0, 2))
        # Already normalized.
        if sy.shape[0] == n_obs and sy.shape[2] == n_param:
            return sy

        raise ValueError(f"Cannot normalize AMICI sensitivity shape {sy.shape}")

    def _concat_time_axis(self, arrays: List[np.ndarray]) -> Optional[np.ndarray]:
        """Concatenate IA-API arrays along their time axis."""
        if not arrays:
            return None
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays, axis=1)

    def batch_simulate(
        self,
        thetas: List[Union[np.ndarray, Dict[str, float]]],
        time_points: Optional[np.ndarray] = None,
        compute_sensitivities: bool = True,
        n_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run multiple simulations, sequentially for AMICI object safety."""
        if n_workers != 1:
            warnings.warn(
                "Parallel AMICI simulation is deferred; running sequentially to avoid "
                "pickling compiled AMICI model objects.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [
            self.simulate(theta, time_points=time_points, compute_sensitivities=compute_sensitivities)
            for theta in thetas
        ]

    def simulate_with_noise(
        self,
        theta: Union[np.ndarray, Dict[str, float]],
        noise_level: float = 0.1,
        time_points: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        compute_sensitivities: bool = True,
    ) -> Dict[str, Any]:
        """Simulate with additive Gaussian noise while preserving sensitivities."""
        result = self.simulate(
            theta,
            time_points=time_points,
            compute_sensitivities=compute_sensitivities,
        )

        trajectories = result.get("trajectories")
        if trajectories is None:
            return result

        rng = np.random.default_rng(seed)
        mean_val = np.mean(np.abs(trajectories[trajectories != 0])) if np.any(trajectories != 0) else 0.0
        noise_std = float(noise_level * mean_val) if mean_val > 0 else 0.0
        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, trajectories.shape)
            trajectories_noisy = trajectories + noise
        else:
            trajectories_noisy = trajectories.copy()

        result["trajectories_noisy"] = trajectories_noisy
        result["noise_std"] = noise_std
        result["sigma"] = noise_std if noise_std > 0 else float(noise_level)
        return result
