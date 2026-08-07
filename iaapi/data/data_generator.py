"""
Training Data Generator: Generate training quadruples (M, theta, trajectory, FIM).

This module handles generation of training data from PEtab problems,
including parameter sampling, forward simulation, FIM computation, sharded
HDF5 writing, manifests, and loading generated data for training.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm

from iaapi.data.fim_compute import FIMComputer
from iaapi.data.forward_sim import AMICISimulator
from iaapi.data.petab_loader import PEtabLoader, PEtabProblem
from iaapi.data.sbml_parser import sbml_to_hetero_data


SCHEMA_VERSION = "iaapi.training_data.v2"
MANIFEST_SCHEMA_VERSION = "iaapi.training_manifest.v1"


def _utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _json_dumps_attr(value: Any) -> str:
    """Serialize a JSON-compatible value for HDF5 attributes."""
    return json.dumps(value, default=str, ensure_ascii=False)


def _json_loads_attr(value: Any, default: Any = None) -> Any:
    """Deserialize a JSON value from HDF5 attributes."""
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def deterministic_seed(seed: int, *parts: Any) -> int:
    """Generate a deterministic uint32 seed from a base seed and labels."""
    key = ":".join([str(seed), *[str(part) for part in parts]])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def assign_split(
    model_id: str,
    sample_id: int,
    split_seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> str:
    """Assign a sample to train/val/test deterministically."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1")
    value = deterministic_seed(split_seed, model_id, sample_id, "split") / float(2**32)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def compute_fim_stats(condition_numbers: List[float], effective_ranks: List[int]) -> Dict[str, Any]:
    """Compute summary statistics for FIM condition numbers and effective ranks."""
    cond = np.asarray(condition_numbers, dtype=float) if condition_numbers else np.array([])
    ranks = np.asarray(effective_ranks, dtype=float) if effective_ranks else np.array([])
    finite_cond = cond[np.isfinite(cond)]

    def percentile(values: np.ndarray, q: float) -> Optional[float]:
        if values.size == 0:
            return None
        return float(np.percentile(values, q))

    return {
        "condition_number": {
            "min": percentile(finite_cond, 0),
            "median": percentile(finite_cond, 50),
            "p95": percentile(finite_cond, 95),
            "max": percentile(finite_cond, 100),
            "n_finite": int(np.isfinite(cond).sum()) if cond.size else 0,
            "n_infinite": int(np.isinf(cond).sum()) if cond.size else 0,
            "n_nan": int(np.isnan(cond).sum()) if cond.size else 0,
        },
        "effective_rank": {
            "min": percentile(ranks, 0),
            "median": percentile(ranks, 50),
            "max": percentile(ranks, 100),
        },
    }


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a training-data manifest JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], path: str | Path) -> None:
    """Save a training-data manifest JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _utc_now()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def validate_hdf5_shard(path: str | Path, strict: bool = True, max_spot_checks: int = 20) -> Dict[str, Any]:
    """Validate a sharded training-data HDF5 file."""
    path = Path(path)
    required = [
        "theta",
        "theta_true",
        "fim",
        "trajectory",
        "trajectory_clean",
        "time_points",
        "observation_values",
        "observation_times",
        "observation_masks",
        "fim_eigenvalues",
        "fim_eigenvectors",
    ]
    errors: List[str] = []
    checked = 0
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema_version") != SCHEMA_VERSION:
            errors.append("missing or unsupported schema_version")
        samples_group = f.get("samples")
        if samples_group is None:
            errors.append("missing /samples group")
            sample_names: List[str] = []
        else:
            sample_names = [name for name in samples_group if samples_group[name].attrs.get("status") == "ok"]
        if int(f.attrs.get("n_samples", 0)) != len(sample_names):
            errors.append("n_samples attr does not match ok sample group count")

        for name in sample_names[:max_spot_checks]:
            grp = samples_group[name]
            missing = [dataset for dataset in required if dataset not in grp]
            if missing:
                errors.append(f"{name}: missing datasets {missing}")
                continue
            theta = grp["theta"][:]
            theta_true = grp["theta_true"][:]
            fim = grp["fim"][:]
            eigvals = grp["fim_eigenvalues"][:]
            eigvecs = grp["fim_eigenvectors"][:]
            obs = grp["observation_values"][:]
            traj = grp["trajectory"][:]
            masks = grp["observation_masks"][:]
            n_params = theta.shape[0]
            if theta_true.shape != theta.shape or not np.allclose(theta, theta_true):
                errors.append(f"{name}: theta/theta_true mismatch")
            if fim.shape != (n_params, n_params):
                errors.append(f"{name}: invalid fim shape {fim.shape}")
            if eigvecs.shape != (n_params, n_params) or eigvals.shape != (n_params,):
                errors.append(f"{name}: invalid eigensystem shapes")
            if obs.shape != traj.shape or masks.shape != obs.shape:
                errors.append(f"{name}: observation shape mismatch")
            if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(traj)) or not np.all(np.isfinite(fim)):
                errors.append(f"{name}: non-finite values")
            if not np.allclose(fim, fim.T, atol=1e-6, rtol=1e-5):
                errors.append(f"{name}: FIM is not symmetric")
            if np.any(np.diff(eigvals) > 1e-8):
                errors.append(f"{name}: eigenvalues are not sorted descending")
            reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            if not np.allclose(fim, reconstructed, atol=1e-4, rtol=1e-3):
                errors.append(f"{name}: FIM/eigendecomposition mismatch")
            checked += 1

    if strict and errors:
        raise ValueError(f"Invalid shard {path}: " + "; ".join(errors))
    return {"path": str(path), "ok": not errors, "errors": errors, "checked": checked}


def validate_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Validate manifest references and split sample entries."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    errors: List[str] = []
    seen = set()
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split, entries in manifest.get("samples", {}).items():
        split_counts[split] = len(entries)
        for entry in entries:
            key = (entry.get("shard"), entry.get("sample"))
            if key in seen:
                errors.append(f"duplicate sample reference {key}")
            seen.add(key)
            shard_path = base_dir / entry["shard"]
            if not shard_path.exists():
                errors.append(f"missing shard {shard_path}")
                continue
            with h5py.File(shard_path, "r") as f:
                sample_path = f"samples/{entry['sample']}"
                if sample_path not in f:
                    errors.append(f"missing sample {entry['sample']} in {shard_path}")

    n_success = sum(model.get("n_success", 0) for model in manifest.get("models", {}).values())
    if sum(split_counts.values()) != n_success:
        errors.append("split sample count does not match model success count")
    if errors:
        raise ValueError("Invalid manifest: " + "; ".join(errors))
    return {"ok": True, "split_counts": split_counts, "n_success": n_success}


class TrainingDataGenerator:
    """
    Generate training data quadruples (M, theta, trajectory, FIM).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize training data generator."""
        self.config = config
        self.data_config = self._get_nested_config(config, "data", "data_generator", default=config)
        self.bio_priors_config = self._get_nested_config(config, "data", "bio_priors", default=self._get_config(config, "bio_priors", {}))
        self.eval_config = self._get_config(config, "evaluation", {})
        self.theta_scale = self._get_config(self.data_config, "theta_scale", "log10")
        self.noise_level = self._get_config(self.data_config, "noise_level", self._get_config(config, "noise_level", 0.1))

        self.petab_loader = PEtabLoader()
        self.fim_computer = FIMComputer(noise_level=self.noise_level)

        self.bio_prior_loader = None
        if self._get_config(self.bio_priors_config, "enabled", False):
            try:
                from iaapi.data.bio_priors import BioPriorLoader

                loader = BioPriorLoader(self._get_config(self.bio_priors_config, "path"))
                if not loader.list_available_priors():
                    raise ValueError("no prior files were found")
                self.bio_prior_loader = loader
            except Exception as exc:
                warnings.warn(
                    f"Biological priors could not be loaded ({exc}); falling back to uniform bounds.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _get_config(self, config: Any, key: str, default: Any = None) -> Any:
        """Read a key from dict-like or OmegaConf-like config objects."""
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _get_nested_config(self, config: Any, *keys: str, default: Any = None) -> Any:
        """Read nested config keys from dict-like or OmegaConf-like objects."""
        current = config
        for key in keys:
            current = self._get_config(current, key, None)
            if current is None:
                return default
        return current

    def generate_from_petab(
        self,
        petab_yaml: str,
        n_samples: int = 500,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate training data from a PEtab problem (legacy in-memory API)."""
        print(f"Loading PEtab problem: {petab_yaml}")
        petab_problem = self.petab_loader.load(petab_yaml)
        simulator = self._create_simulator(petab_problem)

        print(f"Generating {n_samples} samples...")
        samples = []
        model_id = Path(petab_yaml).stem
        for i in tqdm(range(n_samples)):
            sample_seed = deterministic_seed(42, model_id, i, "theta")
            noise_seed = deterministic_seed(42, model_id, i, "noise")
            split = assign_split(model_id, i)
            sample = self._generate_single_sample(
                petab_problem,
                simulator,
                sample_id=i,
                local_sample_id=i,
                split=split,
                rng=np.random.default_rng(sample_seed),
                generation_seed=sample_seed,
                noise_seed=noise_seed,
                attempts=1,
            )
            samples.append(sample)

        if output_path:
            self._save_to_hdf5(samples, output_path, petab_yaml)
        return samples

    def _create_simulator(self, petab_problem: PEtabProblem) -> AMICISimulator:
        """Create an AMICI simulator using current generator config."""
        return AMICISimulator(
            petab_problem,
            compile_model=True,
            cache_dir=self._get_config(self.data_config, "amici_cache_dir", None),
            force_compile=self._get_config(self.data_config, "force_compile", False),
            theta_scale=self.theta_scale,
            time_points=self._default_time_points(),
        )

    def _sample_parameters(
        self,
        petab_problem: PEtabProblem,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample parameters from biological priors or uniform bounds."""
        if self.bio_prior_loader is not None:
            theta = self.bio_prior_loader.sample(petab_problem)
        else:
            theta = self._sample_uniform(petab_problem, rng=rng)
        return theta

    def _sample_uniform(
        self,
        petab_problem: PEtabProblem,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample parameters uniformly from bounds using the configured theta scale."""
        rng = rng or np.random.default_rng()
        bounds = self.petab_loader.get_parameter_bounds(petab_problem)
        scales = self.petab_loader.get_parameter_scales(petab_problem)

        parameters = self.petab_loader._normalize_id_column(petab_problem.parameters, "parameterId")
        if "estimate" in parameters.columns:
            parameters = parameters[parameters["estimate"].astype(bool)]

        theta = []
        for param_id in parameters["parameterId"]:
            lower, upper = bounds[param_id]
            scale = scales[param_id]

            if self.theta_scale == "log10":
                if lower <= 0 or upper <= 0:
                    raise ValueError(
                        f"Cannot sample log10 theta for non-positive bounds of {param_id}: "
                        f"({lower}, {upper})"
                    )
                if scale in {"log", "log10"}:
                    theta.append(rng.uniform(np.log10(lower), np.log10(upper)))
                else:
                    value = rng.uniform(lower, upper)
                    theta.append(np.log10(value))
            else:
                if scale in {"log", "log10"}:
                    log_val = rng.uniform(np.log10(max(lower, 1e-300)), np.log10(upper))
                    theta.append(10**log_val)
                else:
                    theta.append(rng.uniform(lower, upper))

        return np.array(theta)

    def _default_time_points(self) -> np.ndarray:
        """Build default simulation time points from config."""
        min_time_points = int(self._get_config(self.data_config, "min_time_points", 10))
        max_time_points = int(self._get_config(self.data_config, "max_time_points", 100))
        n_time_points = max(min_time_points, max_time_points)
        time_span = self._get_config(self.data_config, "time_span", [0, 100])
        return np.linspace(float(time_span[0]), float(time_span[1]), n_time_points)

    def _resolve_noise_param_map(
        self,
        parameter_ids: List[str],
        observable_ids: List[str],
    ) -> Dict[int, int]:
        """Map each observable index to the index of its sd_* noise parameter.

        PEtab convention: a noise parameter like ``sd_pSTAT5A_rel`` provides the
        additive noise scale for observable ``pSTAT5A_rel``. Matching strips the
        ``sd_`` prefix and a trailing ``_rel`` (or other suffix) and requires the
        remainder to equal (or contain) the observable id.

        Returns ``{obs_idx: sd_param_idx}``. Observables without a matching
        sd_* parameter are omitted; the caller falls back to the global
        ``noise_level`` for them.
        """
        noise_map: Dict[int, int] = {}
        sd_candidates = {pid: i for i, pid in enumerate(parameter_ids) if pid.startswith("sd_")}
        for obs_idx, obs_id in enumerate(observable_ids):
            for pid, pidx in sd_candidates.items():
                stem = pid[len("sd_"):]
                # strip common suffixes (_rel, _abs, _lin) one at a time
                for suffix in ("_rel", "_abs", "_lin"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                if stem == obs_id or obs_id.startswith(stem) or stem.startswith(obs_id):
                    noise_map[obs_idx] = pidx
                    break
        return noise_map

    def _parameter_scales(self, petab_problem: "PEtabProblem", parameter_ids: List[str]) -> List[str]:
        """Return the parameterScale string per parameter id (default 'lin')."""
        scales: Dict[str, str] = {}
        try:
            params = self.petab_loader._normalize_id_column(petab_problem.parameters, "parameterId")
            for _, row in params.iterrows():
                scales[row["parameterId"]] = row.get("parameterScale", "lin")
        except Exception:
            pass
        return [scales.get(pid, "lin") for pid in parameter_ids]

    def _compute_fim_from_result(
        self,
        result: Dict[str, Any],
        theta: Optional[np.ndarray] = None,
        petab_problem: Optional["PEtabProblem"] = None,
    ) -> Dict[str, Any]:
        """Compute Fisher Information Matrix from a simulation result.

        When ``theta`` and ``petab_problem`` are supplied, the FIM is computed
        consistently with the PEtab noise model and the log10 parameterization:

          1. **chain-rule to log10**: AMICI sensitivities are ``d y/d (linear
             theta)``. The posterior lives in log10 space, so the Fisher is
             transformed as ``F_log10 = D F_lin D`` with
             ``D = diag(ln10 * 10**theta_log10)`` for log10-scaled parameters
             (``D = 1`` for linear parameters).
          2. **per-observable additive noise**: ``sigma_k = 10**theta[sd_k]``
             for the noise parameter mapped to observable ``k`` (PEtab
             ``noiseParameter1_<obs>`` convention). Observables without a mapped
             sd_* fall back to the global ``noise_level``.
          3. **noise-parameter Fisher term**: a Gaussian additive-noise
             parameter ``sd`` (log10-scaled) contributes
             ``F[sd, sd] += 2 (ln10)**2 * n_time`` to its own Fisher entry —
             AMICI forward sensitivities ``d y/d sd`` are identically zero
             because ``sd`` only enters the likelihood variance, not the
             trajectory.

        Without ``theta``/``petab_problem`` the legacy behaviour (single global
        ``sigma`` from the simulation result) is preserved.
        """
        if result.get("sensitivities") is None:
            return {
                "fim": np.array([[]]),
                "eigenvalues": np.array([]),
                "eigenvectors": np.array([]),
                "effective_rank": 0,
                "entropy_effective_rank": 0.0,
                "condition_number": float("inf"),
            }

        sensitivities = np.asarray(result["sensitivities"], dtype=float)  # (n_obs, n_time, n_param)
        n_obs, n_time, n_param = sensitivities.shape
        parameter_ids = result.get("parameter_ids", [])
        observable_ids = result.get("observable_ids", [])

        use_petab_noise = theta is not None and petab_problem is not None and parameter_ids and observable_ids

        if use_petab_noise:
            theta_arr = np.asarray(theta, dtype=float)
            scales = self._parameter_scales(petab_problem, parameter_ids)
            # 1. chain-rule: multiply column j by ln10 * 10**theta_j when log10-scaled
            chain = np.ones(n_param, dtype=float)
            for j, sc in enumerate(scales):
                if sc == "log10" and j < len(theta_arr):
                    chain[j] = np.log(10.0) * np.power(10.0, theta_arr[j])
            jacobian = sensitivities.reshape(n_obs * n_time, n_param) * chain[None, :]
            jacobian = np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)

            # 2. per-observable additive sigma = 10**theta[sd_k]; fallback to noise_level
            noise_map = self._resolve_noise_param_map(parameter_ids, observable_ids)
            sigma_per_obs = np.full(n_obs, float(self.noise_level), dtype=float)
            for obs_idx, sd_pidx in noise_map.items():
                if sd_pidx < len(theta_arr):
                    sigma_per_obs[obs_idx] = float(np.power(10.0, theta_arr[sd_pidx]))
            sigma_per_obs = np.maximum(sigma_per_obs, 1e-12)
            weights = np.repeat(1.0 / (sigma_per_obs ** 2), n_time)  # (n_obs*n_time,)
            fim = jacobian.T @ (weights[:, None] * jacobian)
            fim = 0.5 * (fim + fim.T)

            # 3. noise-parameter Fisher term (additive Gaussian, log10-scaled sd)
            ln10_sq = (np.log(10.0)) ** 2
            for obs_idx, sd_pidx in noise_map.items():
                if sd_pidx < len(scales) and scales[sd_pidx] == "log10":
                    fim[sd_pidx, sd_pidx] += 2.0 * ln10_sq * n_time
        else:
            # legacy path: single global sigma
            sigma = result.get("sigma", result.get("noise_std", None))
            fim = self.fim_computer.compute_fim(sensitivities, sigma=sigma)

        fim_result = self.fim_computer.eigendecompose(fim)
        fim_result["fim"] = fim
        return fim_result

    def _generate_single_sample(
        self,
        petab_problem: PEtabProblem,
        simulator: AMICISimulator,
        sample_id: int,
        local_sample_id: int,
        split: str,
        rng: np.random.Generator,
        generation_seed: int,
        noise_seed: int,
        attempts: int,
    ) -> Dict[str, Any]:
        """Generate one training sample dictionary."""
        theta = self._sample_parameters(petab_problem, rng=rng)
        result = simulator.simulate_with_noise(
            theta,
            noise_level=self.noise_level,
            seed=noise_seed,
            compute_sensitivities=self._get_config(self.data_config, "compute_sensitivities", True),
        )
        fim_result = self._compute_fim_from_result(result, theta=theta, petab_problem=petab_problem)
        trajectory = result["trajectories_noisy"]
        time_points = result["time_points"]
        return {
            "model_id": Path(petab_problem.yaml_path).stem,
            "sample_id": sample_id,
            "local_sample_id": local_sample_id,
            "split": split,
            "M": petab_problem.graph_data,
            "parsed_sbml": petab_problem.parsed_sbml,
            "theta": theta,
            "theta_true": theta,
            "trajectory": trajectory,
            "trajectory_clean": result["trajectories"],
            "time_points": time_points,
            "observation_values": trajectory,
            "observation_times": time_points,
            "observation_masks": np.ones_like(trajectory, dtype=np.float32),
            "fim": fim_result["fim"],
            "fim_eigenvalues": fim_result["eigenvalues"],
            "fim_eigenvectors": fim_result["eigenvectors"],
            "effective_rank": fim_result["effective_rank"],
            "entropy_effective_rank": fim_result.get("entropy_effective_rank", 0.0),
            "condition_number": fim_result.get("condition_number", float("inf")),
            "parameter_ids": result.get("parameter_ids", []),
            "observable_ids": result.get("observable_ids", []),
            "theta_scale": self.theta_scale,
            "noise_std": result.get("noise_std", 0.0),
            "n_params": len(theta),
            "generation_seed": generation_seed,
            "noise_seed": noise_seed,
            "attempts": attempts,
        }

    def generate_petab_shard(
        self,
        petab_yaml: str,
        output_path: str,
        model_id: str,
        shard_id: int,
        start_sample_id: int,
        n_samples: int,
        split_seed: int = 42,
        generation_seed: int = 42,
        val_split: float = 0.1,
        test_split: float = 0.1,
        resume: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate one PEtab HDF5 shard incrementally with resume/retry."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        petab_problem = self.petab_loader.load(petab_yaml)
        simulator = self._create_simulator(petab_problem)
        parameter_sampling = self._parameter_sampling_metadata(petab_problem)

        condition_numbers: List[float] = []
        effective_ranks: List[int] = []
        successes = 0
        failures = []
        sample_entries = {"train": [], "val": [], "test": []}

        with h5py.File(output_path, "a") as f:
            self._initialize_shard_file(
                f,
                petab_problem,
                petab_yaml,
                model_id,
                shard_id,
                start_sample_id,
                n_samples,
                split_seed,
                generation_seed,
                parameter_sampling,
            )
            samples_group = f.require_group("samples")

            for offset in tqdm(range(n_samples), desc=f"{model_id} shard {shard_id}"):
                sample_id = start_sample_id + offset
                sample_name = f"sample_{sample_id:012d}"
                if resume and sample_name in samples_group:
                    status = samples_group[sample_name].attrs.get("status")
                    if status == "ok":
                        successes += 1
                        split = samples_group[sample_name].attrs.get("split", "train")
                        sample_entries[split].append(self._sample_manifest_entry(output_path, sample_name, model_id))
                        condition_numbers.append(float(samples_group[sample_name].attrs.get("condition_number", np.nan)))
                        effective_ranks.append(int(samples_group[sample_name].attrs.get("effective_rank", 0)))
                        continue
                    del samples_group[sample_name]

                sample = None
                last_error = None
                for attempt in range(1, max_retries + 1):
                    theta_seed = deterministic_seed(generation_seed, model_id, sample_id, "theta")
                    noise_seed = deterministic_seed(generation_seed, model_id, sample_id, "noise")
                    split = assign_split(model_id, sample_id, split_seed, val_split, test_split)
                    try:
                        sample = self._generate_single_sample(
                            petab_problem,
                            simulator,
                            sample_id=sample_id,
                            local_sample_id=offset,
                            split=split,
                            rng=np.random.default_rng(theta_seed),
                            generation_seed=theta_seed,
                            noise_seed=noise_seed,
                            attempts=attempt,
                        )
                        break
                    except Exception as exc:  # keep generation running for other samples
                        last_error = exc

                if sample is None:
                    self._record_failure(f, sample_id, last_error, max_retries)
                    failures.append({"sample_id": sample_id, "error": repr(last_error), "shard": str(output_path)})
                    continue

                self._write_sample_group(f, sample, f"samples/{sample_name}")
                successes += 1
                condition_numbers.append(float(sample["condition_number"]))
                effective_ranks.append(int(sample["effective_rank"]))
                sample_entries[sample["split"]].append(self._sample_manifest_entry(output_path, sample_name, model_id))

            fim_stats = compute_fim_stats(condition_numbers, effective_ranks)
            f.attrs["n_samples"] = successes
            f.attrs["n_failed"] = len(failures)
            f.attrs["fim_stats_json"] = _json_dumps_attr(fim_stats)
            f.attrs["updated_at"] = _utc_now()

        return {
            "path": str(output_path),
            "relative_path": str(output_path),
            "shard_id": shard_id,
            "sample_id_start": start_sample_id,
            "sample_id_end": start_sample_id + n_samples - 1,
            "n_requested": n_samples,
            "n_success": successes,
            "n_failed": len(failures),
            "status": "complete" if successes + len(failures) == n_samples else "partial",
            "fim_stats": compute_fim_stats(condition_numbers, effective_ranks),
            "parameter_sampling": parameter_sampling,
            "samples": sample_entries,
            "failures": failures,
        }

    def _initialize_shard_file(
        self,
        f: h5py.File,
        petab_problem: PEtabProblem,
        petab_yaml: str,
        model_id: str,
        shard_id: int,
        start_sample_id: int,
        n_samples: int,
        split_seed: int,
        generation_seed: int,
        parameter_sampling: List[Dict[str, Any]],
    ) -> None:
        """Initialize root attrs/groups for a shard."""
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["model_id"] = model_id
        f.attrs["model_source"] = "petab_benchmark"
        f.attrs["petab_yaml"] = str(Path(petab_yaml).resolve())
        f.attrs["shard_id"] = shard_id
        f.attrs["shard_index_start"] = start_sample_id
        f.attrs["shard_index_end"] = start_sample_id + n_samples - 1
        f.attrs["n_requested"] = n_samples
        f.attrs["theta_scale"] = self.theta_scale
        f.attrs["noise_level"] = self.noise_level
        f.attrs["generation_seed"] = generation_seed
        f.attrs["split_seed"] = split_seed
        f.attrs.setdefault("created_at", _utc_now())
        f.attrs["updated_at"] = _utc_now()
        if petab_problem.parsed_sbml is not None:
            f.attrs["parsed_sbml_json"] = _json_dumps_attr(petab_problem.parsed_sbml)
        f.attrs["parameter_sampling_json"] = _json_dumps_attr(parameter_sampling)
        metadata = f.require_group("metadata")
        metadata.attrs["source_type"] = "petab"
        metadata.attrs["source_path"] = str(Path(petab_yaml).resolve())
        metadata.attrs["sbml_path"] = petab_problem.sbml_path or ""
        metadata.attrs["parameter_sampling_json"] = _json_dumps_attr(parameter_sampling)
        f.require_group("samples")
        f.require_group("failures")

    def _sample_manifest_entry(self, output_path: Path, sample_name: str, model_id: str) -> Dict[str, str]:
        """Create a manifest sample entry. Paths are made relative when possible by caller."""
        return {"shard": str(output_path), "sample": sample_name, "model_id": model_id}

    def _write_sample_group(self, f: h5py.File, sample: Dict[str, Any], group_name: str) -> None:
        """Write one sample group atomically enough for resume detection."""
        if group_name in f:
            del f[group_name]
        grp = f.create_group(group_name)
        grp.attrs["status"] = "incomplete"
        for key in [
            "model_id",
            "sample_id",
            "local_sample_id",
            "split",
            "effective_rank",
            "entropy_effective_rank",
            "condition_number",
            "noise_std",
            "theta_scale",
            "generation_seed",
            "noise_seed",
            "attempts",
        ]:
            grp.attrs[key] = sample.get(key)
        grp.attrs["parameter_ids"] = _json_dumps_attr(sample.get("parameter_ids", []))
        grp.attrs["observable_ids"] = _json_dumps_attr(sample.get("observable_ids", []))

        for key in [
            "theta",
            "theta_true",
            "fim",
            "trajectory",
            "trajectory_clean",
            "time_points",
            "observation_values",
            "observation_times",
            "observation_masks",
            "fim_eigenvalues",
            "fim_eigenvectors",
        ]:
            grp.create_dataset(key, data=sample[key])
        f.flush()
        grp.attrs["status"] = "ok"
        f.flush()

    def _record_failure(self, f: h5py.File, sample_id: int, error: Exception, attempts: int) -> None:
        """Record a failed sample."""
        group_name = f"failures/sample_{sample_id:012d}"
        if group_name in f:
            del f[group_name]
        grp = f.create_group(group_name)
        grp.attrs["sample_id"] = sample_id
        grp.attrs["attempts"] = attempts
        grp.attrs["last_error_type"] = type(error).__name__ if error is not None else "Unknown"
        grp.attrs["last_error"] = repr(error)
        grp.attrs["failed_at"] = _utc_now()
        f.flush()

    def _parameter_sampling_metadata(self, petab_problem: PEtabProblem) -> List[Dict[str, Any]]:
        """Return parameter sampling metadata for a PEtab problem."""
        parameters = self.petab_loader._normalize_id_column(petab_problem.parameters, "parameterId")
        if "estimate" in parameters.columns:
            parameters = parameters[parameters["estimate"].astype(bool)]
        rows = []
        for _, row in parameters.iterrows():
            rows.append(
                {
                    "parameter_id": row["parameterId"],
                    "lower_bound": float(row.get("lowerBound", np.nan)),
                    "upper_bound": float(row.get("upperBound", np.nan)),
                    "petab_scale": row.get("parameterScale", "lin"),
                    "theta_scale": self.theta_scale,
                    "sampling_distribution": "uniform" if self.bio_prior_loader is None else "bio_prior",
                }
            )
        return rows

    def _save_to_hdf5(self, samples: List[Dict[str, Any]], output_path: str, petab_yaml: str):
        """Save samples to legacy HDF5 format."""
        with h5py.File(output_path, "w") as f:
            f.attrs["petab_yaml"] = petab_yaml
            f.attrs["n_samples"] = len(samples)
            if samples and samples[0].get("parsed_sbml") is not None:
                f.attrs["parsed_sbml_json"] = _json_dumps_attr(samples[0]["parsed_sbml"])
            for i, sample in enumerate(tqdm(samples, desc="Saving to HDF5")):
                grp = f.create_group(f"sample_{i:06d}")
                grp.attrs["model_id"] = sample["model_id"]
                grp.attrs["sample_id"] = sample["sample_id"]
                grp.attrs["effective_rank"] = sample["effective_rank"]
                grp.attrs["noise_std"] = sample["noise_std"]
                grp.attrs["theta_scale"] = sample.get("theta_scale", "")
                grp.attrs["parameter_ids"] = _json_dumps_attr(sample.get("parameter_ids", []))
                grp.create_dataset("theta", data=sample["theta"])
                grp.create_dataset("fim", data=sample["fim"])
                grp.create_dataset("trajectory", data=sample["trajectory"])
                grp.create_dataset("trajectory_clean", data=sample["trajectory_clean"])
                grp.create_dataset("time_points", data=sample["time_points"])
                grp.create_dataset("fim_eigenvalues", data=sample["fim_eigenvalues"])
                grp.create_dataset("fim_eigenvectors", data=sample["fim_eigenvectors"])
        print(f"Saved {len(samples)} samples to {output_path}")


class ShardedHDF5Dataset:
    """Dataset for loading IA-API HDF5 training shards."""

    def __init__(
        self,
        shard_paths: Optional[List[str]] = None,
        manifest_path: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[callable] = None,
        return_torch: bool = True,
        reconstruct_graph: bool = True,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.split = split
        self.transform = transform
        self.return_torch = return_torch
        self.reconstruct_graph = reconstruct_graph
        self._graph_cache: Dict[str, Any] = {}

        if self.manifest_path is not None:
            self.manifest = load_manifest(self.manifest_path)
            entries = []
            splits = [split] if split is not None else ["train", "val", "test"]
            for split_name in splits:
                for entry in self.manifest.get("samples", {}).get(split_name, []):
                    path = self.manifest_path.parent / entry["shard"]
                    entries.append({"path": str(path), "sample": entry["sample"]})
            self.sample_index = entries
            self.shard_paths = sorted({entry["path"] for entry in entries})
            self.sample_counts = []
            self.total_samples = len(entries)
        else:
            self.manifest = None
            self.shard_paths = shard_paths or []
            self._compute_sample_counts()
            self.sample_index = None

    def _compute_sample_counts(self):
        """Compute total number of samples across direct shard paths."""
        self.sample_counts = []
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                if f.attrs.get("schema_version") == SCHEMA_VERSION and "samples" in f:
                    n_samples = sum(1 for name in f["samples"] if f["samples"][name].attrs.get("status") == "ok")
                else:
                    n_samples = int(f.attrs.get("n_samples", 0))
                self.sample_counts.append(n_samples)
        self.total_samples = sum(self.sample_counts)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")
        if self.sample_index is not None:
            entry = self.sample_index[idx]
            return self._load_v2_sample(entry["path"], entry["sample"])

        cumulative = 0
        for shard_idx, count in enumerate(self.sample_counts):
            if idx < cumulative + count:
                local_idx = idx - cumulative
                return self._load_sample(shard_idx, local_idx)
            cumulative += count
        raise IndexError(f"Index {idx} out of range")

    def _load_sample(self, shard_idx: int, local_idx: int) -> Dict[str, Any]:
        path = self.shard_paths[shard_idx]
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema_version") == SCHEMA_VERSION:
                ok_names = [name for name in f["samples"] if f["samples"][name].attrs.get("status") == "ok"]
                ok_names.sort()
                return self._load_v2_sample(path, ok_names[local_idx])
            grp = f[f"sample_{local_idx:06d}"]
            sample = {
                "model_id": grp.attrs["model_id"],
                "sample_id": int(grp.attrs["sample_id"]),
                "effective_rank": int(grp.attrs["effective_rank"]),
                "noise_std": float(grp.attrs["noise_std"]),
                "theta": grp["theta"][:],
                "theta_true": grp["theta"][:],
                "trajectory": grp["trajectory"][:],
                "trajectory_clean": grp["trajectory_clean"][:],
                "time_points": grp["time_points"][:],
                "fim": grp["fim"][:] if "fim" in grp else np.array([[]]),
                "fim_eigenvalues": grp["fim_eigenvalues"][:],
                "fim_eigenvectors": grp["fim_eigenvectors"][:],
            }
        sample = self._to_training_schema(sample, path)
        return self._finalize_sample(sample)

    def _load_v2_sample(self, path: str, sample_name: str) -> Dict[str, Any]:
        with h5py.File(path, "r") as f:
            grp = f[f"samples/{sample_name}"]
            sample = {
                "model_id": grp.attrs["model_id"],
                "sample_id": int(grp.attrs["sample_id"]),
                "effective_rank": int(grp.attrs["effective_rank"]),
                "entropy_effective_rank": float(grp.attrs.get("entropy_effective_rank", 0.0)),
                "condition_number": float(grp.attrs.get("condition_number", np.inf)),
                "noise_std": float(grp.attrs["noise_std"]),
                "theta": grp["theta"][:],
                "theta_true": grp["theta_true"][:],
                "fim": grp["fim"][:],
                "trajectory": grp["trajectory"][:],
                "trajectory_clean": grp["trajectory_clean"][:],
                "time_points": grp["time_points"][:],
                "observation_values": grp["observation_values"][:],
                "observation_times": grp["observation_times"][:],
                "observation_masks": grp["observation_masks"][:],
                "fim_eigenvalues": grp["fim_eigenvalues"][:],
                "fim_eigenvectors": grp["fim_eigenvectors"][:],
            }
        sample = self._to_training_schema(sample, path)
        return self._finalize_sample(sample)

    def _to_training_schema(self, sample: Dict[str, Any], path: str) -> Dict[str, Any]:
        values = sample.get("observation_values", sample["trajectory"])
        times = sample.get("observation_times", sample["time_points"])
        masks = sample.get("observation_masks", np.ones_like(values, dtype=np.float32))
        sample["observations"] = {"values": values, "times": times, "masks": masks}
        sample["n_params"] = int(sample["theta_true"].shape[0])
        sample["graph_data"] = self._load_graph(path) if self.reconstruct_graph else None
        return sample

    def _load_graph(self, path: str) -> Any:
        if path in self._graph_cache:
            return self._graph_cache[path]
        with h5py.File(path, "r") as f:
            parsed = _json_loads_attr(f.attrs.get("parsed_sbml_json"), None)
        graph = sbml_to_hetero_data(parsed) if parsed is not None else None
        self._graph_cache[path] = graph
        return graph

    def _finalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.return_torch:
            sample = self._to_torch(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _to_torch(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            return sample
        for key in ["theta", "theta_true", "fim", "trajectory", "trajectory_clean", "time_points", "fim_eigenvalues", "fim_eigenvectors"]:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.as_tensor(sample[key], dtype=torch.float32)
        for key in ["values", "times", "masks"]:
            if isinstance(sample["observations"][key], np.ndarray):
                dtype = torch.float32
                sample["observations"][key] = torch.as_tensor(sample["observations"][key], dtype=dtype)
        return sample


def collate_training_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-shape IA-API samples into a padded training batch."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for collate_training_batch") from exc

    batch_size = len(samples)
    max_obs = max(sample["observations"]["values"].shape[0] for sample in samples)
    max_time = max(sample["observations"]["values"].shape[1] for sample in samples)
    max_params = max(int(sample["n_params"]) for sample in samples)

    values = torch.zeros((batch_size, max_obs, max_time), dtype=torch.float32)
    times = torch.zeros((batch_size, max_time), dtype=torch.float32)
    masks = torch.zeros((batch_size, max_obs, max_time), dtype=torch.float32)
    theta = torch.zeros((batch_size, max_params), dtype=torch.float32)
    fim = torch.zeros((batch_size, max_params, max_params), dtype=torch.float32)
    eigvals = torch.zeros((batch_size, max_params), dtype=torch.float32)
    eigvecs = torch.zeros((batch_size, max_params, max_params), dtype=torch.float32)
    effective_rank = torch.zeros((batch_size,), dtype=torch.long)
    n_params = torch.zeros((batch_size,), dtype=torch.long)

    graphs = []
    model_ids = []
    sample_ids = []
    for i, sample in enumerate(samples):
        obs_values = torch.as_tensor(sample["observations"]["values"], dtype=torch.float32)
        obs_times = torch.as_tensor(sample["observations"]["times"], dtype=torch.float32)
        obs_masks = torch.as_tensor(sample["observations"]["masks"], dtype=torch.float32)
        n_obs, n_time = obs_values.shape
        n_param = int(sample["n_params"])
        values[i, :n_obs, :n_time] = obs_values
        times[i, :n_time] = obs_times
        masks[i, :n_obs, :n_time] = obs_masks
        theta[i, :n_param] = torch.as_tensor(sample["theta_true"], dtype=torch.float32)
        fim[i, :n_param, :n_param] = torch.as_tensor(sample["fim"], dtype=torch.float32)
        eigvals[i, :n_param] = torch.as_tensor(sample["fim_eigenvalues"], dtype=torch.float32)
        eigvecs[i, :n_param, :n_param] = torch.as_tensor(sample["fim_eigenvectors"], dtype=torch.float32)
        effective_rank[i] = int(sample["effective_rank"])
        n_params[i] = n_param
        graphs.append(sample.get("graph_data"))
        model_ids.append(sample.get("model_id"))
        sample_ids.append(sample.get("sample_id"))

    graph_data: Any = graphs
    if all(graph is not None for graph in graphs):
        try:
            from torch_geometric.data import Batch

            graph_data = Batch.from_data_list(graphs)
        except Exception:
            graph_data = graphs

    return {
        "model_id": model_ids,
        "sample_id": sample_ids,
        "graph_data": graph_data,
        "observations": {"values": values, "times": times, "masks": masks},
        "theta_true": theta,
        "theta": theta,
        "fim": fim,
        "fim_eigenvalues": eigvals,
        "fim_eigenvectors": eigvecs,
        "effective_rank": effective_rank,
        "n_params": n_params,
    }
