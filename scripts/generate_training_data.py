"""
Training data generation script.

Usage:
    python scripts/generate_training_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from iaapi.data.data_generator import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    TrainingDataGenerator,
    _utc_now,
    load_manifest,
    save_manifest,
    validate_hdf5_shard,
    validate_manifest,
)


def resolve_petab_yaml(petab_path: Path, model_name: str) -> Path | None:
    """Resolve a PEtab YAML path across supported benchmark layouts."""
    candidates = [
        petab_path / model_name / f"{model_name}.yaml",
        petab_path / model_name / "problem.yaml",
        petab_path / "Benchmark-Models" / model_name / f"{model_name}.yaml",
        petab_path / "Benchmark-Models" / model_name / "problem.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for model_dir in [petab_path / model_name, petab_path / "Benchmark-Models" / model_name]:
        if model_dir.exists():
            yaml_files = sorted(model_dir.glob("*.yaml"))
            if len(yaml_files) == 1:
                return yaml_files[0]
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training data for IA-API")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--output", type=str, default="./training_data", help="Output directory")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of model-level worker processes")
    parser.add_argument("--shard-size", type=int, default=None, help="Samples per HDF5 shard")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None, help="Resume completed samples")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume completed samples")
    parser.add_argument("--max-retries", type=int, default=None, help="Retries per failed sample")
    parser.add_argument("--seed", type=int, default=None, help="Generation seed")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to generate")
    parser.add_argument("--n-samples-per-model", type=int, default=None, help="Override samples per model")
    parser.add_argument("--manifest", type=str, default="manifest.json", help="Manifest filename/path")
    parser.add_argument("--dry-run", action="store_true", help="Print generation plan without generating")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing manifest and shards only")
    return parser.parse_args()


def _get_config(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf containers to plain Python objects."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj


def _to_attr_dict(obj: Any) -> Any:
    """Recursively wrap dictionaries with attribute access."""
    if isinstance(obj, dict):
        return AttrDict({key: _to_attr_dict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(value) for value in obj]
    return obj


class AttrDict(dict):
    """Dictionary supporting attribute access for config compatibility."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load YAML config with a minimal Hydra-style defaults: [default] merge."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    defaults = config.get("defaults", []) or []
    merged: Dict[str, Any] = {}
    for entry in defaults:
        default_name = entry if isinstance(entry, str) else next(iter(entry.values()))
        default_path = path.parent / f"{default_name}.yaml"
        if default_path.exists():
            merged = _deep_merge(merged, load_config_file(default_path))
    return _deep_merge(merged, config)


def plan_model_shards(
    model_id: str,
    yaml_path: Path,
    output_dir: Path,
    n_samples: int,
    shard_size: int,
) -> List[Dict[str, Any]]:
    """Build shard plan entries for one model."""
    n_shards = max(1, math.ceil(n_samples / shard_size))
    shards = []
    for shard_id in range(n_shards):
        start = shard_id * shard_size
        count = min(shard_size, n_samples - start)
        if count <= 0:
            continue
        shards.append(
            {
                "model_id": model_id,
                "petab_yaml": str(yaml_path),
                "shard_id": shard_id,
                "start_sample_id": start,
                "n_samples": count,
                "output_path": str(output_dir / "shards" / model_id / f"shard_{shard_id:05d}.h5"),
            }
        )
    return shards


def build_generation_plan(config: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build a model/shard generation plan from config and CLI args."""
    petab_config = config.data.petab_benchmark
    petab_path = Path(petab_config.path)
    models = list(petab_config.models)
    if args.models:
        requested = {model.strip() for model in args.models.split(",") if model.strip()}
        models = [model for model in models if model in requested]
    n_samples = args.n_samples_per_model or int(petab_config.n_samples_per_model)
    data_generator = config.data.data_generator
    shard_size = args.shard_size or int(_get_config(data_generator, "shard_size", 10000))
    output_dir = Path(args.output)

    plan = []
    for model_name in models:
        yaml_path = resolve_petab_yaml(petab_path, model_name)
        if yaml_path is None:
            print(f"Warning: no PEtab YAML found for {model_name}, skipping")
            continue
        plan.extend(plan_model_shards(model_name, yaml_path, output_dir, n_samples, shard_size))
    return plan


def _worker_generate_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate all shards for one model in a worker process."""
    generator = TrainingDataGenerator(payload["config"])
    shard_results = []
    for shard in payload["shards"]:
        result = generator.generate_petab_shard(
            petab_yaml=shard["petab_yaml"],
            output_path=shard["output_path"],
            model_id=shard["model_id"],
            shard_id=shard["shard_id"],
            start_sample_id=shard["start_sample_id"],
            n_samples=shard["n_samples"],
            split_seed=payload["split_seed"],
            generation_seed=payload["seed"],
            val_split=payload["val_split"],
            test_split=payload["test_split"],
            resume=payload["resume"],
            max_retries=payload["max_retries"],
        )
        shard_results.append(result)
    return {"model_id": payload["model_id"], "shards": shard_results}


def _relative(path: str | Path, base: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_manifest_from_results(
    results: List[Dict[str, Any]],
    config: Any,
    config_path: str,
    output_dir: Path,
    shard_size: int,
    seed: int,
    val_split: float,
    test_split: float,
    max_retries: int,
) -> Dict[str, Any]:
    """Merge worker results into one manifest."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "shard_size": shard_size,
        "split_seed": seed,
        "splits": {"train": 1.0 - val_split - test_split, "val": val_split, "test": test_split},
        "generation": {
            "theta_scale": config["data"]["data_generator"].get("theta_scale", "log10"),
            "noise_level": config["data"]["data_generator"].get("noise_level", 0.1),
            "compute_sensitivities": config["data"]["data_generator"].get("compute_sensitivities", True),
            "max_retries": max_retries,
        },
        "models": {},
        "samples": {"train": [], "val": [], "test": []},
        "failures": [],
    }

    for model_result in results:
        model_id = model_result["model_id"]
        model_entry = {
            "source": "petab_benchmark",
            "petab_yaml": "",
            "n_requested": 0,
            "n_success": 0,
            "n_failed": 0,
            "parameter_sampling": [],
            "fim_stats": {},
            "shards": [],
        }
        condition_numbers = []
        effective_ranks = []
        for shard in model_result["shards"]:
            if not model_entry["petab_yaml"]:
                with_h5 = Path(shard["path"])
                model_entry["petab_yaml"] = ""
                if with_h5.exists():
                    import h5py

                    with h5py.File(with_h5, "r") as f:
                        model_entry["petab_yaml"] = f.attrs.get("petab_yaml", "")
            shard_manifest = {
                "path": _relative(shard["path"], output_dir),
                "shard_id": shard["shard_id"],
                "sample_id_start": shard["sample_id_start"],
                "sample_id_end": shard["sample_id_end"],
                "n_success": shard["n_success"],
                "n_failed": shard["n_failed"],
                "status": shard["status"],
            }
            model_entry["shards"].append(shard_manifest)
            model_entry["n_requested"] += shard["n_requested"]
            model_entry["n_success"] += shard["n_success"]
            model_entry["n_failed"] += shard["n_failed"]
            model_entry["parameter_sampling"] = shard.get("parameter_sampling", []) or model_entry["parameter_sampling"]
            model_entry["fim_stats"] = shard.get("fim_stats", {})
            for split, entries in shard.get("samples", {}).items():
                for entry in entries:
                    rel_entry = dict(entry)
                    rel_entry["shard"] = _relative(rel_entry["shard"], output_dir)
                    manifest["samples"][split].append(rel_entry)
            manifest["failures"].extend(shard.get("failures", []))
        manifest["models"][model_id] = model_entry
    return manifest


def validate_existing(output_dir: Path, manifest_name: str) -> None:
    """Validate an existing manifest and all referenced shards."""
    manifest_path = output_dir / manifest_name
    manifest = load_manifest(manifest_path)
    for model in manifest.get("models", {}).values():
        for shard in model.get("shards", []):
            validate_hdf5_shard(output_dir / shard["path"])
    result = validate_manifest(manifest_path)
    print(f"Validation OK: {result}")


def main():
    """Main data generation loop."""
    args = parse_args()

    plain_config = load_config_file(args.config)
    config = _to_attr_dict(plain_config)
    output_dir = Path(args.output)
    manifest_name = args.manifest

    if args.validate_only:
        validate_existing(output_dir, manifest_name)
        return

    data_generator_config = config.data.data_generator
    shard_size = args.shard_size or int(_get_config(data_generator_config, "shard_size", 10000))
    resume = args.resume if args.resume is not None else bool(_get_config(data_generator_config, "resume", True))
    max_retries = args.max_retries or int(_get_config(data_generator_config, "max_retries", 3))
    seed = args.seed or int(_get_config(data_generator_config, "split_seed", 42))
    val_split = float(config.evaluation.get("val_split", 0.1))
    test_split = float(config.evaluation.get("test_split", 0.1))

    plan = build_generation_plan(config, args)
    print(f"Planned {len(plan)} shards")
    for shard in plan:
        print(
            f"  {shard['model_id']} shard {shard['shard_id']}: "
            f"samples {shard['start_sample_id']}..{shard['start_sample_id'] + shard['n_samples'] - 1} "
            f"-> {shard['output_path']}"
        )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for shard in plan:
        by_model.setdefault(shard["model_id"], []).append(shard)

    payloads = [
        {
            "model_id": model_id,
            "shards": shards,
            "config": plain_config,
            "split_seed": seed,
            "seed": seed,
            "val_split": val_split,
            "test_split": test_split,
            "resume": resume,
            "max_retries": max_retries,
        }
        for model_id, shards in by_model.items()
    ]

    results = []
    if args.n_workers == 1:
        for payload in payloads:
            results.append(_worker_generate_model(payload))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            future_map = {executor.submit(_worker_generate_model, payload): payload["model_id"] for payload in payloads}
            for future in as_completed(future_map):
                results.append(future.result())

    manifest = build_manifest_from_results(
        results,
        plain_config,
        args.config,
        output_dir,
        shard_size,
        seed,
        val_split,
        test_split,
        max_retries,
    )
    manifest_path = output_dir / manifest_name
    save_manifest(manifest, manifest_path)

    for model in manifest.get("models", {}).values():
        for shard in model.get("shards", []):
            validate_hdf5_shard(output_dir / shard["path"])
    validation = validate_manifest(manifest_path)
    print(f"Data generation complete: {validation}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
