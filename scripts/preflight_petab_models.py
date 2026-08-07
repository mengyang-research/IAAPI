"""CLI: PEtab preflight scanner (NCS plan task P1-01).

Runs :mod:`iaapi.data.preflight` over a list of candidate PEtab models, writing
versioned, resumable, incrementally-flushed results with a PID/progress record
and a per-model wall-clock timeout.

Usage:
    python iaapi/scripts/preflight_petab_models.py --config iaapi/configs/ncs_preflight.yaml
    python iaapi/scripts/preflight_petab_models.py --smoke   # 1 stable + 1 unstable, n=5

Outputs (under iaapi/runs/model_preflight/<version>/):
    results/<model>.json   per-model result (flushed as each completes -> resume)
    results.json           aggregated list
    results.csv            flat table
    report.md              human-readable summary
    progress.json          pid / start / completed / total / current / last_update
    preflight.log          append-only run log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "iaapi"))

from iaapi.data.preflight import (  # noqa: E402
    SCHEMA_VERSION,
    CAT_COMPILE,
    CAT_TIMEOUT,
    ModelPreflightResult,
    PreflightConfig,
    PreflightRunner,
    run_model_from_yaml,
    soft_timeout,
    _AlarmTimeout,
)

DEFAULT_CONFIG = {
    "petab_root": "../Benchmark-Models-PEtab/Benchmark-Models",
    "models": [],
    "n_samples": 50,
    "seed": 0,
    "noise_level": 0.1,
    "forward_success_threshold": 0.90,
    "sensitivity_success_threshold": 0.80,
    "runtime_p95_threshold_s": 300.0,
    "per_simulate_timeout_s": 120.0,
    "per_model_timeout_s": 1800.0,
    "amici_cache_dir": "~/.cache/iaapi/amici",
    "output_version": "v1",
}

SMOKE_MODELS = ["Boehm_JProteomeRes2014", "Raimundez_PCB2020"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: str) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).is_file():
        with open(path, encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    return cfg


def _resolve_petab_yaml(petab_root: Path, model: str) -> Path:
    candidates = [
        petab_root / model / f"{model}.yaml",
        petab_root / model / "problem.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    ys = sorted((petab_root / model).glob("*.yaml")) if (petab_root / model).is_dir() else []
    if len(ys) == 1:
        return ys[0]
    raise FileNotFoundError(f"no PEtab yaml for {model} under {petab_root}")


def _run_one(model_id: str, yaml_path: Path, config: PreflightConfig,
             cache_dir: str, per_model_timeout: float) -> Dict[str, Any]:
    """Run preflight for one model with a hard wall-clock budget."""
    try:
        with soft_timeout(per_model_timeout):
            result = run_model_from_yaml(
                str(yaml_path), model_id, config, cache_dir=cache_dir, compile_model=True,
            )
        return result.to_dict()
    except _AlarmTimeout:
        return _reject(model_id, config, CAT_TIMEOUT, "per-model wall budget exceeded")
    except Exception as e:  # noqa: BLE001 - categorize, never swallow
        tb = traceback.format_exc(limit=4)
        return _reject(model_id, config, CAT_COMPILE, f"{type(e).__name__}: {e}", tb=tb)


def _reject(model_id: str, config: PreflightConfig, category: str,
            reason: str, tb: str = "") -> Dict[str, Any]:
    r = ModelPreflightResult(model_id=model_id, n_parameters=0, n_samples=config.n_samples)
    r.failure_categories = [category]
    r.gate = "REJECT"
    r.gate_reasons = [reason]
    r.nominal_forward = {"success": False, "failure_category": category, "runtime_s": 0.0}
    d = r.to_dict()
    if tb:
        d["traceback"] = tb
    return d


def _write_progress(out_dir: Path, progress: Dict[str, Any]) -> None:
    (out_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _flatten(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": result["model_id"],
        "n_parameters": result["n_parameters"],
        "gate": result["gate"],
        "nominal_forward_success": result["nominal_forward"].get("success"),
        "pilot_forward_success_rate": result["pilot_forward"].get("success_rate"),
        "pilot_forward_nan_rate": result["pilot_forward"].get("nan_rate"),
        "pilot_forward_runtime_p95_s": result["pilot_forward"].get("runtime_p95_s"),
        "pilot_sensitivity_success_rate": result["pilot_sensitivity"].get("success_rate"),
        "fim_finite": result["fim"].get("finite"),
        "fim_effective_rank": result["fim"].get("effective_rank_participation"),
        "failure_categories": ";".join(result["failure_categories"]),
        "gate_reasons": "; ".join(result["gate_reasons"]),
        "elapsed_s": result["elapsed_s"],
    }


def _write_report(out_dir: Path, results: List[Dict[str, Any]]) -> None:
    lines = [f"# PEtab Preflight Report — {out_dir.name}", "", f"generated: {_utc_now()}", ""]
    admit = sum(1 for r in results if r["gate"] == "ADMIT")
    lines.append(f"Summary: {admit}/{len(results)} ADMIT, {len(results) - admit} REJECT")
    lines.append("")
    lines.append("| model | n_params | gate | fwd% | nan% | p95(s) | sens% | FIM | reasons |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|---|")
    for r in results:
        pf = r["pilot_forward"]; ps = r["pilot_sensitivity"]
        lines.append(
            f"| {r['model_id']} | {r['n_parameters']} | {r['gate']} | "
            f"{pf.get('success_rate')} | {pf.get('nan_rate')} | {pf.get('runtime_p95_s')} | "
            f"{ps.get('success_rate')} | {r['fim'].get('finite')} | {'; '.join(r['gate_reasons'])} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PEtab preflight scanner")
    p.add_argument("--config", default="iaapi/configs/ncs_preflight.yaml")
    p.add_argument("--models", default=None, help="comma-separated model ids (overrides config)")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-version", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--smoke", action="store_true", help="smoke gate: 2 models, n=5")
    p.add_argument("--petab-root", default=None, help="override petab root (abs or rel to iaapi/)")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    if args.smoke:
        cfg["models"] = list(SMOKE_MODELS)
        cfg["n_samples"] = 5
        cfg["output_version"] = f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        cfg["per_model_timeout_s"] = 1200.0
    if args.models:
        cfg["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_version:
        cfg["output_version"] = args.output_version

    models: List[str] = cfg["models"]
    if not models:
        print("no models configured (use --models or config.models)", file=sys.stderr)
        return 2

    # Resolve petab root relative to iaapi/ (the project convention).
    petab_root = Path(cfg["petab_root"])
    if args.petab_root:
        petab_root = Path(args.petab_root)
    if not petab_root.is_absolute():
        petab_root = (REPO_ROOT / "iaapi" / petab_root).resolve()

    out_dir = REPO_ROOT / "iaapi" / "runs" / "model_preflight" / cfg["output_version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    log_path = out_dir / "preflight.log"

    def log(msg: str) -> None:
        line = f"[{_utc_now()}] {msg}"
        print(line, file=sys.stderr)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    config = PreflightConfig(
        n_samples=cfg["n_samples"], seed=cfg["seed"], noise_level=cfg["noise_level"],
        forward_success_threshold=cfg["forward_success_threshold"],
        sensitivity_success_threshold=cfg["sensitivity_success_threshold"],
        runtime_p95_threshold_s=cfg["runtime_p95_threshold_s"],
        per_simulate_timeout_s=cfg["per_simulate_timeout_s"],
    )
    cache_dir = str(Path(cfg["amici_cache_dir"]).expanduser())

    progress = {
        "schema_version": SCHEMA_VERSION, "pid": os.getpid(), "started_at": _utc_now(),
        "total": len(models), "completed": 0, "current": None,
        "last_update": _utc_now(), "output_version": cfg["output_version"],
    }
    _write_progress(out_dir, progress)
    log(f"START preflight version={cfg['output_version']} models={models} "
        f"n_samples={cfg['n_samples']} seed={cfg['seed']} petab_root={petab_root}")

    results: List[Dict[str, Any]] = []
    for i, model in enumerate(models, 1):
        progress["current"] = model
        _write_progress(out_dir, progress)
        res_path = out_dir / "results" / f"{model}.json"
        if args.resume and res_path.is_file():
            log(f"[{i}/{len(models)}] {model}: resume (cached result)")
            results.append(json.loads(res_path.read_text(encoding="utf-8")))
            progress["completed"] = i
            progress["last_update"] = _utc_now()
            _write_progress(out_dir, progress)
            continue

        try:
            yaml_path = _resolve_petab_yaml(petab_root, model)
        except FileNotFoundError as e:
            log(f"[{i}/{len(models)}] {model}: SKIP ({e})")
            continue

        log(f"[{i}/{len(models)}] {model}: running (yaml={yaml_path.name}) ...")
        t0 = time.perf_counter()
        res = _run_one(model, yaml_path, config, cache_dir, cfg["per_model_timeout_s"])
        log(f"[{i}/{len(models)}] {model}: gate={res['gate']} "
            f"categories={res['failure_categories']} ({time.perf_counter() - t0:.1f}s)")
        res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(res)
        progress["completed"] = i
        progress["last_update"] = _utc_now()
        _write_progress(out_dir, progress)

    # Aggregate.
    (out_dir / "results.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "results": results}, indent=2,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        if results:
            w = csv.DictWriter(fh, fieldnames=list(_flatten(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(_flatten(r))
    _write_report(out_dir, results)
    progress["current"] = None
    progress["finished_at"] = _utc_now()
    _write_progress(out_dir, progress)
    log(f"DONE {sum(1 for r in results if r['gate']=='ADMIT')}/{len(results)} ADMIT -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
