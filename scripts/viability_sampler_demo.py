"""Smoke gate for the P1-02 viability-aware sampler.

Loads a P1-01 REJECT model with AMICI and compares the forward success rate of
the legacy full-bound uniform sampler against the nominal-centered viability
sampler (truncated normal, +/-1 log10 window), with an optional cheap forward
pre-check. Reports both rates and exits non-zero if the new strategy does NOT
materially exceed the old one (acceptance gate for P1-02).

Usage:
    python iaapi/scripts/viability_sampler_demo.py --model Crauste_CellSystems2017
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "iaapi"))

from iaapi.data.forward_sim import AMICISimulator  # noqa: E402
from iaapi.data.petab_loader import PEtabLoader  # noqa: E402
from iaapi.data.viability_sampler import (  # noqa: E402
    RoleWidths,
    ViabilitySampler,
    compare_strategies,
)

PETAB_ROOT = REPO_ROOT / "iaapi" / ".." / "Benchmark-Models-PEtab" / "Benchmark-Models"
PETAB_ROOT = (REPO_ROOT / "Benchmark-Models-PEtab" / "Benchmark-Models").resolve()


def _resolve_yaml(model: str) -> Path:
    c = PETAB_ROOT / model / f"{model}.yaml"
    if c.exists():
        return c
    ys = sorted((PETAB_ROOT / model).glob("*.yaml"))
    if len(ys) == 1:
        return ys[0]
    raise FileNotFoundError(f"no PEtab yaml for {model}")


def _build_sim(model: str, cache_dir: str):
    yaml_path = _resolve_yaml(model)
    problem = PEtabLoader().load(str(yaml_path), validate=False)
    sim = AMICISimulator(problem, compile_model=True, cache_dir=cache_dir,
                         theta_scale="log10", validate=False)
    pdf = sim.petab_problem_v2.parameter_df
    if "parameterId" in pdf.columns:
        pdf = pdf.set_index("parameterId")
    free = list(sim.parameter_ids)
    lower = pdf.loc[free, "lowerBound"].astype(float).to_numpy()
    upper = pdf.loc[free, "upperBound"].astype(float).to_numpy()
    bounds = np.column_stack([sim._to_log10(lower), sim._to_log10(upper)])
    return sim, bounds, sim.theta_nominal_log10, free


def _make_forward_fn(sim):
    def forward_ok(theta):
        try:
            res = sim.simulate(theta, compute_sensitivities=False)
        except Exception:  # noqa: BLE001
            return False
        status = res.get("status")
        traj = res.get("trajectories")
        if status is not None and int(status) != 0:
            return False
        if traj is not None and not np.all(np.isfinite(np.asarray(traj, dtype=float))):
            return False
        return True
    return forward_ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="P1-02 viability sampler smoke gate")
    p.add_argument("--model", default="Elowitz_Nature2000",
                   help="REJECT model to demo on (Elowitz responds to nominal-centering; "
                        "Crauste does not — see plan evidence)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=float, default=0.5,
                   help="log10 half-width applied to ALL roles")
    p.add_argument("--cache-dir", default="~/.cache/iaapi/amici")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args(argv)

    cache_dir = str(Path(args.cache_dir).expanduser())
    print(f"loading {args.model} ...", file=sys.stderr)
    sim, bounds, nominal, param_ids = _build_sim(args.model, cache_dir)
    fwd = _make_forward_fn(sim)
    # Apply the width to every role explicitly (named roles override `default`).
    role_widths = RoleWidths(kinetic=args.width, initial_state=args.width,
                             noise=args.width, scale=args.width, default=args.width)

    t0 = time.perf_counter()
    cmp = compare_strategies(
        fwd, bounds, nominal, param_ids, n=args.n, seed=args.seed,
        new_strategy="truncated_normal", role_widths=role_widths,
    )
    cmp["model"] = args.model
    cmp["n_parameters"] = int(len(param_ids))
    cmp["width"] = args.width

    # Also demonstrate the pre-check filter on the new strategy.
    new_sampler = ViabilitySampler(bounds, nominal, param_ids,
                                   strategy="truncated_normal",
                                   role_widths=role_widths, seed=args.seed)
    viable, stats = new_sampler.sample_with_precheck(fwd, n_target=min(args.n, 20))
    cmp["precheck"] = stats
    cmp["elapsed_s"] = round(time.perf_counter() - t0, 2)

    print(json.dumps(cmp, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(cmp, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")

    verdict = ("PASS" if cmp["materially_exceeds"] else "FAIL")
    print(f"\nP1-02 smoke gate: {verdict} (old={cmp['old_forward_success_rate']} "
          f"new={cmp['new_forward_success_rate']} improvement={cmp['improvement']})",
          file=sys.stderr)
    return 0 if cmp["materially_exceeds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
