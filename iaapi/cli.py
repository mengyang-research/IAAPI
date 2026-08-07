"""Command-line entry points for IAAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def inspect_petab(path: str) -> dict:
    """Load a PEtab YAML file and return a compact model summary."""
    from iaapi.data.petab_loader import PEtabLoader

    problem = PEtabLoader().load(path)
    if "parameterId" in problem.parameters.columns:
        parameter_ids = problem.parameters["parameterId"].astype(str).tolist()
    else:
        parameter_ids = problem.parameters.index.astype(str).tolist()
    return {
        "problem": Path(problem.yaml_path).stem,
        "yaml_path": str(problem.yaml_path),
        "sbml_path": problem.sbml_path,
        "n_parameters": problem.n_parameters,
        "n_observables": problem.n_observables,
        "n_conditions": problem.n_conditions,
        "n_measurements": problem.n_measurements,
        "parameter_ids": parameter_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iaapi",
        description="Identifiability-aware tools for mechanistic ODE models.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate and summarize a PEtab problem"
    )
    inspect_parser.add_argument("petab_yaml", help="Path to a PEtab YAML file")
    inspect_parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        summary = inspect_petab(args.petab_yaml)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Problem: {summary['problem']}")
            print(f"Parameters: {summary['n_parameters']}")
            print(f"Observables: {summary['n_observables']}")
            print(f"Conditions: {summary['n_conditions']}")
            print(f"Measurements: {summary['n_measurements']}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
