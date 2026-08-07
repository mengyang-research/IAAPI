"""
PEtab Loader: Load PEtab problems into a unified format.

This module handles loading of PEtab (Parameter Estimation Toolbench) problems,
which are a standardized format for parameter estimation in systems biology.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import copy

import pandas as pd

from iaapi.data.sbml_parser import SBMLParser, sbml_to_hetero_data

try:
    import petab.v1 as petab
except ImportError:
    raise ImportError("PEtab is required. Install with: pip install petab>=0.4.0")


class PEtabProblem:
    """
    Container for PEtab problem data.

    Attributes:
        sbml_model: SBML model object
        measurements: DataFrame with measurement data
        parameters: DataFrame with parameter bounds and scales
        observables: Dictionary with observable definitions
        conditions: DataFrame with experimental conditions
        yaml_path: Path to the PEtab YAML file
        raw_problem: Original petab.Problem object
        parsed_sbml: Parsed SBML graph dictionary
        graph_data: PyG HeteroData graph
        sbml_path: Path to the SBML file, when resolved
    """

    def __init__(
        self,
        sbml_model: Any,
        measurements: pd.DataFrame,
        parameters: pd.DataFrame,
        observables: Dict[str, Any],
        conditions: pd.DataFrame,
        yaml_path: str,
        raw_problem: Any = None,
        parsed_sbml: Optional[Dict[str, Any]] = None,
        graph_data: Any = None,
        sbml_path: Optional[str] = None,
    ):
        self.sbml_model = sbml_model
        self.measurements = measurements
        self.parameters = parameters
        self.observables = observables
        self.conditions = conditions
        self.yaml_path = yaml_path
        self.raw_problem = raw_problem
        self.parsed_sbml = parsed_sbml
        self.graph_data = graph_data
        self.sbml_path = sbml_path

    @property
    def n_observables(self) -> int:
        """Number of observables in the problem."""
        return len(self.observables)

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the problem."""
        return len(self.parameters)

    @property
    def n_conditions(self) -> int:
        """Number of experimental conditions."""
        if self.conditions is None:
            return 1
        return len(self.conditions)

    @property
    def n_measurements(self) -> int:
        """Number of measurements."""
        return len(self.measurements)


class PEtabLoader:
    """
    Load PEtab problems into a unified format.

    This loader handles the PEtab YAML format and converts it into
    a standardized internal representation.
    """

    def __init__(self):
        self.sbml_parser = SBMLParser()

    def load(self, yaml_path: str, validate: bool = True) -> PEtabProblem:
        """
        Load a PEtab problem from a YAML file.

        Args:
            yaml_path: Path to the PEtab YAML file
            validate: Whether to validate the PEtab problem

        Returns:
            PEtabProblem object
        """
        yaml_path = Path(yaml_path).absolute()
        problem = petab.Problem.from_yaml(str(yaml_path))

        if validate:
            self._validate_problem(problem)

        sbml_model = problem.sbml_model
        measurements = self._copy_df(self._get_problem_df(problem, "measurement_df", "measurements_df"))
        parameters = self._normalize_id_column(
            self._get_problem_df(problem, "parameter_df"), "parameterId"
        )
        observable_df = self._normalize_id_column(
            self._get_problem_df(problem, "observable_df"), "observableId"
        )
        conditions = self._normalize_id_column(
            self._get_problem_df(problem, "condition_df"), "conditionId"
        )

        parsed_sbml = self.sbml_parser.parse_model(sbml_model)
        sbml_path = self._resolve_sbml_path(problem, yaml_path)
        if sbml_path is not None:
            parsed_sbml["metadata"]["source_path"] = str(sbml_path)

        observable_dict = self._observables_to_dict(observable_df)
        self._attach_observables_to_graph(parsed_sbml, observable_df)
        graph_data = sbml_to_hetero_data(parsed_sbml)

        return PEtabProblem(
            sbml_model=sbml_model,
            measurements=measurements,
            parameters=parameters,
            observables=observable_dict,
            conditions=conditions,
            yaml_path=str(yaml_path),
            raw_problem=problem,
            parsed_sbml=parsed_sbml,
            graph_data=graph_data,
            sbml_path=str(sbml_path) if sbml_path is not None else None,
        )

    def load_multiple(
        self, yaml_paths: list[str], validate: bool = True
    ) -> Dict[str, PEtabProblem]:
        """
        Load multiple PEtab problems.

        Args:
            yaml_paths: List of paths to PEtab YAML files
            validate: Whether to validate PEtab problems

        Returns:
            Dictionary mapping problem names to PEtabProblem objects
        """
        problems = {}
        for yaml_path in yaml_paths:
            try:
                problem = self.load(yaml_path, validate=validate)
                problem_name = Path(yaml_path).stem
                problems[problem_name] = problem
            except Exception as e:
                print(f"Failed to load {yaml_path}: {e}")

        return problems

    def get_parameter_bounds(self, problem: PEtabProblem) -> Dict[str, tuple]:
        """
        Extract parameter bounds from a PEtab problem.

        Args:
            problem: PEtabProblem object

        Returns:
            Dictionary mapping parameter IDs to (lower, upper) bounds
        """
        bounds = {}
        parameters = self._normalize_id_column(problem.parameters, "parameterId")
        for _, row in parameters.iterrows():
            param_id = row["parameterId"]
            lower = row.get("lowerBound", -float("inf"))
            upper = row.get("upperBound", float("inf"))
            bounds[param_id] = (lower, upper)

        return bounds

    def get_parameter_scales(self, problem: PEtabProblem) -> Dict[str, str]:
        """
        Extract parameter scales from a PEtab problem.

        Args:
            problem: PEtabProblem object

        Returns:
            Dictionary mapping parameter IDs to scale types ("lin" or "log")
        """
        scales = {}
        parameters = self._normalize_id_column(problem.parameters, "parameterId")
        for _, row in parameters.iterrows():
            param_id = row["parameterId"]
            scale = row.get("parameterScale", "lin")
            scales[param_id] = scale

        return scales

    def get_measurement_conditions(self, problem: PEtabProblem) -> pd.DataFrame:
        """
        Get unique measurement conditions.

        Args:
            problem: PEtabProblem object

        Returns:
            DataFrame with unique conditions
        """
        if problem.measurements is None or problem.measurements.empty:
            return pd.DataFrame()

        conditions = problem.measurements[["simulationConditionId"]].drop_duplicates()
        if "preequilibrationConditionId" in problem.measurements.columns:
            conditions = problem.measurements[
                ["simulationConditionId", "preequilibrationConditionId"]
            ].drop_duplicates()

        return conditions

    def _validate_problem(self, problem: Any) -> None:
        """Validate a PEtab problem with version-tolerant API calls."""
        check_problem = getattr(petab, "check_problem", None)
        if check_problem is None:
            try:
                import petab.v1.lint as petab_lint

                check_problem = getattr(petab_lint, "check_problem", None)
            except ImportError:
                check_problem = None
        if check_problem is None:
            return
        try:
            check_problem(problem, petab.CORE_PARAMETERS.format)
        except TypeError:
            check_problem(problem)

    def _get_problem_df(self, problem: Any, *names: str) -> Optional[pd.DataFrame]:
        """Return the first dataframe attribute available on a PEtab problem."""
        for name in names:
            try:
                return getattr(problem, name)
            except AttributeError:
                continue
        return None

    def _copy_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Return a defensive DataFrame copy."""
        if df is None:
            return pd.DataFrame()
        return df.copy()

    def _normalize_id_column(self, df: Optional[pd.DataFrame], id_col: str) -> pd.DataFrame:
        """Ensure a PEtab identifier exists as a dataframe column."""
        if df is None:
            return pd.DataFrame()
        normalized = df.copy()
        if id_col not in normalized.columns:
            normalized.insert(0, id_col, normalized.index.astype(str))
        return normalized

    def _observables_to_dict(self, observables: pd.DataFrame) -> Dict[str, Any]:
        """Convert PEtab observables dataframe to a dictionary."""
        if observables is None or observables.empty:
            return {}

        observable_dict = {}
        for _, row in observables.iterrows():
            observable_id = row["observableId"]
            formula = row.get("observableFormula", row.get("formula", ""))
            observable_dict[observable_id] = {
                "formula": formula,
                "noise_distribution": row.get("noiseDistribution", ""),
                "noise_parameters": row.get("noiseFormula", row.get("noiseParameters", "")),
                "observable_transformation": row.get("observableTransformation", ""),
            }
        return observable_dict

    def _attach_observables_to_graph(self, parsed_sbml: Dict[str, Any], observables: pd.DataFrame) -> None:
        """Attach PEtab observable nodes and formula-reference edges to the parsed graph."""
        if observables is None or observables.empty:
            return

        species_ids = {node["id"] for node in parsed_sbml.get("species_nodes", [])}
        parameter_ids = {node["id"] for node in parsed_sbml.get("parameter_nodes", [])}
        observable_nodes = parsed_sbml.setdefault("observable_nodes", [])
        edges = parsed_sbml.setdefault("edges", [])

        for _, row in observables.iterrows():
            observable_id = row["observableId"]
            formula = row.get("observableFormula", row.get("formula", ""))
            noise_formula = row.get("noiseFormula", row.get("noiseParameters", ""))
            symbols = self.sbml_parser._extract_formula_symbols(formula=formula)
            noise_symbols = self.sbml_parser._extract_formula_symbols(formula=noise_formula)
            all_symbols = symbols | noise_symbols
            unresolved = sorted(
                symbol
                for symbol in all_symbols
                if symbol not in species_ids and symbol not in parameter_ids
            )

            observable_nodes.append(
                {
                    "id": observable_id,
                    "name": row.get("observableName", observable_id),
                    "formula": formula,
                    "noise_formula": noise_formula,
                    "noise_distribution": row.get("noiseDistribution", ""),
                    "observable_transformation": row.get("observableTransformation", ""),
                    "symbols": sorted(symbols),
                    "unresolved_symbols": unresolved,
                }
            )

            for symbol in symbols:
                if symbol in species_ids:
                    edges.append(
                        self.sbml_parser._edge(
                            symbol,
                            observable_id,
                            "species",
                            "observable",
                            "observed_by",
                        )
                    )
                elif symbol in parameter_ids:
                    edges.append(
                        self.sbml_parser._edge(
                            symbol,
                            observable_id,
                            "parameter",
                            "observable",
                            "used_in_observable",
                        )
                    )

            for symbol in noise_symbols:
                if symbol in parameter_ids:
                    edges.append(
                        self.sbml_parser._edge(
                            symbol,
                            observable_id,
                            "parameter",
                            "observable",
                            "used_in_observable",
                        )
                    )

        parsed_sbml.setdefault("metadata", {})["n_observables"] = len(observable_nodes)

    def _resolve_sbml_path(self, problem: Any, yaml_path: Path) -> Optional[Path]:
        """Best-effort resolution of the SBML file referenced by a PEtab YAML."""
        for attr in ("sbml_files", "model_files"):
            value = getattr(problem, attr, None)
            if value:
                first = value[0] if isinstance(value, (list, tuple)) else value
                return self._resolve_relative_path(first, yaml_path.parent)

        value = getattr(problem, "model_file", None)
        if value:
            return self._resolve_relative_path(value, yaml_path.parent)

        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            for key in ("sbml_files", "model_files", "model_files", "model_file"):
                value = yaml_data.get(key)
                if value:
                    first = value[0] if isinstance(value, list) else value
                    return self._resolve_relative_path(first, yaml_path.parent)
            problems = yaml_data.get("problems") or []
            for entry in problems:
                value = entry.get("sbml_files") or entry.get("model_files") or entry.get("model_file")
                if value:
                    first = value[0] if isinstance(value, list) else value
                    return self._resolve_relative_path(first, yaml_path.parent)
        except Exception:
            return None

        return None

    def _resolve_relative_path(self, value: Any, base_dir: Path) -> Path:
        """Resolve a path-like YAML value against a base directory."""
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()
