"""
SBML Parser: Parse SBML files into heterogeneous graph representations.

This module handles the parsing of SBML models into a graph structure
suitable for processing by the MASE encoder.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Set
import re

import libsbml


class SBMLParser:
    """
    Parse SBML files into heterogeneous graph representations.

    The parsed graph contains:
    - Species nodes: Variables in the system
    - Reaction nodes: Transformations in the system
    - Parameter nodes: Model parameters
    - Compartment nodes: Model compartments
    - Edges: Reactant, product, modifier, parameter-to-reaction relationships
    """

    def __init__(self):
        self._sbml_reader = libsbml.SBMLReader()

    def parse(self, sbml_path: str) -> Dict[str, Any]:
        """
        Parse an SBML file into a graph representation.

        Args:
            sbml_path: Path to the SBML file

        Returns:
            Dictionary containing parsed nodes, edges, rules, events, and metadata.
        """
        doc = self._sbml_reader.readSBMLFromFile(str(sbml_path))

        if doc.getNumErrors() > 0:
            raise ValueError(f"SBML parsing errors: {doc.printErrors()}")

        model = doc.getModel()
        if model is None:
            raise ValueError("No model found in SBML document")

        parsed = self.parse_model(model)
        parsed["metadata"]["source_path"] = str(sbml_path)
        return parsed

    def parse_model(self, model) -> Dict[str, Any]:
        """Parse an existing libSBML model object."""
        sbml_info = {
            "level": model.getLevel(),
            "version": model.getVersion(),
            "name": model.getName() or model.getId(),
            "id": model.getId(),
        }

        compartment_nodes = self._parse_compartments(model)
        species_nodes, compartment_edges = self._parse_species(model)
        reaction_nodes, reaction_edges, local_parameters = self._parse_reactions(model)
        parameter_nodes = self._parse_parameters(model) + local_parameters
        parameter_edges = self._parse_parameter_edges(reaction_nodes, parameter_nodes)
        events = self._parse_events(model)
        assignment_rules = self._parse_assignment_rules(model)

        edges = compartment_edges + reaction_edges + parameter_edges

        return {
            "species_nodes": species_nodes,
            "reaction_nodes": reaction_nodes,
            "parameter_nodes": parameter_nodes,
            "parameters": parameter_nodes,
            "compartment_nodes": compartment_nodes,
            "observable_nodes": [],
            "edges": edges,
            "events": events,
            "assignment_rules": assignment_rules,
            "metadata": {
                "model": self._get_sbase_metadata(model),
                "n_species": len(species_nodes),
                "n_reactions": len(reaction_nodes),
                "n_parameters": len(parameter_nodes),
                "n_compartments": len(compartment_nodes),
            },
            "sbml_info": sbml_info,
        }

    def _parse_compartments(self, model) -> List[Dict[str, Any]]:
        """Parse compartments from an SBML model."""
        compartments = []
        for i in range(model.getNumCompartments()):
            compartment = model.getCompartment(i)
            compartments.append(
                {
                    "id": compartment.getId(),
                    "name": compartment.getName() or compartment.getId(),
                    "size": self._safe_get_value(compartment, "getSize"),
                    "spatial_dimensions": self._safe_get_value(
                        compartment, "getSpatialDimensions"
                    ),
                    "units": self._safe_get_value(compartment, "getUnits"),
                    "constant": self._safe_get_value(compartment, "getConstant"),
                    "metadata": self._get_sbase_metadata(compartment),
                }
            )
        return compartments

    def _parse_species(self, model) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse species from an SBML model."""
        species_nodes = []
        compartment_edges = []
        compartment_ids = {model.getCompartment(i).getId() for i in range(model.getNumCompartments())}

        for i in range(model.getNumSpecies()):
            species = model.getSpecies(i)
            species_id = species.getId()
            compartment_id = species.getCompartment()
            species_nodes.append(
                {
                    "id": species_id,
                    "name": species.getName() or species_id,
                    "compartment": compartment_id,
                    "initial_amount": species.getInitialAmount(),
                    "initial_concentration": species.getInitialConcentration(),
                    "boundary_condition": species.getBoundaryCondition(),
                    "constant": species.getConstant(),
                    "has_only_substance_units": species.getHasOnlySubstanceUnits(),
                    "metadata": self._get_sbase_metadata(species),
                }
            )
            if compartment_id in compartment_ids:
                compartment_edges.append(
                    self._edge(
                        species_id,
                        compartment_id,
                        "species",
                        "compartment",
                        "in_compartment",
                    )
                )
        return species_nodes, compartment_edges

    def _parse_reactions(self, model) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse reactions, reaction edges, and local parameters."""
        reaction_nodes = []
        edges = []
        local_parameters = []
        species_ids = {model.getSpecies(i).getId() for i in range(model.getNumSpecies())}

        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId() or f"reaction_{i}"
            kinetic_law_info, local_nodes = self._parse_kinetic_law(reaction, reaction_id)
            local_parameters.extend(local_nodes)

            reaction_nodes.append(
                {
                    "id": reaction_id,
                    "name": reaction.getName() or reaction_id,
                    "reversible": reaction.getReversible(),
                    "fast": reaction.getFast(),
                    "kinetic_law": kinetic_law_info,
                    "num_reactants": reaction.getNumReactants(),
                    "num_products": reaction.getNumProducts(),
                    "num_modifiers": reaction.getNumModifiers(),
                    "metadata": self._get_sbase_metadata(reaction),
                }
            )

            for j in range(reaction.getNumReactants()):
                reactant = reaction.getReactant(j)
                species_ref = reactant.getSpecies()
                if species_ref in species_ids:
                    edges.append(
                        self._edge(
                            species_ref,
                            reaction_id,
                            "species",
                            "reaction",
                            "reactant",
                            stoichiometry=self._stoichiometry(reactant),
                            constant=self._safe_get_value(reactant, "getConstant"),
                        )
                    )

            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                species_ref = product.getSpecies()
                if species_ref in species_ids:
                    edges.append(
                        self._edge(
                            reaction_id,
                            species_ref,
                            "reaction",
                            "species",
                            "product",
                            stoichiometry=self._stoichiometry(product),
                            constant=self._safe_get_value(product, "getConstant"),
                        )
                    )

            for j in range(reaction.getNumModifiers()):
                modifier = reaction.getModifier(j)
                species_ref = modifier.getSpecies()
                if species_ref in species_ids:
                    edges.append(
                        self._edge(
                            species_ref,
                            reaction_id,
                            "species",
                            "reaction",
                            "modifier",
                        )
                    )

        return reaction_nodes, edges, local_parameters

    def _parse_kinetic_law(self, reaction, reaction_id: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse kinetic law and local parameters for a reaction."""
        kinetic_law = reaction.getKineticLaw()
        if kinetic_law is None:
            return None, []

        formula = self._formula_from_math(kinetic_law.getMath()) or kinetic_law.getFormula()
        symbols = sorted(self._extract_formula_symbols(kinetic_law.getMath(), formula))
        local_parameters = []

        for param in self._iter_kinetic_law_parameters(kinetic_law):
            base_id = param.getId()
            scoped_id = f"{reaction_id}:{base_id}"
            local_parameters.append(
                {
                    "id": scoped_id,
                    "base_id": base_id,
                    "name": param.getName() or base_id,
                    "value": param.getValue(),
                    "constant": True,
                    "type": "local",
                    "reaction_id": reaction_id,
                    "metadata": self._get_sbase_metadata(param),
                }
            )

        return (
            {
                "type": self._safe_get_value(kinetic_law, "getType"),
                "formula": formula,
                "symbols": symbols,
                "num_params": len(local_parameters),
                "classification": self._classify_kinetic_law(formula),
            },
            local_parameters,
        )

    def _iter_kinetic_law_parameters(self, kinetic_law):
        """Yield local kinetic-law parameters across libSBML API versions."""
        yielded = False
        if hasattr(kinetic_law, "getNumLocalParameters"):
            for i in range(kinetic_law.getNumLocalParameters()):
                yielded = True
                yield kinetic_law.getLocalParameter(i)
        if not yielded and hasattr(kinetic_law, "getNumParameters"):
            for i in range(kinetic_law.getNumParameters()):
                yield kinetic_law.getParameter(i)

    def _parse_parameters(self, model) -> List[Dict[str, Any]]:
        """Parse global parameters from SBML model."""
        parameter_nodes = []
        for i in range(model.getNumParameters()):
            param = model.getParameter(i)
            parameter_nodes.append(
                {
                    "id": param.getId(),
                    "name": param.getName() or param.getId(),
                    "value": param.getValue(),
                    "constant": param.getConstant(),
                    "type": "global",
                    "metadata": self._get_sbase_metadata(param),
                }
            )
        return parameter_nodes

    def _parse_parameter_edges(
        self, reaction_nodes: List[Dict[str, Any]], parameter_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse parameter-to-reaction edges from kinetic-law symbols."""
        edges = []
        global_ids = {p["id"] for p in parameter_nodes if p.get("type") == "global"}
        local_by_reaction = {
            (p.get("reaction_id"), p.get("base_id")): p["id"]
            for p in parameter_nodes
            if p.get("type") == "local"
        }

        for reaction in reaction_nodes:
            reaction_id = reaction["id"]
            kinetic_law = reaction.get("kinetic_law") or {}
            for symbol in kinetic_law.get("symbols", []):
                source = None
                local_key = (reaction_id, symbol)
                if local_key in local_by_reaction:
                    source = local_by_reaction[local_key]
                elif symbol in global_ids:
                    source = symbol
                if source is not None:
                    edges.append(
                        self._edge(
                            source,
                            reaction_id,
                            "parameter",
                            "reaction",
                            "used_in",
                        )
                    )
        return edges

    def _parse_events(self, model) -> List[Dict[str, Any]]:
        """Parse events from SBML model."""
        events = []
        for i in range(model.getNumEvents()):
            event = model.getEvent(i)
            trigger = event.getTrigger() if event.isSetTrigger() else None
            events.append(
                {
                    "id": event.getId() or f"event_{i}",
                    "name": event.getName() or event.getId() or f"event_{i}",
                    "trigger_formula": self._formula_from_math(trigger.getMath()) if trigger else None,
                    "use_values_from_trigger_time": event.getUseValuesFromTriggerTime(),
                    "metadata": self._get_sbase_metadata(event),
                }
            )
        return events

    def _parse_assignment_rules(self, model) -> List[Dict[str, Any]]:
        """Parse assignment rules from SBML model."""
        rules = []
        for i in range(model.getNumRules()):
            rule = model.getRule(i)
            if rule.getTypeCode() == libsbml.SBML_ASSIGNMENT_RULE:
                formula = self._formula_from_math(rule.getMath()) or rule.getFormula()
                rules.append(
                    {
                        "variable": rule.getVariable(),
                        "formula": formula,
                        "symbols": sorted(self._extract_formula_symbols(rule.getMath(), formula)),
                        "metadata": self._get_sbase_metadata(rule),
                    }
                )
        return rules

    def _edge(
        self,
        source: str,
        target: str,
        source_type: str,
        target_type: str,
        edge_type: str,
        **attrs,
    ) -> Dict[str, Any]:
        """Build a structured edge dictionary."""
        edge = {
            "source": source,
            "target": target,
            "source_type": source_type,
            "target_type": target_type,
            "type": edge_type,
        }
        edge.update(attrs)
        return edge

    def _stoichiometry(self, species_reference) -> float:
        """Return stoichiometry, defaulting to 1.0 when unset."""
        if hasattr(species_reference, "isSetStoichiometry") and species_reference.isSetStoichiometry():
            return species_reference.getStoichiometry()
        return 1.0

    def _get_sbase_metadata(self, sbase) -> Dict[str, Any]:
        """Extract best-effort metadata from a libSBML SBase object."""
        metadata = {}
        for key, method in (
            ("meta_id", "getMetaId"),
            ("sbo_term", "getSBOTermID"),
            ("notes", "getNotesString"),
            ("annotation", "getAnnotationString"),
        ):
            value = self._safe_get_value(sbase, method)
            if value not in (None, ""):
                metadata[key] = value
        return metadata

    def _safe_get_value(self, obj, method_name: str) -> Any:
        """Call a libSBML getter if present, returning None on failure."""
        method = getattr(obj, method_name, None)
        if method is None:
            return None
        try:
            return method()
        except Exception:
            return None

    def _formula_from_math(self, math_ast) -> Optional[str]:
        """Convert a libSBML AST node to an infix formula string."""
        if math_ast is None:
            return None
        try:
            formula = libsbml.formulaToString(math_ast)
        except Exception:
            return None
        return formula or None

    def _extract_formula_symbols(self, math_ast=None, formula: Optional[str] = None) -> Set[str]:
        """Extract symbol names from an AST node or formula string."""
        symbols = set()
        if math_ast is not None:
            self._collect_ast_names(math_ast, symbols)
        if formula:
            tokens = re.findall(r"\b[A-Za-z_]\w*\b", formula)
            symbols.update(
                token
                for token in tokens
                if token not in {"and", "or", "not", "true", "false", "pow", "exp", "log"}
            )
        return symbols

    def _collect_ast_names(self, node, symbols: Set[str]) -> None:
        """Recursively collect names from a libSBML AST node."""
        try:
            if node.isName():
                symbols.add(node.getName())
        except Exception:
            pass
        try:
            n_children = node.getNumChildren()
        except Exception:
            n_children = 0
        for i in range(n_children):
            self._collect_ast_names(node.getChild(i), symbols)

    def _classify_kinetic_law(self, formula: Optional[str]) -> str:
        """Conservatively classify a kinetic-law formula."""
        if not formula:
            return "unknown"
        normalized = formula.lower().replace(" ", "")
        if re.fullmatch(r"[a-z_]\w*|[0-9.eE+-]+", normalized):
            return "constant_flux"
        if "^" in normalized or "pow" in normalized:
            return "hill_like"
        if "/" in normalized and ("+" in normalized or "km" in normalized):
            return "michaelis_menten_like"
        if "*" in normalized and "/" not in normalized:
            return "mass_action_like"
        return "custom"


def sbml_to_hetero_data(parsed_sbml: Dict[str, Any]) -> Any:
    """
    Convert parsed SBML graph to PyTorch Geometric HeteroData.

    Args:
        parsed_sbml: Output from SBMLParser.parse()

    Returns:
        torch_geometric.data.HeteroData object
    """
    try:
        from torch_geometric.data import HeteroData
        import torch
    except ImportError:
        raise ImportError("torch-geometric is required for HeteroData conversion")

    hetero_data = HeteroData()
    node_maps = {}

    node_specs = {
        "species": parsed_sbml.get("species_nodes", []),
        "reaction": parsed_sbml.get("reaction_nodes", []),
        "parameter": parsed_sbml.get("parameter_nodes", []),
        "compartment": parsed_sbml.get("compartment_nodes", []),
        "observable": parsed_sbml.get("observable_nodes", []),
    }

    for node_type, nodes in node_specs.items():
        if not nodes:
            continue
        ids = [node["id"] for node in nodes]
        node_maps[node_type] = {node_id: i for i, node_id in enumerate(ids)}
        hetero_data[node_type].node_id = ids
        hetero_data[node_type].num_nodes = len(ids)

    grouped_edges: Dict[Tuple[str, str, str], List[List[int]]] = {}
    grouped_stoich: Dict[Tuple[str, str, str], List[float]] = {}

    for edge in parsed_sbml.get("edges", []):
        edge = _normalize_edge(edge)
        source_type = edge["source_type"]
        target_type = edge["target_type"]
        source_map = node_maps.get(source_type, {})
        target_map = node_maps.get(target_type, {})
        if edge["source"] not in source_map or edge["target"] not in target_map:
            continue
        key = (source_type, edge["type"], target_type)
        grouped_edges.setdefault(key, []).append(
            [source_map[edge["source"]], target_map[edge["target"]]]
        )
        if "stoichiometry" in edge:
            grouped_stoich.setdefault(key, []).append(float(edge["stoichiometry"]))

    for key, pairs in grouped_edges.items():
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        hetero_data[key].edge_index = edge_index
        if key in grouped_stoich and len(grouped_stoich[key]) == len(pairs):
            hetero_data[key].stoichiometry = torch.tensor(grouped_stoich[key], dtype=torch.float)

    return hetero_data


def _normalize_edge(edge: Any) -> Dict[str, Any]:
    """Normalize legacy tuple edges to structured edge dictionaries."""
    if isinstance(edge, dict):
        return edge
    source, target, edge_type = edge
    if edge_type == "product":
        return {
            "source": source,
            "target": target,
            "source_type": "reaction",
            "target_type": "species",
            "type": edge_type,
        }
    if edge_type in {"reactant", "modifier"}:
        return {
            "source": source,
            "target": target,
            "source_type": "species",
            "target_type": "reaction",
            "type": edge_type,
        }
    return {
        "source": source,
        "target": target,
        "source_type": "parameter",
        "target_type": "reaction",
        "type": edge_type,
    }
