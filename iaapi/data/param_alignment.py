"""Strict PEtab parameter-token alignment (NCS plan task P3-01).

Guarantees that theta, bounds, FIM, graph nodes and posterior outputs all use
ONE canonical PEtab estimated-parameter order, enforced by explicit ID mappings
and assertions — never inferred from graph traversal.

A :class:`ParameterAlignment` is the single source of truth for the canonical
order of free (estimated) PEtab parameters. Vectors/matrices/tokens are aligned
TO it (and re-aligned when they arrive in any other order). Duplicate or missing
parameter IDs fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"


class AlignmentError(ValueError):
    """Raised when theta/bounds/FIM/tokens are inconsistent with the alignment."""


@dataclass(frozen=True)
class LabeledFIM:
    """A FIM whose rows and columns carry the canonical parameter IDs."""
    matrix: np.ndarray
    param_ids: Tuple[str, ...]

    @property
    def n_parameters(self) -> int:
        return len(self.param_ids)


@dataclass
class LabeledPosteriorTokens:
    """Posterior tokens, each tagged with its parameter ID (one per free param)."""
    tokens: List[Any] = field(default_factory=list)
    param_ids: List[str] = field(default_factory=list)

    @property
    def n_parameters(self) -> int:
        return len(self.param_ids)


class ParameterAlignment:
    """Canonical order of free PEtab parameters + alignment assertions.

    Args:
        param_ids: the canonical free-parameter IDs in PEtab order. Must be
            unique and non-empty. This is the ONLY order used downstream.
    """

    def __init__(self, param_ids: Sequence[str]):
        ids = list(param_ids)
        if not ids:
            raise AlignmentError("parameter_ids is empty")
        if len(set(ids)) != len(ids):
            dup = [x for x in ids if ids.count(x) > 1]
            raise AlignmentError(f"duplicate parameter IDs: {sorted(set(dup))}")
        self.param_ids: Tuple[str, ...] = tuple(ids)
        self._index: Dict[str, int] = {pid: i for i, pid in enumerate(ids)}
        self.n_parameters = len(ids)

    # ---- constructors -------------------------------------------------- #
    @classmethod
    def from_simulator(cls, sim: Any) -> "ParameterAlignment":
        """Build from an AMICISimulator (uses its PEtab free-parameter order)."""
        ids = list(getattr(sim, "parameter_ids", []))
        if not ids:
            raise AlignmentError("simulator has no parameter_ids")
        return cls(ids)

    @classmethod
    def from_petab_problem(cls, problem: Any) -> "ParameterAlignment":
        """Build from an IA-API PEtabProblem (estimated params in df order)."""
        params = getattr(problem, "parameters", None)
        if params is None:
            raise AlignmentError("problem has no parameters")
        df = params
        if "parameterId" in getattr(df, "columns", []):
            ids = list(df["parameterId"].astype(str))
        else:
            ids = [str(i) for i in df.index]
        if "estimate" in getattr(df, "columns", []):
            mask = df["estimate"].astype(bool)
            ids = [pid for pid, m in zip(ids, mask) if bool(m)]
        return cls(ids)

    # ---- indexing ------------------------------------------------------ #
    def index_of(self, param_id: str) -> int:
        if param_id not in self._index:
            raise AlignmentError(f"unknown parameter ID: {param_id!r}")
        return self._index[param_id]

    def id_at(self, index: int) -> str:
        return self.param_ids[index]

    # ---- alignment of vectors ----------------------------------------- #
    def align_vector(self, vec: Sequence[float], ids: Sequence[str]) -> np.ndarray:
        """Reorder ``vec`` (given in ``ids`` order) into canonical order.

        Raises on missing or extra IDs (never silently drops/reorders).
        """
        vec = np.asarray(vec, dtype=float).reshape(-1)
        ids = list(ids)
        if len(ids) != len(vec):
            raise AlignmentError(
                f"vector length {len(vec)} != ids length {len(ids)}")
        missing = [p for p in self.param_ids if p not in ids]
        extra = [p for p in ids if p not in self._index]
        if missing:
            raise AlignmentError(f"missing parameter IDs: {missing}")
        if extra:
            raise AlignmentError(f"extra parameter IDs not in alignment: {extra}")
        pos = [ids.index(p) for p in self.param_ids]
        return vec[pos]

    def assert_vector(self, vec: Sequence[float]) -> np.ndarray:
        """Assert a vector is already in canonical order (length == n)."""
        v = np.asarray(vec, dtype=float).reshape(-1)
        if v.shape[0] != self.n_parameters:
            raise AlignmentError(
                f"vector length {v.shape[0]} != canonical {self.n_parameters}")
        return v

    # ---- FIM ----------------------------------------------------------- #
    def label_fim(self, fim: np.ndarray) -> LabeledFIM:
        """Wrap a FIM matrix with the canonical parameter IDs (rows=cols)."""
        m = np.asarray(fim, dtype=float)
        if m.ndim != 2 or m.shape != (self.n_parameters, self.n_parameters):
            raise AlignmentError(
                f"FIM shape {m.shape} != ({self.n_parameters},{self.n_parameters})")
        return LabeledFIM(matrix=m, param_ids=self.param_ids)

    def assert_fim(self, fim: np.ndarray) -> LabeledFIM:
        return self.label_fim(fim)

    # ---- bounds -------------------------------------------------------- #
    def assert_bounds(self, bounds: np.ndarray) -> np.ndarray:
        b = np.asarray(bounds, dtype=float)
        if b.ndim != 2 or b.shape != (self.n_parameters, 2):
            raise AlignmentError(
                f"bounds shape {b.shape} != ({self.n_parameters},2)")
        return b

    # ---- posterior tokens --------------------------------------------- #
    def label_posterior_tokens(self, tokens: Sequence[Any], ids: Sequence[str]) -> LabeledPosteriorTokens:
        """Tag posterior tokens with parameter IDs; assert they cover the alignment
        exactly (one token per free parameter, no missing/extra)."""
        ids = list(ids)
        missing = [p for p in self.param_ids if p not in ids]
        extra = [p for p in ids if p not in self._index]
        if missing:
            raise AlignmentError(f"posterior tokens missing IDs: {missing}")
        if extra:
            raise AlignmentError(f"posterior tokens extra IDs: {extra}")
        if len(tokens) != len(ids):
            raise AlignmentError(
                f"tokens count {len(tokens)} != ids count {len(ids)}")
        # reorder to canonical
        pos = [ids.index(p) for p in self.param_ids]
        return LabeledPosteriorTokens(
            tokens=[tokens[i] for i in pos], param_ids=list(self.param_ids))

    def assert_posterior_tokens(self, labeled: LabeledPosteriorTokens) -> LabeledPosteriorTokens:
        if list(labeled.param_ids) != list(self.param_ids):
            raise AlignmentError("posterior tokens not in canonical order")
        if len(labeled.tokens) != self.n_parameters:
            raise AlignmentError("posterior tokens count mismatch")
        return labeled

    # ---- round-trip helper -------------------------------------------- #
    def round_trip(self, vec: Sequence[float]) -> bool:
        """Permute a canonical vector by a shuffled ID order, re-align, and check
        identity. Returns True iff the round-trip recovers the input exactly."""
        v = self.assert_vector(vec)
        perm = np.random.default_rng(0).permutation(self.n_parameters)
        shuffled_ids = [self.param_ids[i] for i in perm]
        shuffled_vec = v[perm]
        recovered = self.align_vector(shuffled_vec, shuffled_ids)
        return bool(np.array_equal(recovered, v))
