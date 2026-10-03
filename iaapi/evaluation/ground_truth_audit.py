"""Consistency audit for profile-likelihood ground-truth artifacts.

Scientific targets are emitted only for models whose parameters were classified
by converged profile-likelihood optimizations. Timeouts and simulator failures are
missing labels; they are never interpreted as non-identifiability.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping


CLASSIFIED_CODES = frozenset({"FINITE_CI", "BOUNDARY_HIT"})


@dataclass(frozen=True)
class AuditConfig:
    min_classified_fraction: float = 0.80
    require_converged: bool = True


def audit_pl_summary(
    summary: Mapping[str, Any], config: AuditConfig = AuditConfig()
) -> dict[str, Any]:
    """Audit one PL summary without converting failed parameters to class 0."""
    results = list(summary.get("results") or [])
    model = str(summary.get("model") or "UNKNOWN")
    n_total = int(summary.get("n_param") or len(results))
    codes = Counter(str(item.get("code") or "MISSING_CODE") for item in results)

    raw_classified = [item for item in results if item.get("code") in CLASSIFIED_CODES]
    if config.require_converged:
        admissible = [item for item in raw_classified if item.get("converged") is True]
    else:
        admissible = raw_classified

    raw_fraction = len(raw_classified) / n_total if n_total else 0.0
    admissible_fraction = len(admissible) / n_total if n_total else 0.0
    converged_fraction = (
        sum(item.get("converged") is True for item in raw_classified) / len(raw_classified)
        if raw_classified
        else 0.0
    )

    reasons: list[str] = []
    if n_total <= 0:
        reasons.append("no_parameters")
    if len(results) != n_total:
        reasons.append(f"result_count_mismatch:{len(results)}/{n_total}")
    if not raw_classified:
        reasons.append("no_classified_parameters")
    if config.require_converged and raw_classified and not admissible:
        reasons.append("no_converged_classifications")
    if admissible_fraction < config.min_classified_fraction:
        reasons.append(
            f"admissible_coverage_below_threshold:{admissible_fraction:.6f}"
            f"<{config.min_classified_fraction:.6f}"
        )

    if admissible_fraction >= config.min_classified_fraction and admissible:
        status = "VALID"
    elif admissible:
        status = "PARTIAL"
    else:
        status = "INVALID"

    finite = sum(item.get("code") == "FINITE_CI" for item in admissible)
    boundary = sum(item.get("code") == "BOUNDARY_HIT" for item in admissible)
    target = None
    if status == "VALID":
        target = {
            "identifiable_count": finite,
            "classified_count": len(admissible),
            "identifiable_fraction": finite / len(admissible),
            "denominator": "converged_classified_parameters",
        }

    return {
        "model": model,
        "status": status,
        "n_parameters": n_total,
        "n_results": len(results),
        "code_counts": dict(sorted(codes.items())),
        "raw_classified_count": len(raw_classified),
        "raw_classified_fraction": raw_fraction,
        "converged_classified_count": len(admissible),
        "admissible_classified_fraction": admissible_fraction,
        "converged_fraction_among_raw_classified": converged_fraction,
        "failure_or_missing_count": n_total - len(admissible),
        "reasons": reasons,
        "target": target,
    }


def summarize_audits(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["status"]) for row in rows)
    valid_targets = [row["target"] for row in rows if row.get("target") is not None]
    return {
        "n_models": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "n_scientific_targets": len(valid_targets),
        "all_failures_preserved": all(
            row.get("target") is None
            for row in rows
            if row.get("raw_classified_count", 0) == 0
        ),
    }
