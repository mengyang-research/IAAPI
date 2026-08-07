"""Claim freeze (NCS plan task P6-01).

Promotes ONLY claims whose evidence-registry gates pass, and labels circular,
exploratory and failed results accordingly. Rule-based and data-driven so the
freeze decision is auditable and unit-testable.

Promotion rules (from 03_EVIDENCE_REGISTRY.md): a claim may be PROMOTED only if
the metric definition is frozen, the artifact has provenance, the evaluation uses
the frozen whole-model split, failures/exclusions are reported, an independent
reproduction command succeeds, and the manuscript value matches the artifact.

Freeze statuses:
  PROMOTED     gate passed + independent + frozen split + artifact matches
  FROZEN       PROMOTED and locked (no further drift; e.g. control results)
  CIRCULAR     FIM-derived predictors vs FIM-derived targets (not independent)
  EXPLORATORY  small n / unconcluded / not yet gated
  FAILED       gate failed (significance / admission / coverage)
  BLOCKED      awaiting a user scientific-route decision or a pending heavy run
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0"

PROMOTED = "PROMOTED"
FROZEN = "FROZEN"
CIRCULAR = "CIRCULAR"
EXPLORATORY = "EXPLORATORY"
FAILED = "FAILED"
BLOCKED = "BLOCKED"

VALID_STATUSES = {PROMOTED, FROZEN, CIRCULAR, EXPLORATORY, FAILED, BLOCKED}


@dataclass
class ClaimSpec:
    """Inputs for a freeze decision (assessed per the evidence registry + gates)."""
    id: str
    claim: str
    gate_passed: bool = False
    independent: bool = True          # False if FIM-vs-FIM (circular)
    circular: bool = False            # explicit circular flag
    split_frozen: bool = False        # uses the frozen whole-model split
    artifact_matches: bool = False    # manuscript value == artifact
    exploratory: bool = False
    blocked_reason: str = ""          # non-empty => BLOCKED
    artifact: str = ""
    gate_evidence: str = ""


@dataclass
class ClaimFreezeEntry:
    id: str
    claim: str
    freeze_status: str
    reason: str
    artifact: str
    gate_evidence: str


def freeze_claim(spec: ClaimSpec) -> ClaimFreezeEntry:
    """Apply the promotion rules to one claim spec -> a freeze entry."""
    if spec.blocked_reason:
        status, reason = BLOCKED, spec.blocked_reason
    elif spec.circular or not spec.independent:
        status, reason = CIRCULAR, "FIM-derived predictor vs FIM-derived target (not independent)"
    elif spec.exploratory:
        status, reason = EXPLORATORY, "exploratory / unconcluded / small n"
    elif not spec.gate_passed:
        status, reason = FAILED, "gate failed (significance / admission / coverage)"
    elif not (spec.split_frozen and spec.artifact_matches):
        status, reason = BLOCKED, "frozen split or artifact-match not satisfied"
    else:
        status, reason = PROMOTED, "gate passed + independent + frozen split + artifact matches"
    return ClaimFreezeEntry(id=spec.id, claim=spec.claim, freeze_status=status,
                            reason=reason, artifact=spec.artifact,
                            gate_evidence=spec.gate_evidence)


def freeze_all(specs: List[ClaimSpec]) -> Dict[str, object]:
    entries = [freeze_claim(s) for s in specs]
    by_status: Dict[str, int] = {}
    for e in entries:
        by_status[e.freeze_status] = by_status.get(e.freeze_status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "n_claims": len(entries),
        "counts_by_status": by_status,
        "claims": [{"id": e.id, "claim": e.claim, "freeze_status": e.freeze_status,
                    "reason": e.reason, "artifact": e.artifact,
                    "gate_evidence": e.gate_evidence} for e in entries],
        "promotion_rule": "PROMOTED only if gate_passed + independent + split_frozen + artifact_matches",
    }


def lock_promoted_to_frozen(manifest: Dict[str, object], lock_ids: List[str]) -> Dict[str, object]:
    """Promote->FROZEN for explicitly locked control claims (e.g. OED-1 control)."""
    lock = set(lock_ids)
    for c in manifest["claims"]:
        if c["id"] in lock and c["freeze_status"] == PROMOTED:
            c["freeze_status"] = FROZEN
            c["reason"] = "frozen control result (locked)"
    manifest["counts_by_status"] = {}
    for c in manifest["claims"]:
        manifest["counts_by_status"][c["freeze_status"]] = manifest["counts_by_status"].get(c["freeze_status"], 0) + 1
    return manifest
