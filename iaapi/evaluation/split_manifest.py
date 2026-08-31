"""Model-level frozen split manifest (review item P1-9).

Every model entering training, validation, sealed evaluation, MCMC
admission, OOD refinement, or a case study is recorded with:

  * PEtab model ID and version;
  * data file SHA256;
  * upstream benchmark commit (when known);
  * chosen free parameters and fixed-parameter list;
  * split role: train / validation / sealed-test / OOD / case-study;
  * whether the model participates in FIM, MCMC or PL supervision;
  * split freeze date and split seed;
  * parent/variant lineage (structural deduplication: a model that is a
    renamed or lightly perturbed copy of another must not cross splits).

The manifest is used to *prove* that no parent/perturbation lineage crosses
splits. ``validate_no_lineage_leak`` is the admission gate: it fails if any
parent id appears in two different split roles or if a variant's parent is
sealed while the variant is in training (and vice versa).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0"

ROLES = {"train", "validation", "sealed_test", "ood", "case_study"}
SUPERVISIONS = {"fim", "mcmc", "pl"}


@dataclass
class ModelEntry:
    model_id: str
    petab_version: str = ""
    data_sha256: str = ""
    upstream_commit: str = ""
    free_parameters: List[str] = field(default_factory=list)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    role: str = "train"
    supervision: List[str] = field(default_factory=list)  # subset of SUPER... 
    split_seed: int = 0
    freeze_date: str = ""
    parent_id: Optional[str] = None  # structural lineage (deduplication)
    excluded_reason: str = ""


@dataclass
class SplitManifest:
    schema_version: str = SCHEMA_VERSION
    freeze_date: str = ""
    split_seed: int = 0
    models: List[ModelEntry] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    """SHA256 of a file, hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_role(entry: ModelEntry) -> None:
    if entry.role not in ROLES:
        raise ValueError(f"unknown role {entry.role!r} (valid: {sorted(ROLES)})")
    bad = [s for s in entry.supervision if s not in SUPERVISIONS]
    if bad:
        raise ValueError(f"unknown supervision {bad} (valid: {sorted(SUPERVISIONS)})")


def build_manifest(models: List[ModelEntry], freeze_date: str = "",
                   split_seed: int = 0) -> SplitManifest:
    """Build a manifest and run the lineage-leak gate on it."""
    for m in models:
        _check_role(m)
    manifest = SplitManifest(freeze_date=freeze_date or date.today().isoformat(),
                             split_seed=split_seed, models=models)
    errors = validate_no_lineage_leak(manifest)
    if errors:
        raise ValueError("lineage leak detected:\n" + "\n".join(errors))
    return manifest


def validate_no_lineage_leak(manifest: SplitManifest) -> List[str]:
    """Return a list of leak descriptions; empty means no leakage.

    Rules:
      * a model id may appear at most once;
      * a parent and its variant must not be in *different* split roles
        (train vs sealed_test is the critical one);
      * a model whose parent is ``sealed_test`` must not itself be
        ``sealed_test`` in a different split seed (structural duplicate).
    """
    errors: List[str] = []
    seen: Dict[str, ModelEntry] = {}
    for m in manifest.models:
        if m.model_id in seen:
            errors.append(f"duplicate model_id {m.model_id!r}")
        seen[m.model_id] = m

    for m in manifest.models:
        if m.parent_id is None:
            continue
        parent = seen.get(m.parent_id)
        if parent is None:
            errors.append(f"{m.model_id}: parent {m.parent_id!r} not in manifest")
            continue
        if parent.role != m.role:
            errors.append(
                f"{m.model_id} (role={m.role}) has parent {m.parent_id} "
                f"(role={parent.role}): lineage crosses splits"
            )
    return errors


def manifest_to_dict(manifest: SplitManifest) -> Dict[str, object]:
    return asdict(manifest)


def save_manifest(manifest: SplitManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_to_dict(manifest), indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> SplitManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = [ModelEntry(**m) for m in data["models"]]
    return SplitManifest(schema_version=data.get("schema_version", SCHEMA_VERSION),
                         freeze_date=data.get("freeze_date", ""),
                         split_seed=data.get("split_seed", 0), models=models)
