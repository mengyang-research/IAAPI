"""Tests for the model-level frozen split manifest (review item P1-9)."""

from __future__ import annotations

import pytest

from iaapi.evaluation.split_manifest import (
    ModelEntry,
    SplitManifest,
    build_manifest,
    validate_no_lineage_leak,
)


def make_entry(mid, role="train", parent=None, **kw):
    return ModelEntry(model_id=mid, role=role, parent_id=parent, **kw)


def test_clean_manifest_no_leak():
    entries = [
        make_entry("boehm", "train"),
        make_entry("boehm_v2", "train", parent="boehm"),
        make_entry("liu", "sealed_test"),
    ]
    manifest = SplitManifest(models=entries)
    assert validate_no_lineage_leak(manifest) == []


def test_lineage_crossing_split_detected():
    entries = [
        make_entry("boehm", "train"),
        make_entry("boehm_v2", "sealed_test", parent="boehm"),  # leak!
    ]
    manifest = SplitManifest(models=entries)
    errors = validate_no_lineage_leak(manifest)
    assert len(errors) == 1
    assert "crosses splits" in errors[0]


def test_build_manifest_raises_on_leak():
    entries = [
        make_entry("a", "train"),
        make_entry("a_v1", "sealed_test", parent="a"),
    ]
    with pytest.raises(ValueError, match="lineage leak"):
        build_manifest(entries)


def test_duplicate_model_id_detected():
    entries = [make_entry("a", "train"), make_entry("a", "sealed_test")]
    manifest = SplitManifest(models=entries)
    errors = validate_no_lineage_leak(manifest)
    assert any("duplicate" in e for e in errors)


def test_missing_parent_detected():
    entries = [make_entry("orphan", "train", parent="ghost")]
    manifest = SplitManifest(models=entries)
    errors = validate_no_lineage_leak(manifest)
    assert any("parent" in e for e in errors)


def test_invalid_role_rejected():
    with pytest.raises(ValueError, match="role"):
        build_manifest([make_entry("a", role="nonsense")])


def test_invalid_supervision_rejected():
    with pytest.raises(ValueError, match="supervision"):
        build_manifest([make_entry("a", supervision=["nope"])])


def test_roundtrip_save_load(tmp_path):
    entries = [make_entry("boehm", "train", supervision=["fim", "mcmc"])]
    manifest = build_manifest(entries, freeze_date="2026-07-24", split_seed=42)
    from iaapi.evaluation.split_manifest import save_manifest, load_manifest

    p = tmp_path / "manifest.json"
    save_manifest(manifest, p)
    loaded = load_manifest(p)
    assert loaded.models[0].model_id == "boehm"
    assert loaded.models[0].supervision == ["fim", "mcmc"]
    assert loaded.freeze_date == "2026-07-24"
    assert loaded.split_seed == 42
