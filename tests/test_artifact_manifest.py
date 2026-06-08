"""Tests for artifact manifest output."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "reports" / "artifact_manifest.csv"
MANIFEST_MD = ROOT / "reports" / "summary" / "artifact_manifest.md"

REQUIRED_COLUMNS = [
    "artifact_path",
    "artifact_type",
    "created_at",
    "source_script",
    "required_for_thesis",
    "artifact_path",
    "artifact_type",
    "created_at",
    "source_script",
    "required_for_thesis",
    "chapter",
    "notes",
    "exists_on_disk",
]

import importlib.util

@pytest.fixture(scope="module")
def manifest_files(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("manifest")
    reports_dir = tmp_path / "reports"
    summary_dir = reports_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = ROOT / "scripts" / "generate_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_artifact_manifest", str(script_path))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    
    old_reports = gen.REPORTS_DIR
    old_summary = gen.SUMMARY_DIR
    
    gen.REPORTS_DIR = reports_dir
    gen.SUMMARY_DIR = summary_dir
    try:
        gen.main()
    finally:
        gen.REPORTS_DIR = old_reports
        gen.SUMMARY_DIR = old_summary
        
    return reports_dir / "artifact_manifest.csv", summary_dir / "artifact_manifest.md"


def test_manifest_csv_exists_and_has_columns(manifest_files) -> None:
    csv_path, _ = manifest_files
    df = pd.read_csv(csv_path)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_manifest_no_null_artifact_path(manifest_files) -> None:
    csv_path, _ = manifest_files
    df = pd.read_csv(csv_path)
    assert df["artifact_path"].notna().all(), "Some artifact_path values are null"
    assert (df["artifact_path"] != "").all(), "Some artifact_path values are empty"

@pytest.fixture
def dummy_manifest_df():
    return pd.DataFrame({
        "artifact_path": ["present.csv", "missing.csv"],
        "required_for_thesis": [True, True],
        "exists_on_disk": [True, False],
        "created_at": ["2023-01-01", "2023-01-01"],
        "artifact_type": ["dataset", "dataset"],
        "source_script": ["script1", "script2"],
        "chapter": ["1", "1"],
        "notes": ["", ""]
    })

@pytest.mark.acceptance
def test_required_thesis_artifacts_exist_on_disk(dummy_manifest_df, tmp_path) -> None:
    import os
    fixture_root = os.environ.get("AAC_ACCEPTANCE_FIXTURE_ROOT")
    
    if fixture_root:
        # Run against dummy fixture
        fixture_csv = Path(fixture_root) / "artifact_manifest.csv"
        fixture_csv.parent.mkdir(parents=True, exist_ok=True)
        dummy_manifest_df.to_csv(fixture_csv, index=False)
        manifest_path = fixture_csv
    else:
        # Canonical run
        manifest_path = MANIFEST_CSV

    if not manifest_path.exists():
        if os.environ.get("AAC_ACCEPTANCE") == "1":
            pytest.fail("Manifest is required in acceptance mode")
        else:
            pytest.skip("No manifest available")
            
    df = pd.read_csv(manifest_path)
    required = df[df["required_for_thesis"] == True]
    missing = required[required["exists_on_disk"] != True]
    
    if os.environ.get("AAC_ACCEPTANCE") == "1" or fixture_root:
        assert missing.empty, f"Missing required artifacts: {missing['artifact_path'].tolist()}"


def test_manifest_md_exists(manifest_files) -> None:
    _, md_path = manifest_files
    content = md_path.read_text(encoding="utf-8")
    assert "# Thesis Artifact Manifest" in content
    assert "Chapter" in content


def test_manifest_md_has_no_mojibake_markers(manifest_files) -> None:
    _, md_path = manifest_files
    content = md_path.read_text(encoding="utf-8")
    assert "â" not in content
    assert "Status legend: present = present on disk | missing = not yet generated" in content


def test_manifest_includes_local_explanation_artifacts(manifest_files) -> None:
    csv_path, _ = manifest_files
    df = pd.read_csv(csv_path)
    by_path = df.set_index("artifact_path")
    expected = {
        "reports/tables/local_explanation_examples.csv": "table",
        "reports/summary/local_explanation_examples.md": "report",
    }
    for path, artifact_type in expected.items():
        assert path in by_path.index, f"Missing manifest entry: {path}"
        row = by_path.loc[path]
        assert row["artifact_type"] == artifact_type
        assert row["source_script"] == "scripts/generate_evidence_pack.py"
        assert str(row["notes"]).strip(), f"Missing notes for {path}"


def test_manifest_h3_uses_adopted_only_timing_wording(manifest_files) -> None:
    csv_path, _ = manifest_files
    df = pd.read_csv(csv_path)
    h3 = df[df["artifact_path"] == "reports/tables/h3_age_adoption_speed.csv"]
    assert not h3.empty
    notes = str(h3.iloc[0]["notes"]).lower()
    assert "adoption timing among adopted animals" in notes
    assert "adoption speed" not in notes
