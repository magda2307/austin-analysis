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
    "chapter",
    "notes",
    "exists_on_disk",
]


@pytest.mark.skipif(
    not MANIFEST_CSV.exists(),
    reason="artifact_manifest.csv not yet generated - run scripts/generate_artifact_manifest.py",
)
def test_manifest_csv_exists_and_has_columns() -> None:
    df = pd.read_csv(MANIFEST_CSV)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.skipif(
    not MANIFEST_CSV.exists(),
    reason="artifact_manifest.csv not yet generated",
)
def test_manifest_no_null_artifact_path() -> None:
    df = pd.read_csv(MANIFEST_CSV)
    assert df["artifact_path"].notna().all(), "Some artifact_path values are null"
    assert (df["artifact_path"] != "").all(), "Some artifact_path values are empty"


@pytest.mark.skipif(
    not MANIFEST_CSV.exists(),
    reason="artifact_manifest.csv not yet generated",
)
def test_required_thesis_artifacts_exist_on_disk() -> None:
    df = pd.read_csv(MANIFEST_CSV)
    required = df[df["required_for_thesis"].astype(str).isin(["True", "1", "true"])]
    missing = required[~required["exists_on_disk"].astype(str).isin(["True", "1", "true"])]
    if not missing.empty:
        paths = missing["artifact_path"].tolist()
        pytest.fail(f"Required thesis artifacts missing on disk:\n" + "\n".join(paths))


@pytest.mark.skipif(
    not MANIFEST_MD.exists(),
    reason="artifact_manifest.md not yet generated",
)
def test_manifest_md_exists() -> None:
    content = MANIFEST_MD.read_text(encoding="utf-8")
    assert "# Thesis Artifact Manifest" in content
    assert "Chapter" in content


@pytest.mark.skipif(
    not MANIFEST_MD.exists(),
    reason="artifact_manifest.md not yet generated",
)
def test_manifest_md_has_no_mojibake_markers() -> None:
    content = MANIFEST_MD.read_text(encoding="utf-8")
    assert "â" not in content
    assert "Status legend: present = present on disk | missing = not yet generated" in content


@pytest.mark.skipif(
    not MANIFEST_CSV.exists(),
    reason="artifact_manifest.csv not yet generated",
)
def test_manifest_includes_local_explanation_artifacts() -> None:
    df = pd.read_csv(MANIFEST_CSV)
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


@pytest.mark.skipif(
    not MANIFEST_CSV.exists(),
    reason="artifact_manifest.csv not yet generated",
)
def test_manifest_h3_uses_adopted_only_timing_wording() -> None:
    df = pd.read_csv(MANIFEST_CSV)
    h3 = df[df["artifact_path"] == "reports/tables/h3_age_adoption_speed.csv"]
    assert not h3.empty
    notes = str(h3.iloc[0]["notes"]).lower()
    assert "adoption timing among adopted animals" in notes
    assert "adoption speed" not in notes
