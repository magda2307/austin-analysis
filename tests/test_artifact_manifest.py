"""Strict artifact-manifest generation and acceptance validation tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from aac_adoption.acceptance import AcceptanceError, validate_artifact_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_artifact_manifest.py"
MANIFEST_COLUMNS = [
    "artifact_path",
    "artifact_type",
    "created_at",
    "source_script",
    "required_for_thesis",
    "chapter",
    "notes",
    "exists_on_disk",
    "run_id",
    "producer_source_sha",
    "file_hash",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(
    root: Path,
    *,
    run_id: str = "run-a",
    name: str = "producer",
    outputs: list[Path],
    status: str = "ok",
    profile: str = "thesis-full",
    producer_source_sha: str = "source-sha",
) -> Path:
    receipt_path = root / "reports" / "run_receipts" / run_id / f"{name}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "producer_source_sha": producer_source_sha,
                "profile": profile,
                "status": status,
                "output_hashes": {str(path): _sha256(path) for path in outputs},
            }
        ),
        encoding="utf-8",
    )
    return receipt_path


def _manifest_row(
    root: Path,
    artifact: Path,
    *,
    artifact_path: str | None = None,
    source_script: str = "scripts/producer.py",
    run_id: str = "run-a",
    producer_source_sha: str = "source-sha",
    file_hash: str | None = None,
) -> dict[str, object]:
    return {
        "artifact_path": artifact_path or artifact.relative_to(root).as_posix(),
        "artifact_type": "table",
        "created_at": "2026-06-09T00:00:00+00:00",
        "source_script": source_script,
        "required_for_thesis": True,
        "chapter": "Chapter 4 - Model Evaluation",
        "notes": "Fixture artifact",
        "exists_on_disk": True,
        "run_id": run_id,
        "producer_source_sha": producer_source_sha,
        "file_hash": file_hash or _sha256(artifact),
    }


def _valid_fixture(root: Path) -> tuple[Path, Path, Path]:
    artifact = root / "reports" / "tables" / "result.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("metric,value\nroc_auc,0.8\n", encoding="utf-8")

    source = root / "scripts" / "producer.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# fixture producer\n", encoding="utf-8")
    receipt = _write_receipt(root, outputs=[artifact])

    manifest = root / "reports" / "artifact_manifest.csv"
    pd.DataFrame([_manifest_row(root, artifact)], columns=MANIFEST_COLUMNS).to_csv(
        manifest, index=False
    )
    return manifest, artifact, receipt


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_artifact_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_validator_accepts_valid_single_run_lineage_without_mutating_fixture(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _valid_fixture(tmp_path)
    before = _tree_snapshot(tmp_path)

    result = validate_artifact_manifest(tmp_path, manifest, run_id="run-a")

    assert result.required_artifact_count == 1
    assert result.run_id == "run-a"
    assert result.producer_source_sha == "source-sha"
    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "does not exist"),
        ("empty", "is empty"),
        ("missing_source", "source does not exist"),
        ("hash_mismatch", "disk hash"),
        ("missing_receipt", "no receipt"),
        ("failed_receipt", "status"),
        ("cross_run", "run_id"),
        ("mixed_source_sha", "producer_source_sha"),
        ("wrong_profile", "profile"),
        ("receipt_hash_mismatch", "receipt hash"),
    ],
)
def test_validator_rejects_invalid_required_artifact_lineage(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, artifact, receipt = _valid_fixture(tmp_path)
    frame = pd.read_csv(manifest)
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))

    if mutation == "missing":
        artifact.unlink()
    elif mutation == "empty":
        artifact.write_bytes(b"")
    elif mutation == "missing_source":
        (tmp_path / "scripts" / "producer.py").unlink()
    elif mutation == "hash_mismatch":
        frame.loc[0, "file_hash"] = "0" * 64
        frame.to_csv(manifest, index=False)
    elif mutation == "missing_receipt":
        receipt.unlink()
    elif mutation == "failed_receipt":
        receipt_data["status"] = "error"
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    elif mutation == "cross_run":
        receipt_data["run_id"] = "run-b"
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    elif mutation == "mixed_source_sha":
        _write_receipt(
            tmp_path,
            run_id="run-a",
            name="second_producer",
            outputs=[artifact],
            producer_source_sha="other-source",
        )
    elif mutation == "wrong_profile":
        receipt_data["profile"] = "development-no-shap"
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    elif mutation == "receipt_hash_mismatch":
        receipt_data["output_hashes"][str(artifact)] = "f" * 64
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")

    with pytest.raises(AcceptanceError, match=message):
        validate_artifact_manifest(tmp_path, manifest, run_id="run-a")


def test_validator_rejects_duplicate_normalized_required_paths(tmp_path: Path) -> None:
    manifest, artifact, _ = _valid_fixture(tmp_path)
    rows = [
        _manifest_row(tmp_path, artifact),
        _manifest_row(
            tmp_path,
            artifact,
            artifact_path="reports/tables/../tables/result.csv",
        ),
    ]
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    with pytest.raises(AcceptanceError, match="duplicate normalized"):
        validate_artifact_manifest(tmp_path, manifest, run_id="run-a")


def test_validator_rejects_manifest_older_than_required_artifact(tmp_path: Path) -> None:
    manifest, artifact, _ = _valid_fixture(tmp_path)
    manifest_time = manifest.stat().st_mtime
    os.utime(artifact, (manifest_time + 10, manifest_time + 10))

    with pytest.raises(AcceptanceError, match="older than"):
        validate_artifact_manifest(tmp_path, manifest, run_id="run-a")


def test_validator_allows_explicit_manual_document_source_policy(tmp_path: Path) -> None:
    manifest, artifact, receipt = _valid_fixture(tmp_path)
    document = tmp_path / "docs" / "RESULTS.md"
    document.parent.mkdir(parents=True)
    document.write_text("# Final results\n", encoding="utf-8")
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["output_hashes"][str(document)] = _sha256(document)
    receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    rows = [
        _manifest_row(tmp_path, artifact),
        _manifest_row(tmp_path, document, source_script="manual-doc"),
    ]
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    result = validate_artifact_manifest(tmp_path, manifest, run_id="run-a")

    assert result.required_artifact_count == 2


def test_validator_checks_source_path_when_metadata_includes_cli_arguments(
    tmp_path: Path,
) -> None:
    manifest, artifact, _ = _valid_fixture(tmp_path)
    frame = pd.read_csv(manifest)
    frame.loc[0, "source_script"] = "scripts/producer.py --include-shap"
    frame.to_csv(manifest, index=False)

    result = validate_artifact_manifest(tmp_path, manifest, run_id="run-a")

    assert result.required_artifact_count == 1


def test_cli_requires_run_id() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--run-id" in completed.stderr


def test_unknown_run_fails_before_output_mutation(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()
    csv_out = tmp_path / "reports" / "artifact_manifest.csv"
    md_out = tmp_path / "reports" / "summary" / "artifact_manifest.md"
    md_out.parent.mkdir(parents=True)
    csv_out.write_text("old csv\n", encoding="utf-8")
    md_out.write_text("old markdown\n", encoding="utf-8")
    monkeypatch.setattr(generator, "ARTIFACT_METADATA", {})

    with pytest.raises(AcceptanceError, match="unknown run"):
        generator.generate_manifest(tmp_path, "not-present")

    assert csv_out.read_text(encoding="utf-8") == "old csv\n"
    assert md_out.read_text(encoding="utf-8") == "old markdown\n"


def test_generator_uses_only_exact_requested_receipts(tmp_path: Path, monkeypatch) -> None:
    requested = tmp_path / "reports" / "tables" / "requested.csv"
    other = tmp_path / "reports" / "tables" / "other.csv"
    requested.parent.mkdir(parents=True)
    requested.write_text("value\n1\n", encoding="utf-8")
    other.write_text("value\n2\n", encoding="utf-8")
    source = tmp_path / "scripts" / "producer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    _write_receipt(tmp_path, run_id="run-a", outputs=[requested])
    _write_receipt(tmp_path, run_id="newer-run", outputs=[other])
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "ARTIFACT_METADATA",
        {
            "reports/tables/requested.csv": {
                "artifact_type": "table",
                "source_script": "scripts/producer.py",
                "required_for_thesis": True,
                "chapter": "Chapter 4 - Model Evaluation",
                "notes": "Requested output",
            }
        },
    )

    csv_out, _ = generator.generate_manifest(tmp_path, "run-a")
    frame = pd.read_csv(csv_out)

    assert frame["run_id"].unique().tolist() == ["run-a"]
    assert frame["artifact_path"].tolist() == ["reports/tables/requested.csv"]


def test_generator_ignores_conflicting_hashes_for_unregistered_output(
    tmp_path: Path, monkeypatch
) -> None:
    required = tmp_path / "reports" / "tables" / "required.csv"
    screenshot = tmp_path / "reports" / "figures" / "ui.png"
    required.parent.mkdir(parents=True)
    screenshot.parent.mkdir(parents=True)
    required.write_text("value\n1\n", encoding="utf-8")
    screenshot.write_bytes(b"first")
    source = tmp_path / "scripts" / "producer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    _write_receipt(
        tmp_path,
        run_id="run-a",
        name="first",
        outputs=[required, screenshot],
    )
    second = _write_receipt(
        tmp_path,
        run_id="run-a",
        name="second",
        outputs=[screenshot],
    )
    receipt_data = json.loads(second.read_text(encoding="utf-8"))
    receipt_data["output_hashes"][str(screenshot)] = "f" * 64
    second.write_text(json.dumps(receipt_data), encoding="utf-8")
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "ARTIFACT_METADATA",
        {
            "reports/tables/required.csv": {
                "artifact_type": "table",
                "source_script": "scripts/producer.py",
                "required_for_thesis": True,
                "chapter": "Chapter 4 - Model Evaluation",
                "notes": "Required output",
            }
        },
    )

    csv_out, _ = generator.generate_manifest(tmp_path, "run-a")

    assert pd.read_csv(csv_out)["artifact_path"].tolist() == [
        "reports/tables/required.csv"
    ]


def test_generator_rejects_conflicting_hashes_for_required_output(
    tmp_path: Path, monkeypatch
) -> None:
    required = tmp_path / "reports" / "tables" / "required.csv"
    required.parent.mkdir(parents=True)
    required.write_text("value\n1\n", encoding="utf-8")
    source = tmp_path / "scripts" / "producer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    _write_receipt(tmp_path, run_id="run-a", name="first", outputs=[required])
    second = _write_receipt(
        tmp_path,
        run_id="run-a",
        name="second",
        outputs=[required],
    )
    receipt_data = json.loads(second.read_text(encoding="utf-8"))
    receipt_data["output_hashes"][str(required)] = "f" * 64
    second.write_text(json.dumps(receipt_data), encoding="utf-8")
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "ARTIFACT_METADATA",
        {
            "reports/tables/required.csv": {
                "artifact_type": "table",
                "source_script": "scripts/producer.py",
                "required_for_thesis": True,
                "chapter": "Chapter 4 - Model Evaluation",
                "notes": "Required output",
            }
        },
    )

    with pytest.raises(AcceptanceError, match="conflicting receipt hashes"):
        generator.generate_manifest(tmp_path, "run-a")


def test_generator_rejects_conflicting_dynamic_shap_skip_note(
    tmp_path: Path, monkeypatch
) -> None:
    shap = tmp_path / "reports" / "tables" / "shap_global_regression.csv"
    skip_note = tmp_path / "reports" / "tables" / "shap_regression_skip_note.csv"
    shap.parent.mkdir(parents=True)
    shap.write_text("feature,importance\nage,1\n", encoding="utf-8")
    skip_note.write_text("reason\nselected model unsupported\n", encoding="utf-8")
    source = tmp_path / "scripts" / "producer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# producer\n", encoding="utf-8")
    _write_receipt(
        tmp_path,
        run_id="run-a",
        name="first",
        outputs=[shap, skip_note],
    )
    second = _write_receipt(
        tmp_path,
        run_id="run-a",
        name="second",
        outputs=[skip_note],
    )
    receipt_data = json.loads(second.read_text(encoding="utf-8"))
    receipt_data["output_hashes"][str(skip_note)] = "f" * 64
    second.write_text(json.dumps(receipt_data), encoding="utf-8")
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "ARTIFACT_METADATA",
        {
            "reports/tables/shap_global_regression.csv": {
                "artifact_type": "table",
                "source_script": "scripts/producer.py",
                "required_for_thesis": True,
                "chapter": "Chapter 5 - Interpretation",
                "notes": "Regression SHAP",
            }
        },
    )

    with pytest.raises(
        AcceptanceError, match="shap_regression_skip_note.csv"
    ):
        generator.generate_manifest(tmp_path, "run-a")


def test_generator_accepts_shap_skip_note_for_unselected_model() -> None:
    generator = _load_generator()
    registry = {
        "reports/tables/shap_global_regression.csv": {"required_for_thesis": True},
        "reports/tables/shap_feature_families_regression.csv": {
            "required_for_thesis": True
        },
        "reports/tables/feature_family_importance_regression.csv": {
            "required_for_thesis": True
        },
        "reports/figures/feature_family_importance_regression.png": {
            "required_for_thesis": True
        },
    }
    skip_note = "reports/tables/shap_regression_skip_note.csv"

    resolved = generator._resolve_shap_registry(
        registry, {skip_note: "receipt-hash"}
    )

    assert set(resolved) == {skip_note}


def test_generator_derives_selected_model_and_sidecar_from_selection_output(
    tmp_path: Path, monkeypatch
) -> None:
    selection = tmp_path / "reports" / "tables" / "final_model_selection.csv"
    model = tmp_path / "models" / "classification" / "selected.joblib"
    sidecar = model.with_suffix(".json")
    selection.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    sidecar.write_text('{"model": "selected"}', encoding="utf-8")
    pd.DataFrame(
        [{"selected": True, "artifact_path": model.relative_to(tmp_path).as_posix()}]
    ).to_csv(selection, index=False)
    source = tmp_path / "scripts" / "run_analysis.py"
    source.parent.mkdir(parents=True)
    source.write_text("# analysis\n", encoding="utf-8")
    _write_receipt(tmp_path, outputs=[selection, model, sidecar])
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "ARTIFACT_METADATA",
        {
            "reports/tables/final_model_selection.csv": {
                "artifact_type": "table",
                "source_script": "scripts/run_analysis.py",
                "required_for_thesis": True,
                "chapter": "Chapter 4 - Model Evaluation",
                "notes": "Final model selection",
            }
        },
    )

    csv_out, _ = generator.generate_manifest(tmp_path, "run-a")
    paths = set(pd.read_csv(csv_out)["artifact_path"])

    assert model.relative_to(tmp_path).as_posix() in paths
    assert sidecar.relative_to(tmp_path).as_posix() in paths


def test_curated_registry_includes_core_datasets_and_final_documents() -> None:
    generator = _load_generator()
    required = {
        "data/processed/modeling_dataset.csv",
        "data/processed/feature_columns.json",
        "data/processed/target_columns.json",
        "reports/tables/h3_adopted_only_age_speed.csv",
        "reports/tables/final_model_selection.csv",
        "reports/summary/final_model_selection.md",
        "README.md",
        "docs/METHODOLOGY.md",
        "docs/RESULTS.md",
        "docs/target_definitions.md",
    }

    assert required <= set(generator.ARTIFACT_METADATA)
    assert all(
        generator.ARTIFACT_METADATA[path]["required_for_thesis"] for path in required
    )


def test_manifest_h3_preserves_adopted_only_timing_wording() -> None:
    generator = _load_generator()
    notes = generator.ARTIFACT_METADATA[
        "reports/tables/h3_adopted_only_age_speed.csv"
    ]["notes"].lower()

    assert "adoption timing among adopted animals" in notes
    assert "adoption speed" not in notes
