"""Generate a manifest of all thesis artifact files under reports/ and docs/.

Usage:
    python scripts/generate_artifact_manifest.py --run-id RUN_ID

Outputs:
    reports/artifact_manifest.csv
    reports/summary/artifact_manifest.md
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aac_adoption.acceptance import (
    AcceptanceError,
    compute_sha256,
    normalize_relative_path,
    resolve_source_path,
    validate_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUMMARY_DIR = REPORTS_DIR / "summary"

# -------------------------------------------------------------------------
# Curated metadata: for known artifacts, override chapter/source/required.
# Artifacts not in this dict get sensible defaults derived from their path.
# -------------------------------------------------------------------------
ARTIFACT_METADATA: dict[str, dict] = {
    # Model comparison outputs
    "reports/tables/model_comparison_classification.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Primary classification leaderboard",
    },
    "reports/tables/model_comparison_regression.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Primary regression leaderboard",
    },
    # SHAP outputs
    "reports/tables/shap_global_classification.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_diagnostics.py --include-shap",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Global SHAP per feature — classification",
    },
    "reports/tables/shap_global_regression.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_diagnostics.py --include-shap",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Global SHAP per feature — regression",
    },
    "reports/tables/shap_feature_families_classification.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_diagnostics.py --include-shap",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "SHAP aggregated by feature family — classification",
    },
    "reports/tables/shap_feature_families_regression.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_diagnostics.py --include-shap",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "SHAP aggregated by feature family — regression",
    },
    "reports/tables/feature_family_importance_classification.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_feature_family_importance.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Extended family importance with sum/mean/n_features",
    },
    "reports/tables/feature_family_importance_regression.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_feature_family_importance.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Extended family importance with sum/mean/n_features — regression",
    },
    "reports/figures/feature_family_importance_classification.png": {
        "artifact_type": "figure",
        "source_script": "scripts/generate_feature_family_importance.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Feature family bar chart — classification",
    },
    "reports/figures/feature_family_importance_regression.png": {
        "artifact_type": "figure",
        "source_script": "scripts/generate_feature_family_importance.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Feature family bar chart — regression",
    },
    # Hypothesis evidence
    "reports/tables/h1_intake_vs_appearance.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H1 evidence: intake type and condition vs appearance",
    },
    "reports/tables/h2_seasonality_summary.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H2 evidence: seasonality summary table",
    },
    "reports/tables/h3_adopted_only_age_speed.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H3 evidence: age and adoption timing among adopted animals",
    },
    "reports/tables/h4_dark_color_summary.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H4 evidence: dark coat colour summary table",
    },
    "reports/tables/h5_covid_period.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H5 evidence: COVID period adoption dynamics",
    },
    "reports/summary/h2_interpretation.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H2 evidence seasonality summary report",
    },
    "reports/summary/h4_interpretation.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "H4 evidence coat colour summary report",
    },
    "reports/summary/external_validity_limitations.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "External validity and causal limitations report",
    },
    "reports/summary/breed_color_justification.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 — Hypotheses",
        "notes": "Breed and coat colour engineering justification report",
    },
    "reports/summary/descriptive_baseline_comparison.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Non-ML descriptive baseline vs ML model comparison report",
    },
    # Reliability
    "reports/tables/model_evidence_pack.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Summary evidence pack for model trustworthiness",
    },
    "reports/tables/subgroup_reliability.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Per-cohort reliability red flags",
    },
    "reports/tables/metric_confidence_intervals.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Bootstrap confidence intervals for key metrics",
    },
    "reports/tables/local_explanation_examples.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 â€” Interpretation",
        "notes": "Local explanation examples combining animal journeys, nearest neighbours, and model reasons",
    },
    "reports/summary/local_explanation_examples.md": {
        "artifact_type": "report",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 â€” Interpretation",
        "notes": "Narrative summary of local explanation examples with causal limitations",
    },
    # Animal research
    "reports/tables/animal_archetypes.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_animal_research.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Animal profile archetypes with adoption statistics",
    },
    "reports/tables/vulnerable_profiles.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_animal_research.py",
        "required_for_thesis": True,
        "chapter": "Chapter 5 — Interpretation",
        "notes": "Profiles needing targeted visibility",
    },
    # Reproducibility
    "reports/tables/environment_snapshot.csv": {
        "artifact_type": "table",
        "source_script": "scripts/generate_environment_snapshot.py",
        "required_for_thesis": True,
        "chapter": "Appendix",
        "notes": "Library version snapshot for reproducibility",
    },
    "reports/summary/environment_snapshot.md": {
        "artifact_type": "report",
        "source_script": "scripts/generate_environment_snapshot.py",
        "required_for_thesis": True,
        "chapter": "Appendix",
        "notes": "Human-readable environment snapshot for appendix inclusion",
    },
    "reports/summary/model_evidence_pack.md": {
        "artifact_type": "report",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Narrative model evidence summary",
    },
    "reports/summary/subgroup_reliability.md": {
        "artifact_type": "report",
        "source_script": "scripts/generate_evidence_pack.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 — Model Evaluation",
        "notes": "Subgroup reliability narrative summary",
    },
    "data/processed/modeling_dataset.csv": {
        "artifact_type": "dataset",
        "source_script": "scripts/build_dataset.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Canonical matched-episode modeling dataset",
    },
    "data/processed/modeling_dataset_context.csv": {
        "artifact_type": "dataset",
        "source_script": "scripts/build_dataset.py",
        "required_for_thesis": False,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Modeling dataset with intake-time context features",
    },
    "data/processed/feature_columns.json": {
        "artifact_type": "metadata",
        "source_script": "scripts/build_dataset.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Leakage-safe feature registry",
    },
    "data/processed/context_feature_columns.json": {
        "artifact_type": "metadata",
        "source_script": "scripts/build_dataset.py",
        "required_for_thesis": False,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Intake-time context feature registry",
    },
    "data/processed/target_columns.json": {
        "artifact_type": "metadata",
        "source_script": "scripts/build_dataset.py",
        "required_for_thesis": True,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Canonical classification and timing target registry",
    },
    "reports/tables/final_model_selection.csv": {
        "artifact_type": "table",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 - Model Evaluation",
        "notes": "Frozen validation-period model selections",
    },
    "reports/summary/final_model_selection.md": {
        "artifact_type": "report",
        "source_script": "scripts/run_analysis.py",
        "required_for_thesis": True,
        "chapter": "Chapter 4 - Model Evaluation",
        "notes": "Selection rules and selected-model rationale",
    },
    "README.md": {
        "artifact_type": "documentation",
        "source_script": "manual-doc",
        "required_for_thesis": True,
        "chapter": "Front Matter",
        "notes": "Project overview and reproducibility entry point",
    },
    "docs/METHODOLOGY.md": {
        "artifact_type": "documentation",
        "source_script": "manual-doc",
        "required_for_thesis": True,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Final thesis methodology",
    },
    "docs/RESULTS.md": {
        "artifact_type": "documentation",
        "source_script": "manual-doc",
        "required_for_thesis": True,
        "chapter": "Chapter 4 - Results",
        "notes": "Final thesis results",
    },
    "docs/target_definitions.md": {
        "artifact_type": "documentation",
        "source_script": "manual-doc",
        "required_for_thesis": True,
        "chapter": "Chapter 3 - Data And Methods",
        "notes": "Canonical predictive target definitions",
    },
    "reports/artifact_manifest.csv": {
        "artifact_type": "manifest",
        "source_script": "scripts/generate_artifact_manifest.py",
        "required_for_thesis": False,
        "chapter": "Appendix",
        "notes": "Self-referential manifest entry",
    },
}

def _resolve_shap_registry(
    registry: dict[str, dict], receipt_hashes: dict[str, str]
) -> dict[str, dict]:
    resolved = dict(registry)
    for task in ("classification", "regression"):
        skip_note = f"reports/tables/shap_{task}_skip_note.csv"
        if skip_note not in receipt_hashes:
            continue
        for path in (
            f"reports/tables/shap_global_{task}.csv",
            f"reports/tables/shap_feature_families_{task}.csv",
            f"reports/tables/feature_family_importance_{task}.csv",
            f"reports/figures/feature_family_importance_{task}.png",
        ):
            resolved.pop(path, None)
        resolved[skip_note] = {
            "artifact_type": "table",
            "source_script": "scripts/generate_diagnostics.py --include-shap",
            "required_for_thesis": True,
            "chapter": "Chapter 5 - Interpretation",
            "notes": f"SHAP skip rationale for the selected {task} model",
        }
    return resolved


def _infer_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "table"
    if suffix == ".png":
        return "figure"
    if suffix in (".md", ".txt"):
        return "report"
    return "other"


def _infer_chapter(rel_path: str) -> str:
    if "tables" in rel_path or "figures" in rel_path:
        return "Unknown — see notes"
    if "summary" in rel_path:
        return "Unknown — see notes"
    if "diagnostics" in rel_path:
        return "Chapter 4 — Model Evaluation"
    return "Unknown"


def _clean_text(value: object) -> object:
    """Normalize common mojibake dashes in manifest text."""
    if not isinstance(value, str):
        return value
    return (
        value.replace("\u00e2\u20ac\u201d", "-")
        .replace("\u00e2\u20ac\u2014", "-")
        .replace("\u0101\u20ac\u201d", "-")
        .replace("\u0101\u20ac\u2014", "-")
        .replace("â€”", "-")
    )


def _receipt_relative_path(root: Path, value: object) -> str:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise AcceptanceError(f"receipt output is outside project root: {value}") from exc


def _load_requested_receipts(
    root: Path, run_id: str
) -> tuple[Path, dict[str, str], dict[str, str], str]:
    receipts_dir = root / "reports" / "run_receipts" / run_id
    if not receipts_dir.is_dir():
        raise AcceptanceError(f"unknown run: {run_id}")
    receipt_paths = sorted(receipts_dir.glob("*.json"))
    if not receipt_paths:
        raise AcceptanceError(f"unknown run or no receipts: {run_id}")

    output_hashes: dict[str, str] = {}
    output_sources: dict[str, str] = {}
    source_shas: set[str] = set()
    for receipt_path in receipt_paths:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcceptanceError(f"invalid receipt: {receipt_path}") from exc
        if receipt.get("status") != "ok":
            raise AcceptanceError(f"receipt status is not ok: {receipt_path}")
        if receipt.get("run_id") != run_id:
            raise AcceptanceError(f"receipt run_id mismatch: {receipt_path}")
        if receipt.get("profile") != "thesis-full":
            raise AcceptanceError(f"receipt profile must be thesis-full: {receipt_path}")
        source_sha = str(receipt.get("producer_source_sha", "")).strip()
        if not source_sha:
            raise AcceptanceError(f"receipt producer_source_sha is missing: {receipt_path}")
        source_shas.add(source_sha)
        candidate_source = f"scripts/{receipt_path.stem}.py"
        if not (root / candidate_source).is_file():
            candidate_source = ""
        for output_path, expected_hash in receipt.get("output_hashes", {}).items():
            relative = _receipt_relative_path(root, output_path)
            expected = str(expected_hash)
            previous = output_hashes.get(relative)
            if previous is not None and previous != expected:
                raise AcceptanceError(
                    f"conflicting receipt hashes for output: {relative}"
                )
            output_hashes[relative] = expected
            if candidate_source:
                output_sources[relative] = candidate_source

    if len(source_shas) != 1:
        raise AcceptanceError("requested run has mixed producer_source_sha values")
    return receipts_dir, output_hashes, output_sources, source_shas.pop()


def _selected_model_metadata(
    root: Path, output_sources: dict[str, str]
) -> dict[str, dict]:
    selection_path = root / "reports" / "tables" / "final_model_selection.csv"
    if not selection_path.is_file() or selection_path.stat().st_size == 0:
        return {}
    selection = pd.read_csv(selection_path)
    if "artifact_path" not in selection.columns:
        return {}
    if "selected" in selection.columns:
        selected = selection[
            selection["selected"].astype(str).str.strip().str.lower().isin({"true", "1"})
        ]
    else:
        selected = selection

    derived: dict[str, dict] = {}
    for value in selected["artifact_path"].dropna():
        relative = _receipt_relative_path(root, value)
        sidecar = Path(relative).with_suffix(".json").as_posix()
        for artifact_path, artifact_type, notes in (
            (relative, "model", "Selected final model artifact"),
            (sidecar, "metadata", "Selected final model metadata sidecar"),
        ):
            derived[artifact_path] = {
                "artifact_type": artifact_type,
                "source_script": output_sources.get(
                    artifact_path, "scripts/run_analysis.py"
                ),
                "required_for_thesis": True,
                "chapter": "Chapter 4 - Model Evaluation",
                "notes": notes,
            }
    return derived


def collect_artifacts(root: Path, run_id: str) -> tuple[list[dict], Path]:
    receipts_dir, receipt_hashes, output_sources, source_sha = (
        _load_requested_receipts(root, run_id)
    )
    registry = {
        path: metadata
        for path, metadata in ARTIFACT_METADATA.items()
        if metadata.get("required_for_thesis")
    }
    registry = _resolve_shap_registry(registry, receipt_hashes)
    registry.update(_selected_model_metadata(root, output_sources))

    normalized = [normalize_relative_path(path) for path in registry]
    if len(normalized) != len(set(normalized)):
        raise AcceptanceError("registry contains duplicate normalized artifact paths")

    rows = []
    for relative, meta in registry.items():
        relative = normalize_relative_path(relative)
        artifact = root / relative
        if not artifact.is_file():
            raise AcceptanceError(f"required artifact does not exist: {relative}")
        if artifact.stat().st_size == 0:
            raise AcceptanceError(f"required artifact is empty: {relative}")
        actual_hash = compute_sha256(artifact)
        receipt_hash = receipt_hashes.get(relative)
        if receipt_hash is None:
            raise AcceptanceError(f"no receipt output found for required artifact: {relative}")
        if receipt_hash != actual_hash:
            raise AcceptanceError(f"receipt hash mismatch for {relative}")
        source_script = str(meta.get("source_script", "")).strip()
        source_path = resolve_source_path(root, source_script)
        if source_path is not None and not source_path.is_file():
            raise AcceptanceError(f"artifact source does not exist: {source_script}")
        rows.append(
            {
                "artifact_path": relative,
                "artifact_type": _clean_text(
                    meta.get("artifact_type", _infer_type(artifact))
                ),
                "created_at": datetime.fromtimestamp(
                    artifact.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "source_script": _clean_text(source_script),
                "required_for_thesis": True,
                "chapter": _clean_text(meta.get("chapter", _infer_chapter(relative))),
                "notes": _clean_text(meta.get("notes", "")),
                "exists_on_disk": True,
                "run_id": run_id,
                "producer_source_sha": source_sha,
                "file_hash": actual_hash,
            }
        )
    return sorted(rows, key=lambda row: row["artifact_path"]), receipts_dir


def markdown_cell(value: object) -> str:
    """Return ASCII-safe manifest Markdown cell text."""
    return str(_clean_text(value)).replace("—", "-").replace("|", "\\|")


def build_markdown(df: pd.DataFrame) -> str:
    chapters = sorted(df["chapter"].dropna().unique())
    lines = [
        "# Thesis Artifact Manifest",
        "",
        f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        "Status legend: present = present on disk | missing = not yet generated",
        "",
    ]
    for chapter in chapters:
        chapter_df = df[df["chapter"] == chapter].sort_values("artifact_path")
        lines.append(f"## {markdown_cell(chapter)}")
        lines.append("")
        lines.append("| Status | Artifact | Type | Source Script | Notes |")
        lines.append("|--------|----------|------|---------------|-------|")
        for _, row in chapter_df.iterrows():
            status = "present" if row["exists_on_disk"] else "missing"
            lines.append(
                f"| {status} | `{markdown_cell(row['artifact_path'])}` | {markdown_cell(row['artifact_type'])} "
                f"| `{markdown_cell(row['source_script'])}` | {markdown_cell(row['notes'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _temporary_path(parent: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".artifact_manifest.", suffix=suffix, dir=parent
    )
    os.close(descriptor)
    return Path(name)


def generate_manifest(root: str | Path, run_id: str) -> tuple[Path, Path]:
    root_path = Path(root).resolve()
    rows, receipts_dir = collect_artifacts(root_path, run_id)
    frame = pd.DataFrame(rows)
    reports_dir = root_path / "reports"
    summary_dir = reports_dir / "summary"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_out = reports_dir / "artifact_manifest.csv"
    md_out = summary_dir / "artifact_manifest.md"
    csv_temp = _temporary_path(reports_dir, ".csv")
    md_temp = _temporary_path(summary_dir, ".md")
    try:
        frame.to_csv(csv_temp, index=False)
        md_temp.write_text(build_markdown(frame), encoding="utf-8")
        validate_artifact_manifest(
            root_path,
            csv_temp,
            run_id=run_id,
            receipts_dir=receipts_dir,
        )
        os.replace(csv_temp, csv_out)
        os.replace(md_temp, md_out)
    finally:
        csv_temp.unlink(missing_ok=True)
        md_temp.unlink(missing_ok=True)
    return csv_out, md_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a strict final thesis artifact manifest."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Exact reports/run_receipts/<run-id> directory to finalize.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("=== Artifact Manifest Generator ===")
    csv_out, md_out = generate_manifest(ROOT, args.run_id)
    print(f"Saved: {csv_out}")
    print(f"Saved: {md_out}")


if __name__ == "__main__":
    main()
