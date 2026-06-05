"""Generate a manifest of all thesis artifact files under reports/ and docs/.

Usage:
    python scripts/generate_artifact_manifest.py

Outputs:
    reports/artifact_manifest.csv
    reports/summary/artifact_manifest.md
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUMMARY_DIR = REPORTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

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
    "reports/tables/h3_age_adoption_speed.csv": {
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
    "reports/artifact_manifest.csv": {
        "artifact_type": "manifest",
        "source_script": "scripts/generate_artifact_manifest.py",
        "required_for_thesis": False,
        "chapter": "Appendix",
        "notes": "Self-referential manifest entry",
    },
}


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


def collect_artifacts() -> list[dict]:
    rows = []
    scan_dirs = [REPORTS_DIR, ROOT / "docs"]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for fpath in sorted(scan_dir.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.name.startswith(".") or fpath.suffix == ".gitkeep":
                continue
            rel = str(fpath.relative_to(ROOT)).replace("\\", "/")
            meta = ARTIFACT_METADATA.get(rel, {})
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc).isoformat()
            rows.append(
                {
                    "artifact_path": rel,
                    "artifact_type": meta.get("artifact_type", _infer_type(fpath)),
                    "created_at": mtime,
                    "source_script": meta.get("source_script", ""),
                    "required_for_thesis": meta.get("required_for_thesis", False),
                    "chapter": meta.get("chapter", _infer_chapter(rel)),
                    "notes": meta.get("notes", ""),
                    "exists_on_disk": True,
                }
            )
    # Also add known artifacts that don't exist yet
    on_disk = {r["artifact_path"] for r in rows}
    for rel, meta in ARTIFACT_METADATA.items():
        if rel not in on_disk:
            rows.append(
                {
                    "artifact_path": rel,
                    "artifact_type": meta.get("artifact_type", "unknown"),
                    "created_at": "",
                    "source_script": meta.get("source_script", ""),
                    "required_for_thesis": meta.get("required_for_thesis", False),
                    "chapter": meta.get("chapter", ""),
                    "notes": meta.get("notes", ""),
                    "exists_on_disk": False,
                }
            )
    return rows


def markdown_cell(value: object) -> str:
    """Return ASCII-safe manifest Markdown cell text."""
    return str(value).replace("—", "-").replace("|", "\\|")


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


def main() -> None:
    print("=== Artifact Manifest Generator ===")
    rows = collect_artifacts()
    df = pd.DataFrame(rows)
    csv_out = REPORTS_DIR / "artifact_manifest.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out} ({len(df)} entries)")

    md_out = SUMMARY_DIR / "artifact_manifest.md"
    md_out.write_text(build_markdown(df), encoding="utf-8")
    print(f"Saved: {md_out}")


if __name__ == "__main__":
    main()
