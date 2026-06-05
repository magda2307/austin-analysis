"""Generate leakage audit report.

Checks that no outcome-derived column appears in feature_columns.json.
FAILS with exit code 1 if leakage columns are found.

Outputs:
    reports/tables/leakage_audit.csv
    reports/summary/leakage_audit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from aac_adoption.features.feature_sets import (
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    available_intake_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leakage audit report")
    parser.add_argument(
        "--feature-columns",
        default="data/processed/feature_columns.json",
        help="Path to feature_columns.json produced by build_dataset.py",
    )
    parser.add_argument(
        "--data",
        default="data/processed/modeling_dataset.csv",
        help="Path to modeling dataset (used to list all available columns)",
    )
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--summary-dir", default="reports/summary")
    return parser.parse_args()


def build_leakage_audit(
    feature_columns_path: str,
    data_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build leakage audit DataFrame and return detected violations."""
    # Load feature columns from JSON
    feat_path = Path(feature_columns_path)
    if feat_path.exists():
        feature_columns = json.loads(feat_path.read_text(encoding="utf-8"))
    else:
        feature_columns = []

    # Load all dataset columns
    data_path_obj = Path(data_path)
    if data_path_obj.exists():
        all_columns = list(pd.read_csv(data_path_obj, nrows=0).columns)
    else:
        all_columns = feature_columns

    records = []
    violations: list[str] = []

    # Classify every column in the dataset
    for col in sorted(set(all_columns)):
        if col in set(feature_columns):
            category = "predictor"
            allowed = True
            if col in LEAKAGE_COLUMNS:
                category = "LEAKAGE — predictor AND leakage set"
                allowed = False
                violations.append(col)
        elif col in set(TARGET_COLUMNS):
            category = "target"
            allowed = True
            note = "Use as model label/target only"
        elif col in set(METADATA_COLUMNS):
            category = "metadata"
            allowed = True
            note = "Do not use as predictor or target"
        else:
            category = "other / derived"
            allowed = True

        records.append({
            "column": col,
            "category": category,
            "in_feature_columns_json": col in set(feature_columns),
            "in_target_columns": col in set(TARGET_COLUMNS),
            "in_metadata_columns": col in set(METADATA_COLUMNS),
            "in_leakage_set": col in LEAKAGE_COLUMNS,
            "leakage_violation": col in violations,
            "allowed_as_predictor": col in set(feature_columns) and col not in LEAKAGE_COLUMNS,
        })

    return pd.DataFrame(records), violations


def write_summary_md(df: pd.DataFrame, violations: list[str], output_path: Path) -> None:
    predictor_cols = df[df["category"] == "predictor"]["column"].tolist()
    target_cols = df[df["in_target_columns"]]["column"].tolist()
    metadata_cols = df[df["in_metadata_columns"]]["column"].tolist()

    status = "✅ PASS — No leakage violations found." if not violations else \
             f"❌ FAIL — {len(violations)} leakage violation(s) found: {violations}"

    lines = [
        "# Leakage Audit Report",
        "",
        f"## Status: {status}",
        "",
        "## What This Audit Checks",
        "",
        "Verifies that no outcome-derived column (targets, metadata) appears in `feature_columns.json`.",
        "A leakage violation means a column that is determined *after* intake time is being used",
        "as a predictor, which would give the model illegitimate future information.",
        "",
        "## Column Categories",
        "",
        "### Predictor Columns (safe to use as model features)",
        "",
        "These columns are listed in `feature_columns.json` and are intake-time-only.",
        "",
        "```",
        *[f"  {c}" for c in sorted(predictor_cols)],
        "```",
        "",
        "### Target Columns (must never appear in feature_columns.json)",
        "",
        "```",
        *[f"  {c}" for c in sorted(target_cols)],
        "```",
        "",
        "### Metadata Columns (outcome-related; must never be predictors)",
        "",
        "```",
        *[f"  {c}" for c in sorted(metadata_cols)],
        "```",
        "",
        "## Leakage Set Definition",
        "",
        "The `LEAKAGE_COLUMNS` set in `src/aac_adoption/features/feature_sets.py` is the",
        "union of `TARGET_COLUMNS` and `METADATA_COLUMNS`, minus `animal_id` and `intake_datetime`.",
        "The `validate_no_leakage()` function also checks for column names containing 'future',",
        "'_next_', or starting with 'next_'.",
        "",
        "## Methodology Reference",
        "",
        "See `docs/target_definitions.md` for the full leakage control summary.",
        "See `docs/methodology_notes.md` for the intake-time feature set justification.",
    ]

    if violations:
        lines.insert(4, "")
        lines.insert(4, f"**Violations:** {violations}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    summary_dir = Path(args.summary_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("Running leakage audit...")
    df, violations = build_leakage_audit(args.feature_columns, args.data)

    out_csv = tables_dir / "leakage_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_md = summary_dir / "leakage_audit.md"
    write_summary_md(df, violations, out_md)
    print(f"Wrote {out_md}")

    predictor_count = df[df["category"] == "predictor"].shape[0]
    target_count = df[df["in_target_columns"]].shape[0]
    metadata_count = df[df["in_metadata_columns"]].shape[0]

    print(f"\n  Predictor columns (safe): {predictor_count}")
    print(f"  Target columns:           {target_count}")
    print(f"  Metadata columns:         {metadata_count}")

    if violations:
        print(f"\nFAIL: LEAKAGE VIOLATIONS FOUND ({len(violations)}):")
        for v in violations:
            print(f"   - {v}")
        print("\nFix: remove these columns from feature_columns.json and retrain all models.")
        sys.exit(1)
    else:
        print("\nPASS: Leakage audit passed -- no violations found.")


if __name__ == "__main__":
    main()
