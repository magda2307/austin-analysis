"""Generate feature quality audit: missingness and category cardinality tables.

Outputs:
    reports/tables/feature_missingness.csv
    reports/tables/category_cardinality.csv
    reports/summary/feature_quality_audit.md

Justifies why breed/color simplification was necessary and documents
missing value patterns for thesis methodology chapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np

from aac_adoption.features.feature_sets import BASE_INTAKE_TIME_FEATURES


# Features to audit (superset of the modeling feature set — includes raw fields too)
AUDIT_FEATURES = [
    # Categorical raw
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "breed",
    "color",
    # Engineered categorical
    "age_group",
    "primary_breed",
    "simplified_breed_group",
    "primary_color",
    "simplified_color_group",
    # Binary flags
    "is_black_or_dark",
    "has_name",
    "is_named",
    "is_mixed_breed",
    # Numeric
    "age_in_days",
    "age_days",
    "age_in_months",
    "age_months",
    "age_in_years",
    "age_years",
    # Temporal
    "intake_year",
    "intake_month",
    "covid_period",
]

# Category features to include in the cardinality table
CATEGORY_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "breed",
    "color",
    "age_group",
    "primary_breed",
    "simplified_breed_group",
    "primary_color",
    "simplified_color_group",
    "covid_period",
]

# Threshold below which a category value is considered "rare"
RARE_THRESHOLD_PCT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature quality audit tables")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--summary-dir", default="reports/summary")
    return parser.parse_args()


def build_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build missingness statistics for each audited feature."""
    n = len(df)
    records = []
    for feature in AUDIT_FEATURES:
        if feature not in df.columns:
            continue
        col = df[feature]
        missing = int(col.isna().sum())
        dtype = str(col.dtype)
        unique = col.nunique(dropna=True)

        # Example values (up to 5 non-null unique values)
        examples = col.dropna().astype(str).unique()[:5].tolist()

        records.append({
            "feature": feature,
            "dtype": dtype,
            "missing_count": missing,
            "missing_percent": round(missing / n * 100, 2) if n > 0 else 0.0,
            "unique_values": unique,
            "example_values": " | ".join(examples),
        })

    return pd.DataFrame(records).sort_values("missing_percent", ascending=False)


def build_cardinality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build cardinality statistics for categorical features."""
    n = len(df)
    records = []
    for feature in CATEGORY_FEATURES:
        if feature not in df.columns:
            continue
        col = df[feature].dropna().astype(str)
        value_counts = col.value_counts()
        top_10 = " | ".join(value_counts.head(10).index.tolist())
        rare_mask = (value_counts / len(col) * 100) < RARE_THRESHOLD_PCT
        rare_count = int(rare_mask.sum())
        rare_pct = round(rare_mask.mean() * 100, 1)

        records.append({
            "feature": feature,
            "unique_values": len(value_counts),
            "top_10_values": top_10,
            "rare_category_count": rare_count,
            "rare_category_percent": rare_pct,
        })

    return pd.DataFrame(records).sort_values("unique_values", ascending=False)


def write_summary_md(
    miss_df: pd.DataFrame,
    card_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Feature Quality Audit",
        "",
        "## Purpose",
        "",
        "This audit documents missingness and category cardinality for all features",
        "used in the AAC adoption ML pipeline. It justifies feature engineering decisions",
        "(breed/color simplification) and documents that missing values were handled",
        "intentionally, not silently ignored.",
        "",
        "## Missingness Summary",
        "",
        "| Feature | Missing % | Missing Count | Unique Values |",
        "|---|---|---|---|",
    ]

    for _, row in miss_df.iterrows():
        lines.append(
            f"| {row['feature']} | {row['missing_percent']:.1f}% "
            f"| {int(row['missing_count']):,} | {int(row['unique_values']):,} |"
        )

    lines += [
        "",
        "## Category Cardinality Summary",
        "",
        "| Feature | Unique Values | Rare Categories (<1%) | Rare % |",
        "|---|---|---|---|",
    ]
    for _, row in card_df.iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['unique_values']):,} "
            f"| {int(row['rare_category_count']):,} | {row['rare_category_percent']:.1f}% |"
        )

    lines += [
        "",
        "## Why Breed and Color Were Simplified",
        "",
        "Raw `breed` and `color` fields have extremely high cardinality (hundreds of unique values).",
        "Most individual categories have too few records for reliable model training.",
        "The simplification creates stable groups with enough records for meaningful patterns:",
        "",
        "- `simplified_breed_group`: collapses hundreds of breed names into ~10 functional groups",
        "  (pit_bull_type, chihuahua_type, domestic_cat, retriever_type, etc.)",
        "- `simplified_color_group`: collapses color strings into ~7 perceptual groups",
        "  (black_or_dark, brown_tan, white_light, gray_blue, orange_yellow, mixed_other)",
        "- `is_black_or_dark`: binary flag for the H4 black dog/cat syndrome analysis",
        "",
        "High-cardinality raw fields are retained in the dataset for reference but are NOT",
        "used as model features (they would cause sparsity and overfitting problems).",
        "",
        "## Missing Value Handling",
        "",
        "- **age fields**: Missing age is common for strays without known history.",
        "  `age_group = 'unknown'` is a valid category; models handle it via the categorical encoding.",
        "- **name fields**: `has_name` / `is_named` are binary; no missing values expected",
        "  (absence of name = False, not NA).",
        "- **color/breed**: Raw strings are sometimes 'Unknown'; these are passed through as",
        "  a valid category value after simplification.",
        "- **covid_period**: Derived from `intake_datetime`; missing only if datetime is missing.",
        "",
        "## Source Reference",
        "",
        "Feature engineering: `src/aac_adoption/features/feature_engineering.py`",
        "Feature set definition: `src/aac_adoption/features/feature_sets.py`",
        "Leakage audit: `reports/summary/leakage_audit.md`",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    summary_dir = Path(args.summary_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("Building missingness table...")
    miss_df = build_missingness_table(df)
    miss_path = tables_dir / "feature_missingness.csv"
    miss_df.to_csv(miss_path, index=False)
    print(f"Wrote {miss_path}")

    print("Building category cardinality table...")
    card_df = build_cardinality_table(df)
    card_path = tables_dir / "category_cardinality.csv"
    card_df.to_csv(card_path, index=False)
    print(f"Wrote {card_path}")

    print("Writing summary...")
    out_md = summary_dir / "feature_quality_audit.md"
    write_summary_md(miss_df, card_df, out_md)
    print(f"Wrote {out_md}")

    # Print highlights
    print("\n=== Feature Quality Highlights ===")
    high_missing = miss_df[miss_df["missing_percent"] > 5]
    if not high_missing.empty:
        print(f"  Features with >5% missing values ({len(high_missing)}):")
        for _, row in high_missing.iterrows():
            print(f"    {row['feature']:35s}: {row['missing_percent']:.1f}% missing")
    else:
        print("  No features have >5% missing values.")

    high_card = card_df[card_df["unique_values"] > 50]
    if not high_card.empty:
        print(f"\n  High-cardinality features (>50 unique values) — these required simplification:")
        for _, row in high_card.iterrows():
            print(f"    {row['feature']:35s}: {int(row['unique_values']):,} unique values")


if __name__ == "__main__":
    main()
