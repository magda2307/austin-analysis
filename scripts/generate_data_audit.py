"""Generate data attrition table showing row counts at each pipeline stage.

Outputs:
    reports/tables/data_audit_attrition.csv
    reports/summary/data_audit.md

The table documents exactly how raw CSVs became the final modeling dataset,
making the dataset defensible for thesis reviewers and replication.
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

from aac_adoption.data.clean_data import clean_intakes, clean_outcomes
from aac_adoption.data.load_data import load_intakes, load_outcomes
from aac_adoption.data.match_records import match_intakes_to_future_outcomes

HORIZON_DAYS = (7, 30, 60, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data attrition audit table")
    parser.add_argument(
        "--intakes",
        default="data/raw/intakes.csv",
        help="Path to raw intakes CSV",
    )
    parser.add_argument(
        "--outcomes",
        default="data/raw/outcomes.csv",
        help="Path to raw outcomes CSV",
    )
    parser.add_argument(
        "--data",
        default="data/processed/modeling_dataset.csv",
        help="Path to final modeling dataset CSV (for final row count verification)",
    )
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--summary-dir", default="reports/summary")
    return parser.parse_args()


def _count_stats(df: pd.DataFrame, id_col: str = "animal_id") -> dict:
    """Compute summary stats for one stage dataframe."""
    stats: dict = {
        "rows": len(df),
        "dog_rows": int(df["animal_type"].astype(str).str.lower().eq("dog").sum()) if "animal_type" in df.columns else None,
        "cat_rows": int(df["animal_type"].astype(str).str.lower().eq("cat").sum()) if "animal_type" in df.columns else None,
        "unique_animals": df[id_col].nunique() if id_col in df.columns else None,
        "duplicate_animal_ids": int(df.duplicated(subset=[id_col]).sum()) if id_col in df.columns else None,
    }
    if "intake_datetime" in df.columns:
        dates = pd.to_datetime(df["intake_datetime"], errors="coerce").dropna()
        stats["min_intake_date"] = str(dates.min().date()) if not dates.empty else None
        stats["max_intake_date"] = str(dates.max().date()) if not dates.empty else None
    else:
        stats["min_intake_date"] = None
        stats["max_intake_date"] = None
    return stats


def build_attrition_table(
    intakes_path: str,
    outcomes_path: str,
    final_data_path: str,
) -> pd.DataFrame:
    """Run each pipeline stage and record row counts."""
    records = []

    # Stage 1: raw intakes
    raw_intakes = pd.read_csv(intakes_path, low_memory=False)
    raw_intakes.columns = [c.strip().lower().replace(" ", "_") for c in raw_intakes.columns]
    if "datetime" in raw_intakes.columns and "intake_datetime" not in raw_intakes.columns:
        raw_intakes = raw_intakes.rename(columns={"datetime": "intake_datetime"})
    s = _count_stats(raw_intakes)
    records.append({
        "stage": "raw_intakes",
        "rows": s["rows"],
        "rows_removed": None,
        "reason": "Source records from intakes.csv",
        **{k: s[k] for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
    })
    prev_intake_rows = s["rows"]

    # Stage 2: raw outcomes
    raw_outcomes = pd.read_csv(outcomes_path, low_memory=False)
    raw_outcomes.columns = [c.strip().lower().replace(" ", "_") for c in raw_outcomes.columns]
    if "datetime" in raw_outcomes.columns and "outcome_datetime" not in raw_outcomes.columns:
        raw_outcomes = raw_outcomes.rename(columns={"datetime": "outcome_datetime"})
    s2 = _count_stats(raw_outcomes)
    records.append({
        "stage": "raw_outcomes",
        "rows": s2["rows"],
        "rows_removed": None,
        "reason": "Source records from outcomes.csv",
        "dog_rows": s2["dog_rows"],
        "cat_rows": s2["cat_rows"],
        "unique_animals": s2["unique_animals"],
        "duplicate_animal_ids": s2["duplicate_animal_ids"],
        "min_intake_date": None,
        "max_intake_date": None,
    })

    # Stage 3: standardized intakes (after cleaning)
    std_intakes = load_intakes(intakes_path)
    std_intakes_clean = clean_intakes(std_intakes)
    s3 = _count_stats(std_intakes_clean)
    records.append({
        "stage": "standardized_intakes",
        "rows": s3["rows"],
        "rows_removed": prev_intake_rows - s3["rows"],
        "reason": "Removed: invalid datetime, missing required fields, exact duplicates",
        **{k: s3[k] for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
    })

    # Stage 4: standardized outcomes
    std_outcomes = load_outcomes(outcomes_path)
    std_outcomes_clean = clean_outcomes(std_outcomes)
    s4 = _count_stats(std_outcomes_clean)
    records.append({
        "stage": "standardized_outcomes",
        "rows": s4["rows"],
        "rows_removed": s2["rows"] - s4["rows"],
        "reason": "Removed: invalid datetime, missing required fields, exact duplicates",
        "dog_rows": s4["dog_rows"],
        "cat_rows": s4["cat_rows"],
        "unique_animals": s4["unique_animals"],
        "duplicate_animal_ids": s4["duplicate_animal_ids"],
        "min_intake_date": None,
        "max_intake_date": None,
    })

    # Stage 5: dog/cat filter is already applied in clean_intakes (filter_cats_and_dogs)
    # standardized_intakes already = dog/cat only; document this explicitly
    records.append({
        "stage": "dog_cat_intakes",
        "rows": s3["rows"],
        "rows_removed": 0,
        "reason": "filter_cats_and_dogs() applied during standardization; non-dog/cat rows already removed",
        **{k: s3[k] for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
    })

    # Stage 6: matched future outcomes
    INTAKE_COLS = ["animal_id", "name", "animal_type", "intake_datetime", "intake_type",
                   "intake_condition", "sex_upon_intake", "age_upon_intake", "breed", "color", "found_location"]
    OUTCOME_COLS = ["animal_id", "outcome_datetime", "outcome_type", "outcome_subtype",
                    "sex_upon_outcome", "age_upon_outcome"]

    intake_subset = std_intakes_clean[[c for c in INTAKE_COLS if c in std_intakes_clean.columns]]
    outcome_subset = std_outcomes_clean[[c for c in OUTCOME_COLS if c in std_outcomes_clean.columns]]
    match_result = match_intakes_to_future_outcomes(intake_subset, outcome_subset, extract_end_date=pd.Timestamp.now())
    matched = match_result.matched_episodes
    unmatched = match_result.unmatched_intakes
    s6 = _count_stats(matched)
    records.append({
        "stage": "matched_future_outcomes",
        "rows": s6["rows"],
        "rows_removed": s3["rows"] - s6["rows"],
        "reason": f"Removed {unmatched:,} intakes without a valid future outcome (cannot form supervised episode)",
        **{k: s6[k] for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
    })
    
    global _matched_episodes_for_ambiguity, _unresolved_intakes_for_ambiguity
    _matched_episodes_for_ambiguity = s6["rows"]
    _unresolved_intakes_for_ambiguity = unmatched

    # Stage 7: final modeling dataset
    if Path(final_data_path).exists():
        final = pd.read_csv(final_data_path)

        ambiguous = final["is_ambiguous_match"].sum() if "is_ambiguous_match" in final.columns else 0
        records.append({
            "stage": "reintake_ambiguity_audit",
            "rows": len(final),
            "rows_removed": 0,
            "reason": f"Found {ambiguous:,} ambiguous episodes where another intake occurs before the outcome.",
            **{k: None for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
        })

        if "followup_days_available" in final.columns:
            censored_7d = (final["followup_days_available"] < 7).sum()
            censored_30d = (final["followup_days_available"] < 30).sum()
            censored_60d = (final["followup_days_available"] < 60).sum()

            records.append({
                "stage": "horizon_censoring_audit",
                "rows": len(final),
                "rows_removed": 0,
                "reason": f"Censoring exclusions if active: {censored_7d:,} (7d), {censored_30d:,} (30d), {censored_60d:,} (60d) dropped near dataset end.",
                **{k: None for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
            })

        s7 = _count_stats(final)
        records.append({
            "stage": "final_modeling_dataset",
            "rows": s7["rows"],
            "rows_removed": None,
            "reason": "Final episode-level dataset with features and targets; matches build_modeling_dataset() output",
            **{k: s7[k] for k in ["dog_rows", "cat_rows", "unique_animals", "duplicate_animal_ids", "min_intake_date", "max_intake_date"]},
        })
    else:
        records.append({
            "stage": "final_modeling_dataset",
            "rows": None,
            "rows_removed": None,
            "reason": f"Not found at {final_data_path}; run build_dataset.py first",
            "dog_rows": None, "cat_rows": None, "unique_animals": None,
            "duplicate_animal_ids": None, "min_intake_date": None, "max_intake_date": None,
        })

    return pd.DataFrame(records)


def build_horizon_followup_audit(final_data_path: str) -> pd.DataFrame:
    """Summarize horizon-label inclusion and censoring counts."""
    path = Path(final_data_path)
    columns = [
        "horizon_days",
        "extract_end_date",
        "total_rows",
        "included_rows",
        "censored_rows",
        "excluded_rows",
        "full_followup_rows",
        "short_followup_rows",
        "fast_adopted_short_followup_rows",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    final = pd.read_csv(path, parse_dates=["intake_datetime", "outcome_datetime"])
    if final.empty:
        return pd.DataFrame(columns=columns)

    if "followup_days_available" not in final.columns:
        snapshot_date = max(final["intake_datetime"].max(), final["outcome_datetime"].max())
        final["followup_days_available"] = (
            snapshot_date - final["intake_datetime"]
        ).dt.total_seconds() / 86400
    else:
        inferred_snapshot = final["intake_datetime"] + pd.to_timedelta(
            final["followup_days_available"], unit="D"
        )
        snapshot_date = inferred_snapshot.max()

    rows = []
    for horizon in HORIZON_DAYS:
        target_col = f"adopted_in_{horizon}d"
        if target_col not in final.columns:
            rows.append({
                "horizon_days": horizon,
                "extract_end_date": snapshot_date.date().isoformat(),
                "total_rows": len(final),
                "included_rows": 0,
                "censored_rows": len(final),
                "excluded_rows": len(final),
                "full_followup_rows": 0,
                "short_followup_rows": len(final),
                "fast_adopted_short_followup_rows": 0,
            })
            continue

        full_followup = final["followup_days_available"] >= horizon
        adopted = final.get("adopted", pd.Series(False, index=final.index)).astype(bool)
        fast_adopted = adopted & (final["days_to_outcome"] <= horizon)
        observed_target = final[target_col].notna()
        censored = ~observed_target

        rows.append({
            "horizon_days": horizon,
            "extract_end_date": snapshot_date.date().isoformat(),
            "total_rows": len(final),
            "included_rows": int(observed_target.sum()),
            "censored_rows": int(censored.sum()),
            "excluded_rows": int(censored.sum()),
            "full_followup_rows": int(full_followup.sum()),
            "short_followup_rows": int((~full_followup).sum()),
            "fast_adopted_short_followup_rows": int(((~full_followup) & fast_adopted).sum()),
        })

    return pd.DataFrame(rows, columns=columns)


def write_summary_md(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Data Attrition Audit",
        "",
        "## Summary",
        "",
        "This table documents how raw Austin Animal Center CSV files become the final supervised ML dataset.",
        "Removed rows are not arbitrary — each removal is required for methodologically sound episode-level supervised learning.",
        "",
        "## Attrition Table",
        "",
        "| Stage | Rows | Removed | Reason |",
        "|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        removed = str(int(row["rows_removed"])) if pd.notna(row["rows_removed"]) else "—"
        rows = str(int(row["rows"])) if pd.notna(row["rows"]) else "?"
        lines.append(f"| {row['stage']} | {rows:>10} | {removed:>8} | {row['reason']} |")

    lines += [
        "",
        "## Why Rows Are Removed",
        "",
        "- **Invalid datetime / missing fields**: Records without a parseable intake_datetime or animal_id",
        "  cannot be placed on a timeline and cannot form a valid episode.",
        "- **Exact duplicates**: Exact-duplicate rows are administrative data quality issues; they would",
        "  artificially inflate episode counts.",
        "- **Non-dog/cat records**: Thesis scope is limited to dogs and cats (the two dominant species at AAC).",
        "- **No valid future outcome**: An intake without a subsequent outcome record cannot form a",
        "  supervised learning episode (we need both X and y). These intakes may represent ongoing stays",
        "  at data snapshot time or data entry gaps.",
        "",
        "## Matching Logic",
        "",
        "Each intake is matched to the **nearest unused future outcome** for the same animal.",
        "The algorithm is a greedy nearest-future-match: outcomes before the intake are skipped;",
        "each outcome is used at most once. This prevents negative length-of-stay values and",
        "prevents one outcome from being assigned to multiple intake episodes.",
        "",
        "See `docs/METHODOLOGY.md` for full matching logic documentation.",
        "See `reports/tables/matching_examples.csv` for human-readable examples.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def append_horizon_summary(output_path: Path, horizon_df: pd.DataFrame) -> None:
    """Append horizon censoring summary to the data audit Markdown."""
    lines = output_path.read_text(encoding="utf-8").splitlines()
    lines += [
        "",
        "## Horizon Follow-Up Audit",
        "",
        "Horizon adoption targets are observed only when an intake has enough possible follow-up time,",
        "unless the animal was adopted within the horizon before the snapshot boundary.",
        "",
        "| Horizon | Included | Censored / Excluded | Fast adopted with short follow-up | Extract end date |",
        "|---:|---:|---:|---:|---|",
    ]
    if horizon_df.empty:
        lines.append("| ? | 0 | 0 | 0 | unavailable |")
    else:
        for _, row in horizon_df.iterrows():
            lines.append(
                f"| {int(row['horizon_days'])}d | {int(row['included_rows'])} | "
                f"{int(row['censored_rows'])} | {int(row['fast_adopted_short_followup_rows'])} | "
                f"{row['extract_end_date']} |"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_matching_ambiguity_md(df: pd.DataFrame, final_data_path: str, output_path: Path) -> None:
    try:
        final = pd.read_csv(final_data_path)
        ambiguous_count = final["is_ambiguous_match"].sum() if "is_ambiguous_match" in final.columns else 0
    except Exception:
        ambiguous_count = 0

    total_count = _matched_episodes_for_ambiguity
    unresolved_count = _unresolved_intakes_for_ambiguity
    clean_count = total_count - ambiguous_count

    report = f"""## Re-Intake Matching Ambiguity Audit

This audit checks if the greedy outcome-matching process improperly assigns outcomes by searching for episodes where an animal has *another intake* recorded before its assigned outcome.

### Findings

| Metric | Count |
|--------|-------|
| Total Matched Episodes | {total_count:,} |
| Unresolved Intakes | {unresolved_count:,} |
| Clean Episodes | {clean_count:,} |
| Ambiguous/Overlapping Episodes | {ambiguous_count:,} |

**Conclusion:** 
"""
    if ambiguous_count == 0:
        report += "The greedy matching correctly avoids overlap. All matched episodes are clean.\n"
    elif ambiguous_count < (total_count * 0.05):
        report += f"A small fraction ({(ambiguous_count/total_count)*100:.2f}%) of episodes overlap. This is acceptable for modeling, but these rows represent data entry anomalies at the shelter (e.g., animal returned before previous outcome was recorded).\n"
    else:
        report += "A significant number of episodes overlap. The matching logic (`data/match_records.py`) should be reviewed to reject these overlapping spans.\n"

    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    summary_dir = Path(args.summary_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("Building data attrition table...")
    df = build_attrition_table(args.intakes, args.outcomes, args.data)

    out_csv = tables_dir / "data_audit_attrition.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_md = summary_dir / "data_audit.md"
    write_summary_md(df, out_md)
    horizon_df = build_horizon_followup_audit(args.data)
    horizon_csv = tables_dir / "horizon_followup_audit.csv"
    horizon_df.to_csv(horizon_csv, index=False)
    append_horizon_summary(out_md, horizon_df)
    print(f"Wrote {horizon_csv}")
    print(f"Wrote {out_md}")
    
    ambiguity_md = summary_dir / "matching_ambiguity.md"
    write_matching_ambiguity_md(df, args.data, ambiguity_md)
    print(f"Wrote {ambiguity_md}")

    # Print summary to console
    print("\n=== Data Attrition Summary ===")
    for _, row in df.iterrows():
        rows = f"{int(row['rows']):,}" if pd.notna(row["rows"]) else "?"
        removed = f"(-{int(row['rows_removed']):,})" if pd.notna(row["rows_removed"]) and row["rows_removed"] else ""
        print(f"  {row['stage']:35s}: {rows:>10} rows {removed}")


if __name__ == "__main__":
    main()
