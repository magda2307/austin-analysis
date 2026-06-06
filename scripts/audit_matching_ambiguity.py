"""Audit the modeling dataset for ambiguous re-intakes (overlapping stays)."""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading modeling dataset for ambiguity audit...")
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        logger.error(f"Missing {data_path}")
        return

    df = pd.read_csv(data_path, parse_dates=["intake_datetime", "outcome_datetime"])
    
    # Sort by animal and intake time
    df = df.sort_values(by=["animal_id", "intake_datetime"])

    # Find next intake datetime for the same animal
    df["next_intake"] = df.groupby("animal_id")["intake_datetime"].shift(-1)

    # An episode is ambiguous if the next intake occurs BEFORE the outcome of the current intake
    df["is_ambiguous"] = df["next_intake"].notna() & (df["next_intake"] < df["outcome_datetime"])

    ambiguous_count = df["is_ambiguous"].sum()
    total_count = len(df)
    clean_count = total_count - ambiguous_count

    logger.info(f"Audit complete: {ambiguous_count} ambiguous overlapping episodes found out of {total_count}.")

    report = f"""## Re-Intake Matching Ambiguity Audit

This audit checks if the greedy outcome-matching process improperly assigns outcomes by searching for episodes where an animal has *another intake* recorded before its assigned outcome.

### Findings

| Metric | Count |
|--------|-------|
| Total Matched Episodes | {total_count:,} |
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

    report_path = Path("reports/summary/matching_ambiguity.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Wrote report to {report_path}")

if __name__ == "__main__":
    main()
