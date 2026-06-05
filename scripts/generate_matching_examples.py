"""Generate human-readable intake-outcome matching examples.

Produces examples of the matching algorithm behavior across five categories:
  1. Single intake + single future outcome
  2. Multiple intakes + multiple outcomes (repeat stays)
  3. Outcome before intake (must be skipped)
  4. Intake with no valid future outcome (unmatched)
  5. Repeated animal stay treated as separate episode

Outputs:
    reports/tables/matching_examples.csv
    reports/summary/matching_logic_examples.md
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matching logic examples")
    parser.add_argument("--intakes", default="data/raw/intakes.csv")
    parser.add_argument("--outcomes", default="data/raw/outcomes.csv")
    parser.add_argument(
        "--data",
        default="data/processed/modeling_dataset.csv",
        help="Final modeling dataset for cross-referencing",
    )
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--summary-dir", default="reports/summary")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max examples per category",
    )
    return parser.parse_args()


def _fmt_dt(dt) -> str:
    if pd.isna(dt):
        return "NaT"
    return str(pd.Timestamp(dt))[:19]


def _fmt_days(days) -> str:
    if pd.isna(days):
        return "N/A"
    return f"{float(days):.1f}"


def find_examples(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
    final_df: pd.DataFrame | None,
    max_examples: int = 5,
) -> list[dict]:
    """Find representative examples for each matching scenario."""
    examples: list[dict] = []

    # Group outcomes by animal
    outcomes_by_animal = {
        animal_id: group.sort_values("outcome_datetime").reset_index(drop=True)
        for animal_id, group in outcomes.groupby("animal_id", sort=False)
    }
    intakes_by_animal = {
        animal_id: group.sort_values("intake_datetime").reset_index(drop=True)
        for animal_id, group in intakes.groupby("animal_id", sort=False)
    }

    found = {
        "single_stay": 0,
        "multiple_stays": 0,
        "skipped_past_outcome": 0,
        "unmatched_intake": 0,
        "repeated_episode": 0,
    }
    targets = {k: max_examples for k in found}

    for animal_id, animal_intakes in intakes_by_animal.items():
        if all(v >= targets[k] for k, v in found.items()):
            break

        animal_outcomes = outcomes_by_animal.get(animal_id, pd.DataFrame())

        n_intakes = len(animal_intakes)
        n_outcomes = len(animal_outcomes)

        # Category 1: single intake, single future outcome
        if found["single_stay"] < targets["single_stay"] and n_intakes == 1 and n_outcomes == 1:
            intake_dt = animal_intakes.iloc[0]["intake_datetime"]
            outcome_dt = animal_outcomes.iloc[0]["outcome_datetime"]
            if outcome_dt >= intake_dt:
                days = (outcome_dt - intake_dt).total_seconds() / 86400
                if days >= 0:
                    examples.append({
                        "example_type": "single_stay",
                        "animal_id": animal_id,
                        "intake_datetime": _fmt_dt(intake_dt),
                        "candidate_outcome_datetimes": _fmt_dt(outcome_dt),
                        "selected_outcome_datetime": _fmt_dt(outcome_dt),
                        "days_to_outcome": _fmt_days(days),
                        "why_selected": "Only one outcome; it is after intake; matched directly.",
                    })
                    found["single_stay"] += 1

        # Category 2: multiple intakes and multiple outcomes (repeat stays as separate episodes)
        if found["multiple_stays"] < targets["multiple_stays"] and n_intakes >= 2 and n_outcomes >= 2:
            candidate_outcomes = animal_outcomes["outcome_datetime"].tolist()
            outcome_index = 0
            episode_count = 0
            episode_examples = []
            for _, intake_row in animal_intakes.iterrows():
                intake_dt = intake_row["intake_datetime"]
                # Skip past outcomes
                while outcome_index < len(candidate_outcomes) and candidate_outcomes[outcome_index] < intake_dt:
                    outcome_index += 1
                if outcome_index >= len(candidate_outcomes):
                    break
                selected_dt = candidate_outcomes[outcome_index]
                days = (selected_dt - intake_dt).total_seconds() / 86400
                episode_examples.append({
                    "intake_datetime": _fmt_dt(intake_dt),
                    "selected_outcome_datetime": _fmt_dt(selected_dt),
                    "days_to_outcome": _fmt_days(days),
                })
                outcome_index += 1
                episode_count += 1

            if episode_count >= 2:
                for i, ep in enumerate(episode_examples[:3], 1):
                    examples.append({
                        "example_type": "repeated_episode" if i > 1 else "multiple_stays",
                        "animal_id": animal_id,
                        "intake_datetime": ep["intake_datetime"],
                        "candidate_outcome_datetimes": f"Episode {i} of {episode_count} stays",
                        "selected_outcome_datetime": ep["selected_outcome_datetime"],
                        "days_to_outcome": ep["days_to_outcome"],
                        "why_selected": (
                            f"Episode {i}/{episode_count}: nearest unused future outcome assigned. "
                            "Each stay is a separate operational episode."
                        ),
                    })
                found["multiple_stays"] += 1
                found["repeated_episode"] += 1

        # Category 3: outcome before intake that must be skipped
        if found["skipped_past_outcome"] < targets["skipped_past_outcome"] and n_outcomes >= 2:
            if not animal_outcomes.empty and not animal_intakes.empty:
                first_intake = animal_intakes.iloc[0]["intake_datetime"]
                past_outcomes = animal_outcomes[animal_outcomes["outcome_datetime"] < first_intake]
                if not past_outcomes.empty:
                    past_dt = past_outcomes.iloc[0]["outcome_datetime"]
                    # Find a valid future outcome
                    future_outcomes = animal_outcomes[animal_outcomes["outcome_datetime"] >= first_intake]
                    if not future_outcomes.empty:
                        selected_dt = future_outcomes.iloc[0]["outcome_datetime"]
                        days = (selected_dt - first_intake).total_seconds() / 86400
                        all_candidates = " | ".join(
                            [_fmt_dt(past_dt)] + [_fmt_dt(d) for d in future_outcomes["outcome_datetime"].head(2)]
                        )
                        examples.append({
                            "example_type": "skipped_past_outcome",
                            "animal_id": animal_id,
                            "intake_datetime": _fmt_dt(first_intake),
                            "candidate_outcome_datetimes": all_candidates,
                            "selected_outcome_datetime": _fmt_dt(selected_dt),
                            "days_to_outcome": _fmt_days(days),
                            "why_selected": (
                                f"Outcome at {_fmt_dt(past_dt)} is BEFORE intake; skipped to avoid "
                                "negative days_to_outcome. Next valid outcome selected."
                            ),
                        })
                        found["skipped_past_outcome"] += 1

        # Category 4: unmatched intake (no valid future outcome)
        if found["unmatched_intake"] < targets["unmatched_intake"] and n_intakes >= 1:
            for _, intake_row in animal_intakes.iterrows():
                intake_dt = intake_row["intake_datetime"]
                if animal_outcomes.empty:
                    examples.append({
                        "example_type": "unmatched_intake",
                        "animal_id": animal_id,
                        "intake_datetime": _fmt_dt(intake_dt),
                        "candidate_outcome_datetimes": "none",
                        "selected_outcome_datetime": "none — excluded from dataset",
                        "days_to_outcome": "N/A",
                        "why_selected": (
                            "No outcome record exists for this animal. "
                            "Cannot form a supervised episode (no label y). Excluded from final dataset."
                        ),
                    })
                    found["unmatched_intake"] += 1
                    break
                else:
                    # All outcomes are before this intake
                    future = animal_outcomes[animal_outcomes["outcome_datetime"] >= intake_dt]
                    if future.empty:
                        all_outcomes = " | ".join(
                            _fmt_dt(d) for d in animal_outcomes["outcome_datetime"].head(3)
                        )
                        examples.append({
                            "example_type": "unmatched_intake",
                            "animal_id": animal_id,
                            "intake_datetime": _fmt_dt(intake_dt),
                            "candidate_outcome_datetimes": all_outcomes,
                            "selected_outcome_datetime": "none — excluded from dataset",
                            "days_to_outcome": "N/A",
                            "why_selected": (
                                "All outcome records are before this intake datetime. "
                                "No future outcome available; intake excluded from final dataset."
                            ),
                        })
                        found["unmatched_intake"] += 1
                        break

    return examples


def write_summary_md(examples: list[dict], output_path: Path) -> None:
    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        by_type.setdefault(ex["example_type"], []).append(ex)

    descriptions = {
        "single_stay": "## Category 1: Single Intake + Single Future Outcome\n\nThe simplest case. One intake, one outcome after intake. Matched directly.",
        "multiple_stays": "## Category 2: Multiple Intakes + Multiple Outcomes\n\nAnimal had several shelter stays. Each intake is matched to the nearest unused future outcome.",
        "repeated_episode": "## Category 5: Repeated Animal Stay as Separate Episode\n\nSame animal as above, later stay episode. Each stay is treated independently.",
        "skipped_past_outcome": "## Category 3: Outcome Before Intake — Skipped\n\nAn outcome record exists before this intake (from a prior stay). The algorithm skips it to avoid negative days_to_outcome.",
        "unmatched_intake": "## Category 4: Intake with No Valid Future Outcome\n\nNo future outcome found. This intake is excluded from the final modeling dataset.",
    }

    lines = [
        "# Matching Logic Examples",
        "",
        "## Purpose",
        "",
        "These examples demonstrate the **nearest-unused-future-outcome matching** algorithm",
        "used to construct intake/outcome episode pairs.",
        "",
        "**Key invariants guaranteed by the algorithm:**",
        "- No negative `days_to_outcome` values.",
        "- No outcome record is reused for multiple intake episodes.",
        "- Each repeat shelter stay is treated as a separate operational episode.",
        "- Intakes without any future outcome are excluded from the final dataset.",
        "",
    ]

    for cat_type, cat_desc in descriptions.items():
        exs = by_type.get(cat_type, [])
        lines.append(cat_desc)
        lines.append("")
        if not exs:
            lines.append("_No examples found in current dataset._")
            lines.append("")
            continue
        for ex in exs:
            lines += [
                f"- **Animal ID:** {ex['animal_id']}",
                f"  - Intake: {ex['intake_datetime']}",
                f"  - Candidates: {ex['candidate_outcome_datetimes']}",
                f"  - **Selected:** {ex['selected_outcome_datetime']}",
                f"  - Days to outcome: {ex['days_to_outcome']}",
                f"  - Why: {ex['why_selected']}",
                "",
            ]

    lines += [
        "## Algorithm Reference",
        "",
        "```python",
        "# src/aac_adoption/data/match_records.py",
        "# For each animal_id:",
        "#   Sort intakes by intake_datetime",
        "#   Sort outcomes by outcome_datetime",
        "#   For each intake (in order):",
        "#     Skip outcomes where outcome_datetime < intake_datetime",
        "#     Assign next available outcome (mark as used)",
        "#     If no future outcome: count as unmatched_intake (excluded)",
        "```",
        "",
        "Full implementation: `src/aac_adoption/data/match_records.py`",
        "Test coverage: `tests/`",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    summary_dir = Path(args.summary_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    intakes = load_intakes(args.intakes)
    intakes = __import__("aac_adoption.data.clean_data", fromlist=["clean_intakes"]).clean_intakes(intakes)
    outcomes = load_outcomes(args.outcomes)
    outcomes = __import__("aac_adoption.data.clean_data", fromlist=["clean_outcomes"]).clean_outcomes(outcomes)

    final_df = None
    if Path(args.data).exists():
        final_df = pd.read_csv(args.data)
        print(f"  Modeling dataset: {len(final_df):,} rows")

    print("Searching for matching examples...")
    examples = find_examples(intakes, outcomes, final_df, max_examples=args.max_examples)

    if not examples:
        print("No examples found — check that intakes/outcomes files are loaded correctly.")
        return

    df = pd.DataFrame(examples)

    # Verify: no negative days_to_outcome
    numeric_days = pd.to_numeric(df["days_to_outcome"], errors="coerce")
    negatives = (numeric_days < 0).sum()
    if negatives > 0:
        print(f"WARNING: {negatives} example(s) have negative days_to_outcome — check matching logic!")
    else:
        print(f"PASS: Verified: no negative days_to_outcome in examples.")

    out_csv = tables_dir / "matching_examples.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_md = summary_dir / "matching_logic_examples.md"
    write_summary_md(examples, out_md)
    print(f"Wrote {out_md}")

    print(f"\n  Examples by category:")
    for cat, count in df["example_type"].value_counts().items():
        print(f"    {cat:35s}: {count}")


if __name__ == "__main__":
    main()
