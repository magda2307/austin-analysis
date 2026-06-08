"""Date-aware intake/outcome record matching."""

import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchResult:
    matched_episodes: pd.DataFrame
    unresolved_intakes: pd.DataFrame
    unmatched_intakes: int


def verify_reintake_patterns(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Verify re-intake patterns and track episodes."""
    episodes = []
    
    intakes_sorted = intakes.sort_values("intake_datetime").to_dict("records")
    intakes_grouped = {}
    for row in intakes_sorted:
        intakes_grouped.setdefault(row["animal_id"], []).append(row)

    outcomes_sorted = outcomes.sort_values("outcome_datetime").to_dict("records")
    outcomes_grouped = {}
    for row in outcomes_sorted:
        outcomes_grouped.setdefault(row["animal_id"], []).append(row)

    for animal_id, animal_intakes in intakes_grouped.items():
        animal_outcomes = outcomes_grouped.get(animal_id, [])
        
        for intake in animal_intakes:
            intake_time = intake["intake_datetime"]
            
            # Find prev outcomes using loop over sorted list
            prev_outcomes = [o for o in animal_outcomes if o["outcome_datetime"] < intake_time]
            
            if len(prev_outcomes) > 0:
                last_outcome = prev_outcomes[-1]
                days_since_last_stay = (intake_time - last_outcome["outcome_datetime"]).days
            else:
                days_since_last_stay = None
            
            episodes.append({
                "animal_id": animal_id,
                "intake_datetime": intake_time,
                "episode_number": len(prev_outcomes) + 1,
                "days_since_last_stay": days_since_last_stay,
                "is_reintake": len(prev_outcomes) > 0,
            })
    
    return pd.DataFrame(episodes)


def infer_censoring_reason(outcome_type: str) -> str:
    """Determine censoring reason based on outcome type."""
    outcome_lower = str(outcome_type).lower().strip() if pd.notna(outcome_type) else ""
    if outcome_lower == "adoption":
        return "adopted"
    elif outcome_lower in ["transfer", "moved", "reintake"]:
        return "censored_transfer"
    elif outcome_lower in ["euthanize", "euthanasia"]:
        return "censored_euthanasia"
    elif outcome_lower in ["return_to_owner", "returned"]:
        return "censored_return"
    elif outcome_lower in ["din", "disappear", "missing"]:
        return "censored_lost"
    else:
        return "censored_unknown"


def match_intakes_to_future_outcomes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    extract_end_date: pd.Timestamp | None = None,
) -> MatchResult:
    """Match each intake to nearest unused future outcome for same animal.

    AAC animals can have repeat stays. This creates one row per intake episode
    when a valid future outcome exists. Outcome rows are not reused for the same
    animal. Intakes without a future outcome are skipped and counted.
    """
    rows: list[dict] = []
    unmatched_intakes = 0

    episodes = verify_reintake_patterns(intakes, outcomes)
    episodes_sorted = episodes.to_dict("records")
    episodes_by_animal = {}
    for row in episodes_sorted:
        episodes_by_animal.setdefault(row["animal_id"], []).append(row)

    outcomes_sorted = outcomes.sort_values("outcome_datetime").to_dict("records")
    outcomes_by_animal = {}
    for row in outcomes_sorted:
        outcomes_by_animal.setdefault(row["animal_id"], []).append(row)
    
    if "intake_datetime" not in intakes.columns:
        raise ValueError("Intake data must contain intake_datetime column")
    if "outcome_datetime" not in outcomes.columns:
        raise ValueError("Outcome data must contain outcome_datetime column")

    intakes_sorted = intakes.sort_values("intake_datetime").to_dict("records")
    intakes_by_animal = {}
    for row in intakes_sorted:
        intakes_by_animal.setdefault(row["animal_id"], []).append(row)

    unresolved_rows = []

    for animal_id, animal_intakes_list in intakes_by_animal.items():
        outcome_records = outcomes_by_animal.get(animal_id, [])
        outcome_index = 0

        animal_episodes = episodes_by_animal.get(animal_id, [])
        episodes_by_time = {row["intake_datetime"]: row for row in animal_episodes}

        for i, intake in enumerate(animal_intakes_list):
            next_intake_time = None
            if i + 1 < len(animal_intakes_list):
                next_intake_time = animal_intakes_list[i + 1]["intake_datetime"]

            while (
                outcome_index < len(outcome_records)
                and outcome_records[outcome_index]["outcome_datetime"] < intake["intake_datetime"]
            ):
                outcome_index += 1

            if outcome_index < len(outcome_records):
                outcome = outcome_records[outcome_index]
                
                # Check if outcome belongs to next intake episode
                if next_intake_time is not None and outcome["outcome_datetime"] >= next_intake_time:
                    unmatched_intakes += 1
                    if extract_end_date is None:
                        raise ValueError("extract_end_date is required to process unresolved intakes")
                    
                    row = intake.copy()
                    row["unresolved_reason"] = "no_unused_future_outcome"
                    row["observation_end"] = extract_end_date
                    row["followup_days_available"] = (extract_end_date - intake["intake_datetime"]).days
                    
                    episode_info = episodes_by_time.get(intake["intake_datetime"], {})
                    row["episode_number"] = episode_info.get("episode_number", 1)
                    row["is_reintake"] = episode_info.get("is_reintake", False)
                    row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
                    unresolved_rows.append(row)
                    continue

                row = intake.copy()
                for key, value in outcome.items():
                    if key != "animal_id":
                        row[key] = value
                
                episode_info = episodes_by_time.get(intake["intake_datetime"], {})
                row["episode_number"] = episode_info.get("episode_number", 1)
                row["is_reintake"] = episode_info.get("is_reintake", False)
                row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
                
                reason = infer_censoring_reason(outcome.get("outcome_type"))
                row["censoring_reason"] = reason
                row["event_type"] = "adoption" if reason == "adopted" else "censored"
                row["censoring_date"] = outcome.get("outcome_datetime")
                
                rows.append(row)
                outcome_index += 1
            else:
                unmatched_intakes += 1
                if extract_end_date is None:
                    raise ValueError("extract_end_date is required to process unresolved intakes")
                
                row = intake.copy()
                row["unresolved_reason"] = "no_unused_future_outcome"
                row["observation_end"] = extract_end_date
                row["followup_days_available"] = (extract_end_date - intake["intake_datetime"]).days
                
                episode_info = episodes_by_time.get(intake["intake_datetime"], {})
                row["episode_number"] = episode_info.get("episode_number", 1)
                row["is_reintake"] = episode_info.get("is_reintake", False)
                row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
                unresolved_rows.append(row)

    return MatchResult(
        matched_episodes=pd.DataFrame(rows),
        unresolved_intakes=pd.DataFrame(unresolved_rows) if unresolved_rows else pd.DataFrame(columns=list(intakes.columns) + ["unresolved_reason", "observation_end", "followup_days_available", "episode_number", "is_reintake", "days_since_last_stay"]),
        unmatched_intakes=unmatched_intakes
    )
