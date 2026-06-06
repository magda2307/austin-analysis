"""Date-aware intake/outcome record matching."""

import pandas as pd


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


def match_intakes_to_future_outcomes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
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

    intakes_sorted = intakes.sort_values("intake_datetime").to_dict("records")
    intakes_by_animal = {}
    for row in intakes_sorted:
        intakes_by_animal.setdefault(row["animal_id"], []).append(row)

    for animal_id, animal_intakes_list in intakes_by_animal.items():
        outcome_records = outcomes_by_animal.get(animal_id, [])
        outcome_index = 0

        animal_episodes = episodes_by_animal.get(animal_id, [])
        episodes_by_time = {row["intake_datetime"]: row for row in animal_episodes}

        for intake in animal_intakes_list:
            while (
                outcome_index < len(outcome_records)
                and outcome_records[outcome_index]["outcome_datetime"] < intake["intake_datetime"]
            ):
                outcome_index += 1

            if outcome_index < len(outcome_records):
                outcome = outcome_records[outcome_index]
                
                row = intake.copy()
                for key, value in outcome.items():
                    if key != "animal_id":
                        row[key] = value
                
                episode_info = episodes_by_time.get(intake["intake_datetime"], {})
                row["episode_number"] = episode_info.get("episode_number", 1)
                row["is_reintake"] = episode_info.get("is_reintake", False)
                row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
                
                # Check for ambiguity: does another intake happen before this outcome?
                row["is_ambiguous_match"] = False
                for next_intake in animal_intakes_list:
                    if next_intake["intake_datetime"] > intake["intake_datetime"] and next_intake["intake_datetime"] < outcome["outcome_datetime"]:
                        row["is_ambiguous_match"] = True
                        break
                
                rows.append(row)
                outcome_index += 1
            else:
                unmatched_intakes += 1

    return pd.DataFrame(rows), unmatched_intakes
