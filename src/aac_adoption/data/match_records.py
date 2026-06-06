"""Date-aware intake/outcome record matching."""

import pandas as pd


def verify_reintake_patterns(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Verify re-intake patterns and track episodes."""
    episodes = []
    
    for animal_id in intakes["animal_id"].unique():
        animal_intakes = intakes[intakes["animal_id"] == animal_id].sort_values("intake_datetime")
        animal_outcomes = outcomes[outcomes["animal_id"] == animal_id].sort_values("outcome_datetime")
        
        for idx, intake in animal_intakes.iterrows():
            intake_time = intake["intake_datetime"]
            
            prev_outcomes = animal_outcomes[animal_outcomes["outcome_datetime"] < intake_time]
            if len(prev_outcomes) > 0:
                last_outcome = prev_outcomes.iloc[-1]
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

    outcomes_by_animal = {
        animal_id: group.sort_values("outcome_datetime").to_dict("records")
        for animal_id, group in outcomes.groupby("animal_id", sort=False)
    }

    for animal_id, animal_intakes in intakes.sort_values("intake_datetime").groupby(
        "animal_id", sort=False
    ):
        outcome_records = outcomes_by_animal.get(animal_id, [])
        outcome_index = 0

        for intake in animal_intakes.to_dict("records"):
            while (
                outcome_index < len(outcome_records)
                and outcome_records[outcome_index]["outcome_datetime"] < intake["intake_datetime"]
            ):
                outcome_index += 1

            if outcome_index >= len(outcome_records):
                unmatched_intakes += 1
                continue

            outcome = outcome_records[outcome_index]
            outcome_index += 1

            row = dict(intake)
            for key, value in outcome.items():
                if key != "animal_id":
                    row[key] = value
            
            episode_info = episodes[episodes["intake_datetime"] == intake["intake_datetime"]].iloc[0].to_dict()
            row["episode_number"] = episode_info.get("episode_number", 1)
            row["is_reintake"] = episode_info.get("is_reintake", False)
            row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
            
            # Check for ambiguity: does another intake happen before this outcome?
            animal_intakes_list = animal_intakes.to_dict("records")
            row["is_ambiguous_match"] = False
            for next_intake in animal_intakes_list:
                if next_intake["intake_datetime"] > intake["intake_datetime"] and next_intake["intake_datetime"] < outcome["outcome_datetime"]:
                    row["is_ambiguous_match"] = True
                    break
            
            rows.append(row)

    return pd.DataFrame(rows), unmatched_intakes

