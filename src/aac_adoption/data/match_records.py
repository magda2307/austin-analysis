"""Date-aware intake/outcome record matching."""

import pandas as pd


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
            rows.append(row)

    return pd.DataFrame(rows), unmatched_intakes

