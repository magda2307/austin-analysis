import pandas as pd

from aac_adoption.analysis.animal_profiles import add_animal_descriptors, animal_archetypes, profile_contrasts


def _animal_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat", "Cat"],
            "age_group": ["senior", "baby", "senior", "baby"],
            "intake_type": ["Stray", "Stray", "Owner Surrender", "Stray"],
            "intake_condition": ["Behavior", "Normal", "Sick", "Nursing"],
            "outcome_subtype": ["Aggressive", None, "Medical", "Foster"],
            "simplified_breed_group": ["pit_bull_type", "retriever_type", "domestic_cat", "domestic_cat"],
            "simplified_color_group": ["black_or_dark", "brown_tan", "black_or_dark", "white_light"],
            "sex_upon_intake": ["Intact Male", "Spayed Female", "Unknown", "Unknown"],
            "is_named": [True, False, True, False],
            "classification_target": [0, 1, 0, 1],
            "days_to_outcome": [30.0, 5.0, 20.0, 3.0],
            "outcome_type": ["Euthanasia", "Adoption", "Transfer", "Adoption"],
        }
    )


def test_animal_descriptors_and_archetypes_include_health_behavior():
    df = add_animal_descriptors(_animal_rows())
    archetypes = animal_archetypes(df, min_records=1)

    assert "health_profile" in archetypes.columns
    assert "behavior_support_flag" in archetypes.columns
    assert "behavior_support_signal" in set(df["behavior_support_flag"])


def test_profile_contrasts_include_cat_health_and_named_views():
    df = add_animal_descriptors(_animal_rows())
    contrasts = profile_contrasts(df)

    assert "black_or_dark_cats_vs_other_cats" in set(contrasts["contrast"])
    assert "health_profile_by_species" in set(contrasts["contrast"])
    assert "behavior_support_signal_by_species" in set(contrasts["contrast"])
