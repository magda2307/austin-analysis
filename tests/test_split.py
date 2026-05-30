import pandas as pd

from aac_adoption.models.split import make_time_split


def test_make_time_split_uses_thesis_year_boundaries():
    df = pd.DataFrame(
        {
            "intake_year": [2019, 2021, 2022, 2023, 2024, 2025],
            "animal_type": ["Dog", "Cat", "Dog", "Cat", "Dog", "Cat"],
            "classification_target": [1, 0, 1, 0, 1, 0],
            "animal_id": list("ABCDEF"),
        }
    )

    split = make_time_split(df, "classification_target")

    assert set(split.train["intake_year"]) == {2019, 2021}
    assert set(split.validation["intake_year"]) == {2022, 2023}
    assert set(split.test["intake_year"]) == {2024, 2025}
    assert split.strategy == "time"
    assert split.train_period == "2013-2021"
    assert split.validation_period == "2022-2023"
    assert split.test_period == "2024-2025"


def test_make_time_split_filters_animal_subset():
    df = pd.DataFrame(
        {
            "intake_year": [2020, 2021, 2024, 2025],
            "animal_type": ["Dog", "Cat", "Dog", "Cat"],
            "classification_target": [1, 0, 0, 1],
            "animal_id": list("ABCD"),
        }
    )

    split = make_time_split(df, "classification_target", animal_subset="dog")

    assert set(split.full_data["animal_type"]) == {"Dog"}
    assert set(split.train["animal_type"]) == {"Dog"}
    assert set(split.test["animal_type"]) == {"Dog"}

