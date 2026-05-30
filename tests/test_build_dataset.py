import pandas as pd
import pytest

from aac_adoption.data.build_dataset import build_modeling_dataset, validate_modeling_dataset
from aac_adoption.data.load_data import standardize_column_names


def _sample_intakes() -> pd.DataFrame:
    return standardize_column_names(
        pd.DataFrame(
            [
                {
                    "Animal ID": "A1",
                    "Name": "Max",
                    "Animal Type": "Dog",
                    "Intake DateTime": "2021-01-01 10:00:00",
                    "Intake Type": "Stray",
                    "Intake Condition": "Normal",
                    "Sex upon Intake": "Neutered Male",
                    "Age upon Intake": "2 years",
                    "Breed": "Labrador Retriever Mix",
                    "Color": "Black/White",
                },
                {
                    "Animal ID": "A2",
                    "Name": None,
                    "Animal Type": "Cat",
                    "Intake DateTime": "2021-02-01 10:00:00",
                    "Intake Type": "Owner Surrender",
                    "Intake Condition": "Normal",
                    "Sex upon Intake": "Spayed Female",
                    "Age upon Intake": "7 months",
                    "Breed": "Domestic Shorthair",
                    "Color": "Brown Tabby",
                },
                {
                    "Animal ID": "A3",
                    "Name": "Bun",
                    "Animal Type": "Rabbit",
                    "Intake DateTime": "2021-03-01 10:00:00",
                    "Intake Type": "Stray",
                    "Intake Condition": "Normal",
                    "Sex upon Intake": "Unknown",
                    "Age upon Intake": "1 year",
                    "Breed": "Rabbit",
                    "Color": "White",
                },
            ]
        )
    )


def _sample_outcomes() -> pd.DataFrame:
    return standardize_column_names(
        pd.DataFrame(
            [
                {
                    "Animal ID": "A1",
                    "Animal Type": "Dog",
                    "Outcome DateTime": "2021-01-10 10:00:00",
                    "Outcome Type": "Adoption",
                    "Outcome Subtype": "",
                    "Sex upon Outcome": "Neutered Male",
                    "Age upon Outcome": "2 years",
                },
                {
                    "Animal ID": "A2",
                    "Animal Type": "Cat",
                    "Outcome DateTime": "2021-03-01 10:00:00",
                    "Outcome Type": "Transfer",
                    "Outcome Subtype": "Partner",
                    "Sex upon Outcome": "Spayed Female",
                    "Age upon Outcome": "8 months",
                },
                {
                    "Animal ID": "A3",
                    "Animal Type": "Rabbit",
                    "Outcome DateTime": "2021-03-05 10:00:00",
                    "Outcome Type": "Adoption",
                    "Outcome Subtype": "",
                    "Sex upon Outcome": "Unknown",
                    "Age upon Outcome": "1 year",
                },
            ]
        )
    )


def test_build_modeling_dataset_filters_and_creates_targets():
    result = build_modeling_dataset(_sample_intakes(), _sample_outcomes())
    dataset = result.dataset

    assert result.matched_rows == 2
    assert set(dataset["animal_type"]) == {"Dog", "Cat"}
    assert dataset["days_to_outcome"].min() >= 0
    assert dataset.loc[dataset["animal_id"] == "A1", "classification_target"].item() == 1
    assert dataset.loc[dataset["animal_id"] == "A2", "classification_target"].item() == 0
    assert dataset.loc[dataset["animal_id"] == "A1", "simplified_color_group"].item() == "black_or_dark"
    assert bool(dataset.loc[dataset["animal_id"] == "A1", "is_black_or_dark"].item()) is True
    assert bool(dataset.loc[dataset["animal_id"] == "A2", "has_name"].item()) is False


def test_validate_rejects_negative_los():
    result = build_modeling_dataset(_sample_intakes(), _sample_outcomes())
    dataset = result.dataset.copy()
    dataset.loc[0, "days_to_outcome"] = -1

    with pytest.raises(ValueError, match="negative days_to_outcome"):
        validate_modeling_dataset(dataset)


def test_repeated_animal_matches_each_intake_to_next_unused_outcome():
    intakes = standardize_column_names(
        pd.DataFrame(
            [
                {
                    "Animal ID": "A1",
                    "Animal Type": "Dog",
                    "Intake DateTime": "2021-01-01 10:00:00",
                    "Intake Type": "Stray",
                    "Intake Condition": "Normal",
                    "Sex upon Intake": "Neutered Male",
                    "Age upon Intake": "2 years",
                    "Breed": "Mix",
                    "Color": "Black",
                },
                {
                    "Animal ID": "A1",
                    "Animal Type": "Dog",
                    "Intake DateTime": "2021-02-01 10:00:00",
                    "Intake Type": "Stray",
                    "Intake Condition": "Normal",
                    "Sex upon Intake": "Neutered Male",
                    "Age upon Intake": "2 years",
                    "Breed": "Mix",
                    "Color": "Black",
                },
            ]
        )
    )
    outcomes = standardize_column_names(
        pd.DataFrame(
            [
                {
                    "Animal ID": "A1",
                    "Animal Type": "Dog",
                    "Outcome DateTime": "2021-01-03 10:00:00",
                    "Outcome Type": "Return to Owner",
                },
                {
                    "Animal ID": "A1",
                    "Animal Type": "Dog",
                    "Outcome DateTime": "2021-02-10 10:00:00",
                    "Outcome Type": "Adoption",
                },
            ]
        )
    )

    dataset = build_modeling_dataset(intakes, outcomes).dataset

    assert dataset["days_to_outcome"].tolist() == [2.0, 9.0]
    assert dataset["classification_target"].tolist() == [0, 1]
