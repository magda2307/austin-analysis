import pandas as pd
import pytest

from aac_adoption.data.build_dataset import (
    build_modeling_dataset,
    build_modeling_dataset_from_files,
    validate_modeling_dataset,
)
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
                    "Found Location": "Austin (TX)",
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
                    "Found Location": "Airport And Denson in Austin (TX)",
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
                    "Found Location": "Outside Jurisdiction",
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
    assert dataset.loc[dataset["animal_id"] == "A1", "found_location_kind"].item() == "austin_city"
    assert dataset.loc[dataset["animal_id"] == "A2", "found_location_kind"].item() == "intersection"
    assert bool(dataset.loc[dataset["animal_id"] == "A2", "is_airport_location"].item()) is True
    assert "found_location" not in dataset.columns
    assert bool(dataset.loc[dataset["animal_id"] == "A1", "is_black_or_dark"].item()) is True
    assert bool(dataset.loc[dataset["animal_id"] == "A2", "is_named"].item()) is False


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

    assert abs(dataset["days_to_outcome"].iloc[0] - 2.0) < 0.1
    assert abs(dataset["days_to_outcome"].iloc[1] - 9.0) < 0.1
    assert dataset["classification_target"].tolist() == [0, 1]


def test_build_modeling_dataset_from_files_adds_context_features(tmp_path):
    intakes_path = tmp_path / "intakes.csv"
    outcomes_path = tmp_path / "outcomes.csv"
    output_path = tmp_path / "processed" / "modeling_dataset.csv"
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    _sample_intakes().to_csv(intakes_path, index=False)
    _sample_outcomes().to_csv(outcomes_path, index=False)
    pd.DataFrame({"DATE": ["2021-01-01", "2021-02-01"], "TMAX": [96, 70], "TMIN": [55, 40], "PRCP": [0.2, 0]}).to_csv(
        context_dir / "austin_weather_daily.csv",
        index=False,
    )
    pd.DataFrame({"request_date": ["2020-12-31", "2021-01-31"], "animal_311_requests": [3, 4]}).to_csv(
        context_dir / "austin_311_animal_requests.csv",
        index=False,
    )

    result = build_modeling_dataset_from_files(intakes_path, outcomes_path, output_path, context_data_dir=context_dir)
    dataset = result.dataset

    assert "daily_temp_max" in dataset.columns
    assert dataset.loc[dataset["animal_id"] == "A1", "daily_temp_max"].item() == 96
    assert dataset.loc[dataset["animal_id"] == "A1", "animal_311_requests_7d"].item() == 3
    assert "found_location_kind" in dataset.columns
    assert (tmp_path / "processed" / "context_feature_columns.json").exists()
