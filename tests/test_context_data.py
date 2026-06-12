import pandas as pd
import pytest

from aac_adoption.data.context_data import (
    add_context_features,
    normalize_311_animal_requests,
    normalize_weather_daily,
)


def test_context_parsers_normalize_dates_and_counts():
    weather = normalize_weather_daily(
        pd.DataFrame(
            {
                "DATE": ["2021-01-01T00:00:00", "2021-01-02"],
                "TMAX": [96, 70],
                "TMIN": [40, 50],
                "PRCP": [0.25, 0],
            }
        )
    )
    requests = normalize_311_animal_requests(
        pd.DataFrame(
            {
                "Created Date": ["2021-01-01 12:00:00", "2021-01-01 13:00:00", "2021-01-02"],
                "SR Description": ["Animal bite", "Street light", "Loose Animal"],
            }
        )
    )

    assert weather["context_date"].tolist() == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    assert weather.loc[0, "daily_temp_max"] == 96
    assert requests["animal_311_requests"].tolist() == [1, 1]


def test_context_features_use_prior_windows_only_and_stable_defaults():
    dataset = pd.DataFrame(
        {
            "animal_id": ["A1", "A2", "A3"],
            "intake_datetime": pd.to_datetime(["2021-01-02 09:00", "2021-01-03 09:00", "2021-01-10 09:00"]),
        }
    )
    weather = pd.DataFrame({"DATE": ["2021-01-01"], "TMAX": [96], "TMIN": [60], "PRCP": [0.1]})
    requests = pd.DataFrame(
        {
            "request_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "animal_311_requests": [2, 5, 11],
        }
    )

    enriched = add_context_features(
        dataset,
        raw_intakes=dataset,
        weather_daily=weather,
        requests_311=requests,
    )

    first = enriched[enriched["animal_id"] == "A1"].iloc[0]
    second = enriched[enriched["animal_id"] == "A2"].iloc[0]
    third = enriched[enriched["animal_id"] == "A3"].iloc[0]

    assert first["animal_311_requests_7d"] == 2
    assert second["animal_311_requests_7d"] == 7
    assert first["intake_volume_7d"] == 0
    assert second["intake_volume_7d"] == 1
    assert bool(first["is_extreme_heat"]) is True
    assert bool(first["is_rainy_day"]) is True
    assert third["animal_311_requests_30d"] == 18
    assert pd.isna(third["daily_temp_max"])
    assert pd.isna(third["is_rainy_day"])


def test_context_features_reject_missing_focal_intake_history():
    modeling = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "intake_datetime": pd.to_datetime(["2024-01-02 08:00:00"]),
        }
    )
    raw_intakes = pd.DataFrame(
        {
            "animal_id": ["A2"],
            "intake_datetime": pd.to_datetime(["2024-01-01 08:00:00"]),
        }
    )

    with pytest.raises(ValueError, match="raw intake history"):
        add_context_features(
            modeling,
            raw_intakes=raw_intakes,
            weather_daily=None,
            requests_311=None,
        )


def test_weather_availability_and_flags_use_field_level_missingness():
    modeling = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "intake_datetime": pd.to_datetime(["2024-01-02 08:00:00"]),
        }
    )
    weather = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "TMAX": [pd.NA],
            "TMIN": [50],
            "PRCP": [0.2],
        }
    )

    result = add_context_features(
        modeling,
        raw_intakes=modeling,
        weather_daily=weather,
        requests_311=None,
    ).iloc[0]

    assert bool(result["weather_available"]) is True
    assert pd.isna(result["is_extreme_heat"])
    assert bool(result["is_rainy_day"]) is True


def test_intake_volume_is_independent_of_outcome_columns():
    raw_intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A2"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    dataset_a = raw_intakes.assign(outcome_type=["Adoption", "Transfer"])
    dataset_b = raw_intakes.assign(outcome_type=["Euthanasia", "Adoption"])

    enriched_a = add_context_features(
        dataset_a,
        raw_intakes=raw_intakes,
        weather_daily=None,
        requests_311=None,
    )
    enriched_b = add_context_features(
        dataset_b,
        raw_intakes=raw_intakes,
        weather_daily=None,
        requests_311=None,
    )

    assert enriched_a["intake_volume_7d"].tolist() == enriched_b["intake_volume_7d"].tolist()
