import pandas as pd

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
    weather = pd.DataFrame({"DATE": ["2021-01-02"], "TMAX": [96], "TMIN": [60], "PRCP": [0.1]})
    requests = pd.DataFrame(
        {
            "request_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "animal_311_requests": [2, 5, 11],
        }
    )

    enriched = add_context_features(dataset, weather=weather, animal_requests=requests)

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
    assert bool(third["is_rainy_day"]) is False
