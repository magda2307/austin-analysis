import math

import pandas as pd

from aac_adoption.features.feature_engineering import (
    add_intake_features,
    age_group_from_days,
    covid_period_from_date,
    extract_found_location_area,
    found_location_flags,
    found_location_kind,
    parse_age_to_days,
    primary_breed,
    season_from_month,
    simplify_color,
)


def test_parse_age_to_days():
    assert parse_age_to_days("2 years") == 730.5
    assert math.isclose(parse_age_to_days("7 months"), 213.0625)
    assert parse_age_to_days("-1 years") != parse_age_to_days("-1 years")


def test_time_and_group_features():
    assert season_from_month(1) == "winter"
    assert season_from_month(7) == "summer"
    assert age_group_from_days(100) == "baby"
    assert age_group_from_days(365.25 * 9) == "senior"
    assert covid_period_from_date(pd.Timestamp("2020-02-29")) == "pre_covid"
    assert covid_period_from_date(pd.Timestamp("2020-03-01")) == "covid"
    assert covid_period_from_date(pd.Timestamp("2022-01-01")) == "post_covid"


def test_simplify_color():
    assert simplify_color("Black/White") == "black_or_dark"
    assert simplify_color("Brown Tabby") == "brown_tan"
    assert simplify_color("") == "unknown"


def test_primary_breed():
    assert primary_breed("Labrador Retriever Mix") == "labrador_retriever"
    assert primary_breed("Border Terrier/Border Collie") == "border_terrier"


def test_found_location_kind_and_area_rules():
    assert found_location_kind("Austin (TX)") == "austin_city"
    assert extract_found_location_area("Austin (TX)") == "austin"
    assert found_location_kind("Travis (TX)") == "county_or_region"
    assert extract_found_location_area("812 Intersection Of 183 in Travis (TX)") == "travis"
    assert found_location_kind("Outside Jurisdiction") == "outside_jurisdiction"
    assert found_location_kind("Airport And Denson in Austin (TX)") == "intersection"
    assert found_location_kind("3600 Presidential Blvd (Airport) in Austin (TX)") == "address_like"
    assert found_location_kind(None) == "other"
    assert extract_found_location_area("No stable label") == "unknown"


def test_found_location_flags():
    flags = found_location_flags("3600 Presidential Blvd (Airport) in Austin (TX)")

    assert flags["is_austin_found_location"] is True
    assert flags["is_address_like_location"] is True
    assert flags["is_airport_location"] is True
    assert flags["is_intersection_location"] is False


def test_add_intake_features_adds_found_location_fields():
    df = pd.DataFrame(
        {
            "name": ["Max"],
            "age_upon_intake": ["2 years"],
            "intake_datetime": [pd.Timestamp("2021-01-01 10:00:00")],
            "color": ["Black/White"],
            "breed": ["Labrador Retriever Mix"],
            "found_location": ["Berkman Dr & Briarcliff Blvd in Austin (TX)"],
        }
    )

    result = add_intake_features(df)

    assert result.loc[0, "found_location_kind"] == "intersection"
    assert result.loc[0, "found_location_area"] == "austin"
    assert bool(result.loc[0, "is_intersection_location"]) is True
