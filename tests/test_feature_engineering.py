import math

import pandas as pd

from aac_adoption.features.feature_engineering import (
    age_group_from_days,
    covid_period_from_date,
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
