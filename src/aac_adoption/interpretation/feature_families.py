"""Feature family mapping for thesis interpretation chapter.

Maps raw model feature names to human-readable family groups,
aggregates SHAP or permutation importance per family.
"""

from __future__ import annotations

import pandas as pd

# -------------------------------------------------------------------------
# Canonical feature-family mapping
# Keys match the `feature_family` column already in shap_global_*.csv.
# Each value is the ordered list of raw feature names belonging to that family.
# -------------------------------------------------------------------------
FEATURE_FAMILY_MAP: dict[str, list[str]] = {
    "age": [
        "age_upon_intake",
        "age_days",
        "age_months",
        "age_years",
        "age_in_days",
        "age_in_months",
        "age_in_years",
        "age_group",
    ],
    "sex_reproductive_status": [
        "sex_upon_intake",
    ],
    "intake_circumstances": [
        "intake_type",
    ],
    "intake_condition_health": [
        "intake_condition",
    ],
    "breed_appearance": [
        "breed",
        "primary_breed",
        "is_mixed_breed",
        "simplified_breed_group",
    ],
    "color": [
        "color",
        "primary_color",
        "simplified_color_group",
        "is_black_or_dark",
    ],
    "name_identity": [
        "has_name",
        "is_named",
    ],
    "location": [
        "found_location_kind",
        "found_location_area",
        "is_austin_found_location",
        "is_outside_jurisdiction",
        "is_intersection_location",
        "is_address_like_location",
        "is_airport_location",
    ],
    "timing_seasonality": [
        "intake_year",
        "intake_month",
        "intake_quarter",
        "intake_season",
    ],
    "covid_period": [
        "covid_period",
    ],
    "animal_type": [
        "animal_type",
    ],
    "weather_context": [
        "daily_temp_max",
        "daily_temp_min",
        "daily_precipitation",
        "is_extreme_heat",
        "is_rainy_day",
    ],
    "shelter_demand_context": [
        "animal_311_requests_7d",
        "animal_311_requests_30d",
        "intake_volume_7d",
        "intake_volume_30d",
    ],
}

# Reverse lookup: feature name -> family name
_FEATURE_TO_FAMILY: dict[str, str] = {
    feature: family
    for family, features in FEATURE_FAMILY_MAP.items()
    for feature in features
}

# Human-readable display labels for each family (used in figures and thesis text)
FAMILY_LABELS: dict[str, str] = {
    "age": "Age",
    "sex_reproductive_status": "Sex / Reproductive Status",
    "intake_circumstances": "Intake Circumstances",
    "intake_condition_health": "Intake Condition / Health",
    "breed_appearance": "Breed / Appearance",
    "color": "Coat Color",
    "name_identity": "Name / Identity",
    "location": "Found Location",
    "timing_seasonality": "Timing & Seasonality",
    "covid_period": "COVID Period",
    "animal_type": "Animal Type",
    "weather_context": "Weather Context",
    "shelter_demand_context": "Shelter Demand Context",
}


def assign_feature_family(col_name: str) -> str:
    """Return the family name for a raw feature column, or 'other'."""
    return _FEATURE_TO_FAMILY.get(col_name, "other")


def aggregate_importance_by_family(
    importance_df: pd.DataFrame,
    feature_col: str = "feature",
    importance_col: str = "mean_abs_shap",
) -> pd.DataFrame:
    """Group and sum importance values by feature family.

    Args:
        importance_df: DataFrame with at least `feature_col` and `importance_col`.
        feature_col: Name of the column holding raw feature names.
        importance_col: Name of the importance metric column.

    Returns:
        DataFrame with columns: family, display_label, sum_importance,
        mean_importance, n_features, feature_list.
        Sorted descending by sum_importance.
    """
    df = importance_df.copy()
    df["family"] = df[feature_col].map(assign_feature_family)
    grouped = (
        df.groupby("family")[importance_col]
        .agg(sum_importance="sum", mean_importance="mean", n_features="count")
        .reset_index()
    )
    # Add feature list per family for transparency
    feature_lists = (
        df.groupby("family")[feature_col]
        .apply(lambda s: ", ".join(s.tolist()))
        .reset_index()
        .rename(columns={feature_col: "feature_list"})
    )
    grouped = grouped.merge(feature_lists, on="family", how="left")
    grouped["display_label"] = grouped["family"].map(
        lambda f: FAMILY_LABELS.get(f, f.replace("_", " ").title())
    )
    return grouped.sort_values("sum_importance", ascending=False).reset_index(drop=True)
