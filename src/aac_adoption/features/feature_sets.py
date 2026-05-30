"""Feature-set definitions and leakage checks."""

INTAKE_TIME_FEATURES = [
    "animal_type",
    "intake_type",
    "intake_condition",
    "sex_upon_intake",
    "age_upon_intake",
    "age_days",
    "age_months",
    "age_years",
    "age_group",
    "breed",
    "primary_breed",
    "is_mixed_breed",
    "simplified_breed_group",
    "color",
    "primary_color",
    "simplified_color_group",
    "is_black_or_dark",
    "has_name",
    "is_named",
    "intake_year",
    "intake_month",
    "intake_quarter",
    "intake_season",
    "covid_period",
]

TARGET_COLUMNS = [
    "adopted",
    "is_adopted",
    "target_adopted",
    "classification_target",
    "regression_target_days",
    "days_to_outcome",
    "days_to_adoption",
    "length_of_stay",
]

METADATA_COLUMNS = [
    "animal_id",
    "intake_datetime",
    "outcome_datetime",
    "outcome_type",
    "outcome_subtype",
    "sex_upon_outcome",
    "age_upon_outcome",
]

LEAKAGE_COLUMNS = set(TARGET_COLUMNS + METADATA_COLUMNS) - {"animal_id", "intake_datetime"}


def available_intake_features(columns: list[str] | set[str]) -> list[str]:
    """Return configured intake-time features present in a dataset."""
    column_set = set(columns)
    return [column for column in INTAKE_TIME_FEATURES if column in column_set]


def validate_no_leakage(feature_columns: list[str]) -> None:
    """Raise when outcome-derived columns are used as model features."""
    leakage = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leakage:
        raise ValueError(f"Leakage columns cannot be model features: {leakage}")

