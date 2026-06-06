import pytest

from aac_adoption.features.feature_sets import (
    INTAKE_TIME_FEATURES,
    available_intake_features,
    feature_set_label,
    validate_no_leakage,
)


def test_available_intake_features_excludes_outcome_columns():
    columns = {
        "animal_type",
        "intake_type",
        "age_days",
        "age_years",
        "found_location_kind",
        "found_location",
        "outcome_type",
        "days_to_outcome",
        "classification_target",
    }

    features = available_intake_features(columns)

    assert features == ["animal_type", "intake_type", "age_days", "found_location_kind"]


def test_model_feature_set_prunes_duplicate_aliases():
    assert "age_days" in INTAKE_TIME_FEATURES
    assert "age_group" in INTAKE_TIME_FEATURES
    assert "is_named" in INTAKE_TIME_FEATURES
    assert "primary_color" in INTAKE_TIME_FEATURES
    assert "simplified_color_group" in INTAKE_TIME_FEATURES
    assert {
        "age_upon_intake",
        "breed",
        "color",
        "age_months",
        "age_years",
        "has_name",
        "intake_quarter",
        "intake_season",
        "color_group",
    }.isdisjoint(INTAKE_TIME_FEATURES)


def test_validate_no_leakage_rejects_outcome_features():
    with pytest.raises(ValueError, match="Leakage columns"):
        validate_no_leakage(["animal_type", "outcome_type", "days_to_outcome"])


def test_validate_no_leakage_rejects_future_context_windows():
    with pytest.raises(ValueError, match="Leakage columns"):
        validate_no_leakage(["animal_type", "animal_311_requests_next_7d"])


def test_feature_set_label_detects_context_features():
    assert feature_set_label(["animal_type", "age_days"]) == "intake_time_v2"
    assert feature_set_label(["animal_type", "daily_temp_max"]) == "intake_time_context_v2"
