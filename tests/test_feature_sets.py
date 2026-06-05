import pytest

from aac_adoption.features.feature_sets import available_intake_features, feature_set_label, validate_no_leakage


def test_available_intake_features_excludes_outcome_columns():
    columns = {
        "animal_type",
        "intake_type",
        "age_years",
        "outcome_type",
        "days_to_outcome",
        "classification_target",
    }

    features = available_intake_features(columns)

    assert features == ["animal_type", "intake_type", "age_years"]


def test_validate_no_leakage_rejects_outcome_features():
    with pytest.raises(ValueError, match="Leakage columns"):
        validate_no_leakage(["animal_type", "outcome_type", "days_to_outcome"])


def test_validate_no_leakage_rejects_future_context_windows():
    with pytest.raises(ValueError, match="Leakage columns"):
        validate_no_leakage(["animal_type", "animal_311_requests_next_7d"])


def test_feature_set_label_detects_context_features():
    assert feature_set_label(["animal_type", "age_years"]) == "intake_time_v1"
    assert feature_set_label(["animal_type", "daily_temp_max"]) == "intake_time_context_v1"
