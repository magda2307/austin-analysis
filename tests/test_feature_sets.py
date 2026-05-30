import pytest

from aac_adoption.features.feature_sets import available_intake_features, validate_no_leakage


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

