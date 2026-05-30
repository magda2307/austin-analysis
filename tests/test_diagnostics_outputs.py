import pandas as pd

from aac_adoption.diagnostics.feature_families import feature_family
from aac_adoption.diagnostics.model_diagnostics import calibration_table, error_slices, placement_risk_table, threshold_table


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat", "Cat", "Dog", "Cat"],
            "age_group": ["adult", "senior", "baby", "adult", "adult", "baby"],
            "intake_type": ["Stray", "Stray", "Owner Surrender", "Stray", "Public Assist", "Owner Surrender"],
            "intake_condition": ["Normal", "Sick", "Normal", "Normal", "Normal", "Normal"],
            "covid_period": ["post_covid"] * 6,
            "simplified_color_group": ["black_or_dark", "brown_tan", "black_or_dark", "white_light", "brown_tan", "black_or_dark"],
            "is_black_or_dark": [True, False, True, False, False, True],
            "classification_target": [1, 0, 1, 1, 0, 1],
            "predicted_adoption_probability": [0.9, 0.7, 0.8, 0.4, 0.2, 0.6],
            "predicted_adopted": [1, 1, 1, 0, 0, 1],
            "regression_target_days": [5.0, 30.0, 4.0, 12.0, 15.0, 6.0],
            "predicted_days_to_outcome": [6.0, 12.0, 5.0, 8.0, 9.0, 7.0],
            "regression_residual": [-1.0, 18.0, -1.0, 4.0, 6.0, -1.0],
            "absolute_error": [1.0, 18.0, 1.0, 4.0, 6.0, 1.0],
        }
    )


def test_threshold_calibration_and_error_slices():
    predictions = _prediction_frame()

    thresholds = threshold_table(predictions)
    calibration = calibration_table(predictions, bins=5)
    cls_slices, reg_slices = error_slices(predictions, min_records=1)
    risk = placement_risk_table(predictions)

    assert {"threshold", "precision", "recall", "f1", "flagged_for_adoption_share"}.issubset(thresholds.columns)
    assert {"probability_bin", "records", "mean_predicted_probability", "observed_adoption_rate"}.issubset(calibration.columns)
    assert {"slice", "value", "false_positive_rate", "false_negative_rate"}.issubset(cls_slices.columns)
    assert {"slice", "value", "mae", "median_absolute_error"}.issubset(reg_slices.columns)
    assert {"risk_quadrant", "records", "observed_adoption_rate"}.issubset(risk.columns)


def test_feature_family_mapping():
    assert feature_family("age_days") == "age"
    assert feature_family("intake_type") == "intake_circumstances"
    assert feature_family("simplified_color_group") == "color"
    assert feature_family("surprise_column") == "other"
