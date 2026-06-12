import pandas as pd
import joblib

from aac_adoption.diagnostics.feature_families import feature_family
from aac_adoption.interpretation.feature_families import assign_feature_family
from aac_adoption.diagnostics.model_diagnostics import (
    _load_model,
    calibration_table,
    diagnostics_validation_tactics,
    error_slices,
    placement_risk_table,
    shap_outputs,
    threshold_table,
)
from aac_adoption.models.evaluate import classification_metrics, expected_calibration_error


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


def test_load_model_uses_final_selection_for_artifact_family(tmp_path):
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    models_root = tmp_path / "models"
    selected_path = models_root / "boosting" / "classification" / "combined" / "hist_gradient_boosting.joblib"
    selected_path.parent.mkdir(parents=True)
    selected_model = {"kind": "selected_hist_gradient_boosting"}
    joblib.dump(selected_model, selected_path)

    pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "animal_subset": "combined",
                "subset": "combined",
                "selected": True,
                "task": "classification",
            }
        ]
    ).to_csv(tables / "final_model_selection.csv", index=False)

    model, metadata = _load_model(models_root / "advanced", "classification", "combined", tables)

    assert model == selected_model
    assert metadata["model_name"] == "hist_gradient_boosting"
    assert metadata["artifact_path"].endswith("models\\boosting\\classification\\combined\\hist_gradient_boosting.joblib") or metadata[
        "artifact_path"
    ].endswith("models/boosting/classification/combined/hist_gradient_boosting.joblib")
    assert "final_model_selection.csv" in metadata["selection_source"]


def test_diagnostics_validation_tactics_cover_generated_parts():
    tactics = diagnostics_validation_tactics(include_shap=True)

    assert {"diagnostic_part", "validation_tactic"}.issubset(tactics.columns)
    assert {"selected_model_resolution", "calibration_table", "shap_outputs"}.issubset(set(tactics["diagnostic_part"]))
    assert tactics["validation_tactic"].str.len().min() > 20


def test_shap_skip_removes_stale_unselected_regression_outputs(
    tmp_path, monkeypatch
):
    tables = tmp_path / "reports" / "tables"
    figures = tmp_path / "reports" / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    stale_paths = [
        tables / "shap_global_regression.csv",
        tables / "shap_feature_families_regression.csv",
        tables / "feature_family_importance_regression.csv",
        figures / "shap_summary_regression.png",
        figures / "shap_feature_families_regression.png",
        figures / "feature_family_importance_regression.png",
    ]
    for path in stale_paths:
        path.write_text("stale", encoding="utf-8")
    data_path = tmp_path / "modeling.csv"
    pd.DataFrame({"classification_target": [0], "regression_target_days": [1]}).to_csv(
        data_path, index=False
    )
    monkeypatch.setattr(
        "aac_adoption.diagnostics.model_diagnostics._selected_model_row",
        lambda _tables, _task, _subset: {"model_name": "random_forest"},
    )

    shap_outputs(
        data_path,
        tmp_path / "models",
        tables,
        figures,
        10,
        tables,
    )

    assert all(not path.exists() for path in stale_paths)
    assert (tables / "shap_regression_skip_note.csv").exists()


def test_feature_family_mapping():
    assert feature_family("age_days") == "age"
    assert feature_family("intake_type") == "intake_circumstances"
    assert feature_family("simplified_color_group") == "color"
    assert feature_family("daily_temp_max") == "weather_context"
    assert feature_family("animal_311_requests_7d") == "shelter_demand_context"
    assert feature_family("intake_volume_30d") == "shelter_demand_context"
    assert feature_family("surprise_column") == "other"


def test_interpretation_feature_family_mapping_splits_context():
    assert assign_feature_family("daily_precipitation") == "weather_context"
    assert assign_feature_family("intake_volume_7d") == "shelter_demand_context"


def test_classification_metrics_include_pr_auc_and_calibration_metrics():
    metrics = classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0.1, 0.6, 0.7, 0.9],
    )

    assert "pr_auc" in metrics
    assert metrics["pr_auc"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["expected_calibration_error"] is not None
    assert 0 <= metrics["expected_calibration_error"] <= 1


def test_expected_calibration_error_is_zero_for_perfect_bins():
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], bins=2) == 0.0
