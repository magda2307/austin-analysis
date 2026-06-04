import pandas as pd

from aac_adoption.reporting.evidence_pack import (
    bootstrap_metric_intervals,
    create_evidence_pack,
    model_limitations_by_cohort,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat", "Cat", "Dog", "Cat"],
            "age_group": ["senior", "adult", "baby", "adult", "senior", "baby"],
            "intake_type": ["Stray", "Stray", "Owner Surrender", "Stray", "Public Assist", "Owner Surrender"],
            "intake_condition": ["Normal", "Sick", "Normal", "Normal", "Normal", "Normal"],
            "simplified_breed_group": ["pit_bull_type", "retriever_type", "domestic_cat", "domestic_cat", "pit_bull_type", "domestic_cat"],
            "simplified_color_group": ["black_or_dark", "brown_tan", "black_or_dark", "white_light", "brown_tan", "black_or_dark"],
            "is_named": [True, True, False, True, False, True],
            "classification_target": [1, 0, 1, 1, 0, 1],
            "predicted_adoption_probability": [0.9, 0.7, 0.8, 0.4, 0.2, 0.6],
            "predicted_adopted": [1, 1, 1, 0, 0, 1],
            "regression_target_days": [5.0, 30.0, 4.0, 12.0, 15.0, 6.0],
            "predicted_days_to_outcome": [6.0, 12.0, 5.0, 8.0, 9.0, 7.0],
            "absolute_error": [1.0, 18.0, 1.0, 4.0, 6.0, 1.0],
        }
    )


def test_bootstrap_intervals_have_expected_schema():
    intervals = bootstrap_metric_intervals(_predictions(), n_bootstrap=10, random_state=1)

    assert {"metric", "animal_subset", "lower", "estimate", "upper", "bootstrap_samples"}.issubset(intervals.columns)
    assert {"roc_auc", "pr_auc", "f1_at_0_50", "mae"}.issubset(set(intervals["metric"]))


def test_model_limitations_flag_small_cohorts():
    limitations = model_limitations_by_cohort(_predictions(), min_records=4)

    assert {"cohort", "records", "small_cohort_flag", "calibration_gap", "mae"}.issubset(limitations.columns)
    assert limitations["small_cohort_flag"].any()


def test_create_evidence_pack_writes_artifacts(tmp_path):
    tables = tmp_path / "tables"
    diagnostics = tmp_path / "diagnostics"
    summary = tmp_path / "summary"
    tables.mkdir()
    diagnostics.mkdir()

    pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "catboost", "roc_auc": 0.8, "pr_auc": 0.85, "f1": 0.7},
            {"animal_subset": "dogs", "model_name": "hist_gradient_boosting", "roc_auc": 0.75, "pr_auc": 0.8, "f1": 0.65},
        ]
    ).to_csv(tables / "model_comparison_classification.csv", index=False)
    pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "catboost", "mae": 18.0, "rmse": 30.0},
            {"animal_subset": "cats", "model_name": "catboost", "mae": 15.0, "rmse": 25.0},
        ]
    ).to_csv(tables / "model_comparison_regression.csv", index=False)
    pd.DataFrame(
        [{"feature_family": "age", "mean_abs_shap": 0.5}, {"feature_family": "intake", "mean_abs_shap": 0.3}]
    ).to_csv(tables / "shap_feature_families_classification.csv", index=False)
    pd.DataFrame(
        [{"feature": "age_upon_intake", "mean_abs_shap": 0.5}, {"feature": "intake_type", "mean_abs_shap": 0.3}]
    ).to_csv(tables / "shap_global_classification.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile_label": "senior named Dog",
                "records": 100,
                "adoption_rate_pct": 45.0,
                "median_days_to_outcome": 20.0,
                "animal_type": "Dog",
                "age_group": "senior",
                "intake_type": "Stray",
                "intake_condition": "Normal",
                "simplified_breed_group": "pit_bull_type",
                "simplified_color_group": "black_or_dark",
                "sex_upon_intake": "Intact Male",
                "is_named": True,
                "visibility_need": "needs visibility",
            }
        ]
    ).to_csv(tables / "animal_archetypes.csv", index=False)
    _predictions().to_csv(diagnostics / "diagnostic_predictions_sample.csv", index=False)

    paths = create_evidence_pack(
        data_path=tmp_path / "missing.csv",
        tables_dir=tables,
        diagnostics_dir=diagnostics,
        summary_dir=summary,
        models_dir=tmp_path / "models",
        bootstrap_samples=10,
        min_cohort_records=4,
    )

    assert paths["evidence"].exists()
    assert paths["limitations"].exists()
    assert paths["intervals"].exists()
    assert paths["journeys"].exists()
    assert paths["summary"].exists()
    assert "associated with model behavior" in paths["summary"].read_text(encoding="utf-8")
