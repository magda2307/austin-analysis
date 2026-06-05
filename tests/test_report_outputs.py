import pandas as pd

from aac_adoption.reporting.report import create_report_outputs


def test_create_report_outputs_writes_summary_and_figures(tmp_path):
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()

    pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "logistic_regression", "roc_auc": 0.75, "f1": 0.7},
            {"animal_subset": "combined", "model_name": "hist_gradient_boosting", "roc_auc": 0.84, "f1": 0.8},
            {"animal_subset": "dogs", "model_name": "hist_gradient_boosting", "roc_auc": 0.81, "f1": 0.78},
        ]
    ).to_csv(tables_dir / "model_comparison_classification.csv", index=False)
    pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "ridge", "mae": 24.0, "rmse": 35.0},
            {"animal_subset": "combined", "model_name": "hist_gradient_boosting", "mae": 20.2, "rmse": 32.0},
            {"animal_subset": "cats", "model_name": "hist_gradient_boosting", "mae": 18.3, "rmse": 29.0},
        ]
    ).to_csv(tables_dir / "model_comparison_regression.csv", index=False)
    pd.DataFrame(
        [
            {
                "value": "Stray",
                "records": 100,
                "adoption_rate_pct": 45.0,
                "median_days_to_outcome": 6.0,
                "variable": "intake_type",
            },
            {
                "value": "Normal",
                "records": 150,
                "adoption_rate_pct": 55.0,
                "median_days_to_outcome": 7.0,
                "variable": "intake_condition",
            },
        ]
    ).to_csv(tables_dir / "h1_intake_vs_appearance.csv", index=False)
    pd.DataFrame(
        [
            {"value": "baby", "records": 80, "adoption_rate_pct": 60.0, "median_days_to_outcome": 5.5},
            {"value": "senior", "records": 20, "adoption_rate_pct": 30.0, "median_days_to_outcome": 8.0},
        ]
    ).to_csv(tables_dir / "h3_age_adoption_speed.csv", index=False)
    pd.DataFrame(
        [
            {"value": "pre_covid", "records": 100, "adoption_rate_pct": 46.0, "median_days_to_outcome": 5.0},
            {"value": "covid", "records": 40, "adoption_rate_pct": 57.0, "median_days_to_outcome": 8.0},
        ]
    ).to_csv(tables_dir / "h5_covid_period.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": "classification",
                "animal_subset": "combined",
                "model_name": "hist_gradient_boosting",
                "primary_metric": "roc_auc",
                "base_score": 0.84,
                "context_score": 0.85,
                "delta": 0.01,
                "higher_is_better": True,
            }
        ]
    ).to_csv(tables_dir / "context_model_comparison.csv", index=False)

    summary_path = create_report_outputs(tables_dir, figures_dir, summary_dir)

    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Best classification models by ROC-AUC" in summary_text
    assert "External Context Feature Test" in summary_text
    assert "H3 age-group patterns" in summary_text
    assert (figures_dir / "model_comparison_classification_roc_auc.png").exists()
    assert (figures_dir / "model_comparison_regression_mae.png").exists()
    assert (figures_dir / "h1_intake_type_adoption_rate.png").exists()
    assert (figures_dir / "h3_age_group_adoption_rate.png").exists()
    assert (figures_dir / "h5_covid_period_adoption_rate.png").exists()
    assert (figures_dir / "context_model_delta.png").exists()
