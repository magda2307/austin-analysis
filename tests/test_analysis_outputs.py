import pandas as pd

from aac_adoption.analysis.hypothesis_tables import create_hypothesis_support_tables
from aac_adoption.analysis.model_comparison import create_context_model_comparison_table, create_model_comparison_tables


def test_model_comparison_ranks_classification_and_regression(tmp_path):
    metrics_dir = tmp_path / "metrics"
    tables_dir = tmp_path / "tables"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {"model_name": "a", "task": "classification", "animal_subset": "combined", "roc_auc": 0.7, "f1": 0.6},
            {"model_name": "b", "task": "classification", "animal_subset": "combined", "roc_auc": 0.8, "f1": 0.5},
            {"model_name": "a", "task": "classification", "animal_subset": "dogs", "roc_auc": 0.7, "f1": 0.6, "feature_set": "intake_time_v1"},
            {"model_name": "a", "task": "classification", "animal_subset": "dogs", "roc_auc": 0.75, "f1": 0.65, "feature_set": "intake_time_context_v1"},
        ]
    ).to_csv(metrics_dir / "classification_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": "c", "task": "classification", "animal_subset": "combined", "roc_auc": 0.9, "f1": 0.7}
        ]
    ).to_csv(metrics_dir / "boosting_classification_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": "a", "task": "regression", "animal_subset": "combined", "mae": 10.0, "rmse": 20.0},
            {"model_name": "b", "task": "regression", "animal_subset": "combined", "mae": 8.0, "rmse": 22.0},
            {"model_name": "a", "task": "regression", "animal_subset": "dogs", "mae": 10.0, "rmse": 20.0, "feature_set": "intake_time_v1"},
            {"model_name": "a", "task": "regression", "animal_subset": "dogs", "mae": 9.0, "rmse": 19.0, "feature_set": "intake_time_context_v1"},
        ]
    ).to_csv(metrics_dir / "regression_metrics.csv", index=False)
    pd.DataFrame(
        [{"model_name": "c", "task": "regression", "animal_subset": "combined", "mae": 9.0, "rmse": 18.0}]
    ).to_csv(metrics_dir / "boosting_regression_metrics.csv", index=False)

    create_model_comparison_tables(metrics_dir, tables_dir)

    cls = pd.read_csv(tables_dir / "model_comparison_classification.csv")
    reg = pd.read_csv(tables_dir / "model_comparison_regression.csv")
    assert cls.iloc[0]["model_name"] == "c"
    assert reg.iloc[0]["model_name"] == "b"
    assert {"roc_auc_rank", "f1_rank"}.issubset(cls.columns)
    assert {"mae_rank", "rmse_rank"}.issubset(reg.columns)
    context = pd.read_csv(tables_dir / "context_model_comparison.csv")
    assert set(context["task"]) == {"classification", "regression"}
    assert round(context.loc[context["task"] == "classification", "delta"].item(), 2) == 0.05
    assert context.loc[context["task"] == "regression", "delta"].item() == -1.0


def test_hypothesis_support_tables_are_created(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    pd.DataFrame(
        {
            "adopted": [True, False, True, False],
            "days_to_outcome": [3.0, 20.0, 5.0, 12.0],
            "intake_type": ["Stray", "Owner Surrender", "Stray", "Owner Surrender"],
            "intake_condition": ["Normal", "Normal", "Injured", "Normal"],
            "simplified_breed_group": ["retriever_type", "domestic_cat", "retriever_type", "domestic_cat"],
            "simplified_color_group": ["black_or_dark", "brown_tan", "white_light", "black_or_dark"],
            "age_group": ["baby", "adult", "senior", "adult"],
            "covid_period": ["pre_covid", "covid", "post_covid", "covid"],
        }
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        {
            "feature": [
                "intake_type_Stray",
                "age_days",
                "covid_period_covid",
                "simplified_color_group_black_or_dark",
            ],
            "importance": [0.3, 0.2, 0.1, 0.05],
            "task": ["classification"] * 4,
        }
    ).to_csv(tables_dir / "permutation_importance_classification.csv", index=False)

    create_hypothesis_support_tables(data_path, tables_dir)

    for filename in ["h1_intake_vs_appearance.csv", "h3_age_length_of_stay.csv", "h5_covid_period.csv"]:
        table = pd.read_csv(tables_dir / filename)
        assert {"records", "adoption_rate_pct", "median_days_to_outcome"}.issubset(table.columns)


def test_context_model_comparison_from_separate_metric_dirs(tmp_path):
    base_metrics = tmp_path / "base"
    context_metrics = tmp_path / "context"
    tables_dir = tmp_path / "tables"
    base_metrics.mkdir()
    context_metrics.mkdir()
    pd.DataFrame(
        [
            {
                "model_name": "catboost",
                "task": "classification",
                "animal_subset": "combined",
                "feature_set": "intake_time_v1",
                "roc_auc": 0.84,
                "f1": 0.8,
            }
        ]
    ).to_csv(base_metrics / "advanced_classification_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "catboost",
                "task": "classification",
                "animal_subset": "combined",
                "feature_set": "intake_time_context_v1",
                "roc_auc": 0.85,
                "f1": 0.81,
            }
        ]
    ).to_csv(context_metrics / "advanced_classification_metrics.csv", index=False)

    comparison = create_context_model_comparison_table(base_metrics, context_metrics, tables_dir)

    assert round(comparison["delta"].item(), 2) == 0.01
    assert (tables_dir / "context_model_comparison.csv").exists()
