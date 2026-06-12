import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aac_adoption.analysis.calibration_summary import create_calibration_summary
from aac_adoption.analysis.hypothesis_tables import create_adopted_only_timing_tables
from aac_adoption.analysis.model_selection import create_final_model_selection
from aac_adoption.analysis.reliability_red_flags import create_reliability_red_flags
from aac_adoption.analysis.threshold_analysis import (
    _evaluate_thresholds,
    _find_best_model,
    _selection_selected_thresholds,
    create_threshold_analysis,
)

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.generate_leakage_audit import build_leakage_audit


def test_leakage_audit_keeps_existing_columns_and_adds_acceptance_aliases(tmp_path):
    feature_columns = tmp_path / "feature_columns.json"
    data_path = tmp_path / "modeling_dataset.csv"
    feature_columns.write_text('["age_days", "classification_target"]', encoding="utf-8")
    pd.DataFrame(columns=["age_days", "classification_target", "outcome_type"]).to_csv(data_path, index=False)

    audit, violations = build_leakage_audit(str(feature_columns), str(data_path))

    assert "classification_target" in violations
    assert {
        "category",
        "allowed_as_predictor",
        "in_leakage_set",
        "leakage_violation",
        "role",
        "allowed_as_feature",
        "leakage_risk",
        "notes",
    }.issubset(audit.columns)
    leaking = audit.set_index("column").loc["classification_target"]
    assert leaking["allowed_as_feature"] == False
    assert leaking["leakage_risk"] == True
    assert "Leakage violation" in leaking["notes"]


def test_h3_adopted_only_table_adds_age_group_and_records_aliases(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "adopted": [True, True, False, True],
            "days_to_outcome": [4.0, 8.0, 12.0, 2.0],
            "age_group": ["baby", "adult", "adult", "baby"],
            "animal_type": ["Dog", "Dog", "Dog", "Cat"],
        }
    ).to_csv(data_path, index=False)

    create_adopted_only_timing_tables(data_path, tables_dir, figures_dir)
    table = pd.read_csv(tables_dir / "h3_adopted_only_age_speed.csv")

    assert {"group_variable", "group_value", "all_records", "age_group", "records"}.issubset(table.columns)
    age_rows = table[table["group_variable"] == "age_group"]
    assert age_rows["age_group"].equals(age_rows["group_value"])
    assert age_rows["records"].equals(age_rows["all_records"])


def test_final_model_selection_adds_subset_alias(tmp_path):
    tables_dir = tmp_path / "tables"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "animal_subset": "combined",
                "roc_auc": 0.7,
                "pr_auc": 0.75,
                "f1": 0.6,
                "brier_score": 0.22,
                "expected_calibration_error": 0.12,
                "artifact_path": "path", "split_strategy": "time", "metric_split": "selection"
            },
            {
                "model_name": "catboost",
                "animal_subset": "combined",
                "roc_auc": 0.8,
                "pr_auc": 0.85,
                "f1": 0.7,
                "brier_score": 0.18,
                "expected_calibration_error": 0.08,
                "artifact_path": "path", "split_strategy": "time", "metric_split": "selection"
            },
        ]
    ).to_csv(tables_dir / "model_comparison_classification.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": "ridge", "animal_subset": "combined", "mae": 10.0, "rmse": 20.0, "artifact_path": "path", "split_strategy": "time", "metric_split": "selection"},
            {"model_name": "catboost", "animal_subset": "combined", "mae": 8.0, "rmse": 18.0, "artifact_path": "path", "split_strategy": "time", "metric_split": "selection"},
        ]
    ).to_csv(tables_dir / "model_comparison_regression.csv", index=False)

    create_final_model_selection(tables_dir, summary_dir)
    table = pd.read_csv(tables_dir / "final_model_selection.csv")

    assert {"animal_subset", "subset", "brier_score", "expected_calibration_error"}.issubset(table.columns)
    assert table["subset"].equals(table["animal_subset"])


def test_final_model_selection_uses_pr_auc_before_roc_auc(tmp_path):
    tables_dir = tmp_path / "tables"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model_name": "higher_roc", "animal_subset": "combined", "roc_auc": 0.90, "pr_auc": 0.70, "f1": 0.6,
                "artifact_path": "path/a", "split_strategy": "time", "metric_split": "selection", "expected_calibration_error": 0.1
            },
            {
                "model_name": "higher_pr", "animal_subset": "combined", "roc_auc": 0.85, "pr_auc": 0.80, "f1": 0.7,
                "artifact_path": "path/b", "split_strategy": "time", "metric_split": "selection", "expected_calibration_error": 0.1
            },
        ]
    ).to_csv(tables_dir / "model_comparison_classification.csv", index=False)

    create_final_model_selection(tables_dir, summary_dir)
    table = pd.read_csv(tables_dir / "final_model_selection.csv")
    selected = table[(table["task"] == "classification") & (table["selected"] == True)]

    assert selected.iloc[0]["model_name"] == "higher_pr"


def test_threshold_analysis_adds_threshold_name_alias():
    table = _evaluate_thresholds(
        np.array([0, 0, 1, 1, 1, 0]),
        np.array([0.1, 0.4, 0.7, 0.8, 0.9, 0.2]),
    )

    assert {"threshold_label", "threshold_name", "selection_source"}.issubset(table.columns)
    assert table["threshold_name"].equals(table["threshold_label"])
    assert {"youden_j", "top_10_percent_capacity"}.issubset(set(table["threshold_label"]))
    assert set(table["selection_source"]) == {"selection"}


def test_selection_selected_thresholds_freeze_thresholds_for_test():
    table = _selection_selected_thresholds(
        selection_true=np.array([0, 0, 1, 1, 1, 0]),
        selection_score=np.array([0.1, 0.4, 0.7, 0.8, 0.9, 0.2]),
        test_true=np.array([0, 1, 1, 0, 1, 0]),
        test_score=np.array([0.7, 0.6, 0.55, 0.5, 0.45, 0.1]),
    )

    assert {
        "selection_precision",
        "selection_recall",
        "selection_f1",
        "test_precision",
        "test_recall",
        "test_f1",
        "selection_tactic",
    }.issubset(table.columns)
    assert table["selection_tactic"].str.contains("selection period only").all()


def test_create_threshold_analysis_writes_validation_and_test_metrics(tmp_path):
    data_path = tmp_path / "modeling_dataset.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    summary_dir = tmp_path / "summary"
    models_root = tmp_path / "models"
    model_path = models_root / "baseline" / "classification" / "combined" / "logistic_regression.joblib"
    model_path.parent.mkdir(parents=True)
    tables_dir.mkdir()

    rows = []
    for year in range(2013, 2026):
        for idx in range(4):
            age_days = year * 10 + idx
            rows.append(
                {
                    "animal_type": "Dog",
                    "intake_year": year,
                    "age_days": age_days,
                    "classification_target": int(idx % 2 == 0),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)

    pipeline = Pipeline(
        [
            ("select_scale", ColumnTransformer([("age", StandardScaler(), ["age_days"])])),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(df[["animal_type", "intake_year", "age_days"]], df["classification_target"])
    joblib.dump(pipeline, model_path)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "animal_subset": "combined",
                "subset": "combined",
                "selected": True,
                "task": "classification",
                "artifact_path": str(model_path)
            }
        ]
    ).to_csv(tables_dir / "final_model_selection.csv", index=False)

    create_threshold_analysis(data_path, tables_dir, figures_dir, summary_dir, models_root)
    thresholds = pd.read_csv(tables_dir / "final_classifier_thresholds.csv")

    assert set(thresholds["threshold_selection_period"]) == {"selection"}
    assert set(thresholds["evaluation_period"]) == {"test"}
    assert {"selection_f1", "test_f1", "selection_tactic"}.issubset(thresholds.columns)


def test_threshold_model_finder_accepts_string_selected_and_artifact_path(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    explicit = tmp_path / "custom" / "chosen.joblib"
    explicit.parent.mkdir()
    explicit.write_bytes(b"placeholder")
    pd.DataFrame(
        [
            {
                "model_name": "custom_model",
                "animal_subset": "combined",
                "subset": "combined",
                "selected": "true",
                "task": "classification",
                "artifact_path": str(explicit),
            }
        ]
    ).to_csv(tables_dir / "final_model_selection.csv", index=False)

    found = _find_best_model(tables_dir, tmp_path / "models")

    assert found == (explicit, "combined", "custom_model")


def test_calibration_summary_adds_subset_and_records_aliases(tmp_path):
    tables_dir = tmp_path / "tables"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()
    pd.DataFrame(
        [
            {
                "cohort": "age_group",
                "value": "adult",
                "records": 12,
                "observed_adoption_rate": 0.5,
                "mean_predicted_adoption_probability": 0.6,
                "calibration_gap": 0.1,
                "small_cohort_flag": False,
            }
        ]
    ).to_csv(tables_dir / "subgroup_reliability.csv", index=False)

    result = create_calibration_summary(tables_dir, summary_dir)

    assert {"animal_subset", "subset", "records"}.issubset(result.columns)
    assert result["subset"].equals(result["animal_subset"])
    assert set(result["records"]) == {12}


def test_reliability_red_flags_add_acceptance_aliases(tmp_path):
    tables_dir = tmp_path / "tables"
    summary_dir = tmp_path / "summary"
    tables_dir.mkdir()
    pd.DataFrame(
        [
            {
                "cohort": "age_group",
                "value": "senior",
                "records": 5,
                "observed_adoption_rate": 0.3,
                "mean_predicted_adoption_probability": 0.55,
                "calibration_gap": 0.25,
                "false_positive_rate": 0.2,
                "false_negative_rate": 0.1,
                "mae": 7.0,
                "small_cohort_flag": True,
            }
        ]
    ).to_csv(tables_dir / "subgroup_reliability.csv", index=False)

    result = create_reliability_red_flags(tables_dir, summary_dir)

    assert {
        "cohort",
        "value",
        "mean_predicted_adoption_probability",
        "subgroup_field",
        "subgroup_value",
        "mean_predicted_probability",
    }.issubset(result.columns)
    row = result.iloc[0]
    assert row["subgroup_field"] == row["cohort"]
    assert row["subgroup_value"] == row["value"]
    assert row["mean_predicted_probability"] == row["mean_predicted_adoption_probability"]
