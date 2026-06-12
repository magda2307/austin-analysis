import pandas as pd

from aac_adoption.reporting.evidence_pack import (
    best_model_evidence,
    bootstrap_metric_intervals,
    create_evidence_pack,
    model_limitations_by_cohort,
    model_failure_modes,
    subgroup_adoption_milestones,
    subgroup_metric_intervals,
    subgroup_reliability,
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


def test_subgroup_outputs_have_expected_schema(tmp_path):
    predictions = _predictions()
    reliability = subgroup_reliability(predictions, min_records=2)
    intervals = subgroup_metric_intervals(predictions, n_bootstrap=5, min_records=2, random_state=1)
    failures = model_failure_modes(reliability, top_n=2)

    assert {"cohort", "value", "records", "calibration_gap", "mae", "false_positive_rate", "false_negative_rate"}.issubset(reliability.columns)
    assert {"cohort", "value", "metric", "records", "lower", "estimate", "upper", "status"}.issubset(intervals.columns)
    assert {"ok", "insufficient_class_variety"} & set(intervals["status"])
    assert {"failure_mode", "cohort", "metric", "value_score", "interpretation"}.issubset(failures.columns)


def test_subgroup_adoption_milestones_include_day_60(tmp_path):
    data = pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat", "Cat"],
            "age_group": ["baby", "senior", "baby", "senior"],
            "intake_type": ["Stray", "Stray", "Owner Surrender", "Owner Surrender"],
            "intake_condition": ["Normal", "Sick", "Normal", "Normal"],
            "outcome_subtype": ["Foster", "Medical", "Foster", "Foster"],
            "classification_target": [1, 1, 0, 1],
            "days_to_adoption": [5.0, 45.0, 0.0, 80.0],
            "days_to_outcome": [5.0, 45.0, 10.0, 80.0],
        }
    )
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    milestones = subgroup_adoption_milestones(data_path, min_records=1)

    assert {"adopted_by_day_7_pct", "adopted_by_day_30_pct", "adopted_by_day_60_pct", "adopted_by_day_90_pct"}.issubset(milestones.columns)
    assert "animal_type" in set(milestones["cohort"])


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
    pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Cat"],
            "age_group": ["senior", "baby", "baby"],
            "intake_type": ["Stray", "Stray", "Owner Surrender"],
            "intake_condition": ["Normal", "Normal", "Normal"],
            "outcome_subtype": ["Foster", "Foster", "Foster"],
            "classification_target": [1, 0, 1],
            "days_to_adoption": [10.0, 0.0, 30.0],
            "days_to_outcome": [10.0, 5.0, 30.0],
        }
    ).to_csv(tmp_path / "modeling.csv", index=False)

    paths = create_evidence_pack(
        data_path=tmp_path / "modeling.csv",
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
    assert paths["subgroup_reliability"].exists()
    assert paths["subgroup_intervals"].exists()
    assert paths["subgroup_milestones"].exists()
    assert paths["failure_modes"].exists()
    assert paths["journeys"].exists()
    assert paths["local_explanations"].exists()
    assert paths["summary"].exists()
    assert paths["subgroup_summary"].exists()
    assert paths["local_explanation_summary"].exists()
    assert "associated with model behavior" in paths["summary"].read_text(encoding="utf-8")

    local_examples = pd.read_csv(paths["local_explanations"])
    assert not local_examples.empty
    assert {
        "explanation_type",
        "profile_label",
        "similar_historical_cases",
        "shap_model_reasons",
        "limitation_note",
    }.issubset(local_examples.columns)
    assert local_examples["similar_historical_cases"].str.contains("cases").any()
    assert local_examples["shap_model_reasons"].str.len().gt(0).all()
    assert local_examples["limitation_note"].str.contains("non-causal", case=False).all()

    local_summary = paths["local_explanation_summary"].read_text(encoding="utf-8")
    assert "illustrative and non-causal" in local_summary
    assert "similar historical cases" in local_summary
    assert "SHAP/model reasons" in local_summary
    assert "Limitations" in local_summary


def test_bootstrap_handles_cluster_and_fallback():
    predictions = _predictions().copy()
    predictions["animal_id"] = [1, 1, 2, 2, 3, 3]
    intervals = bootstrap_metric_intervals(predictions, n_bootstrap=10, random_state=1)
    
    assert not intervals.empty
    assert {"metric", "animal_subset", "lower", "estimate", "upper", "bootstrap_samples", "pr_auc_first"}.issubset(intervals.columns)


def test_best_model_evidence_prioritizes_pr_auc():
    classification = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "catboost", "roc_auc": 0.8, "pr_auc": 0.75},
            {"animal_subset": "combined", "model_name": "logistic", "roc_auc": 0.85, "pr_auc": 0.65},
        ]
    )
    regression = pd.DataFrame()
    
    evidence = best_model_evidence(classification, regression)
    
    assert not evidence.empty
    assert evidence.iloc[0]["metric"] == "pr_auc"
    assert "PR-AUC" in evidence.iloc[0]["interpretation"]


def test_model_limitations_respects_min_records():
    predictions = _predictions()
    limitations = model_limitations_by_cohort(predictions, min_records=5)
    
    assert (limitations[limitations["records"] < 5]["small_cohort_flag"] == True).all()


def test_subgroup_intervals_enforces_cohort_threshold():
    predictions = _predictions()
    intervals = subgroup_metric_intervals(predictions, n_bootstrap=5, min_records=2, cohort_threshold=0.10, random_state=1)
    
    assert {"cohort", "value", "metric", "records", "lower", "estimate", "upper", "bootstrap_samples", "status", "interpretation_status"}.issubset(intervals.columns)
    if not intervals.empty:
        assert {"small_cohort", "insufficient_class_variety", "ok"}.issubset(set(intervals["status"]))
        assert intervals["interpretation_status"].notna().all()


def test_subgroup_adoption_milestones_missing_targets_fails(tmp_path):
    import pytest
    data = pd.DataFrame(
        {
            "animal_type": ["Dog", "Cat"],
            "adopted": [True, False],
            "days_to_outcome": [5.0, 10.0],
        }
    )
    data_path = tmp_path / "modeling.csv"
    data.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="modeling dataset missing target columns"):
        subgroup_adoption_milestones(data_path, min_records=1)

