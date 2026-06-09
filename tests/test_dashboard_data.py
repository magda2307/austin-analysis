import math
import json
import pandas as pd
import pytest

from aac_adoption.dashboard.data import (
    best_model_rows,
    build_prediction_record,
    build_profile_prediction_record,
    model_feature_columns,
    profile_global_shap_reasons,
    similar_historical_cases,
    visibility_need_from_prediction,
    los_days_to_bucket,
    load_model_metadata,
    predict_from_record,
)
from aac_adoption.models.metadata import REQUIRED_MODEL_METADATA


def test_best_model_rows_selects_expected_metrics():
    # 1. Test case when pr_auc is present: should sort by pr_auc desc, then roc_auc desc as tie-breaker
    # and set primary_metric to "pr_auc".
    classification_with_pr = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "logistic", "roc_auc": 0.7, "pr_auc": 0.6},
            {"animal_subset": "combined", "model_name": "boosting", "roc_auc": 0.8, "pr_auc": 0.7},
            # Tie breaker: same pr_auc, higher roc_auc should win
            {"animal_subset": "combined", "model_name": "random_forest", "roc_auc": 0.9, "pr_auc": 0.7},
        ]
    )
    regression = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "ridge", "mae": 30.0},
            {"animal_subset": "combined", "model_name": "boosting", "mae": 20.0},
        ]
    )

    result_with_pr = best_model_rows(classification_with_pr, regression)
    assert len(result_with_pr) == 2
    # For classification, random_forest should be chosen because pr_auc is 0.7 (higher than 0.6)
    # and roc_auc is 0.9 (higher than boosting's 0.8)
    assert set(result_with_pr["model_name"]) == {"random_forest", "boosting"}
    assert set(result_with_pr["primary_metric"]) == {"pr_auc", "mae"}

    clf_row = result_with_pr[result_with_pr["task"] == "classification"].iloc[0]
    assert clf_row["model_name"] == "random_forest"
    assert clf_row["primary_metric"] == "pr_auc"
    assert clf_row["score"] == 0.7

    # 2. Test fallback to roc_auc sorting when pr_auc is not present
    classification_no_pr = pd.DataFrame(
        [
            {"animal_subset": "combined", "model_name": "logistic", "roc_auc": 0.7},
            {"animal_subset": "combined", "model_name": "boosting", "roc_auc": 0.8},
        ]
    )
    result_no_pr = best_model_rows(classification_no_pr, regression)
    assert len(result_no_pr) == 2
    assert set(result_no_pr["model_name"]) == {"boosting"}
    assert set(result_no_pr["primary_metric"]) == {"roc_auc", "mae"}

    clf_row_no_pr = result_no_pr[result_no_pr["task"] == "classification"].iloc[0]
    assert clf_row_no_pr["model_name"] == "boosting"
    assert clf_row_no_pr["primary_metric"] == "roc_auc"
    assert clf_row_no_pr["score"] == 0.8


def test_build_prediction_record_creates_model_features():
    record = build_prediction_record(
        animal_type="Dog",
        intake_type="Stray",
        intake_condition="Normal",
        sex_upon_intake="Intact Male",
        age_days=365.25 * 2,
        breed="Labrador Retriever Mix",
        color="Black/White",
        has_name=True,
        intake_date=pd.Timestamp("2024-06-01"),
    )

    assert record.loc[0, "age_group"] == "young"
    assert record.loc[0, "simplified_breed_group"] == "retriever_type"
    assert record.loc[0, "simplified_color_group"] == "black_or_dark"
    assert record.loc[0, "covid_period"] == "post_covid"
    assert record.loc[0, "intake_season"] == "summer"
    assert "is_extreme_heat" in record.columns


def test_build_profile_prediction_record_uses_representative_values():
    profile = pd.Series(
        {
            "animal_type": "Dog",
            "age_group": "senior",
            "intake_type": "Owner Surrender",
            "intake_condition": "Normal",
            "sex_upon_intake": "Neutered Male",
            "simplified_breed_group": "pit_bull_type",
            "simplified_color_group": "brown_tan",
            "is_named": True,
        }
    )

    record = build_profile_prediction_record(profile)

    assert record.loc[0, "age_group"] == "senior"
    assert record.loc[0, "simplified_breed_group"] == "pit_bull_type"
    assert record.loc[0, "simplified_color_group"] == "brown_tan"
    assert bool(record.loc[0, "is_named"]) is True


def test_model_feature_columns_uses_artifact_metadata(tmp_path):
    models_dir = tmp_path / "models"
    metadata_path = models_dir / "classification" / "combined" / "catboost.json"
    metadata_path.parent.mkdir(parents=True)
    metadata = {key: None for key in REQUIRED_MODEL_METADATA}
    metadata.update(
        {
            "feature_columns": ["animal_type", "intake_type", "age_days"],
            "model_name": "catboost",
            "task": "classification",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    record = pd.DataFrame(
        [
            {
                "animal_type": "Cat",
                "intake_type": "Stray",
                "age_days": 100,
                "is_extreme_heat": False,
            }
        ]
    )

    assert model_feature_columns(record, models_dir, "classification") == [
        "animal_type",
        "intake_type",
        "age_days",
    ]


def test_load_model_metadata_rejects_invalid_sidecar(tmp_path):
    metadata_path = (
        tmp_path
        / "models"
        / "classification"
        / "combined"
        / "catboost.json"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        '{"feature_columns": ["animal_type"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required model metadata fields"):
        load_model_metadata(tmp_path / "models", "classification")


def test_model_feature_columns_requires_metadata(tmp_path):
    record = pd.DataFrame([{"animal_type": "Dog"}])

    with pytest.raises(FileNotFoundError, match="Missing model metadata"):
        model_feature_columns(
            record,
            tmp_path / "models",
            "classification",
        )


def test_similar_historical_cases_returns_outcome_mix(tmp_path):
    data = pd.DataFrame(
        {
            "animal_type": ["Dog", "Dog", "Dog"],
            "age_group": ["senior", "senior", "senior"],
            "intake_type": ["Stray", "Stray", "Stray"],
            "intake_condition": ["Normal", "Normal", "Normal"],
            "simplified_breed_group": ["pit_bull_type", "pit_bull_type", "pit_bull_type"],
            "simplified_color_group": ["brown_tan", "brown_tan", "brown_tan"],
            "sex_upon_intake": ["Neutered Male", "Neutered Male", "Neutered Male"],
            "is_named": [True, True, True],
            "classification_target": [1, 0, 0],
            "days_to_outcome": [10.0, 20.0, 30.0],
            "outcome_type": ["Adoption", "Transfer", "Euthanasia"],
        }
    )
    path = tmp_path / "modeling.csv"
    data.to_csv(path, index=False)
    record = build_prediction_record(
        animal_type="Dog",
        intake_type="Stray",
        intake_condition="Normal",
        sex_upon_intake="Neutered Male",
        age_days=365.25 * 10,
        breed="Pit Bull Mix",
        color="Brown/White",
        has_name=True,
        intake_date=pd.Timestamp("2024-06-01"),
    )

    similar = similar_historical_cases(path, record)

    assert similar.loc[0, "similar_records"] == 3
    assert similar.loc[0, "historical_adoption_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "median_days_to_outcome"] == 20.0
    assert similar.loc[0, "adoption_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "transfer_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "euthanasia_rate_pct"] == pytest.approx(100 / 3)
    assert similar.loc[0, "matching_level"] == "exact visible profile"


def test_profile_global_shap_reasons_maps_profile_values():
    profile = pd.Series(
        {
            "age_group": "baby",
            "intake_type": "Stray",
            "simplified_breed_group": "domestic_cat",
            "is_named": False,
        }
    )
    shap_global = pd.DataFrame(
        {
            "feature": ["age_upon_intake", "intake_type", "surprise"],
            "mean_abs_shap": [0.5, 0.4, 0.9],
        }
    )

    reasons = profile_global_shap_reasons(profile, shap_global, top_n=2)

    assert list(reasons["feature"]) == ["age_upon_intake", "intake_type"]
    assert "profile_value" in reasons.columns


def test_visibility_need_from_prediction_labels_quadrants():
    assert visibility_need_from_prediction(0.7, 5) == "quick placement likely"
    assert visibility_need_from_prediction(0.7, 20) == "needs visibility"
    assert visibility_need_from_prediction(0.2, 20) == "long-stay risk"


def test_los_days_to_bucket():
    # <= 7
    assert los_days_to_bucket(0.0) == "0-7d"
    assert los_days_to_bucket(5.0) == "0-7d"
    assert los_days_to_bucket(7.0) == "0-7d"

    # 7 < days <= 30
    assert los_days_to_bucket(7.1) == "8-30d"
    assert los_days_to_bucket(15.0) == "8-30d"
    assert los_days_to_bucket(30.0) == "8-30d"

    # 30 < days <= 60
    assert los_days_to_bucket(30.1) == "31-60d"
    assert los_days_to_bucket(45.0) == "31-60d"
    assert los_days_to_bucket(60.0) == "31-60d"

    # 60 < days <= 90
    assert los_days_to_bucket(60.1) == "61-90d"
    assert los_days_to_bucket(75.0) == "61-90d"
    assert los_days_to_bucket(90.0) == "61-90d"

    # days > 90
    assert los_days_to_bucket(90.1) == "90+d"
    assert los_days_to_bucket(120.0) == "90+d"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_from_record_handles_calibration():
    from unittest.mock import patch, MagicMock
    import numpy as np

    with patch("aac_adoption.dashboard.data.load_table") as mock_load_table, \
         patch("aac_adoption.dashboard.data.load_model_metadata") as mock_load_model_metadata, \
         patch("aac_adoption.dashboard.data.load_model") as mock_load_model, \
         patch("aac_adoption.dashboard.data.artifact_path") as mock_artifact_path, \
         patch("aac_adoption.dashboard.data.joblib.load") as mock_joblib_load:

        # 1. Setup mock data
        selection_df = pd.DataFrame([
            {"selected": True, "task": "classification", "animal_subset": "combined", "model_name": "catboost"},
            {"selected": True, "task": "regression", "animal_subset": "combined", "model_name": "catboost"},
        ])
        mock_load_table.return_value = selection_df

        def metadata_for_task(models_dir, task, subset="combined", model_name="catboost"):
            metadata = {
                "feature_columns": ["animal_type", "intake_type", "age_days"],
            }
            if task == "regression":
                metadata["prediction_inverse_transform"] = "expm1"
            return metadata

        mock_load_model_metadata.side_effect = metadata_for_task

        # 2. Setup mock models
        mock_calibrated_clf = MagicMock()
        mock_calibrated_clf.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_joblib_load.return_value = mock_calibrated_clf

        mock_base_clf = MagicMock()
        mock_base_clf.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_regressor = MagicMock()
        mock_regressor.predict.return_value = np.array([math.log1p(15)])

        def side_effect_load_model(models_dir, task, subset="combined", model_name="catboost"):
            if task == "classification":
                return mock_base_clf
            else:
                return mock_regressor

        mock_load_model.side_effect = side_effect_load_model

        # 3. Setup mock path
        mock_path = MagicMock()
        mock_artifact_path.return_value = mock_path

        # Create test record
        record = build_prediction_record(
            animal_type="Dog",
            intake_type="Stray",
            intake_condition="Normal",
            sex_upon_intake="Intact Male",
            age_days=365.25 * 2,
            breed="Labrador Retriever Mix",
            color="Black/White",
            has_name=True,
            intake_date=pd.Timestamp("2024-06-01"),
        )

        # CASE A: Calibrated model exists
        mock_path.exists.return_value = True

        res_calibrated = predict_from_record(record, models_dir="models/advanced", subset="combined")

        # Assertions for calibrated
        assert res_calibrated.adoption_probability == 0.9
        assert res_calibrated.predicted_days_to_outcome == pytest.approx(15.0)
        assert res_calibrated.los_bucket == "8-30d"
        mock_joblib_load.assert_called_once_with(mock_path)

        # CASE B: Calibrated model does NOT exist (fallback to base classifier)
        mock_joblib_load.reset_mock()
        mock_path.exists.return_value = False

        res_fallback = predict_from_record(record, models_dir="models/advanced", subset="combined")

        # Assertions for fallback
        assert res_fallback.adoption_probability == 0.7
        assert res_fallback.predicted_days_to_outcome == pytest.approx(15.0)
        assert res_fallback.los_bucket == "8-30d"
        mock_joblib_load.assert_not_called()
