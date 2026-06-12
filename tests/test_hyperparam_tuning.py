"""Tests for hyperparameter tuning."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from aac_adoption.optimization.hyperparam_tuning import (
    tune_histgradient_boosting_classification,
    tune_histgradient_boosting_regression,
)
from aac_adoption.models.tune import tune_models
from aac_adoption.models.split import make_time_split
from aac_adoption.features.feature_sets import model_feature_columns


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "age_upon_intake_days": np.random.randn(n_samples) * 100,
        "intake_condition": np.random.choice(["Normal", "Injured"], n_samples),
        "intake_year": np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples),
        "classification_target": np.random.choice([0, 1], n_samples),
        "regression_target_days": np.random.randint(1, 30, n_samples),
    })
    
    return df


@pytest.fixture
def divergent_row_data():
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "animal_type": np.random.choice(["Dog", "Cat"], n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender"], n_samples),
        "intake_condition": np.random.choice(["Normal", "Injured", "Critical", "Deceased"], n_samples),
        "sex_upon_intake": np.random.choice(["Neutered Male", "Spayed Female", "Intact Male", "Intact Female"], n_samples),
        "age_days": np.random.choice([1, 5, 10, 100, 500, 1500, 2500, 3500, 4500, 5000], n_samples).astype(float),
        "age_group": np.random.choice(["Kitten/Puppy", "Adult", "Senior"], n_samples),
        "intake_year": np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples),
        "intake_datetime": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        "classification_target": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "regression_target_days": np.random.choice([1, 2, 3, 5, 7, 14, 21, 30], n_samples).astype(float),
    })
    
    divergent_indices = [5, 25, 50, 75, 100, 125, 150, 175]
    for idx in divergent_indices:
        df.loc[idx, "age_days"] = np.random.choice([0, 10000, 15000, 20000]).astype(float)
        df.loc[idx, "regression_target_days"] = np.random.choice([0, 45, 60, 90]).astype(float)
    
    return df


def test_tune_histgradient_classification(sample_data):
    result = tune_histgradient_boosting_classification(
        sample_data,
        n_splits=3,
        max_iter_options=[10, 20],
        max_leaf_nodes_options=[5, 10],
        learning_rate_options=[0.05, 0.1],
    )
    
    assert "best_score" in result
    assert "best_params" in result
    assert result["best_params"] is not None
    assert "max_iter" in result["best_params"]
    assert "max_leaf_nodes" in result["best_params"]
    assert "learning_rate" in result["best_params"]


def test_tune_histgradient_regression(sample_data):
    result = tune_histgradient_boosting_regression(
        sample_data,
        n_splits=2,
        max_iter_options=[10, 20],
        max_leaf_nodes_options=[5, 10],
        learning_rate_options=[0.05, 0.1],
    )
    
    assert "best_score" in result
    assert "best_params" in result
    assert result["best_params"] is not None


def test_tune_empty_data():
    empty_df = pd.DataFrame({
        "age_upon_intake_days": pd.Series([], dtype="float64"),
        "intake_condition": pd.Series([], dtype="object"),
        "intake_year": pd.Series([], dtype="int64"),
        "classification_target": pd.Series([], dtype="int64"),
    })
    
    result = tune_histgradient_boosting_classification(empty_df)
    
    assert result.get("status") == "failed" or result.get("best_score", -float("inf")) > -float("inf")


def test_tune_models_runs_successfully():
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "animal_type": np.random.choice(["Dog", "Cat"], n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender"], n_samples),
        "intake_condition": np.random.choice(["Normal", "Injured"], n_samples),
        "sex_upon_intake": np.random.choice(["Neutered Male", "Spayed Female"], n_samples),
        "age_days": np.random.randint(1, 3000, n_samples).astype(float),
        "age_group": np.random.choice(["Adult", "Kitten/Puppy"], n_samples),
        "intake_year": np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples),
        "intake_datetime": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        "classification_target": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "regression_target_days": np.random.randint(1, 30, n_samples).astype(float),
    })
    
    best_params, studies = tune_models(df, n_trials=2)
    
    assert isinstance(best_params, dict)
    assert isinstance(studies, dict)
    assert "catboost_classification" in best_params
    assert "catboost_regression" in best_params
    assert "hist_gradient_boosting_classification" in best_params
    assert "hist_gradient_boosting_regression" in best_params
    
    assert "catboost_classification" in studies
    assert "catboost_regression" in studies
    assert "hist_gradient_boosting_classification" in studies
    assert "hist_gradient_boosting_regression" in studies


def test_tune_models_regression_feature_alignment(divergent_row_data):
    df = divergent_row_data
    
    split_clf = make_time_split(df, "classification_target", animal_subset="combined")
    train_df_clf = split_clf.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns_clf = model_feature_columns(train_df_clf)
    X_clf = train_df_clf[feature_columns_clf]
    y_clf = train_df_clf["classification_target"]
    
    split_reg = make_time_split(df, "regression_target_days", animal_subset="combined")
    train_df_reg = split_reg.train.sort_values("intake_datetime").reset_index(drop=True)
    feature_columns_reg = model_feature_columns(train_df_reg)
    X_reg = train_df_reg[feature_columns_reg]
    y_reg = train_df_reg["regression_target_days"]
    
    assert X_reg.index.equals(y_reg.index), "Regression feature frame index must match regression target index"
    
    assert len(X_reg) == len(y_reg), "Regression feature frame row count must match regression target row count"
    
    assert X_reg.index.intersection(y_reg.index).equals(X_reg.index), "No misaligned rows in regression feature frame and target"


def test_tune_models_catboost_regression_fit_spy(divergent_row_data):
    df = divergent_row_data
    
    split_reg = make_time_split(df, "regression_target_days", animal_subset="combined")
    train_df_reg = split_reg.train.sort_values("intake_datetime").reset_index(drop=True)
    y_reg_original = train_df_reg["regression_target_days"]
    
    with patch("catboost.CatBoostRegressor.fit") as mock_fit:
        best_params, studies = tune_models(df, n_trials=2)
        
        assert mock_fit.call_count >= 1
        
        for call in mock_fit.call_args_list:
            X_tr = call[0][0]
            y_tr = call[0][1]
            assert isinstance(X_tr, pd.DataFrame), "X_tr must be a DataFrame"
            assert isinstance(y_tr, pd.Series), "y_tr must be a Series"
            assert X_tr.index.equals(y_tr.index), "X_tr and y_tr indices must match for each CV fold"
        
        actual_y_values = [call[0][1] for call in mock_fit.call_args_list]
        
        for y in actual_y_values:
            original_y_tr = y_reg_original[y.index]
            assert np.allclose(y, np.log1p(original_y_tr)), "y_tr passed to fit must be log-transformed"


def test_tune_failure_payloads_rejected_by_trainer(tmp_path):
    import json
    from aac_adoption.models.train_advanced import train_all_advanced
    from aac_adoption.models.train_boosting import train_all_boosting
    from aac_adoption.models.train_adopted_regression import train_all_adopted
    
    dataset_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "classification_target": [0, 1, 0, 1],
        "regression_target_days": [1, 2, 3, 4],
        "days_to_adoption": [1, 2, 3, 4],
        "adopted": [1, 1, 1, 1],
        "intake_datetime": pd.date_range("2020-01-01", periods=4, freq="D")
    })
    df.to_csv(dataset_path, index=False)
    
    failed_payload = {
        "catboost_classification": {"status": "failed", "error": "mock failure", "best_params": None},
        "hist_gradient_boosting_classification": {"status": "failed"},
        "catboost_adopted_regression": {"status": "failed"}
    }
    tuned_params_path = tmp_path / "tuned_params.json"
    tuned_params_path.write_text(json.dumps(failed_payload))
    
    with pytest.raises(ValueError, match="Tuning failed or missing/malformed parameters"):
        train_all_advanced(
            data_path=dataset_path,
            metrics_dir=tmp_path / "metrics",
            models_dir=tmp_path / "models",
            tuned_params_path=tuned_params_path
        )
        
    with pytest.raises(ValueError, match="Tuning failed or missing/malformed parameters"):
        train_all_boosting(
            data_path=dataset_path,
            metrics_dir=tmp_path / "metrics",
            models_dir=tmp_path / "models",
            tuned_params_path=tuned_params_path
        )


def test_subgroup_analysis_standalone_helper_runs():
    from aac_adoption.models.evaluate import subgroup_analysis
    
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1]
    y_score = [0.1, 0.9, 0.2, 0.4, 0.6, 0.8]
    subgroup = ["dog", "dog", "cat", "cat", "bird", "bird"]
    
    df_metrics = subgroup_analysis(y_true, y_pred, y_score, subgroup)
    
    # Should return a valid pandas DataFrame
    assert isinstance(df_metrics, pd.DataFrame)
    assert len(df_metrics) == 3
    assert "subgroup" in df_metrics.columns
    assert set(df_metrics["subgroup"]) == {"bird", "cat", "dog"}

import pytest
pytestmark = pytest.mark.slow
