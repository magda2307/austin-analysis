"""Tests for hyperparameter tuning."""

import pandas as pd
import numpy as np
import pytest

from aac_adoption.optimization.hyperparam_tuning import (
    tune_histgradient_boosting_classification,
    tune_histgradient_boosting_regression,
)


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
    
    assert result["best_score"] > -float("inf") or result["best_params"] is None
