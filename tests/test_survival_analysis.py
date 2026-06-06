"""Tests for survival analysis utilities."""

import pandas as pd
import numpy as np
import pytest

from aac_adoption.analysis.survival_analysis import (
    compute_kaplan_meier_survival,
    fit_cox_proportional_hazards,
    compute_concordance_index,
    compute_LOS_quantiles,
    add_censoring_indicators,
    log_transform_LOS,
)


@pytest.fixture
def sample_los_data():
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(1, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
    })
    
    return df


def test_kaplan_meier_survival(sample_los_data):
    result = compute_kaplan_meier_survival(sample_los_data, time_points=[7, 14, 30, 60])
    
    assert "days" in result.columns
    assert "survival_probability" in result.columns
    assert len(result) == 4
    assert result["survival_probability"].iloc[0] <= 1.0


def test_kaplan_meier_empty():
    empty_df = pd.DataFrame()
    result = compute_kaplan_meier_survival(empty_df)
    
    assert result.empty


def test_cox_proportional_hazards_fitting(sample_los_data):
    cph, summary = fit_cox_proportional_hazards(
        sample_los_data,
        feature_cols=["animal_type", "age_group"],
    )
    
    assert cph is not None
    assert len(summary) > 0
    assert "coefficient" in summary.columns
    assert "hazard_ratio" in summary.columns


def test_cox_empty_data():
    empty_df = pd.DataFrame()
    cph, summary = fit_cox_proportional_hazards(empty_df)
    
    assert cph is None
    assert summary.empty


def test_concordance_index(sample_los_data):
    sample_los_data["predicted_days"] = sample_los_data["days_to_outcome"] + np.random.randn(100) * 5
    
    c_index = compute_concordance_index(sample_los_data, predicted_col="predicted_days")
    
    assert 0.0 <= c_index <= 1.0


def test_los_quantiles(sample_los_data):
    result = compute_LOS_quantiles(sample_los_data, [0.5, 0.9])
    
    assert len(result) == 4
    assert "LOS_days" in result.columns
    assert result[result["animal_type"] == "Dog"]["quantile"].iloc[0] == 0.5


def test_add_censoring_indicators(sample_los_data):
    result = add_censoring_indicators(sample_los_data, max_observation_period=30)
    
    assert "is_censored" in result.columns
    assert "tracked_days" in result.columns
    assert result["tracked_days"].max() == 30


def test_log_transform_LOS(sample_los_data):
    result = log_transform_LOS(sample_los_data)
    
    assert "log_days_to_outcome" in result.columns
    assert result["log_days_to_outcome"].dtype == float
    assert result["log_days_to_outcome"].min() >= 0
