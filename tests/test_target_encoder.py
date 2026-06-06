"""Tests for target encoder with leakage prevention."""

import pandas as pd
import numpy as np
import pytest

from aac_adoption.features.target_encoder import OOFBayesianTargetEncoder


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "high_card_col": [f"cat_{i % 20}" for i in range(n_samples)],
        "low_card_col": [f"cat_{i % 3}" for i in range(n_samples)],
        "continuous_col": np.random.randn(n_samples),
    })
    
    target = pd.Series(np.random.choice([0, 1], size=n_samples))
    
    return df, target


def test_target_encoder_filters_leakage_columns():
    encoder = OOFBayesianTargetEncoder(
        columns=["high_card_col", "adopted", "intake_datetime"],
        smoothing=10.0,
    )
    
    assert "adopted" not in encoder.columns


def test_target_encoder_high_cardinality_detection(sample_data):
    df, target = sample_data
    encoder = OOFBayesianTargetEncoder(
        columns=["high_card_col", "low_card_col"],
        smoothing=10.0,
    )
    
    encoder.fit(df, target)
    
    assert "high_card_col" in encoder.high_cardinality_cols_
    assert "low_card_col" not in encoder.high_cardinality_cols_
    assert len(encoder.high_cardinality_cols_) == 1


def test_target_encoder_oof_fitting(sample_data):
    df, target = sample_data
    encoder = OOFBayesianTargetEncoder(
        columns=["high_card_col", "low_card_col"],
        smoothing=10.0,
        n_splits=5,
    )
    
    result = encoder.fit_transform(df, target)
    
    assert result.shape == df.shape
    assert result["high_card_col"].dtype == float
    assert result["low_card_col"].dtype == float


def test_target_encoder_transform_new_data(sample_data):
    df, target = sample_data
    encoder = OOFBayesianTargetEncoder(
        columns=["high_card_col", "low_card_col"],
        smoothing=10.0,
    )
    
    encoder.fit(df, target)
    
    new_df = pd.DataFrame({
        "high_card_col": ["cat_0", "cat_10", "cat_15"],
        "low_card_col": ["cat_0", "cat_1", "cat_2"],
        "continuous_col": [1.0, 2.0, 3.0],
    })
    
    result = encoder.transform(new_df)
    
    assert result.shape == (3, 3)
    assert result["high_card_col"].dtype == float


def test_target_encoder_unknown_handling(sample_data):
    df, target = sample_data
    encoder = OOFBayesianTargetEncoder(
        columns=["high_card_col"],
        smoothing=10.0,
        handle_unknown="return_nan",
    )
    
    encoder.fit(df, target)
    
    new_df = pd.DataFrame({
        "high_card_col": ["unknown_category", "cat_0"],
    })
    
    result = encoder.transform(new_df)
    
    assert pd.isna(result["high_card_col"].iloc[0])
    assert not pd.isna(result["high_card_col"].iloc[1])
