"""Tests for feature engineering outliers and winsorization."""

import pandas as pd
import numpy as np

from aac_adoption.features.feature_engineering import winsorize_outliers


def test_winsorize_outliers_reduces_extreme_values():
    data = pd.Series([1, 2, 3, 4, 5, 100, 200, 300, 50, 40])
    result = winsorize_outliers(data, 0.1, 0.9)
    
    assert result.max() <= data.quantile(0.9)
    assert result.min() >= data.quantile(0.1)
    assert len(result) == len(data)


def test_winsorize_outliers_no_extreme_after():
    data = pd.Series([1, 2, 3, 4, 5, 500])
    result = winsorize_outliers(data, 0.05, 0.95)
    
    assert result.max() < 500
    assert result.max() >= 5


def test_winsorize_outliers_preserves_non_extreme():
    data = pd.Series([10, 20, 30, 40, 50, 60, 70])
    result = winsorize_outliers(data, 0.1, 0.9)
    
    lower = data.quantile(0.1)
    upper = data.quantile(0.9)
    assert result.min() >= lower
    assert result.max() <= upper
    assert len(result) == len(data)


def test_winsorize_outliers_handles_nans():
    data = pd.Series([1, 2, np.nan, 4, 5, 1000])
    result = winsorize_outliers(data, 0.1, 0.9)
    
    assert pd.isna(result.iloc[2])
    assert result.max() <= data.quantile(0.9)
    assert result.min() >= data.quantile(0.1)
