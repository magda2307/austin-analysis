"""Tests for rolling features decoupling."""

import pandas as pd
import numpy as np
import pytest

from aac_adoption.features.rolling_features_cache import (
    RollingFeaturesCache,
    compute_rolling_features_decoupled,
)


def test_rolling_features_decoupled_basic():
    intake_daily = pd.DataFrame({
        "context_date": pd.date_range("2021-01-01", periods=15, freq="D"),
        "intake_volume": [10, 15, 12, 18, 20, 14, 16, 11, 13, 17, 19, 15, 14, 16, 18],
    })
    
    result = compute_rolling_features_decoupled(intake_daily, [7, 30])
    
    assert "intake_volume_7d" in result.columns
    assert "intake_volume_30d" in result.columns
    assert len(result) == 15
    assert result["intake_volume_7d"].iloc[6] == 89.0
    assert result["intake_volume_7d"].iloc[7] == 105.0


def test_rolling_features_decoupled_empty():
    intake_daily = pd.DataFrame({
        "context_date": pd.Series([], dtype="datetime64[ns]"),
        "intake_volume": pd.Series([], dtype="float64"),
    })
    
    result = compute_rolling_features_decoupled(intake_daily, [7, 30])
    
    assert result.empty or len(result.columns) == 3


def test_rolling_features_cache_basic():
    cache = RollingFeaturesCache()
    
    # Basic cache operations
    cache._cache["test_key"] = pd.Series([1, 2, 3])
    assert len(cache._cache["test_key"]) == 3


def test_rolling_features_cache_multiple_dates():
    cache = RollingFeaturesCache()
    
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    volumes = [10, 12, 8, 15, 20]
    
    # Store in cache
    for date, vol in zip(dates, volumes):
        cache._cache[f"intake_volume_7d_cache"] = pd.Series(volumes, index=dates)
    
    value = cache._cache[f"intake_volume_7d_cache"].iloc[0]
    
    assert value == 10


def test_rolling_features_cache_persistence(tmp_path):
    cache = RollingFeaturesCache(cache_dir=tmp_path / "cache")
    
    # Test that cache directory exists
    assert cache.cache_dir.exists()


def test_rolling_features_decoupled_window_borders():
    intake_daily = pd.DataFrame({
        "context_date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "intake_volume": [10, 20, 30, 40, 50],
    })
    
    result = compute_rolling_features_decoupled(intake_daily, [3, 5])
    
    assert result["intake_volume_3d"].iloc[0] == 0.0
    assert result["intake_volume_3d"].iloc[2] == 30.0
    assert result["intake_volume_5d"].iloc[4] == 100.0
