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
    assert result["intake_volume_7d"].iloc[6] == 106.0


def test_rolling_features_decoupled_empty():
    intake_daily = pd.DataFrame({
        "context_date": pd.Series([], dtype="datetime64[ns]"),
        "intake_volume": pd.Series([], dtype="float64"),
    })
    
    result = compute_rolling_features_decoupled(intake_daily, [7, 30])
    
    assert result.empty or len(result.columns) == 3


def test_rolling_features_cache_store_and_get():
    cache = RollingFeaturesCache()
    
    date = pd.Timestamp("2021-01-15")
    cache.store_intake_volume(date, 50.0, 7)
    
    value = cache.get_intake_volume(date + pd.Timedelta(days=1), 7)
    
    assert value == 50.0


def test_rolling_features_cache_multiple_dates():
    cache = RollingFeaturesCache()
    
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    volumes = [10, 12, 8, 15, 20, 14, 16, 11, 13, 17]
    
    for date, vol in zip(dates, volumes):
        cache.store_intake_volume(date, vol, 7)
    
    last_date = dates[-1] + pd.Timedelta(days=1)
    value = cache.get_intake_volume(last_date, 7)
    
    assert value == sum(volumes)


def test_rolling_features_cache_persistence(tmp_path):
    cache = RollingFeaturesCache(cache_dir=tmp_path / "cache")
    
    date = pd.Timestamp("2021-01-15")
    cache.store_intake_volume(date, 50.0, 7)
    
    new_cache = RollingFeaturesCache(cache_dir=tmp_path / "cache")
    value = new_cache.get_intake_volume(date + pd.Timedelta(days=1), 7)
    
    assert value == 50.0


def test_rolling_features_decoupled_window_borders():
    intake_daily = pd.DataFrame({
        "context_date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "intake_volume": [10, 20, 30, 40, 50],
    })
    
    result = compute_rolling_features_decoupled(intake_daily, [3, 5])
    
    assert result["intake_volume_3d"].iloc[0] == 0.0
    assert result["intake_volume_3d"].iloc[2] == 30.0
    assert result["intake_volume_5d"].iloc[4] == 100.0
