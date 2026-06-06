"""Tests for rolling context features to ensure no data leakage."""

import pandas as pd
import pytest
from aac_adoption.data.context_data import _rolling_prior_counts


def test_rolling_prior_counts_excludes_current_day():
    """Verify that a rolling sum strictly uses prior days and excludes the current day."""
    
    # Simulate intakes on Jan 1, Jan 2, Jan 3
    daily = pd.DataFrame([
        {"context_date": "2024-01-01", "intake_volume": 10},
        {"context_date": "2024-01-02", "intake_volume": 20},
        {"context_date": "2024-01-03", "intake_volume": 30},
        {"context_date": "2024-01-06", "intake_volume": 100},
    ])
    
    target_dates = pd.Series(pd.to_datetime(["2024-01-07"]))
    result = _rolling_prior_counts(daily, "intake_volume", [2, 7], target_dates=target_dates)
    
    # Make lookup easy
    res_dict = result.set_index("context_date").to_dict(orient="index")
    
    # 2024-01-01: No prior data -> 0
    assert res_dict[pd.Timestamp("2024-01-01")]["intake_volume_2d"] == 0
    assert res_dict[pd.Timestamp("2024-01-01")]["intake_volume_7d"] == 0
    
    # 2024-01-02: Prior 2 days is just Jan 1 (10) -> 10. (Excludes Jan 2's 20)
    assert res_dict[pd.Timestamp("2024-01-02")]["intake_volume_2d"] == 10
    
    # 2024-01-03: Prior 2 days are Jan 1 (10) + Jan 2 (20) -> 30. (Excludes Jan 3's 30)
    assert res_dict[pd.Timestamp("2024-01-03")]["intake_volume_2d"] == 30
    
    # 2024-01-04 (implicit 0): Prior 2 days are Jan 2 (20) + Jan 3 (30) -> 50
    assert res_dict[pd.Timestamp("2024-01-04")]["intake_volume_2d"] == 50
    
    # 2024-01-05 (implicit 0): Prior 2 days are Jan 3 (30) + Jan 4 (0) -> 30
    assert res_dict[pd.Timestamp("2024-01-05")]["intake_volume_2d"] == 30
    
    # 2024-01-06: Prior 2 days are Jan 4 (0) + Jan 5 (0) -> 0. (Excludes Jan 6's 100)
    assert res_dict[pd.Timestamp("2024-01-06")]["intake_volume_2d"] == 0
    
    # Let's check 7d window for Jan 6: Prior 7 days are Dec 30 to Jan 5.
    # Included: Jan 1 (10), Jan 2 (20), Jan 3 (30) -> 60
    assert res_dict[pd.Timestamp("2024-01-06")]["intake_volume_7d"] == 60
    
    # Check Jan 7: Prior 7 days are Dec 31 to Jan 6. Includes Jan 6 (100) -> 160
    assert res_dict[pd.Timestamp("2024-01-07")]["intake_volume_7d"] == 160
