"""Tests for recency comparison strategies module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from aac_adoption.analysis.recency_comparison import run_recency_comparison, compute_recency_weights


@pytest.fixture
def recency_data_fixture():
    """Create small test dataset spanning 2013-2025."""
    np.random.seed(42)
    
    # Generate years from 2013 to 2025
    years = list(range(2013, 2026))
    rows = []
    animal_id = 1
    
    for year in years:
        # Generate 15 animals per year to ensure enough samples
        for _ in range(15):
            animal_type = np.random.choice(["Dog", "Cat"])
            classification_target = np.random.choice([0, 1], p=[0.5, 0.5])
            regression_target = np.random.exponential(30) + 1
            
            rows.append({
                "animal_id": f"A{animal_id:04d}",
                "animal_type": animal_type,
                "intake_year": year,
                "intake_age_days": np.random.randint(0, 3650),
                "classification_target": classification_target,
                "regression_target_days": round(regression_target, 1),
                "is_named": np.random.choice([0, 1], p=[0.3, 0.7]),
                "intake_condition": np.random.choice(["Normal", "Sick"]),
                "intake_type": np.random.choice(["Stray", "Owner Surrender"]),
                "sex_upon_intake": np.random.choice(["Neutered Male", "Intact Female"]),
                "age_group": np.random.choice(["Adult", "Young Adult"]),
                "simplified_breed_group": np.random.choice(["Mixed", "Other"]),
                "simplified_color_group": np.random.choice(["Brown", "Black"]),
                "found_location_kind": np.random.choice(["Street", "Shelter"]),
                "intake_season": np.random.choice(["Spring", "Summer"]),
                "covid_period": np.random.choice(["Pre", "During", "Post"]),
            })
            animal_id += 1
            
    return pd.DataFrame(rows)


def test_compute_recency_weights():
    """Test dynamic computation of sample weights based on intake_year."""
    df = pd.DataFrame({"intake_year": [2013, 2017, 2021]})
    weights = compute_recency_weights(df, start_year=2013, end_year=2021)
    
    assert len(weights) == 3
    assert weights[0] == 1.0  # (2013 - 2013) / 8 -> weight 1.0
    assert weights[1] == 1.25 # (2017 - 2013) / 8 -> 1.0 + 0.5*4/8 = 1.25
    assert weights[2] == 1.5  # (2021 - 2013) / 8 -> 1.0 + 0.5*8/8 = 1.5


def test_recency_comparison_output_schema(recency_data_fixture):
    """Verify recency comparison outputs expected columns and strategies."""
    df = recency_data_fixture
    
    result = run_recency_comparison(
        df,
        n_bootstraps=3,
        iterations=5,
        test_period="2024-2025",
        quick=True,
    )
    
    assert result is not None
    assert len(result) > 0
    
    # Expected columns
    expected_cols = {
        "strategy", "train_years", "test_years", "subset", "model",
        "pr_auc", "pr_auc_lower", "pr_auc_upper", "roc_auc", "roc_auc_lower", "roc_auc_upper",
        "brier", "ece", "mae", "mae_lower", "mae_upper"
    }
    for col in expected_cols:
        assert col in result.columns
        
    # Expected strategies for combined subset
    combined_results = result[result["subset"] == "combined"]
    strategies = set(combined_results["strategy"].unique())
    assert "full_history" in strategies
    assert "recent_5yr" in strategies
    assert "recent_3yr" in strategies
    assert "recency_weighted" in strategies


def test_recency_comparison_quick_mode_overrides(recency_data_fixture):
    """Test quick mode uses lower default bootstraps/iterations if not specified."""
    df = recency_data_fixture
    
    # In quick mode, default parameters should be overridden internally to be fast
    result = run_recency_comparison(
        df,
        test_period="2024-2025",
        quick=True,
    )
    
    assert len(result) > 0
