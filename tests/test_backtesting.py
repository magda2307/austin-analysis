import pytest
import pandas as pd


def test_backtesting_output_schema():
    """Test that yearly backtesting produces correct output schema with sufficient data."""
    df = pd.DataFrame({
        "animal_id": list(range(12)),
        "intake_year": [2018]*4 + [2019]*4 + [2020]*4,
        "classification_target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "regression_target_days": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
        "animal_type": ["Dog", "Cat", "Dog", "Cat"] * 3,
        "intake_age_days": [100, 200, 100, 200] * 3,
    })
    
    from aac_adoption.models.yearly_backtesting import run_yearly_backtesting
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=None,
        compute_ci=False,
    )
    
    assert result is not None
    assert len(result) > 0
    assert "train_years" in result.columns
    assert "test_year" in result.columns
    assert "subset" in result.columns
    assert "model" in result.columns
    assert "pr_auc" in result.columns
    assert "roc_auc" in result.columns
