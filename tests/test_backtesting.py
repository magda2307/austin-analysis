import pytest
import pandas as pd


def test_backtesting_output_schema():
    """Test that yearly backtesting produces correct output schema with sufficient data."""
    df = pd.DataFrame({
        "animal_id": list(range(30)),
        "intake_year": [2018]*10 + [2019]*10 + [2020]*10,
        "classification_target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3,
        "regression_target_days": [10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0] * 3,
        "animal_type": ["Dog", "Cat"] * 15,
        "intake_age_days": [100, 200] * 15,
        "intake_datetime": pd.to_datetime(["2018-01-01", "2018-02-01"] * 15),
        "outcome_datetime": pd.to_datetime(["2018-01-05", "2018-02-05"] * 15),
    })
    
    from aac_adoption.models.yearly_backtesting import run_yearly_backtesting, _detect_categorical_features
    
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
    
    X_train = df.drop(columns=["classification_target", "animal_id", "intake_year"])
    categorical_features = _detect_categorical_features(X_train)
    assert "intake_datetime" not in categorical_features
    assert "outcome_datetime" not in categorical_features
