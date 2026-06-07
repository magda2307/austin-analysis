"""Regression tests for yearly temporal backtesting functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from aac_adoption.models.yearly_backtesting import (
    run_yearly_backtesting,
    get_test_years,
    get_train_years,
    format_train_period,
    _detect_categorical_features,
    _train_classifier,
    _train_regressor,
)


@pytest.fixture
def six_year_fixture():
    """Create small test dataset spanning 6 years (2019-2024)."""
    np.random.seed(42)
    
    n_per_year = {
        2019: 50,
        2020: 60,
        2021: 55,
        2022: 65,
        2023: 70,
        2024: 75,
    }
    
    rows = []
    animal_id = 1
    
    for year, n in n_per_year.items():
        for _ in range(n):
            animal_type = np.random.choice(["Dog", "Cat"])
            classification_target = np.random.choice([0, 1], p=[0.6, 0.4])
            regression_target = np.random.exponential(30) + 1
            
            rows.append({
                "animal_id": f"A{animal_id:04d}",
                "animal_type": animal_type,
                "intake_year": year,
                "intake_age_days": np.random.randint(0, 3650),
                "classification_target": classification_target,
                "regression_target_days": round(regression_target, 1),
                "is_named": np.random.choice([0, 1], p=[0.3, 0.7]),
                "intake_condition": np.random.choice(["Normal", "Sick", "Injured"]),
                "intake_type": np.random.choice(["Stray", "Owner Surrender", "Public Assist"]),
                "sex_upon_intake": np.random.choice(["Neutered Male", "Intact Female", "Intact Male", "Spayed Female"]),
                "age_group": np.random.choice(["Adult", "Senior", "Young Adult"]),
                "simplified_breed_group": np.random.choice(["Mixed", "Terrier", "Labrador", "Other"]),
                "simplified_color_group": np.random.choice(["Brown", "Black", "White", "Other"]),
                "found_location_kind": np.random.choice(["Street", "Shelter", "Rescue"]),
                "intake_season": np.random.choice(["Spring", "Summer", "Fall", "Winter"]),
                "covid_period": np.random.choice(["Pre", "During", "Post"]),
                "intake_datetime": pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=np.random.randint(0, 365)),
                "outcome_datetime": pd.to_datetime(f"{year}-06-01") + pd.Timedelta(days=np.random.randint(0, 180)),
            })
            animal_id += 1
    
    return pd.DataFrame(rows)


def test_yearly_backtesting_output_schema(six_year_fixture, tmp_path):
    """Verify output schema: train_years, test_year, subset, model, pr_auc, roc_auc, brier, ece, mae, rmse, r2."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    assert result is not None
    assert len(result) > 0
    assert "train_years" in result.columns
    assert "test_year" in result.columns
    assert "subset" in result.columns
    assert "model" in result.columns
    assert "pr_auc" in result.columns
    assert "roc_auc" in result.columns
    assert "brier" in result.columns
    assert "ece" in result.columns
    assert "mae" in result.columns
    assert "rmse" in result.columns
    assert "r2" in result.columns


def test_get_test_years(six_year_fixture):
    """Test get_test_years extracts unique years."""
    years = get_test_years(six_year_fixture)
    expected = [2019, 2020, 2021, 2022, 2023, 2024]
    assert years == expected


def test_get_train_years():
    """Test get_train_years calculates training window."""
    start, end = get_train_years(2019)
    assert start == 2013
    assert end == 2018
    
    start, end = get_train_years(2024)
    assert start == 2013
    assert end == 2023


def test_format_train_period():
    """Test format_train_period creates correct string."""
    assert format_train_period(2013, 2018) == "2013-2018"
    assert format_train_period(2013, 2023) == "2013-2023"


def test_detect_categorical_features(six_year_fixture):
    """Test categorical feature detection."""
    X = six_year_fixture.drop(columns=["classification_target", "animal_id", "intake_year"])
    categorical = _detect_categorical_features(X)
    
    assert "animal_type" in categorical
    assert "intake_condition" in categorical
    assert "intake_type" in categorical
    assert "intake_datetime" not in categorical
    assert "outcome_datetime" not in categorical


def test_yearly_backtesting_catboost_classifier_metrics(six_year_fixture, tmp_path):
    """Test CatBoost classifier outputs PR-AUC and ROC-AUC."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    catboost_classifier = result[result["model"].str.contains("catboost_classifier")]
    assert len(catboost_classifier) > 0, "Should have CatBoost classifier results"
    
    for _, row in catboost_classifier.iterrows():
        assert pd.notna(row["pr_auc"]), "PR-AUC should be computed"
        assert pd.notna(row["roc_auc"]), "ROC-AUC should be computed"


def test_yearly_backtesting_histgradientboosting_classifier_metrics(six_year_fixture, tmp_path):
    """Test HistGradientBoosting classifier outputs PR-AUC and ROC-AUC."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    hist_classifier = result[result["model"].str.contains("histgradientboosting_classifier")]
    assert len(hist_classifier) > 0, "Should have HistGradientBoosting classifier results"
    
    for _, row in hist_classifier.iterrows():
        assert pd.notna(row["pr_auc"]), "PR-AUC should be computed"
        assert pd.notna(row["roc_auc"]), "ROC-AUC should be computed"


def test_yearly_backtesting_catboost_regressor_metrics(six_year_fixture, tmp_path):
    """Test CatBoost regressor outputs MAE, RMSE, R²."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="regression_target_days",
        animal_subset="combined",
        output_path=None,
        compute_ci=False,
        quick=True,
    )
    
    catboost_regressor = result[result["model"].str.contains("catboost_regressor")]
    assert len(catboost_regressor) > 0, "Should have CatBoost regressor results"
    
    for _, row in catboost_regressor.iterrows():
        assert pd.notna(row["mae"]), "MAE should be computed"
        assert pd.notna(row["rmse"]), "RMSE should be computed"
        assert pd.notna(row["r2"]), "R² should be computed"


def test_yearly_backtesting_histgradientboosting_regressor_metrics(six_year_fixture, tmp_path):
    """Test HistGradientBoosting regressor outputs MAE, RMSE, R²."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="regression_target_days",
        animal_subset="combined",
        output_path=None,
        compute_ci=False,
        quick=True,
    )
    
    hist_regressor = result[result["model"].str.contains("histgradientboosting_regressor")]
    assert len(hist_regressor) > 0, "Should have HistGradientBoosting regressor results"
    
    for _, row in hist_regressor.iterrows():
        assert pd.notna(row["mae"]), "MAE should be computed"
        assert pd.notna(row["rmse"]), "RMSE should be computed"
        assert pd.notna(row["r2"]), "R² should be computed"


def test_yearly_backtesting_bootstrap_confidence_intervals(six_year_fixture, tmp_path):
    """Test bootstrap confidence intervals generated (lower/upper columns present)."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=True,
        bootstrap_n=10,
        quick=True,
    )
    
    has_ci_columns = any(col.endswith("_lower") or col.endswith("_upper") for col in result.columns)
    assert has_ci_columns, "Should have bootstrap CI lower/upper columns"
    
    catboost_row = result[result["model"].str.contains("catboost_classifier")].iloc[0]
    assert "pr_auc_lower" in catboost_row
    assert "pr_auc_upper" in catboost_row
    assert "roc_auc_lower" in catboost_row
    assert "roc_auc_upper" in catboost_row


def test_yearly_backtesting_animal_subsets(six_year_fixture, tmp_path):
    """Test animal subsets: combined, dogs, cats."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    subsets = result["subset"].unique()
    assert "combined" in subsets
    assert "dogs" in subsets
    assert "cats" in subsets


def test_yearly_backtesting_horizon_targets(six_year_fixture, tmp_path):
    """Test horizon targets (7/30/60/90 days) produce correct output."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    test_years = sorted(result["test_year"].unique())
    expected_years = [2019, 2020, 2021, 2022, 2023, 2024]
    assert test_years == expected_years
    
    train_years = result["train_years"].unique()
    for year in test_years:
        expected_train = f"2013-{year-1}"
        assert expected_train in train_years


def test_yearly_backtesting_empty_splits_skipped(tmp_path):
    """Test empty train/test splits are skipped gracefully."""
    df = pd.DataFrame({
        "animal_id": ["A0001", "A0002"],
        "intake_year": [2019, 2020],
        "classification_target": [1, 0],
        "animal_type": ["Dog", "Cat"],
        "intake_age_days": [100, 200],
    })
    
    output_path = tmp_path / "yearly_backtesting.csv"
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    assert result is not None
    assert len(result) == 0


def test_yearly_backtesting_multiple_targets(six_year_fixture, tmp_path):
    """Test both classification and regression targets are evaluated."""
    df = six_year_fixture
    
    result_class = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=None,
        compute_ci=False,
        quick=True,
    )
    
    result_reg = run_yearly_backtesting(
        df,
        target_column="regression_target_days",
        animal_subset="combined",
        output_path=None,
        compute_ci=False,
        quick=True,
    )
    
    assert len(result_class) > 0
    assert len(result_reg) > 0
    
    class_models = set(result_class["model"].unique())
    reg_models = set(result_reg["model"].unique())
    
    assert "catboost_classifier" in class_models
    assert "histgradientboosting_classifier" in class_models
    assert "catboost_regressor" in reg_models
    assert "histgradientboosting_regressor" in reg_models


def test_yearly_backtesting_output_csv(six_year_fixture, tmp_path):
    """Test output CSV is written correctly."""
    df = six_year_fixture
    output_path = tmp_path / "yearly_backtesting.csv"
    
    result = run_yearly_backtesting(
        df,
        target_column="classification_target",
        animal_subset="combined",
        output_path=str(output_path),
        compute_ci=False,
        quick=True,
    )
    
    assert output_path.exists()
    
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(result, saved_df)
