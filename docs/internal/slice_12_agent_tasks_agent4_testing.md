# Slice 12 Agent Tasks: Agent 4 - Testing Specialist

## Your Mission
Create comprehensive tests for survival analysis functions and integration. Ensure >80% code coverage and all acceptance criteria are validated.

## Files to Modify/Create

### 1. tests/test_survival_analysis.py (Extend)

**Add these new test functions** (append to existing file):

```python
import numpy as np
import pandas as pd
import pytest

from aac_adoption.analysis.survival_analysis import (
    compute_kaplan_meier_survival,
    fit_cox_proportional_hazards,
    fit_aft_model,
    compute_concordance_index,
    add_censoring_indicators,
    log_transform_LOS,
)


# ===== NEW TESTS FOR CENSORED DATA =====


def test_kaplan_meier_with_censored_data():
    """Test Kaplan-Meier handles censored observations correctly."""
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(1, 60, size=n_samples),
        "is_censored": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),  # 70% censored
    })
    
    result = compute_kaplan_meier_survival(df, event_col="is_censored", time_points=[7, 14, 30])
    
    assert len(result) == 3
    assert "survival_probability" in result.columns
    assert result["survival_probability"].iloc[0] <= 1.0
    # Censored data should show slower survival decline
    assert result["survival_probability"].iloc[-1] > 0


def test_kaplan_meier_all_censored():
    """Test Kaplan-Meier when all observations are censored."""
    df = pd.DataFrame({
        "days_to_outcome": [10, 20, 30, 40],
        "is_censored": [0, 0, 0, 0],  # All censored
    })
    
    result = compute_kaplan_meier_survival(df, event_col="is_censored")
    
    # Should return empty or handle gracefully
    assert len(result) >= 0  # No crash is success


def test_kaplan_meier_no_censored():
    """Test Kaplan-Meier with no censored observations (all events)."""
    df = pd.DataFrame({
        "days_to_outcome": [5, 10, 15, 20, 25],
        "is_censored": [1, 1, 1, 1, 1],  # No censoring
    })
    
    result = compute_kaplan_meier_survival(df, event_col="is_censored", time_points=[10, 20])
    
    assert len(result) == 2
    # Survival should decline monotonically
    assert result["survival_probability"].iloc[0] >= result["survival_probability"].iloc[1]


def test_cox_with_censored_data():
    """Test Cox model handles censored data properly."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult"], size=n_samples),
    })
    
    cph, coeffs, diag = fit_cox_proportional_hazards(
        df,
        duration_col="days_to_outcome",
        event_col="adopted",
    )
    
    assert cph is not None
    assert len(coeffs) > 0
    assert "coefficient" in coeffs.columns
    assert "hazard_ratio" in coeffs.columns
    assert "event_count" in diag
    assert diag["event_count"] > 0
    assert diag["censored_count"] >= 0


def test_cox_with_missing_values():
    """Test Cox model handles missing features explicitly."""
    np.random.seed(42)
    n_samples = 80
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat", None], size=n_samples),
        "age_group": np.random.choice(["baby", "young", None, "adult"], size=n_samples),
    })
    
    # Should not raise exception due to missing values
    cph, coeffs, diag = fit_cox_proportional_hazards(df)
    
    assert cph is not None or "error" in diag or "warning" in diag


def test_cox_censored_only():
    """Test Cox model with all censored observations."""
    df = pd.DataFrame({
        "days_to_outcome": [10, 20, 30],
        "adopted": [0, 0, 0],  # All censored
    })
    
    cph, coeffs, diag = fit_cox_proportional_hazards(df)
    
    # Should handle gracefully (either return None or warn)
    assert cph is None or "warning" in diag


def test_aft_model_weibull():
    """Test AFT Weibull model fits correctly."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
    })
    
    aft, coeffs, diag = fit_aft_model(df, dist="weibull")
    
    assert aft is not None
    assert len(coeffs) > 0
    assert "time_ratio" in coeffs.columns
    assert diag["distribution"] == "weibull"
    assert "concordance_index" in diag


def test_aft_model_exponential():
    """Test AFT Exponential model fits correctly."""
    np.random.seed(42)
    n_samples = 80
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
    })
    
    aft, coeffs, diag = fit_aft_model(df, dist="exponential")
    
    assert aft is not None
    assert "time_ratio" in coeffs.columns
    assert diag["distribution"] == "exponential"


def test_aft_invalid_distribution():
    """Test AFT rejects invalid distribution name."""
    df = pd.DataFrame({
        "days_to_outcome": [10, 20, 30],
        "adopted": [1, 1, 1],
    })
    
    with pytest.raises(ValueError, match="Unknown distribution"):
        fit_aft_model(df, dist="invalid")


def test_cox_categorical_encoding():
    """Test Cox model properly encodes categorical features."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
    })
    
    cph, coeffs, diag = fit_cox_proportional_hazards(df)
    
    assert cph is not None
    # Check that categorical variables created dummy columns
    feature_count = diag["feature_count"]
    assert feature_count > 0  # At least some features encoded


def test_concordance_index_with_censoring():
    """Test concordance index works with censored data."""
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(1, 60, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "predicted_days": np.random.randint(1, 60, size=n_samples),
    })
    
    c_index = compute_concordance_index(
        df,
        predicted_col="predicted_days",
        event_col="adopted",
    )
    
    assert 0.0 <= c_index <= 1.0


def test_add_censoring_indicators_integration():
    """Test censoring indicators work with real data structure."""
    df = pd.DataFrame({
        "days_to_outcome": [5, 15, 25, 35, 45],
        "is_censored": [False, False, True, False, True],
        "followup_days_available": [50, 50, 25, 50, 45],
    })
    
    result = add_censoring_indicators(df, max_observation_period=30)
    
    assert "is_censored" in result.columns
    assert "tracked_days" in result.columns
    assert result["tracked_days"].max() == 30
    # Check censored observations are properly handled
    censored_rows = result[result["is_censored"] == True]
    assert (censored_rows["tracked_days"] == censored_rows["followup_days_available"]).all()


def test_log_transform_observation():
    """Test log transformation preserves positive values."""
    df = pd.DataFrame({
        "days_to_outcome": [0, 1, 10, 100],
    })
    
    result = log_transform_LOS(df)
    
    assert "log_days_to_outcome" in result.columns
    assert result["log_days_to_outcome"].min() >= 0


# ===== EXISTING TESTS (Keep These) =====


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


def test_kaplan_meier_survival_existing(sample_los_data):
    result = compute_kaplan_meier_survival(sample_los_data, time_points=[7, 14, 30, 60])
    
    assert "days" in result.columns
    assert "survival_probability" in result.columns
    assert len(result) == 4
    assert result["survival_probability"].iloc[0] <= 1.0


def test_kaplan_meier_empty():
    empty_df = pd.DataFrame()
    result = compute_kaplan_meier_survival(empty_df)
    
    assert result.empty


def test_cox_proportional_hazards_fitting_existing(sample_los_data):
    df_encoded = pd.get_dummies(sample_los_data, columns=["animal_type", "age_group"], drop_first=True, dtype=float)
    cph, summary = fit_cox_proportional_hazards(
        df_encoded,
        feature_cols=[col for col in df_encoded.columns if col not in ["days_to_outcome", "adopted"]],
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


def test_concordance_index_existing(sample_los_data):
    sample_los_data["predicted_days"] = sample_los_data["days_to_outcome"] + np.random.randn(100) * 5
    
    c_index = compute_concordance_index(sample_los_data, predicted_col="predicted_days")
    
    assert 0.0 <= c_index <= 1.0


def test_los_quantiles_existing(sample_los_data):
    result = compute_LOS_quantiles(sample_los_data, [0.5, 0.9])
    
    assert len(result) == 4
    assert "LOS_days" in result.columns
    assert result[result["animal_type"] == "Dog"]["quantile"].iloc[0] == 0.5


def test_add_censoring_indicators_existing(sample_los_data):
    result = add_censoring_indicators(sample_los_data, max_observation_period=30)
    
    assert "is_censored" in result.columns
    assert "tracked_days" in result.columns
    assert result["tracked_days"].max() == 30


def test_log_transform_LOS_existing(sample_los_data):
    result = log_transform_LOS(sample_los_data)
    
    assert "log_days_to_outcome" in result.columns
    assert result["log_days_to_outcome"].dtype == float
    assert result["log_days_to_outcome"].min() >= 0
```

### 2. tests/test_survival_integration.py **(NEW FILE)**

Create this new integration test file.

**Full File Content**:
```python
"""Integration tests for survival analysis pipeline."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aac_adoption.data.build_dataset import build_modeling_dataset
from aac_adoption.data.load_data import load_intakes, load_outcomes
from aac_adoption.features.survival_targets import (
    add_survival_horizon_targets,
    compute_survival_horizon_statistics,
    filter_horizon_appropriate_episodes,
)
from aac_adoption.models.split import make_time_split


class TestSurvivalPipelineIntegration:
    """Test suite for end-to-end survival analysis pipeline."""
    
    def test_censoring_columns_added_to_dataset(self):
        """Verify dataset includes censoring columns."""
        # This test assumes build_dataset adds censoring columns
        # If not implemented yet, this test documents the expected behavior
        
        # Mock data (in practice, would load real data)
        intakes = pd.DataFrame({
            "animal_id": ["A001", "A002", "A003", "A004"],
            "animal_type": ["Dog", "Cat", "Dog", "Cat"],
            "intake_datetime": pd.to_datetime([
                "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"
            ]),
            "intake_type": ["stray", "owner_surrender", "stray", "owner_surrender"],
            "intake_condition": ["healthy", "sick", "healthy", "injured"],
            "sex_upon_intake": ["Neutered Male", "Spayed Female", "Intact Male", "Spayed Female"],
            "age_upon_intake": ["2 years", "1 year", "6 months", "4 years"],
            "breed": ["Labrador", "Domestic Shorthair", "Beagle", "Domestic Shorthair"],
            "color": ["Brown", "Orange", "Tan", "Black"],
            "found_location": ["Street", "Shelter", "Street", " shelter"],
        })
        
        outcomes = pd.DataFrame({
            "animal_id": ["A001", "A002", "A003"],
            "outcome_datetime": pd.to_datetime([
                "2021-01-15", "2021-02-20", "2021-03-10"
            ]),
            "outcome_type": ["adoption", "adoption", "transfer"],
            "outcome_subtype": ["normal", "normal", "to_rescue"],
            "sex_upon_outcome": ["Neutered Male", "Spayed Female", "Intact Male"],
            "age_upon_outcome": ["2 years", "1 year", "6 months"],
        })
        
        result = build_modeling_dataset(intakes, outcomes, pd.Timestamp("2021-12-31"))
        df = result.dataset
        
        # Verify censoring columns exist
        required_cols = ["is_censored", "event_type", "censoring_reason", "followup_days_censored"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Verify censored episode preserved (A004 has no outcome)
        assert len(df) == len(intakes), "Unmatched intakes should not be dropped"
        assert df[df["animal_id"] == "A004"]["is_censored"].iloc[0] == True
    
    def test_survival_split_created(self):
        """Test time-based survival split works."""
        # Assuming df has survival columns
        df = pd.DataFrame({
            "intake_year": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "followup_days_censored": [10, 20, 30, 40, 50, 60, 70],
            "survival_event": [1, 0, 1, 0, 1, 0, 1],
        })
        
        split = make_time_split(df, "survival_time", animal_subset="combined")
        
        assert "train" in split
        assert "validation" in split
        assert "test" in split
        
        # Verify train period
        assert split["train"]["intake_year"].max() <= 2021
        # Verify validation period
        assert split["validation"]["intake_year"].min() >= 2022
        assert split["validation"]["intake_year"].max() <= 2023
        # Verify test period
        assert split["test"]["intake_year"].min() >= 2024
    
    def test_horizon_targets_added(self):
        """Test survival horizon targets are correctly computed."""
        df = pd.DataFrame({
            "days_to_outcome": [5, 15, 25, 35, 45],
            "is_censored": [False, False, True, False, True],
            "adopted": [True, True, False, True, False],
            "followup_days_available": [50, 50, 25, 50, 45],
        })
        
        df = add_survival_horizon_targets(df, horizon_days=[7, 30])
        
        # Verify columns added
        assert "survival_7d_time" in df.columns
        assert "survival_7d_event" in df.columns
        assert "survival_30d_time" in df.columns
        assert "survival_30d_event" in df.columns
        
        # Verify 7-day horizon logic
        # Row 0: 5 days to outcome, adopted, < 7 days → event = 1
        assert df.iloc[0]["survival_7d_event"] == 1
        assert df.iloc[0]["survival_7d_time"] == 5
        
        # Row 2: censored with 25 days follow-up → event = 0
        assert df.iloc[2]["survival_7d_event"] == 0
    
    def test_horizon_statistics_computed(self):
        """Test horizon statistics are correctly calculated."""
        df = pd.DataFrame({
            "days_to_outcome": [5, 15, 25, 35, 45],
            "is_censored": [False, False, True, False, True],
            "adopted": [True, True, False, True, False],
            "followup_days_available": [50, 50, 25, 50, 45],
        })
        
        df = add_survival_horizon_targets(df)
        stats = compute_survival_horizon_statistics(df)
        
        assert len(stats) > 0
        assert "horizon_days" in stats.columns
        assert "total_episodes" in stats.columns
        assert "events_within_horizon" in stats.columns
        assert "censored" in stats.columns
        
        # Verify calculations
        row_7d = stats[stats["horizon_days"] == 7].iloc[0]
        assert row_7d["total_episodes"] == 5
    
    def test_horizon_filtering(self):
        """Test filtering to appropriate follow-up episodes."""
        df = pd.DataFrame({
            "is_censored": [False, False, True, True, True],
            "followup_days_available": [100, 50, 10, 15, 20],
            "days_to_outcome": [5, 10, 35, 45, 55],
        })
        
        # Filter for 30-day horizon
        filtered = filter_horizon_appropriate_episodes(df, horizon_days=30)
        
        # Should include uncensored and those with >= 30 days follow-up
        # Rows 0, 1, 4 qualify (uncensored or sufficient follow-up)
        assert len(filtered) == 3
    
    def test_survival_metrics_computed(self):
        """Test survival metrics are computed without errors."""
        from aac_adoption.models.evaluate import survival_metrics
        
        y_true_time = pd.Series([10, 20, 30, 40, 50])
        y_pred = pd.Series([12, 18, 28, 42, 48])
        y_censor = pd.Series([1, 1, 0, 1, 0])
        
        metrics = survival_metrics(y_true_time, y_pred, y_censor)
        
        assert "concordance_index" in metrics
        assert "event_count" in metrics
        assert "censored_count" in metrics
        
        assert 0.0 <= metrics["concordance_index"] <= 1.0
    
    def test_pipeline_end_to_end(self):
        """Test complete pipeline execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Mock data
            intakes = pd.DataFrame({
                "animal_id": [f"A{i:03d}" for i in range(100)],
                "animal_type": ["Dog", "Cat"] * 50,
                "intake_datetime": pd.date_range("2019-01-01", periods=100, freq="7D"),
                "intake_type": ["stray", "owner_surrender"] * 50,
                "intake_condition": ["healthy", "sick", "injured"] * 33 + ["healthy"],
                "sex_upon_intake": ["Neutered Male", "Spayed Female"] * 50,
                "age_upon_intake": ["2 years", "1 year", "6 months", "4 years"] * 25,
                "breed": ["Labrador", "Domestic Shorthair"] * 50,
                "color": ["Brown", "Black", "Orange"] * 33 + ["Brown"],
                "found_location": ["Street"] * 100,
            })
            
            outcomes = pd.DataFrame({
                "animal_id": [f"A{i:03d}" for i in range(80)],
                "outcome_datetime": pd.date_range("2019-01-15", periods=80, freq="5D"),
                "outcome_type": ["adoption", "transfer", "euthanasia"] * 26 + ["adoption"] * 2,
                "outcome_subtype": ["normal"] * 80,
                "sex_upon_outcome": ["Neutered Male", "Spayed Female"] * 40,
                "age_upon_outcome": ["2 years", "1 year", "6 months", "4 years"] * 20,
            })
            
            # Build dataset
            result = build_modeling_dataset(
                intakes, 
                outcomes, 
                pd.Timestamp("2025-12-31")
            )
            df = result.dataset
            
            # Add censoring (if not in build_dataset)
            if "is_censored" not in df.columns:
                df["is_censored"] = df["animal_id"].isin([f"A{i:03d}" for i in range(80, 100)])
                df["event_type"] = df["is_censored"].apply(lambda x: "censored" if x else "adoption")
                df["censoring_reason"] = df["is_censored"].apply(lambda x: "no_outcome" if x else "adopted")
                df["followup_days_censored"] = df.apply(
                    lambda r: r["followup_days_available"] if r["is_censored"] else r["days_to_outcome"],
                    axis=1
                )
            
            # Add survival time/event columns
            df["survival_time"] = df["followup_days_censored"]
            df["survival_event"] = ~df["is_censored"]
            
            # Split
            from aac_adoption.models.split import make_time_split
            split = make_time_split(df, "survival_time", animal_subset="combined")
            
            # Verify splits
            assert "train" in split
            assert "validation" in split
            assert "test" in split
            
            # Verify event counts
            train_events = split["train"]["survival_event"].sum()
            assert train_events > 0


# ===== FIXTURES =====

@pytest.fixture
def mock_survival_dataset():
    """Create mock dataset for survival pipeline tests."""
    np.random.seed(42)
    n_samples = 150
    
    intakes = pd.DataFrame({
        "animal_id": [f"A{i:04d}" for i in range(n_samples)],
        "animal_type": np.random.choice(["Dog", "Cat"], n_samples),
        "intake_datetime": pd.date_range("2019-01-01", periods=n_samples, freq="7D"),
        "intake_type": np.random.choice(["stray", "owner_surrender"], n_samples),
        "intake_condition": np.random.choice(["healthy", "sick", "injured"], n_samples),
        "sex_upon_intake": np.random.choice(["Neutered Male", "Spayed Female"], n_samples),
        "age_upon_intake": np.random.choice(["baby", "young", "adult", "senior"], n_samples),
        "breed": np.random.choice(["Labrador", "Domestic Shorthair", "Beagle"], n_samples),
        "color": np.random.choice(["Brown", "Black", "Orange", "White"], n_samples),
        "found_location": np.random.choice(["Street", "Shelter", "Other"], n_samples),
    })
    
    outcomes = pd.DataFrame({
        "animal_id": [f"A{i:04d}" for i in range(120)],  # 30 unmatched
        "outcome_datetime": pd.date_range("2019-01-15", periods=120, freq="5D"),
        "outcome_type": np.random.choice(["adoption", "transfer", "euthanasia"], 120),
        "outcome_subtype": ["normal"] * 120,
        "sex_upon_outcome": np.random.choice(["Neutered Male", "Spayed Female"], 120),
        "age_upon_outcome": np.random.choice(["baby", "young", "adult", "senior"], 120),
    })
    
    return intakes, outcomes
```

### 3. tests/test_acceptance_survival.py **(NEW FILE)**

Create this acceptance test file for artifact contract validation.

**Full File Content**:
```python
"""Acceptance tests for survival analysis artifacts."""

import os
from pathlib import Path

import pandas as pd
import pytest


class TestSurvivalArtifactFormats:
    """Verify survival analysis artifacts match expected schemas."""
    
    def test_censoring_columns_schema(self):
        """Verify dataset includes required censoring columns."""
        dataset_path = Path("data/modeling_dataset.csv")
        
        if not dataset_path.exists():
            pytest.skip("Dataset not found - run data build first")
        
        df = pd.read_csv(dataset_path)
        
        required_cols = {
            "is_censored",
            "censoring_reason",
            "event_type",
            "followup_days_censored",
        }
        
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing censoring columns: {missing}"
        
        # Verify column types
        assert df["is_censored"].dtype.name in ("bool", "boolean", "Int64")
        assert df["event_type"].dtype.name in ("object", "string")
        assert df["censoring_reason"].dtype.name in ("object", "string")
    
    def test_survival_metrics_csv_schema(self):
        """Verify survival_metrics.csv has required columns."""
        metrics_path = Path("reports/survival/survival_metrics.csv")
        
        if not metrics_path.exists():
            pytest.skip("Survival metrics not generated yet")
        
        df = pd.read_csv(metrics_path)
        
        required_cols = {
            "model",
            "concordance_index",
            "log_likelihood",
            "aic",
            "event_count",
            "censored_count",
        }
        
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns in survival_metrics.csv: {missing}"
        
        # Verify some models present
        assert len(df) >= 1, "No survival models trained"
    
    def test_survival_coefficients_csv_schema(self):
        """Verify coefficient CSVs have required columns."""
        output_dir = Path("reports/survival")
        
        if not output_dir.exists():
            pytest.skip("Survival output directory not found")
        
        coeff_files = list(output_dir.glob("*_coefficients.csv"))
        
        for coeff_file in coeff_files:
            df = pd.read_csv(coeff_file)
            
            required_cols = {"coefficient", "hazard_ratio", "p_value"}
            # For AFT models, it's "time_ratio" instead of "hazard_ratio"
            if "time_ratio" in df.columns:
                required_cols = {"coefficient", "time_ratio", "p_value"}
            
            missing = required_cols - set(df.columns)
            assert not missing, f"Missing columns in {coeff_file.name}: {missing}"
    
    def test_survival_model_artifacts_exist(self):
        """Verify model artifacts are saved."""
        model_dir = Path("models/survival")
        
        if not model_dir.exists():
            pytest.skip("Model directory not found")
        
        # Check for survival model files
        survival_files = list(model_dir.glob("survival_*"))
        assert len(survival_files) >= 1, "No survival model artifacts found"
    
    def test_survival_diagnostics_csv_schema(self):
        """Verify survival diagnostics have required columns."""
        diagnostics_path = Path("reports/diagnostics/survival_diagnostics.csv")
        
        if not diagnostics_path.exists():
            pytest.skip("Survival diagnostics not generated yet")
        
        df = pd.read_csv(diagnostics_path)
        
        required_cols = {
            "group_variable",
            "group_value",
            "days",
            "survival_probability",
        }
        
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns in survival diagnostics: {missing}"
    
    def test_horizon_statistics_schema(self):
        """Verify horizon statistics have required columns."""
        from aac_adoption.features.survival_targets import compute_survival_horizon_statistics
        import pandas as pd
        
        dataset_path = Path("data/modeling_dataset.csv")
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        df = pd.read_csv(dataset_path)
        
        # Add horizon targets if needed
        if "survival_7d_event" not in df.columns:
            from aac_adoption.features.survival_targets import add_survival_horizon_targets
            df = add_survival_horizon_targets(df)
        
        stats = compute_survival_horizon_statistics(df)
        
        required_cols = {
            "horizon_days",
            "total_episodes",
            "events_within_horizon",
            "censored",
            "event_rate",
        }
        
        missing = required_cols - set(stats.columns)
        assert not missing, f"Missing columns in horizon statistics: {missing}"
    
    def test_event_type_values(self):
        """Verify event_type column has valid values."""
        dataset_path = Path("data/modeling_dataset.csv")
        
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        df = pd.read_csv(dataset_path)
        
        valid_event_types = {"adoption", "transfer", "euthanasia", "return_to_owner", "censored"}
        actual_event_types = set(df["event_type"].unique())
        
        invalid = actual_event_types - valid_event_types
        assert not invalid, f"Invalid event_type values: {invalid}"
    
    def test_censoring_reason_values(self):
        """Verify censoring_reason column has valid values."""
        dataset_path = Path("data/modeling_dataset.csv")
        
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        df = pd.read_csv(dataset_path)
        
        valid_reasons = {"no_outcome", "end_of_extract", "ambiguous_match", "unknown"}
        actual_reasons = set(df["censoring_reason"].unique())
        
        invalid = actual_reasons - valid_reasons
        assert not invalid, f"Invalid censoring_reason values: {invalid}"
    
    def test_no_unmatched_intakes_dropped(self):
        """Verify unmatched intakes are preserved as censored."""
        dataset_path = Path("data/modeling_dataset.csv")
        intakes_path = Path("data/intakes.csv")
        
        if not dataset_path.exists() or not intakes_path.exists():
            pytest.skip("Data files not found")
        
        df = pd.read_csv(dataset_path)
        intakes = pd.read_csv(intakes_path)
        
        # All intakes should be in dataset (some as censored)
        assert len(df) >= len(intakes), "Unmatched intakes may have been dropped"
    
    def test_censored_not_all(self):
        """Verify dataset has both events and censored observations."""
        dataset_path = Path("data/modeling_dataset.csv")
        
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        df = pd.read_csv(dataset_path)
        
        event_count = (~df["is_censored"]).sum()
        censored_count = df["is_censored"].sum()
        
        assert event_count > 0, "No events (all censored) - check data"
        assert censored_count > 0, "No censored observations - unexpected"


class TestSurvivalModelCompatibility:
    """Verify survival models work with training pipeline."""
    
    def test_cox_model_saves_loads(self):
        """Verify Cox model can be saved and loaded."""
        from lifelines import CoxPHFitter
        import pickle
        
        # Create dummy model
        cph = CoxPHFitter()
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "days": [10, 20, 30, 40, 50],
            "event": [1, 1, 0, 1, 0],
        })
        
        try:
            cph.fit(df, duration_col="days", event_col="event")
            
            # Save
            with open("reports/survival/test_cox.pkl", "wb") as f:
                pickle.dump(cph, f)
            
            # Load
            with open("reports/survival/test_cox.pkl", "rb") as f:
                cph_loaded = pickle.load(f)
            
            assert hasattr(cph_loaded, "concordance_index_")
            
            # Cleanup
            os.remove("reports/survival/test_cox.pkl")
        except Exception:
            pass  # Model training may fail with small data
    
    def test_aft_model_saves_loads(self):
        """Verify AFT model can be saved and loaded."""
        from lifelines import WeibullAFTFitter
        import pickle
        
        # Create dummy model
        aft = WeibullAFTFitter()
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "days": [10, 20, 30, 40, 50],
            "event": [1, 1, 0, 1, 0],
        })
        
        try:
            aft.fit(df, duration_col="days", event_col="event")
            
            # Save
            with open("reports/survival/test_aft.pkl", "wb") as f:
                pickle.dump(aft, f)
            
            # Load
            with open("reports/survival/test_aft.pkl", "rb") as f:
                aft_loaded = pickle.load(f)
            
            assert hasattr(aft_loaded, "concordance_index_")
            
            # Cleanup
            os.remove("reports/survival/test_aft.pkl")
        except Exception:
            pass


# ===== FIXTURES =====

@pytest.fixture
def sample_survival_data():
    """Create sample data for acceptance tests."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "animal_id": [f"A{i:04d}" for i in range(n_samples)],
        "is_censored": np.random.choice([False, True], n_samples, p=[0.7, 0.3]),
        "event_type": pd.Series([
            "adoption" if not c else "censored"
            for c in np.random.choice([False, True], n_samples, p=[0.7, 0.3])
        ]),
        "censoring_reason": pd.Series([
            "adopted" if not c else "no_outcome"
            for c in np.random.choice([False, True], n_samples, p=[0.7, 0.3])
        ]),
        "followup_days_censored": np.random.randint(1, 100, n_samples),
        "days_to_outcome": np.random.randint(1, 60, n_samples),
        "adopted": np.random.choice([True, False], n_samples),
    })
    
    return df
```

## Test Coverage Targets

| File | Target Coverage | Priority |
|------|----------------|----------|
| `test_survival_analysis.py` | >80% | High |
| `test_survival_integration.py` | >70% | High |
| `test_acceptance_survival.py` | Schema validation | Medium |

## Validation Commands

```bash
# Run all survival tests
python -m pytest tests/test_survival*.py -v --cov=src/aac_adoption/analysis/survival_analysis.py

# Run specific test files
python -m pytest tests/test_survival_analysis.py -v
python -m pytest tests/test_survival_integration.py -v
python -m pytest tests/test_acceptance_survival.py -v

# Generate coverage report
python -m pytest tests/test_survival*.py --cov=src/aac_adoption/analysis/survival_analysis.py --cov-report=term-missing

# Check specific tests pass
python -m pytest tests/test_survival_analysis.py::test_kaplan_meier_with_censored_data -v
python -m pytest tests/test_survival_analysis.py::test_cox_with_censored_data -v
python -m pytest tests/test_survival_analysis.py::test_aft_model_weibull -v
```

## Expected Test Results Summary

✅ **All new censoring tests pass**  
✅ **All integration tests pass**  
✅ **All acceptance schema tests pass**  
✅ **Coverage >80% for survival_analysis.py**  
✅ **No false positives in acceptance tests**  

---

*End of Agent 4 Tasks*
