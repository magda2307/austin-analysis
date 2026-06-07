"""Comprehensive tests for survival analysis - simplified version.

Tests the main survival analysis functions with synthetic data.
"""

import pandas as pd
import numpy as np
import pytest

from aac_adoption.analysis.survival_analysis import (
    compute_kaplan_meier_survival,
    fit_cox_with_censoring,
    compute_concordance_index,
    compute_LOS_quantiles,
    add_censoring_indicators,
    log_transform_LOS,
    validate_proportional_hazards,
    encode_categorical_features,
    compute_subDistribution_hazard,
    compute_cumulative_hazard,
    test_logrank_difference as logrank_test_difference,
)


@pytest.fixture
def survival_data_with_all_features():
    """Generate comprehensive synthetic survival data."""
    np.random.seed(42)
    n_samples = 150
    
    df = pd.DataFrame({
        "animal_id": [f"A{i:03d}" for i in range(n_samples)],
        "days_to_outcome": np.random.randint(1, 90, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender"], size=n_samples),
    })
    
    return df


@pytest.fixture  
def survival_data_encoded(survival_data_with_all_features):
    """Generate encoded version of survival data."""
    result, _ = encode_categorical_features(
        survival_data_with_all_features,
        categorical_cols=["animal_type", "age_group"],
        drop_first=True,
        return_encoders=False
    )
    return result


class TestCensoringColumnCreation:
    """Tests for censoring column creation and validation."""
    
    def test_add_censoring_indicators_creates_expected_columns(self):
        df = pd.DataFrame({
            "days_to_outcome": [5, 15, 35, 50, 100],
            "adopted": [1, 1, 0, 0, 0]
        })
        
        result = add_censoring_indicators(df, max_observation_period=30)
        
        assert "is_censored" in result.columns
        assert "followup_days_censored" in result.columns
        assert "event_observed" in result.columns
        assert "event_observed_native" in result.columns
    
    def test_add_censoring_indicators_all_censored(self):
        df = pd.DataFrame({
            "days_to_outcome": [100, 150, 200],
            "adopted": [0, 0, 0]
        })
        
        result = add_censoring_indicators(df, max_observation_period=30)
        
        assert result["is_censored"].all()
        assert result["event_observed"].sum() == 0
        assert result["followup_days_censored"].max() == 30
    
    def test_add_censoring_indicators_no_censoring(self):
        df = pd.DataFrame({
            "days_to_outcome": [5, 10, 15],
            "adopted": [1, 1, 1]
        })
        
        result = add_censoring_indicators(df, max_observation_period=100)
        
        assert not result["is_censored"].any()
        assert result["event_observed"].sum() == 3
    
    def test_add_censoring_indicators_mixed_censoring(self, survival_data_with_all_features):
        result = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        
        n_censored = result["is_censored"].sum()
        n_events = result["event_observed"].sum()
        
        assert n_censored + n_events == len(result)
        assert n_censored > 0
        assert n_events > 0


class TestKaplanMeierEstimation:
    """Tests for Kaplan-Meier estimation."""
    
    def test_kaplan_meier_basic_survival(self, survival_data_with_all_features):
        result = compute_kaplan_meier_survival(
            survival_data_with_all_features, 
            time_points=[7, 14, 30, 60]
        )
        
        assert "days" in result.columns
        assert "survival_probability" in result.columns
        assert len(result) == 4
        assert (result["survival_probability"] >= 0).all()
        assert (result["survival_probability"] <= 1.0).all()
    
    def test_kaplan_meier_with_censoring_column(self, survival_data_with_all_features):
        df_with_censoring = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        
        result = compute_kaplan_meier_survival(
            df_with_censoring,
            censoring_col="event_observed",
            time_points=[7, 14, 30]
        )
        
        assert "days" in result.columns
        assert "survival_probability" in result.columns
        assert len(result) == 3
    
    def test_kaplan_meier_grouped_curves(self, survival_data_with_all_features):
        result = compute_kaplan_meier_survival(
            survival_data_with_all_features,
            time_points=[7, 14, 30],
            group_by="animal_type"
        )
        
        assert "animal_type" in result.columns
        assert len(result["animal_type"].unique()) == 2
        assert len(result) >= 6
    
    def test_kaplan_meier_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = compute_kaplan_meier_survival(empty_df)
        
        assert result.empty


class TestCoxProportionalHazards:
    """Tests for Cox Proportional Hazards model."""
    
    def test_cox_basic_fitting(self, survival_data_encoded):
        cph, summary = fit_cox_with_censoring(
            survival_data_encoded,
            feature_cols=["animal_type_encoded_1", "age_group_encoded_1", "age_group_encoded_2", "age_group_encoded_3"]
        )
        
        assert cph is not None
        assert len(summary) > 0
        assert "coefficient" in summary.columns
        assert "hazard_ratio" in summary.columns
    
    def test_cox_empty_dataframe(self):
        empty_df = pd.DataFrame()
        cph, summary = fit_cox_with_censoring(empty_df)
        
        assert cph is None
        assert summary.empty
    
    def test_cox_with_strata(self, survival_data_encoded):
        cph, summary = fit_cox_with_censoring(
            survival_data_encoded,
            strata=["animal_type"]
        )
        
        assert cph is not None
        assert len(summary) > 0
    
    def test_cox_all_events(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["adopted"] = 1
        
        result, _ = encode_categorical_features(df, drop_first=True, return_encoders=False)
        
        cph, summary = fit_cox_with_censoring(
            result,
            feature_cols=["animal_type_encoded_1", "age_group_encoded_1", "age_group_encoded_2", "age_group_encoded_3"]
        )
        
        assert cph is not None
        assert len(summary) > 0
    
    def test_cox_no_events(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["adopted"] = 0
        
        result, _ = encode_categorical_features(df, drop_first=True, return_encoders=False)
        
        cph, summary = fit_cox_with_censoring(
            result,
            feature_cols=["animal_type_encoded_1", "age_group_encoded_1", "age_group_encoded_2", "age_group_encoded_3"]
        )
        
        assert cph is not None
        assert len(summary) > 0


class TestProportionalHazardsAssumption:
    """Tests for proportional hazards assumption validation."""
    
    def test_ph_validation_basic(self, survival_data_encoded):
        cph, _ = fit_cox_with_censoring(survival_data_encoded)
        
        if cph is not None:
            result = validate_proportional_hazards(cph)
            
            assert "variable" in result.columns
            assert "chi_square_statistic" in result.columns
            assert "ph_test_p_value" in result.columns
            assert "ph_assumption_passes" in result.columns
    
    def test_ph_validation_none_model(self):
        result = validate_proportional_hazards(None)
        
        assert result.empty


class TestCompetingRisksAnalysis:
    """Tests for competing risks analysis."""
    
    def test_subdistribution_hazard_basic(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["event_type"] = np.where(
            df["adopted"],
            "adoption",
            "censored"
        )
        
        result = compute_subDistribution_hazard(
            df,
            event_col="adopted",
            competing_events_col="event_type"
        )
        
        assert result is not None
        assert "coefficient" in result.columns
        assert "subdistribution_hazard_ratio" in result.columns
    
    def test_subdistribution_hazard_no_competing_col(self, survival_data_with_all_features):
        result = compute_subDistribution_hazard(
            survival_data_with_all_features,
            event_col="adopted"
        )
        
        assert result is None
    
    def test_subdistribution_hazard_empty_data(self):
        df = pd.DataFrame()
        result = compute_subDistribution_hazard(df)
        
        assert result is None


class TestConcordanceIndex:
    """Tests for concordance index computation."""
    
    def test_concordance_index_basic(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["predicted_days"] = df["days_to_outcome"] + np.random.randn(len(df)) * 5
        
        c_index = compute_concordance_index(df, predicted_col="predicted_days")
        
        assert 0.0 <= c_index <= 1.0
    
    def test_concordance_index_with_censoring(self, survival_data_with_all_features):
        df = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        df["predicted_risk"] = np.random.rand(len(df))
        
        c_index = compute_concordance_index(df, use_censoring=True)
        
        assert 0.0 <= c_index <= 1.0
    
    def test_concordance_index_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_concordance_index(df)
        
        assert result == 0.5


class TestLOSQuantiles:
    """Tests for LOS quantile computation."""
    
    def test_los_quantiles_basic(self, survival_data_with_all_features):
        result = compute_LOS_quantiles(survival_data_with_all_features, [0.25, 0.5, 0.75])
        
        assert "animal_type" in result.columns
        assert "quantile" in result.columns
        assert len(result) == 8
    
    def test_los_quantiles_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_LOS_quantiles(df)
        
        assert result.empty
    
    def test_los_quantiles_custom_duration(self, survival_data_with_all_features):
        df = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        
        result = compute_LOS_quantiles(df, [0.5], duration_col_override="followup_days_censored")
        
        assert "LOS_days" in result.columns
        assert len(result) == 2


class TestCumulativeHazard:
    """Tests for cumulative hazard computation."""
    
    def test_cumulative_hazard_basic(self, survival_data_with_all_features):
        df = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        
        result = compute_cumulative_hazard(df)
        
        assert "time" in result.columns
        assert "cumulative_hazard" in result.columns
        assert len(result) > 0
    
    def test_cumulative_hazard_with_groups(self, survival_data_with_all_features):
        df = add_censoring_indicators(survival_data_with_all_features, max_observation_period=30)
        
        result = compute_cumulative_hazard(df, groups="animal_type")
        
        assert "animal_type" in result.columns
        assert len(result) > 0
    
    def test_cumulative_hazard_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_cumulative_hazard(df)
        
        assert result.empty


class TestLogRankTest:
    """Tests for log-rank test."""
    
    def test_logrank_basic(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["group"] = np.random.choice(["control", "treatment"], len(df))
        
        result = logrank_test_difference(df, group_col="group")
        
        assert "test_statistic" in result
        assert "p_value" in result
        assert "groups" in result
    
    def test_logrank_single_group(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df["group"] = "single"
        
        result = logrank_test_difference(df, group_col="group")
        
        assert result["test_statistic"] is None
        assert result["p_value"] is None
    
    def test_logrank_empty_dataframe(self):
        df = pd.DataFrame()
        result = logrank_test_difference(df)
        
        assert result["test_statistic"] is None
        assert result["p_value"] is None


class TestEncodeCategoricalFeatures:
    """Tests for categorical feature encoding."""
    
    def test_encode_basic(self, survival_data_with_all_features):
        result, encoders = encode_categorical_features(
            survival_data_with_all_features,
            categorical_cols=["animal_type", "age_group"],
            return_encoders=True
        )
        
        assert "animal_type_encoded" in result.columns
        assert "age_group_encoded" in result.columns
        assert len(encoders) == 2
    
    def test_encode_drop_first(self, survival_data_with_all_features):
        result, _ = encode_categorical_features(
            survival_data_with_all_features,
            categorical_cols=["animal_type"],
            drop_first=True
        )
        
        assert "animal_type_encoded_1" in result.columns
        assert result["animal_type_encoded_1"].dtype == bool
    
    def test_encode_empty_dataframe(self):
        df = pd.DataFrame()
        result, encoders = encode_categorical_features(df)
        
        assert result.empty
        assert len(encoders) == 0


class TestIntegrationBuildDatasetSurvival:
    """Integration tests between build_dataset.py and survival_analysis.py."""
    
    def test_survival_analysis_with_censoring(self, survival_data_with_all_features):
        df = survival_data_with_all_features.copy()
        df = add_censoring_indicators(df, max_observation_period=90)
        
        result = compute_kaplan_meier_survival(
            df,
            time_points=[7, 14, 30, 60, 90]
        )
        
        assert len(result) == 5
        assert (result["survival_probability"] >= 0).all()
    
    def test_cox_model_with_encoded_features(self, survival_data_with_all_features):
        result, _ = encode_categorical_features(
            survival_data_with_all_features,
            drop_first=True,
            return_encoders=False
        )
        
        cph, summary = fit_cox_with_censoring(
            result,
            feature_cols=["animal_type_encoded_1", "age_group_encoded_1", "age_group_encoded_2", "age_group_encoded_3"]
        )
        
        assert cph is not None
        assert len(summary) > 0


# Basic tests for edge cases

def test_kaplan_meier_empty():
    """Test Kaplan-Meier with empty dataframe."""
    empty_df = pd.DataFrame()
    result = compute_kaplan_meier_survival(empty_df)
    
    assert result.empty


def test_add_censoring_indicators_empty():
    """Test censoring indicator addition with empty dataframe."""
    empty_df = pd.DataFrame()
    result = add_censoring_indicators(empty_df, max_observation_period=30)
    
    assert result.empty


def test_log_transform_LOS(survival_data_with_all_features):
    """Test log transformation of LOS."""
    result = log_transform_LOS(survival_data_with_all_features)
    
    assert "log_days_to_outcome" in result.columns
    assert result["log_days_to_outcome"].dtype == float
    assert result["log_days_to_outcome"].min() >= 0


def test_cox_proportional_hazards_empty():
    """Test Cox PH with empty dataframe."""
    empty_df = pd.DataFrame()
    cph, summary = fit_cox_with_censoring(empty_df)
    
    assert cph is None
    assert summary.empty


def test_cox_with_censoring_native(survival_data_with_all_features):
    """Test Cox PH with native censoring support."""
    result = add_censoring_indicators(survival_data_with_all_features, max_observation_period=60)
    
    model, summary = fit_cox_with_censoring(
        result,
        duration_col="followup_days_censored",
        event_col="event_observed",
        feature_cols=["animal_type", "age_group"],
    )
    
    assert model is not None
    assert len(summary) > 0
    assert "coefficient" in summary.columns
    assert "hazard_ratio" in summary.columns


def test_proportional_hazards_validation(survival_data_with_all_features):
    """Test proportional hazards validation."""
    encoded_df, _ = encode_categorical_features(
        survival_data_with_all_features,
        categorical_cols=["animal_type", "age_group"],
        drop_first=True,
        return_encoders=False
    )
    
    cph, summary = fit_cox_with_censoring(encoded_df)
    
    if cph is not None:
        validation_result = validate_proportional_hazards(cph, p_value_threshold=0.05)
        
        assert "variable" in validation_result.columns
        assert "ph_test_p_value" in validation_result.columns
        assert "ph_assumption_passes" in validation_result.columns
        assert len(validation_result) > 0


def test_survival_metrics_with_censoring(survival_data_with_all_features):
    """Test survival metrics with censoring."""
    result = add_censoring_indicators(survival_data_with_all_features, max_observation_period=60)
    
    result["predicted_days"] = result["followup_days_censored"] + np.random.randn(len(result)) * 10
    
    c_index_native = compute_concordance_index(
        result,
        predicted_col="predicted_days",
        use_censoring=False,
    )
    
    c_index_censored = compute_concordance_index(
        result,
        predicted_col="predicted_days",
        use_censoring=True,
    )
    
    assert 0.0 <= c_index_native <= 1.0
    assert 0.0 <= c_index_censored <= 1.0
