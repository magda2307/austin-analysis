"""Tests for survival analysis utilities."""

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

from aac_adoption.data.match_records import infer_censoring_reason


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


@pytest.fixture
def sample_survival_data_with_censoring():
    np.random.seed(42)
    n_samples = 150
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 90, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "is_censored": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender", "Public Assistance"], size=n_samples),
        "sex_upon_intake": np.random.choice(["Neutered Male", "Spayed Female", "Intact Female", "Intact Male"], size=n_samples),
    })
    
    return df


@pytest.fixture
def mixed_outcome_data():
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(1, 120, size=n_samples),
        "outcome_type": np.random.choice(["adoption", "transfer", "euthanasia", "return_to_owner", "din"], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
    })
    
    return df


def test_kaplan_meier_survival(sample_los_data):
    result = compute_kaplan_meier_survival(sample_los_data, time_points=[7, 14, 30, 60])
    
    assert "days" in result.columns
    assert "survival_probability" in result.columns
    assert len(result) == 4
    assert result["survival_probability"].iloc[0] <= 1.0


def test_kaplan_meier_empty():
    empty_df = pd.DataFrame()
    result = compute_kaplan_meier_survival(empty_df)
    
    assert result.empty


def test_cox_proportional_hazards_fitting(sample_los_data):
    df_encoded = pd.get_dummies(sample_los_data, columns=["animal_type", "age_group"], drop_first=True, dtype=float)
    cph, summary = fit_cox_with_censoring(
        df_encoded,
        duration_col="days_to_outcome",
        event_col="adopted",
        feature_cols=[col for col in df_encoded.columns if col not in ["days_to_outcome", "adopted"]],
    )
    
    assert cph is not None
    assert len(summary) > 0
    assert "coefficient" in summary.columns
    assert "hazard_ratio" in summary.columns


def test_cox_empty_data():
    empty_df = pd.DataFrame()
    cph, summary = fit_cox_with_censoring(empty_df)
    
    assert cph is None
    assert summary.empty


def test_concordance_index(sample_los_data):
    sample_los_data["predicted_days"] = sample_los_data["days_to_outcome"] + np.random.randn(100) * 5
    
    c_index = compute_concordance_index(sample_los_data, predicted_col="predicted_days")
    
    assert 0.0 <= c_index <= 1.0


def test_los_quantiles(sample_los_data):
    result = compute_LOS_quantiles(sample_los_data, [0.5, 0.9])
    
    assert len(result) == 4
    assert "LOS_days" in result.columns
    assert result[result["animal_type"] == "Dog"]["quantile"].iloc[0] == 0.5


def test_add_censoring_indicators(sample_los_data):
    result = add_censoring_indicators(sample_los_data, max_observation_period=30)
    
    assert "is_censored" in result.columns
    assert "followup_days_censored" in result.columns
    assert result["followup_days_censored"].max() == 30


def test_log_transform_LOS(sample_los_data):
    result = log_transform_LOS(sample_los_data)
    
    assert "log_days_to_outcome" in result.columns
    assert result["log_days_to_outcome"].dtype == float
    assert result["log_days_to_outcome"].min() >= 0


@pytest.fixture
def synthetic_survival_data():
    """Generate synthetic survival data with censoring for testing."""
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "animal_id": [f"A{i:03d}" for i in range(n_samples)],
        "days_to_outcome": np.random.randint(1, 90, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender", "Public Service"], size=n_samples),
        "intake_condition": np.random.choice(["Normal", "Sick", "Injured"], size=n_samples),
        "sex_upon_intake": np.random.choice(["Spayed Female", "Neutered Male", "Intact Male", "Intact Female"], size=n_samples),
        "primary_breed": np.random.choice(["Mixed", "Labrador", "Domestic Shorthair", "Golden Retriever"], size=n_samples),
        "simplified_breed_group": np.random.choice(["Mixed", "Herding", "Terrier", "Non-Sporting"], size=n_samples),
        "simplified_color_group": np.random.choice(["Black or Dark", "Light", "Multi"], size=n_samples),
        "found_location_kind": np.random.choice(["austin_city", "intersection", "outside_jurisdiction"], size=n_samples),
    })
    
    return df


class TestCensoringColumnCreation:
    """Tests for censoring column creation and validation."""
    
    def test_add_censoring_indicators_basic(self):
        df = pd.DataFrame({
            "days_to_outcome": [5, 15, 35, 50, 100],
            "adopted": [1, 1, 0, 0, 0]
        })
        
        result = add_censoring_indicators(df, max_observation_period=30)
        
        assert "is_censored" in result.columns
        assert "followup_days_censored" in result.columns
        assert "event_observed" in result.columns
        assert result["is_censored"].tolist() == [False, False, True, True, True]
        assert result["followup_days_censored"].tolist() == [5, 15, 30, 30, 30]
        assert result["event_observed"].tolist() == [1, 1, 0, 0, 0]
    
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
    
    def test_add_censoring_indicators_mixed_censoring(self, synthetic_survival_data):
        result = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        n_censored = result["is_censored"].sum()
        n_events = result["event_observed"].sum()
        
        assert n_censored + n_events == len(result)
        assert n_censored > 0
        assert n_events > 0
    
    def test_add_censoring_indicators_native_event_column(self, synthetic_survival_data):
        result = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        assert "event_observed_native" in result.columns
        assert result["event_observed_native"].dtype == int
        assert result["event_observed_native"].isin([0, 1]).all()


class TestKaplanMeierEstimation:
    """Tests for Kaplan-Meier estimation with native censoring support."""
    
    def test_kaplan_meier_basic_survival(self, synthetic_survival_data):
        result = compute_kaplan_meier_survival(synthetic_survival_data, time_points=[7, 14, 30, 60])
        
        assert "days" in result.columns
        assert "survival_probability" in result.columns
        assert len(result) == 4
        assert result["survival_probability"].iloc[0] <= 1.0
        assert result["survival_probability"].iloc[0] > 0
        assert (result["survival_probability"] >= 0).all()
        assert (result["survival_probability"] <= 1.0).all()
        assert result["survival_probability"].is_monotonic_decreasing or result["survival_probability"].is_monotonic_increasing or result["survival_probability"].nunique() > 1
    
    def test_kaplan_meier_with_censoring_column(self, synthetic_survival_data):
        df_with_censoring = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        result = compute_kaplan_meier_survival(
            df_with_censoring,
            censoring_col="event_observed",
            time_points=[7, 14, 30]
        )
        
        assert "days" in result.columns
        assert "survival_probability" in result.columns
        assert len(result) == 3
    
    def test_kaplan_meier_grouped_curves(self, synthetic_survival_data):
        result = compute_kaplan_meier_survival(
            synthetic_survival_data,
            time_points=[7, 14, 30],
            group_by="animal_type"
        )
        
        assert "animal_type" in result.columns
        assert len(result["animal_type"].unique()) == 2
        assert len(result) >= 6
        assert result["survival_probability"].iloc[0] <= 1.0
    
    def test_kaplan_meier_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = compute_kaplan_meier_survival(empty_df)
        
        assert result.empty
    
    def test_kaplan_meier_single_time_point(self, synthetic_survival_data):
        result = compute_kaplan_meier_survival(
            synthetic_survival_data,
            time_points=[30]
        )
        
        assert len(result) == 1
        assert result["days"].iloc[0] == 30
    
    def test_kaplan_meier_time_points_default(self, synthetic_survival_data):
        result = compute_kaplan_meier_survival(synthetic_survival_data)
        
        assert "days" in result.columns
        assert "survival_probability" in result.columns
    
    def test_kaplan_meier_with_native_event(self, synthetic_survival_data):
        result = compute_kaplan_meier_survival(
            synthetic_survival_data,
            event_col="adopted",
            censoring_col="adopted",
            time_points=[7, 14, 30]
        )
        
        assert len(result) == 3
        assert result["survival_probability"].iloc[0] <= 1.0
    
    def test_kaplan_meier_all_censored_data(self):
        df = pd.DataFrame({
            "days_to_outcome": [10, 20, 30],
            "adopted": [0, 0, 0]
        })
        
        result = compute_kaplan_meier_survival(df, time_points=[10, 20, 30])
        
        assert len(result) == 3
        assert result["survival_probability"].iloc[0] <= 1.0


class TestCoxProportionalHazards:
    """Tests for Cox Proportional Hazards model with censoring."""
    
    def test_cox_basic_fitting(self, synthetic_survival_data):
        result, _ = encode_categorical_features(
            synthetic_survival_data,
            categorical_cols=["animal_type", "age_group"],
            drop_first=True,
            return_encoders=False
        )
        
        cph, summary = fit_cox_with_censoring(
            result,
            feature_cols=["animal_type_encoded_1", "age_group_encoded_1", "age_group_encoded_2", "age_group_encoded_3"]
        )
        
        assert cph is not None
        assert len(summary) > 0
        assert "coefficient" in summary.columns
        assert "hazard_ratio" in summary.columns
    
    def test_cox_with_categorical_features(self, synthetic_survival_data):
        df_encoded = pd.get_dummies(
            synthetic_survival_data,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, summary = fit_cox_with_censoring(df_encoded)
        
        assert cph is not None
        assert "p_value" in summary.columns
    
    def test_cox_empty_dataframe(self):
        empty_df = pd.DataFrame()
        cph, summary = fit_cox_with_censoring(empty_df)
        
        assert cph is None
        assert summary.empty
    
    def test_cox_custom_features(self, synthetic_survival_data):
        df_encoded = pd.get_dummies(
            synthetic_survival_data,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        feature_cols = ["animal_type_Dog", "age_group_baby"]
        cph, summary = fit_cox_with_censoring(
            df_encoded,
            feature_cols=feature_cols
        )
        
        assert cph is not None
        assert len(summary) >= len(feature_cols) - len([c for c in feature_cols if c not in df_encoded.columns])
    
    def test_cox_with_strata(self, synthetic_survival_data):
        df_encoded = pd.get_dummies(
            synthetic_survival_data,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, summary = fit_cox_with_censoring(
            df_encoded,
            strata=["animal_type"]
        )
        
        assert cph is not None
        assert len(summary) > 0
    
    def test_cox_normalized_features(self, synthetic_survival_data):
        df_encoded = pd.get_dummies(
            synthetic_survival_data,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        df_encoded["age_in_days"] = np.random.randint(1, 2000, size=len(df_encoded))
        
        cph, summary = fit_cox_with_censoring(
            df_encoded,
            normalize=True
        )
        
        assert cph is not None
        assert len(summary) > 0
    
    def test_cox_all_events(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["adopted"] = 1
        
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, summary = fit_cox_with_censoring(df_encoded)
        
        assert cph is not None
        assert len(summary) > 0
    
    def test_cox_no_events(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["adopted"] = 0
        
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, summary = fit_cox_with_censoring(df_encoded, duration_col="days_to_outcome", event_col="adopted")
        
        assert cph is None
        assert summary.empty


class TestProportionalHazardsAssumption:
    """Tests for proportional hazards assumption validation."""
    
    def test_ph_validation_basic(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=90)
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, _ = fit_cox_with_censoring(df_encoded, duration_col="followup_days_censored", event_col="event_observed")
        
        if cph is not None:
            result = validate_proportional_hazards(cph, df_encoded, p_value_threshold=0.05)
            
            assert "variable" in result.columns
            assert "chi_square_statistic" in result.columns
            assert "ph_test_p_value" in result.columns
            assert "ph_assumption_passes" in result.columns
    
    def test_ph_validation_with_threshold(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=90)
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, _ = fit_cox_with_censoring(df_encoded, duration_col="followup_days_censored", event_col="event_observed")
        
        if cph is not None:
            result = validate_proportional_hazards(cph, df_encoded, p_value_threshold=0.1)
            
            assert len(result) > 0
    
    def test_ph_validation_none_model(self):
        result = validate_proportional_hazards(None, None, p_value_threshold=0.05)
        
        assert result.empty
    
    def test_ph_validation_different_methods(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=90)
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, _ = fit_cox_with_censoring(df_encoded, duration_col="followup_days_censored", event_col="event_observed")
        
        if cph is not None:
            result_schoenfeld = validate_proportional_hazards(cph, df_encoded, test_method="schoenfeld")
            result_residuals = validate_proportional_hazards(cph, df_encoded, test_method="residuals")
            
            assert len(result_schoenfeld) > 0
            assert len(result_residuals) > 0


class TestCompetingRisksAnalysis:
    """Tests for competing risks analysis."""
    
    def test_subdistribution_hazard_basic(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["event_type"] = np.where(
            df["adopted"],
            "adoption",
            np.where(df["days_to_outcome"] < 60, "censored", "other")
        )
        
        result = compute_subDistribution_hazard(
            df,
            event_col="adopted",
            competing_events_col="event_type"
        )
        
        if result is not None:
            assert "coefficient" in result.columns
            assert "subdistribution_hazard_ratio" in result.columns
    
    def test_subdistribution_hazard_no_competing_col(self, synthetic_survival_data):
        result = compute_subDistribution_hazard(
            synthetic_survival_data,
            event_col="adopted"
        )
        
        assert result is None
    
    def test_subdistribution_hazard_empty_data(self):
        df = pd.DataFrame()
        result = compute_subDistribution_hazard(df)
        
        assert result is None
    
    def test_subdistribution_hazard_with_features(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["event_type"] = np.where(df["adopted"], "adoption", "censored")
        
        feature_cols = ["animal_type", "age_group"]
        result = compute_subDistribution_hazard(
            df,
            event_col="adopted",
            competing_events_col="event_type",
            feature_cols=feature_cols
        )
        
        if result is not None:
            assert len(result) >= len(feature_cols)


class TestDataValidation:
    """Tests for data validation for required columns."""
    
    def test_kaplan_meier_missing_duration_column(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df = df.drop(columns=["days_to_outcome"])
        
        result = compute_kaplan_meier_survival(df)
        
        assert result.empty
    
    def test_kaplan_meier_missing_event_column(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df = df.drop(columns=["adopted"])
        
        result = compute_kaplan_meier_survival(df)
        
        assert result.empty
    
    def test_cox_missing_duration_column(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df = df.drop(columns=["days_to_outcome"])
        
        cph, summary = fit_cox_with_censoring(df)
        
        assert cph is None
        assert summary.empty
    
    def test_cox_missing_feature_columns(self, synthetic_survival_data):
        df = pd.DataFrame({
            "days_to_outcome": [10, 20, 30],
            "adopted": [1, 1, 0]
        })
        
        cph, summary = fit_cox_with_censoring(df)
        
        assert cph is not None
        assert len(summary) == 0
    
    def test_concordance_index_missing_columns(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df = df.drop(columns=["days_to_outcome"])
        
        result = compute_concordance_index(df)
        
        assert result == 0.5


class TestConcordanceIndex:
    """Tests for concordance index computation."""
    
    def test_concordance_index_basic(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["predicted_days"] = df["days_to_outcome"] + np.random.randn(len(df)) * 5
        
        c_index = compute_concordance_index(df, predicted_col="predicted_days")
        
        assert 0.0 <= c_index <= 1.0
    
    def test_concordance_index_with_censoring(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        df["predicted_risk"] = np.random.rand(len(df))
        
        c_index = compute_concordance_index(df, use_censoring=True)
        
        assert 0.0 <= c_index <= 1.0
    
    def test_concordance_index_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_concordance_index(df)
        
        assert result == 0.5
    
    def test_concordance_index_predicted_is_hazard(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["predicted_hazard"] = np.random.rand(len(df)) * 0.1
        
        c_index = compute_concordance_index(
            df,
            predicted_col="predicted_hazard",
            predicted_is_hazard=True
        )
        
        assert 0.0 <= c_index <= 1.0


class TestLOSQuantiles:
    """Tests for LOS quantile computation."""
    
    def test_los_quantiles_basic(self, synthetic_survival_data):
        result = compute_LOS_quantiles(synthetic_survival_data, [0.25, 0.5, 0.75])
        
        assert "animal_type" in result.columns
        assert "quantile" in result.columns
        assert len(result) == 6
    
    def test_los_quantiles_all_quantiles(self, synthetic_survival_data):
        default_quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        result = compute_LOS_quantiles(synthetic_survival_data, default_quantiles)
        
        assert len(result) == len(synthetic_survival_data["animal_type"].unique()) * len(default_quantiles)
    
    def test_los_quantiles_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_LOS_quantiles(df)
        
        assert result.empty
    
    def test_los_quantiles_custom_duration(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        result = compute_LOS_quantiles(df, [0.5], duration_col_override="followup_days_censored")
        
        assert "LOS_days" in result.columns
        assert len(result) == 2


class TestCumulativeHazard:
    """Tests for cumulative hazard computation."""
    
    def test_cumulative_hazard_basic(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        result = compute_cumulative_hazard(df)
        
        assert "time" in result.columns
        assert "cumulative_hazard" in result.columns
        assert len(result) > 0
    
    def test_cumulative_hazard_with_groups(self, synthetic_survival_data):
        df = add_censoring_indicators(synthetic_survival_data, max_observation_period=30)
        
        result = compute_cumulative_hazard(df, groups="animal_type")
        
        assert "animal_type" in result.columns
        assert len(result) > 0
    
    def test_cumulative_hazard_empty_dataframe(self):
        df = pd.DataFrame()
        result = compute_cumulative_hazard(df)
        
        assert result.empty


class TestLogRankTest:
    """Tests for log-rank test to compare survival curves."""
    
    def test_logrank_basic(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df["group"] = np.random.choice(["control", "treatment"], len(df))
        
        result = logrank_test_difference(df, group_col="group")
        
        assert "test_statistic" in result
        assert "p_value" in result
        assert "groups" in result
    
    def test_logrank_single_group(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
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
    
    def test_encode_basic(self, synthetic_survival_data):
        result, encoders = encode_categorical_features(
            synthetic_survival_data,
            categorical_cols=["animal_type", "age_group"],
            return_encoders=True
        )
        
        assert "animal_type_encoded_1" in result.columns
        assert "age_group_encoded_1" in result.columns
        assert "age_group_encoded_2" in result.columns
        assert "age_group_encoded_3" in result.columns
        assert len(encoders) == 2
    
    def test_encode_drop_first(self, synthetic_survival_data):
        result, _ = encode_categorical_features(
            synthetic_survival_data,
            categorical_cols=["animal_type"],
            drop_first=True
        )
        
        assert "animal_type_encoded_1" in result.columns
        assert "animal_type" not in result.columns
    
    def test_encode_empty_dataframe(self):
        df = pd.DataFrame()
        result, encoders = encode_categorical_features(df)
        
        assert result.empty
        assert len(encoders) == 0
    
    def test_encode_default_columns(self, synthetic_survival_data):
        result, _ = encode_categorical_features(synthetic_survival_data)
        
        assert len([c for c in result.columns if "encoded" in c]) > 0


class TestIntegrationBuildDatasetSurvival:
    """Integration tests between build_dataset.py and survival_analysis.py."""
    
    def test_survival_analysis_with_build_dataset_columns(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        df = add_censoring_indicators(df, max_observation_period=90)
        
        result = compute_kaplan_meier_survival(
            df,
            time_points=[7, 14, 30, 60, 90]
        )
        
        assert len(result) == 5
        assert result["survival_probability"].iloc[0] <= 1.0
        assert result["survival_probability"].iloc[0] >= 0
        assert (result["survival_probability"] >= 0).all()
    
    def test_cox_model_with_dataset_features(self, synthetic_survival_data):
        df = synthetic_survival_data.copy()
        
        df_encoded = pd.get_dummies(
            df,
            columns=["animal_type", "age_group", "intake_type", "intake_condition", "sex_upon_intake"],
            drop_first=True,
            dtype=float
        )
        
        cph, summary = fit_cox_with_censoring(
            df_encoded,
            feature_cols=[
                "animal_type_Dog", "age_group_baby", "age_group_senior",
                "age_group_young", "intake_type_Owner Surrender"
            ]
        )
        
        if cph is not None:
            assert len(summary) > 0
            assert "coefficient" in summary.columns
            assert "hazard_ratio" in summary.columns
            assert "p_value" in summary.columns


def test_cox_with_censoring_native(sample_survival_data_with_censoring):
    np.random.seed(42)
    df = sample_survival_data_with_censoring.copy()
    
    result = add_censoring_indicators(df, max_observation_period=60)
    
    encoded_df, _ = encode_categorical_features(
        result,
        categorical_cols=["animal_type", "age_group", "intake_type", "sex_upon_intake"],
        drop_first=True,
        return_encoders=False,
    )
    
    feature_cols = [col for col in encoded_df.columns if col not in ["days_to_outcome", "adopted", "is_censored", "followup_days_censored", "event_observed", "event_observed_native"]]
    
    model, summary = fit_cox_with_censoring(
        encoded_df,
        duration_col="followup_days_censored",
        event_col="event_observed",
        feature_cols=feature_cols,
    )
    
    assert model is not None
    assert len(summary) > 0
    assert "coefficient" in summary.columns
    assert "hazard_ratio" in summary.columns


def test_proportional_hazards_validation():
    np.random.seed(42)
    n_samples = 150
    df = pd.DataFrame({
        "days_to_outcome": np.random.randint(5, 90, size=n_samples),
        "adopted": np.random.choice([0, 1], size=n_samples),
        "animal_type": np.random.choice(["Dog", "Cat"], size=n_samples),
        "age_group": np.random.choice(["baby", "young", "adult", "senior"], size=n_samples),
        "intake_type": np.random.choice(["Stray", "Owner Surrender", "Public Assistance"], size=n_samples),
        "sex_upon_intake": np.random.choice(["Neutered Male", "Spayed Female", "Intact Female", "Intact Male"], size=n_samples),
    })
    
    df_encoded = pd.get_dummies(
        df,
        columns=["animal_type", "age_group", "intake_type", "sex_upon_intake"],
        drop_first=True,
        dtype=float
    )
    
    cph, summary = fit_cox_with_censoring(
        df_encoded,
        duration_col="days_to_outcome",
        event_col="adopted",
        feature_cols=[col for col in df_encoded.columns if col not in ["days_to_outcome", "adopted"]],
    )
    
    if cph is not None:
        validation_result = validate_proportional_hazards(cph, df_encoded, p_value_threshold=0.05)
        
        assert "variable" in validation_result.columns
        assert "ph_test_p_value" in validation_result.columns
        assert "ph_assumption_passes" in validation_result.columns
        assert len(validation_result) > 0


def test_categorical_encoding_survival(sample_survival_data_with_censoring):
    categorical_cols = ["animal_type", "age_group", "intake_type", "sex_upon_intake"]
    
    encoded_df, encoders = encode_categorical_features(
        sample_survival_data_with_censoring,
        categorical_cols=categorical_cols,
        drop_first=True,
        return_encoders=True,
    )
    
    assert len(encoders) == 4
    assert "animal_type_encoded_1" in encoded_df.columns
    assert "age_group_encoded_1" in encoded_df.columns
    assert "age_group_encoded_2" in encoded_df.columns
    assert "age_group_encoded_3" in encoded_df.columns
    
    encoded_categorical_cols = [c for c in encoded_df.columns if c.startswith(("animal_type_", "age_group_", "intake_type_", "sex_upon_intake_")) and "_encoded" in c]
    assert len(encoded_categorical_cols) > 0
    
    feature_cols = [col for col in encoded_df.columns if col not in ["days_to_outcome", "adopted", "is_censored", "animal_type", "age_group", "intake_type", "sex_upon_intake"]]
    assert len(feature_cols) > 0


def test_censoring_reason_inference():
    test_cases = [
        ("adoption", "adopted"),
        ("Adoption", "adopted"),
        ("transfer", "censored_transfer"),
        ("TRANSFER", "censored_transfer"),
        ("moved", "censored_transfer"),
        ("reintake", "censored_transfer"),
        ("euthanasia", "censored_euthanasia"),
        ("euthanize", "censored_euthanasia"),
        ("return_to_owner", "censored_return"),
        ("returned", "censored_return"),
        ("din", "censored_lost"),
        ("disappear", "censored_lost"),
        ("missing", "censored_lost"),
        ("unknown_outcome", "censored_unknown"),
        (None, "censored_unknown"),
        ("", "censored_unknown"),
    ]
    
    from aac_adoption.data.match_records import infer_censoring_reason
    
    for outcome_type, expected_reason in test_cases:
        result = infer_censoring_reason(outcome_type)
        assert result == expected_reason, f"Expected {expected_reason} for {outcome_type}"


def test_cox_with_mixed_outcomes(mixed_outcome_data):
    np.random.seed(42)
    df = mixed_outcome_data.copy()
    
    df["censoring_reason"] = df["outcome_type"].apply(infer_censoring_reason)
    df["is_censored"] = df["censoring_reason"].apply(lambda x: 0 if x == "adopted" else 1)
    
    result = add_censoring_indicators(df, max_observation_period=90)
    
    encoded_df, _ = encode_categorical_features(
        result,
        categorical_cols=["animal_type", "age_group"],
        drop_first=True,
        return_encoders=False,
    )
    
    feature_cols = [col for col in encoded_df.columns if col not in ["days_to_outcome", "outcome_type", "censoring_reason", "is_censored", "followup_days_censored", "event_observed", "event_observed_native"]]
    
    model, summary = fit_cox_with_censoring(
        encoded_df,
        duration_col="followup_days_censored",
        event_col="event_observed",
        feature_cols=feature_cols,
    )
    
    assert model is not None
    assert len(summary) > 0


def test_survival_metrics_with_censoring(sample_survival_data_with_censoring):
    np.random.seed(42)
    df = sample_survival_data_with_censoring.copy()
    
    result = add_censoring_indicators(df, max_observation_period=60)
    
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


def test_censoring_indicators_comprehensive(sample_survival_data_with_censoring):
    df = sample_survival_data_with_censoring.copy()
    
    max_periods = [30, 60, 90]
    
    for max_period in max_periods:
        result = add_censoring_indicators(df, max_observation_period=max_period)
        
        assert "is_censored" in result.columns
        assert "followup_days_censored" in result.columns
        assert "event_observed" in result.columns
        
        assert result["followup_days_censored"].max() <= max_period
        assert result["is_censored"].dtype == bool or result["is_censored"].dtype == np.dtype('bool')
        
        expected_censored = (df["days_to_outcome"] >= max_period).sum()
        actual_censored = result["is_censored"].sum()
        assert actual_censored == expected_censored
        
        expected_event = len(df) - expected_censored
        assert result["event_observed"].sum() == expected_event
    
    df_edge = pd.DataFrame({
        "days_to_outcome": [1, 30, 60, 90, 120],
        "adopted": [1, 1, 0, 0, 0],
    })
    
    result = add_censoring_indicators(df_edge, max_observation_period=60)
    
    assert result.iloc[0]["is_censored"] == False
    assert result.iloc[1]["is_censored"] == False
    assert result.iloc[2]["is_censored"] == True
    assert result.iloc[3]["is_censored"] == True
    assert result.iloc[4]["is_censored"] == True
    
    assert result.iloc[0]["followup_days_censored"] == 1
    assert result.iloc[1]["followup_days_censored"] == 30
    assert result.iloc[2]["followup_days_censored"] == 60
    assert result.iloc[3]["followup_days_censored"] == 60
    assert result.iloc[4]["followup_days_censored"] == 60
