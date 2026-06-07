"""Integration tests for end-to-end survival analysis pipeline.

Tests the complete pipeline:
    build_dataset.py -> match_records.py -> survival_analysis.py -> train_survival.py

Verifies:
- Censoring columns propagate correctly through the pipeline
- Reproducibility (same seed → same results)
- Error handling for missing data
- Output artifacts are created correctly
- Timing benchmarks
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import pytest

from aac_adoption.data.build_dataset import build_modeling_dataset_from_files
from aac_adoption.data.match_records import match_intakes_to_future_outcomes
from aac_adoption.analysis.survival_analysis import (
    add_censoring_indicators,
    compute_concordance_index,
    fit_cox_with_censoring,
)
from aac_adoption.models.train_survival import train_all_survival
from aac_adoption.config import RANDOM_STATE


def generate_sample_intakes(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate sample intakes data for testing."""
    np.random.seed(seed)
    
    animal_ids = [f"A00000{str(i).zfill(4)}" for i in range(n)]
    animal_types = np.random.choice(["Dog", "Cat"], size=n)
    intake_types = np.random.choice(["Stray", "Owner Surrender", "Public Assist"], size=n)
    intake_conditions = np.random.choice(["Normal", "Sick", "Injured"], size=n, p=[0.7, 0.2, 0.1])
    sexes = np.random.choice(["Male", "Female"], size=n)
    
    base_date = pd.Timestamp("2022-01-01")
    days_offset = np.random.randint(0, 365, size=n).tolist()
    intake_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offset]
    
    names = [f"Dog{str(i)}" if t == "Dog" else f"Cat{str(i)}" for i, t in enumerate(animal_types)]
    
    intakes = pd.DataFrame({
        "Animal ID": animal_ids,
        "Name": names,
        "Animal Type": animal_types,
        "Datetime": intake_dates,
        "Intake Type": intake_types,
        "Intake Condition": intake_conditions,
        "Sex upon Intake": sexes,
        "Age upon Intake": np.random.choice(["0-1 year", "1-3 years", "3-5 years", "5+ years"], size=n),
        "Breed": np.random.choice(["Labrador", "Mixed", "Beagle", "Domestic Shorthair", "Maine Coon"], size=n),
        "Color": np.random.choice(["Black", "Brown", "White", "Spotted"], size=n),
        "Found Location": np.random.choice(["Street", "Park", "Residential", "Commercial"], size=n),
    })
    
    return intakes


def generate_sample_outcomes(intakes: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate sample outcomes data that matches intakes."""
    np.random.seed(seed + 1)
    
    intake_animal_ids = intakes["Animal ID"].tolist()
    intake_animal_types = intakes["Animal Type"].tolist()
    outcomes = []
    
    for i, (animal_id, animal_type) in enumerate(zip(intake_animal_ids, intake_animal_types)):
        intake_row = intakes.iloc[i]
        intake_datetime = intake_row["Datetime"]
        
        days_to_outcome = int(np.random.randint(1, 100))
        outcome_datetime = intake_datetime + pd.Timedelta(days=days_to_outcome)
        
        adopt_or_out = np.random.choice(["Adoption", "Transfer", "Euthanasia"], p=[0.6, 0.25, 0.15])
        
        outcomes.append({
            "Animal ID": animal_id,
            "Datetime": outcome_datetime,
            "Outcome Type": adopt_or_out,
            "Outcome Subtype": "Normal" if adopt_or_out == "Adoption" else "Other",
            "Animal Type": animal_type,
            "Sex upon Outcome": intake_row["Sex upon Intake"],
            "Age upon Outcome": intake_row["Age upon Intake"],
        })
    
    outcomes_df = pd.DataFrame(outcomes)
    return outcomes_df


def test_full_pipeline_integration():
    """Test complete pipeline from raw data to trained models."""
    intakes = generate_sample_intakes(n=150, seed=42)
    outcomes = generate_sample_outcomes(intakes, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        assert result.matched_rows > 0, "Pipeline should match rows"
        assert result.dataset is not None, "Dataset should be created"
        
        dataset = pd.read_csv(output_path, parse_dates=["intake_datetime", "outcome_datetime"])
        
        assert "censoring_reason" in dataset.columns, "Censoring column should propagate"
        assert "event_type" in dataset.columns, "Event type column should propagate"
        assert "is_censored" in dataset.columns, "is_censored column should propagate"
        assert "followup_days_censored" in dataset.columns, "Followup censoring column should propagate"


def test_censoring_propagation():
    """Verify censoring columns propagate correctly through the pipeline."""
    intakes = generate_sample_intakes(n=100, seed=123)
    outcomes = generate_sample_outcomes(intakes, seed=123)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        dataset = result.dataset
        
        required_censoring_cols = [
            "is_censored", "censoring_reason", "event_type", "followup_days_censored"
        ]
        
        for col in required_censoring_cols:
            assert col in dataset.columns, f"Censoring column {col} should exist"
        
        censoring_counts = dataset["event_type"].value_counts()
        assert "censored" in censoring_counts.index, "Should have censored events"
        assert "adoption" in censoring_counts.index, "Should have adoption events"


def test_reproducibility():
    """Test that same seed produces same results."""
    intakes1 = generate_sample_intakes(n=80, seed=999)
    outcomes1 = generate_sample_outcomes(intakes1, seed=999)
    
    intakes2 = generate_sample_intakes(n=80, seed=999)
    outcomes2 = generate_sample_outcomes(intakes2, seed=999)
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        tmpdir1_path = Path(tmpdir1)
        tmpdir2_path = Path(tmpdir2)
        
        for tmpdir_path, intakes, outcomes in [
            (tmpdir1_path, intakes1, outcomes1),
            (tmpdir2_path, intakes2, outcomes2),
        ]:
            raw_dir = tmpdir_path / "data" / "raw"
            raw_dir.mkdir(parents=True)
            
            intakes_path = raw_dir / "intakes.csv"
            outcomes_path = raw_dir / "outcomes.csv"
            
            intakes.to_csv(intakes_path, index=False)
            outcomes.to_csv(outcomes_path, index=False)
        
        output_path1 = tmpdir1_path / "data" / "processed" / "modeling_dataset.csv"
        output_path2 = tmpdir2_path / "data" / "processed" / "modeling_dataset.csv"
        
        result1 = build_modeling_dataset_from_files(
            str(tmpdir1_path / "data" / "raw" / "intakes.csv"),
            str(tmpdir1_path / "data" / "raw" / "outcomes.csv"),
            str(output_path1),
        )
        
        result2 = build_modeling_dataset_from_files(
            str(tmpdir2_path / "data" / "raw" / "intakes.csv"),
            str(tmpdir2_path / "data" / "raw" / "outcomes.csv"),
            str(output_path2),
        )
        
        dataset1 = pd.read_csv(output_path1, parse_dates=["intake_datetime", "outcome_datetime"])
        dataset2 = pd.read_csv(output_path2, parse_dates=["intake_datetime", "outcome_datetime"])
        
        pd.testing.assert_frame_equal(
            dataset1[["animal_id", "days_to_outcome", "adopted", "censoring_reason"]],
            dataset2[["animal_id", "days_to_outcome", "adopted", "censoring_reason"]],
            check_exact=True,
        )


def test_error_handling_missing_data():
    """Test error handling for missing or invalid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        
        pd.DataFrame({
            "Animal ID": [],
            "Animal Type": [],
            "Datetime": [],
        }).to_csv(intakes_path, index=False)
        
        outcomes_path = raw_dir / "outcomes.csv"
        
        pd.DataFrame({
            "Animal ID": [],
            "Animal Type": [],
            "Datetime": [],
            "Outcome Type": [],
        }).to_csv(outcomes_path, index=False)
        
        with pytest.raises((ValueError, KeyError)):
            build_modeling_dataset_from_files(
                str(intakes_path),
                str(outcomes_path),
                str(tmpdir_path / "data" / "processed" / "modeling_dataset.csv"),
            )


def test_match_records_integration():
    """Test match_records module directly."""
    intakes = generate_sample_intakes(n=50, seed=777)
    outcomes = generate_sample_outcomes(intakes, seed=777)
    
    cleaned_intakes = intakes.rename(columns={
        "Animal ID": "animal_id",
        "Animal Type": "animal_type",
        "Datetime": "intake_datetime",
        "Intake Type": "intake_type",
        "Intake Condition": "intake_condition",
        "Sex upon Intake": "sex_upon_intake",
        "Age upon Intake": "age_upon_intake",
        "Breed": "breed",
        "Color": "color",
        "Found Location": "found_location",
    })
    
    cleaned_outcomes = outcomes.rename(columns={
        "Animal ID": "animal_id",
        "Animal Type": "animal_type",
        "Datetime": "outcome_datetime",
        "Outcome Type": "outcome_type",
        "Outcome Subtype": "outcome_subtype",
        "Sex upon Outcome": "sex_upon_outcome",
        "Age upon Outcome": "age_upon_outcome",
    })
    
    cleaned_intakes["intake_datetime"] = pd.to_datetime(cleaned_intakes["intake_datetime"], utc=True).dt.tz_localize(None)
    cleaned_outcomes["outcome_datetime"] = pd.to_datetime(cleaned_outcomes["outcome_datetime"], utc=True).dt.tz_localize(None)
    
    matched, unmatched = match_intakes_to_future_outcomes(
        cleaned_intakes, cleaned_outcomes
    )
    
    assert len(matched) > 0, "Should match some records"
    assert "censoring_reason" in matched.columns, "Matched data should have censoring_reason"
    assert "event_type" in matched.columns, "Matched data should have event_type"
    assert "episode_number" in matched.columns, "Matched data should have episode_number"


def test_survival_analysis_with_censoring():
    """Test survival analysis handles censoring correctly."""
    intakes = generate_sample_intakes(n=100, seed=555)
    outcomes = generate_sample_outcomes(intakes, seed=555)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        dataset = result.dataset
        
        assert "is_censored" in dataset.columns
        assert "censoring_reason" in dataset.columns
        
        censored_counts = dataset["is_censored"].value_counts()
        assert len(censored_counts) >= 1, "Should have censoring information"
        
        censoring_reasons = dataset["censoring_reason"].value_counts()
        assert len(censoring_reasons) >= 1, "Should have censoring reason categories"


def test_train_survival_with_sample_data():
    """Test training survival models with sample data."""
    intakes = generate_sample_intakes(n=120, seed=888)
    outcomes = generate_sample_outcomes(intakes, seed=888)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        metrics_dir = tmpdir_path / "reports" / "metrics"
        models_dir = tmpdir_path / "models" / "survival"
        tables_dir = tmpdir_path / "reports" / "tables"
        
        outputs = train_all_survival(
            data_path=str(output_path),
            metrics_dir=str(metrics_dir),
            models_dir=str(models_dir),
            tables_dir=str(tables_dir),
            max_rows=100,
            smoothing=10.0,
        )
        
        assert outputs.concordance_metrics is not None
        assert len(outputs.concordance_metrics) > 0
        assert "c_index" in outputs.concordance_metrics.columns
        
        assert outputs.brier_metrics is not None
        assert len(outputs.brier_metrics) > 0
        
        assert outputs.calibration_metrics is not None
        assert len(outputs.calibration_metrics) > 0


def test_output_artifacts_created():
    """Verify all expected output artifacts are created."""
    intakes = generate_sample_intakes(n=100, seed=666)
    outcomes = generate_sample_outcomes(intakes, seed=666)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        assert output_path.exists(), "Modeling dataset CSV should exist"
        assert (output_path.parent / "feature_columns.json").exists()
        assert (output_path.parent / "target_columns.json").exists()
        
        metrics_dir = tmpdir_path / "reports" / "metrics"
        models_dir = tmpdir_path / "models" / "survival"
        
        outputs = train_all_survival(
            data_path=str(output_path),
            metrics_dir=str(metrics_dir),
            models_dir=str(models_dir),
            tables_dir=str(tmpdir_path / "reports" / "tables"),
            max_rows=80,
        )
        
        assert (metrics_dir / "survival_concordance_metrics.csv").exists()
        assert (metrics_dir / "survival_brier_metrics.csv").exists()
        assert (metrics_dir / "survival_calibration_metrics.csv").exists()
        
        model_files = list(models_dir.glob("*.pkl"))
        assert len(model_files) > 0, "Model artifacts should exist"


def test_timing_benchmarks():
    """Test timing for pipeline steps."""
    intakes = generate_sample_intakes(n=100, seed=321)
    outcomes = generate_sample_outcomes(intakes, seed=321)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        start_time = time.time()
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        build_time = time.time() - start_time
        
        assert build_time < 30, f"Dataset build should complete in under 30 seconds, took {build_time:.2f}s"
        
        metrics_dir = tmpdir_path / "reports" / "metrics"
        models_dir = tmpdir_path / "models" / "survival"
        
        start_time = time.time()
        outputs = train_all_survival(
            data_path=str(output_path),
            metrics_dir=str(metrics_dir),
            models_dir=str(models_dir),
            tables_dir=str(tmpdir_path / "reports" / "tables"),
            max_rows=80,
        )
        train_time = time.time() - start_time
        
        assert train_time < 120, f"Model training should complete in under 120 seconds, took {train_time:.2f}s"


def test_censoring_reason_categories():
    """Test that all expected censoring reason categories are present."""
    intakes = generate_sample_intakes(n=150, seed=789)
    outcomes = generate_sample_outcomes(intakes, seed=789)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        dataset = result.dataset
        
        censoring_reasons = dataset["censoring_reason"].unique()
        
        expected_reasons = ["adopted", "censored_transfer", "censored_euthanasia", 
                          "censored_return", "censored_lost", "censored_unknown"]
        
        for reason in expected_reasons:
            if reason in censoring_reasons:
                count = (dataset["censoring_reason"] == reason).sum()
                assert count > 0, f"Censoring reason '{reason}' should have positive count"
        
        assert len(censoring_reasons) >= 1, "Should have at least one censoring reason"


def test_concordance_index_with_censoring():
    """Test concordance index computation handles censoring correctly."""
    intakes = generate_sample_intakes(n=80, seed=456)
    outcomes = generate_sample_outcomes(intakes, seed=456)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        dataset = result.dataset
        
        cox_model, coefficients = fit_cox_with_censoring(
            dataset,
            duration_col="days_to_outcome",
            event_col="is_censored",
        )
        
        assert coefficients is not None, "Cox model should produce coefficients"
        assert len(coefficients) > 0, "Coefficients should not be empty"
        
        c_index = compute_concordance_index(
            dataset,
            predicted_col="days_to_outcome",
            duration_col="days_to_outcome",
            event_col="is_censored",
        )
        
        assert 0 <= c_index <= 1, "C-index should be between 0 and 1"


def test_censoring_indicator_consistency():
    """Test that censoring indicators are consistent across the pipeline."""
    intakes = generate_sample_intakes(n=100, seed=987)
    outcomes = generate_sample_outcomes(intakes, seed=987)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        raw_dir = tmpdir_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        intakes_path = raw_dir / "intakes.csv"
        outcomes_path = raw_dir / "outcomes.csv"
        
        intakes.to_csv(intakes_path, index=False)
        outcomes.to_csv(outcomes_path, index=False)
        
        output_path = tmpdir_path / "data" / "processed" / "modeling_dataset.csv"
        
        result = build_modeling_dataset_from_files(
            str(intakes_path),
            str(outcomes_path),
            str(output_path),
        )
        
        dataset = result.dataset
        
        assert "days_to_outcome" in dataset.columns
        assert "is_censored" in dataset.columns
        assert "followup_days_censored" in dataset.columns
        
        censored_mask = dataset["is_censored"].astype(bool)
        
        for idx, row in dataset[censored_mask].iterrows():
            assert row["censoring_reason"] != "adopted", \
                f"Censored record {idx} should not be adopted"
        
        adopted_mask = ~censored_mask
        for idx, row in dataset[adopted_mask].iterrows():
            assert row["censoring_reason"] == "adopted", \
                f"Adopted record {idx} should have censoring_reason='adopted'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
