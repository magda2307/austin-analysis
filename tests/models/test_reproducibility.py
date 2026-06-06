"""Golden end-to-end metric reproducibility test to ensure no upstream refactoring breaks model semantics."""

from pathlib import Path

import pandas as pd
import pytest

from aac_adoption.models.train_baseline import limit_rows
from aac_adoption.models.train_boosting import train_boosting_classification


def test_boosting_classification_reproducibility(tmp_path: Path):
    """Ensure ROC-AUC for HistGradientBoosting matches the golden value exactly."""
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        pytest.skip(f"Modeling dataset not found at {data_path}")

    # Use a small deterministic slice of data
    df = pd.read_csv(data_path, parse_dates=["intake_datetime", "outcome_datetime"])
    df = limit_rows(df, 500)

    models_dir = tmp_path / "models"
    tables_dir = tmp_path / "tables"

    # Train boosting
    results = train_boosting_classification(
        df=df,
        models_dir=models_dir,
        tables_dir=tables_dir,
        run_timestamp="2024-01-01T00:00:00Z",
        permutation_repeats=1,
        permutation_max_rows=100,
    )

    combined_result = next(r for r in results if r["animal_subset"] == "combined")
    
    # Assert exactly 5 decimal places match to ensure absolute deterministic output
    assert round(combined_result["roc_auc"], 5) == 0.66029
