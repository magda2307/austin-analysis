"""Tests for data audit and dataset build outputs."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"

# Tables that must exist after running the pipeline
REQUIRED_TABLES = [
    ("model_comparison_classification.csv", ["model_name", "task", "animal_subset"]),
    ("model_comparison_regression.csv", ["model_name", "task", "animal_subset"]),
    ("h1_intake_vs_appearance.csv", []),
    ("h3_age_adoption_speed.csv", []),
    ("h5_covid_period.csv", []),
    ("subgroup_reliability.csv", ["cohort"]),
    ("metric_confidence_intervals.csv", ["metric"]),
]


@pytest.mark.parametrize("filename,required_cols", REQUIRED_TABLES)
def test_required_table_exists_and_nonempty(filename: str, required_cols: list[str]) -> None:
    path = TABLES_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not yet generated")
    df = pd.read_csv(path)
    assert not df.empty, f"{filename} is empty"
    for col in required_cols:
        assert col in df.columns, f"{filename} missing column: {col}"


def test_shap_global_classification_has_family_column() -> None:
    path = TABLES_DIR / "shap_global_classification.csv"
    if not path.exists():
        pytest.skip("shap_global_classification.csv not yet generated")
    df = pd.read_csv(path)
    assert "feature_family" in df.columns, (
        "shap_global_classification.csv must have a feature_family column for interpretation"
    )


def test_shap_global_regression_has_family_column() -> None:
    path = TABLES_DIR / "shap_global_regression.csv"
    if not path.exists():
        pytest.skip("shap_global_regression.csv not yet generated")
    df = pd.read_csv(path)
    assert "feature_family" in df.columns, (
        "shap_global_regression.csv must have a feature_family column for interpretation"
    )
