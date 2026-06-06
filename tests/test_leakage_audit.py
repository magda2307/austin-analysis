"""Tests that no leakage columns appear in training feature sets and risk levels are assigned correctly.

Auto-skips data checks if the modeling dataset is not yet built.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import pytest

from aac_adoption.data.leakage_audit import audit_leakage_columns, DataLeakageError

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_audit_leakage_columns_risk_levels():
    """Test that audit correctly assigns risk levels to features."""
    df = pd.DataFrame({
        "safe_col": [1, 2, 3],
        "intake_condition": ["Normal", "Injured", "Adopt"],  # Adopt makes it needs_audit
        "sex_upon_intake": ["Male", "Female", "Spayed Female"],  # Spayed makes it needs_audit
    })
    
    # We add a constant column to trigger unsafe
    df["constant_col"] = "same"
    
    # Fake LEAKAGE_COLUMNS_TO_AUDIT by monkeypatching
    import aac_adoption.data.leakage_audit as leakage_audit
    original_audit_cols = leakage_audit.LEAKAGE_COLUMNS_TO_AUDIT
    leakage_audit.LEAKAGE_COLUMNS_TO_AUDIT = ["constant_col"]
    
    try:
        # Unsafe col will raise DataLeakageError
        with pytest.raises(DataLeakageError, match="Unsafe leakage columns detected"):
            audit_leakage_columns(df)
    finally:
        leakage_audit.LEAKAGE_COLUMNS_TO_AUDIT = original_audit_cols
        
    # Drop the constant col to test needs_audit
    df_safe = df.drop(columns=["constant_col"])
    risk_levels = audit_leakage_columns(df_safe)
    
    assert risk_levels["safe_col"] == "safe"
    assert risk_levels["intake_condition"] == "needs_audit"
    assert risk_levels["sex_upon_intake"] == "needs_audit"


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="modeling_dataset.csv not yet built — run scripts/build_dataset.py",
)
def test_no_leakage_columns_in_dataset() -> None:
    from aac_adoption.features.feature_sets import (
        LEAKAGE_COLUMNS,
        available_intake_features,
    )

    df = pd.read_csv(DATA_PATH, nrows=100)
    features = available_intake_features(df.columns.tolist())
    leakage_present = sorted(set(features) & LEAKAGE_COLUMNS)
    assert not leakage_present, (
        f"Leakage columns found in training feature set: {leakage_present}"
    )


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="modeling_dataset.csv not yet built",
)
def test_no_future_columns_in_dataset() -> None:
    df = pd.read_csv(DATA_PATH, nrows=100)
    future_cols = [
        col for col in df.columns
        if "future" in col.lower()
        or "_next_" in col.lower()
        or col.lower().startswith("next_")
    ]
    assert not future_cols, (
        f"Future-leaking column names found in dataset: {future_cols}"
    )
