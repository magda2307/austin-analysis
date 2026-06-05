"""Tests that no leakage columns appear in training feature sets.

Auto-skips if the modeling dataset is not yet built.
"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="modeling_dataset.csv not yet built — run scripts/build_dataset.py",
)
def test_no_leakage_columns_in_dataset() -> None:
    import pandas as pd
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
    import pandas as pd

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
