from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _scalar_metadata(metadata: dict) -> dict:
    """Convert list/dict metadata values to JSON strings for safe DataFrame broadcasting."""
    out = {}
    for key, val in metadata.items():
        if isinstance(val, (list, dict)):
            out[key] = json.dumps(val)
        else:
            out[key] = val
    return out

def get_feature_names(pipeline) -> list[str]:
    """Get transformed feature names from fitted sklearn pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    return [str(name).replace("numeric__", "").replace("categorical__", "") for name in names]


def logistic_regression_coefficients(pipeline, metadata: dict) -> pd.DataFrame:
    """Create coefficient table for fitted LogisticRegression pipeline."""
    model = pipeline.named_steps["model"]
    features = get_feature_names(pipeline)
    coefficients = model.coef_[0]
    table = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    for key, val in _scalar_metadata(metadata).items():
        table[key] = val
    return table.sort_values("abs_coefficient", ascending=False)


def random_forest_feature_importance(pipeline, metadata: dict) -> pd.DataFrame:
    """Create feature importance table for fitted RandomForest pipeline."""
    model = pipeline.named_steps["model"]
    features = get_feature_names(pipeline)
    table = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )
    for key, val in _scalar_metadata(metadata).items():
        table[key] = val
    return table.sort_values("importance", ascending=False)


def append_table(table: pd.DataFrame, path: str | Path) -> None:
    """Append table to CSV, writing header only for new file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, mode="a", header=not output.exists(), index=False)

