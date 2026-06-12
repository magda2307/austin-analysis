"""Shared model training metadata helpers."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import feature_set_label
from aac_adoption.models.split import DatasetSplit

REQUIRED_MODEL_METADATA = {
    "schema_version",
    "model_name",
    "task",
    "animal_subset",
    "artifact_path",
    "artifact_sha256",
    "dataset_path",
    "dataset_sha256",
    "feature_columns",
    "target_column",
    "target_transform",
    "prediction_inverse_transform",
    "split_strategy",
    "is_thesis_evaluation",
    "train_period",
    "calibration_period",
    "selection_period",
    "test_period",
    "random_state",
    "run_id",
    "run_timestamp",
    "producer_source_sha",
    "packages",
}

def _get_package_versions() -> dict[str, str]:
    versions = {}
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import catboost
        versions["catboost"] = catboost.__version__
    except ImportError:
        pass
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    return versions

def compute_file_sha256(path: str | Path) -> str:
    """Compute real SHA256 hashes using streaming reads."""
    path_obj = Path(path)
    if not path_obj.exists():
        return "unavailable_not_found"
    
    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_source_sha() -> str:
    """Resolve source SHA using a short subprocess call."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"unavailable (git failed: {e})"

def validate_model_metadata(metadata: dict[str, Any]) -> None:
    """Ensure metadata contains all required schema fields."""
    missing = REQUIRED_MODEL_METADATA - set(metadata.keys())
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

def base_training_metadata(
    *,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    run_timestamp: str,
    target_column: str,
    dataset_path: str,
    run_id: str = "dev",
    target_transform: str = "none",
    prediction_inverse_transform: str = "none",
    **extra: Any,
) -> dict[str, Any]:
    """Build a standard metadata dictionary for any trained model."""
    result = {
        "schema_version": "1.0",
        "model_name": model_name,
        "task": task,
        "animal_subset": split.animal_subset or "all",
        "dataset_path": str(dataset_path),
        "dataset_sha256": compute_file_sha256(dataset_path),
        "feature_columns": feature_columns,
        "feature_set": feature_set_label(feature_columns),
        "target_column": target_column,
        "target_transform": target_transform,
        "prediction_inverse_transform": prediction_inverse_transform,
        "split_strategy": split.strategy,
        "is_thesis_evaluation": bool(split.is_thesis_evaluation),
        "train_period": split.train_period,
        "calibration_period": split.calibration_period,
        "selection_period": split.selection_period,
        "test_period": split.test_period,
        "random_state": RANDOM_STATE,
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "producer_source_sha": get_source_sha(),
        "packages": _get_package_versions(),
    }
    result.update(extra)
    return result
