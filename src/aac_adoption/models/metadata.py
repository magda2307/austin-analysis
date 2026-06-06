"""Shared model training metadata helpers."""

from typing import Any
from aac_adoption.config import RANDOM_STATE
from aac_adoption.features.feature_sets import feature_set_label
from aac_adoption.models.split import DatasetSplit


def base_training_metadata(
    *,
    model_name: str,
    task: str,
    split: DatasetSplit,
    feature_columns: list[str],
    run_timestamp: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a standard metadata dictionary for any trained model."""
    result = {
        "model_name": model_name,
        "task": task,
        "animal_subset": split.animal_subset,
        "split_strategy": split.strategy,
        "train_period": split.train_period,
        "validation_period": split.validation_period,
        "test_period": split.test_period,
        "feature_set": feature_set_label(feature_columns),
        "random_state": RANDOM_STATE,
        "run_timestamp": run_timestamp,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
    }
    result.update(extra)
    return result
