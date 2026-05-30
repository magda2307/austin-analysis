"""Model artifact path and persistence helpers."""

import json
from pathlib import Path
import re
from typing import Any

import joblib


def safe_name(value: str) -> str:
    """Normalize text for filenames."""
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    return re.sub(r"_+", "_", cleaned).strip("_")


def artifact_path(base_dir: str | Path, task: str, animal_subset: str, model_name: str) -> Path:
    """Return canonical model artifact path."""
    return Path(base_dir) / safe_name(task) / safe_name(animal_subset) / f"{safe_name(model_name)}.joblib"


def save_model_artifact(
    pipeline: Any,
    base_dir: str | Path,
    task: str,
    animal_subset: str,
    model_name: str,
    metadata: dict[str, Any],
) -> Path:
    """Save fitted pipeline and sidecar metadata."""
    path = artifact_path(base_dir, task, animal_subset, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path

