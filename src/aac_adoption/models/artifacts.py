"""Model artifact path and persistence helpers."""

import json
from pathlib import Path
import re
from typing import Any

import joblib
from aac_adoption.models.metadata import validate_model_metadata, compute_file_sha256


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
    model_temp = path.with_name(f".{path.name}.tmp")
    sidecar_path = path.with_suffix(".json")
    sidecar_temp = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    model_backup = path.with_name(f".{path.name}.bak")
    sidecar_backup = sidecar_path.with_name(f".{sidecar_path.name}.bak")

    sidecar_metadata = metadata.copy()
    sidecar_metadata["artifact_path"] = str(path)
    sidecar_metadata["artifact_sha256"] = "pending"
    validate_model_metadata(sidecar_metadata)

    success = False
    try:
        joblib.dump(pipeline, model_temp)
        sidecar_metadata["artifact_sha256"] = compute_file_sha256(model_temp)
        sidecar_temp.write_text(
            json.dumps(sidecar_metadata, indent=2, default=str),
            encoding="utf-8",
        )

        model_backup.unlink(missing_ok=True)
        sidecar_backup.unlink(missing_ok=True)
        if path.exists():
            path.replace(model_backup)
        if sidecar_path.exists():
            sidecar_path.replace(sidecar_backup)

        model_temp.replace(path)
        sidecar_temp.replace(sidecar_path)
        success = True
    except Exception:
        path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        if model_backup.exists():
            model_backup.replace(path)
        if sidecar_backup.exists():
            sidecar_backup.replace(sidecar_path)
        raise
    finally:
        model_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
        if success:
            model_backup.unlink(missing_ok=True)
            sidecar_backup.unlink(missing_ok=True)

    return path
