"""Read-only validation for final thesis artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

MANUAL_SOURCE_POLICY = "manual-doc"
REQUIRED_COLUMNS = {
    "artifact_path",
    "source_script",
    "required_for_thesis",
    "run_id",
    "producer_source_sha",
    "file_hash",
}


class AcceptanceError(ValueError):
    """Raised when artifact acceptance evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class AcceptanceResult:
    run_id: str
    producer_source_sha: str
    required_artifact_count: int


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_relative_path(value: object) -> str:
    raw = str(value).strip().replace("\\", "/")
    normalized = os.path.normpath(raw).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not raw or path.is_absolute() or normalized == ".." or normalized.startswith("../"):
        raise AcceptanceError(f"artifact path is not a safe relative path: {raw!r}")
    return path.as_posix()


def resolve_source_path(root: Path, source: object) -> Path | None:
    source_text = str(source).strip()
    if source_text == MANUAL_SOURCE_POLICY:
        return None
    try:
        command = shlex.split(source_text, posix=True)
    except ValueError as exc:
        raise AcceptanceError(f"invalid artifact source command: {source_text}") from exc
    if not command:
        raise AcceptanceError("artifact source command is empty")
    return root / normalize_relative_path(command[0])


def _is_required(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _receipt_output_path(root: Path, value: object) -> str:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise AcceptanceError(f"receipt output is outside fixture root: {value}") from exc


def _load_receipts(
    root: Path, receipts_dir: Path, run_id: str
) -> tuple[list[dict], dict[str, set[str]], str]:
    if not receipts_dir.is_dir():
        raise AcceptanceError(f"unknown run or no receipt directory: {run_id}")

    receipt_paths = sorted(receipts_dir.glob("*.json"))
    if not receipt_paths:
        raise AcceptanceError(f"no receipt files found for run_id {run_id}")

    receipts = []
    output_hashes: dict[str, set[str]] = {}
    source_shas: set[str] = set()
    for receipt_path in receipt_paths:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcceptanceError(f"invalid receipt: {receipt_path}") from exc
        if receipt.get("status") != "ok":
            raise AcceptanceError(f"receipt status is not ok: {receipt_path}")
        if receipt.get("run_id") != run_id:
            raise AcceptanceError(
                f"receipt run_id {receipt.get('run_id')!r} does not match {run_id!r}"
            )
        if receipt.get("profile") != "thesis-full":
            raise AcceptanceError(
                f"receipt profile must be thesis-full: {receipt_path}"
            )
        source_sha = str(receipt.get("producer_source_sha", "")).strip()
        if not source_sha:
            raise AcceptanceError(f"receipt producer_source_sha is missing: {receipt_path}")
        source_shas.add(source_sha)
        for output_path, output_hash in receipt.get("output_hashes", {}).items():
            relative = _receipt_output_path(root, output_path)
            output_hashes.setdefault(relative, set()).add(str(output_hash))
        receipts.append(receipt)

    if len(source_shas) != 1:
        raise AcceptanceError("receipts contain mixed producer_source_sha values")
    return receipts, output_hashes, source_shas.pop()


def validate_artifact_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    run_id: str,
    receipts_dir: str | Path | None = None,
) -> AcceptanceResult:
    """Validate a manifest and its selected run receipts without mutating disk."""

    root_path = Path(root).resolve()
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise AcceptanceError(f"manifest does not exist: {manifest}")

    frame = pd.read_csv(manifest)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise AcceptanceError(
            f"manifest is missing required columns: {sorted(missing_columns)}"
        )
    required = frame[frame["required_for_thesis"].map(_is_required)].copy()
    if required.empty:
        raise AcceptanceError("manifest has no required thesis artifacts")

    normalized_paths = [
        normalize_relative_path(value) for value in required["artifact_path"]
    ]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise AcceptanceError("manifest contains duplicate normalized required paths")
    required["_normalized_path"] = normalized_paths

    selected_receipts = (
        Path(receipts_dir).resolve()
        if receipts_dir is not None
        else root_path / "reports" / "run_receipts" / run_id
    )
    _, receipt_hashes, receipt_source_sha = _load_receipts(
        root_path, selected_receipts, run_id
    )

    manifest_mtime = manifest.stat().st_mtime_ns
    for row in required.to_dict(orient="records"):
        relative = row["_normalized_path"]
        artifact = root_path / relative
        if not artifact.is_file():
            raise AcceptanceError(f"required artifact does not exist: {relative}")
        if artifact.stat().st_size == 0:
            raise AcceptanceError(f"required artifact is empty: {relative}")
        if artifact.stat().st_mtime_ns > manifest_mtime:
            raise AcceptanceError(f"manifest is older than required artifact: {relative}")

        source = str(row["source_script"]).strip()
        source_path = resolve_source_path(root_path, source)
        if source_path is not None and not source_path.is_file():
            raise AcceptanceError(f"artifact source does not exist: {source}")

        if str(row["run_id"]).strip() != run_id:
            raise AcceptanceError(
                f"manifest run_id does not match explicit run_id for {relative}"
            )
        if str(row["producer_source_sha"]).strip() != receipt_source_sha:
            raise AcceptanceError(
                f"manifest producer_source_sha does not match receipts for {relative}"
            )

        disk_hash = compute_sha256(artifact)
        manifest_hash = str(row["file_hash"]).strip()
        if manifest_hash != disk_hash:
            raise AcceptanceError(f"manifest disk hash mismatch for {relative}")
        hashes = receipt_hashes.get(relative)
        if not hashes:
            raise AcceptanceError(f"no receipt output found for required artifact: {relative}")
        if hashes != {disk_hash}:
            raise AcceptanceError(f"receipt hash mismatch for {relative}")

    return AcceptanceResult(
        run_id=run_id,
        producer_source_sha=receipt_source_sha,
        required_artifact_count=len(required),
    )
