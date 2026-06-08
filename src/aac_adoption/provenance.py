import json
import os
import sys
import uuid
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class RunContext:
    run_id: str
    producer_source_sha: str
    profile: str
    started_at: str
    command: list[str]
    input_hashes: dict[str, str]

def compute_file_sha256(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        return "unavailable_not_found"
    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_current_run_context(command: list[str] | None = None, inputs: list[str | Path] | None = None) -> RunContext:
    run_id = os.environ.get("AAC_RUN_ID", f"dev-{uuid.uuid4().hex[:8]}")
    profile = os.environ.get("AAC_RUN_PROFILE", "development-no-shap")
    source_sha = os.environ.get("AAC_PRODUCER_SOURCE_SHA", "unavailable")
    
    input_hashes = {}
    if inputs:
        for inp in inputs:
            p = Path(inp)
            input_hashes[str(p)] = compute_file_sha256(p)
            
    return RunContext(
        run_id=run_id,
        producer_source_sha=source_sha,
        profile=profile,
        started_at=datetime.now(timezone.utc).isoformat(),
        command=command or sys.argv,
        input_hashes=input_hashes
    )

def write_producer_receipt(
    step_name: str,
    context: RunContext,
    outputs: list[str | Path] | None = None,
    status: str = "ok",
    error_message: str | None = None
) -> Path | None:
    receipts_dir = os.environ.get("AAC_RECEIPTS_DIR")
    if not receipts_dir:
        # Development mode without explicit receipt dir
        return None
        
    out_dir = Path(receipts_dir) / context.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-discover outputs from ARTIFACT_METADATA if not provided
    if outputs is None:
        outputs = []
        try:
            # We must import dynamically to avoid circular dependencies if any
            import sys
            import importlib.util
            project_root = Path(__file__).resolve().parents[2]
            manifest_script = project_root / "scripts" / "generate_artifact_manifest.py"
            spec = importlib.util.spec_from_file_location("manifest", str(manifest_script))
            manifest_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(manifest_mod)
            for path, meta in manifest_mod.ARTIFACT_METADATA.items():
                if meta.get("source_script", "").startswith(f"scripts/{step_name}.py"):
                    outputs.append(project_root / path)
        except Exception as e:
            print(f"Warning: could not auto-discover outputs for {step_name}: {e}")
    
    output_hashes = {}
    for out in outputs:
        p = Path(out)
        output_hashes[str(p)] = compute_file_sha256(p)
        
    receipt = {
        "run_id": context.run_id,
        "producer_source_sha": context.producer_source_sha,
        "profile": context.profile,
        "started_at": context.started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": context.command,
        "input_hashes": context.input_hashes,
        "output_hashes": output_hashes,
        "status": status,
        "error_message": error_message
    }
    
    safe_step_name = step_name.replace(" ", "_").replace("/", "_").lower()
    temp_path = out_dir / f".tmp_{safe_step_name}.json"
    final_path = out_dir / f"{safe_step_name}.json"
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        
    temp_path.replace(final_path)
    return final_path
