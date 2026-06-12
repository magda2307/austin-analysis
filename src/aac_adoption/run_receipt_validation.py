import json
import os
from pathlib import Path

def validate_run_receipts(run_id=None, receipts_dir=None, allow_running: bool = False) -> bool:
    """
    Validates run receipts for AAC adoption pipeline.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    if receipts_dir is None:
        receipts_base = project_root / "reports" / "run_receipts"
    else:
        receipts_base = Path(receipts_dir)
        
    # If run_id is None, we try to use reports/run_receipt.json
    overall_receipt_path = project_root / "reports" / "run_receipt.json"
    
    if run_id is None:
        if not overall_receipt_path.exists():
            print(f"ERROR: No run_id provided and {overall_receipt_path} not found.")
            return False
            
        try:
            with open(overall_receipt_path, "r", encoding="utf-8") as f:
                overall_data = json.load(f)
            run_id = overall_data.get("run_id")
            if not run_id:
                print("ERROR: run_receipt.json missing run_id")
                return False
        except Exception as e:
            print(f"ERROR reading run_receipt.json: {e}")
            return False
    else:
        # User explicitly provided a run_id. If overall_receipt_path has this run_id, use it,
        # else we can't fully validate the overall pipeline state easily unless we just check steps.
        # But wait, the instruction says: "loads run_receipt.json (overall) and verifies..."
        # If they give a run_id, maybe we should just check if run_receipt.json matches it.
        overall_data = None
        if overall_receipt_path.exists():
            with open(overall_receipt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("run_id") == run_id:
                    overall_data = data
        
        if not overall_data:
            print(f"ERROR: run_receipt.json does not match the requested run_id {run_id} or is missing.")
            return False

    # 1. Verify overall receipt
    allowed_statuses = {"ok", "running"} if allow_running else {"ok"}
    if overall_data.get("status") not in allowed_statuses:
        expected = "'ok' or 'running'" if allow_running else "'ok'"
        print(f"ERROR: overall status is {overall_data.get('status')}, expected {expected}")
        return False
        
    # Wait, overall receipt from run_full_pipeline.py doesn't have "profile" field explicitly.
    # Ah, let's look at `run_full_pipeline.py`'s receipt_data:
    # receipt_data = { "run_id": run_id, "producer_source_sha": shortsha, ... "status": "ok", "skipped_steps": [], "failed_step": None }
    # Let me check if `profile` was added. Oh, run_full_pipeline didn't add profile!
    # Wait, the prompt says: "profile == 'thesis-full'". I need to add profile to `run_receipt.json` in `run_full_pipeline.py` if it doesn't have it, or it might be in step receipts.
    # Actually, let's add profile to overall receipt in run_full_pipeline.py. I'll do that shortly.

    if overall_data.get("profile") != "thesis-full":
        print(f"ERROR: overall profile is {overall_data.get('profile')}, expected 'thesis-full'")
        return False
        
    overall_sha = overall_data.get("producer_source_sha")
    if not overall_sha:
        print("ERROR: overall receipt missing producer_source_sha")
        return False
        
    if overall_data.get("skipped_steps"):
        print(f"ERROR: run has skipped steps: {overall_data.get('skipped_steps')}")
        return False
        
    if overall_data.get("failed_step"):
        print(f"ERROR: run has failed step: {overall_data.get('failed_step')}")
        return False

    # 2. Scan individual step receipts
    run_dir = receipts_base / run_id
    if not run_dir.exists():
        print(f"ERROR: receipts directory for run {run_id} not found: {run_dir}")
        return False
        
    step_files = list(run_dir.glob("*.json"))
    if not step_files:
        print(f"ERROR: no step receipts found in {run_dir}")
        return False
        
    for step_file in step_files:
        try:
            with open(step_file, "r", encoding="utf-8") as f:
                step_data = json.load(f)
                
            if step_data.get("status") != "ok":
                print(f"ERROR: step receipt {step_file.name} has status {step_data.get('status')}")
                return False
                
            if step_data.get("producer_source_sha") != overall_sha:
                print(f"ERROR: step receipt {step_file.name} SHA {step_data.get('producer_source_sha')} does not match overall SHA {overall_sha}")
                return False
                
            if step_data.get("profile") != "thesis-full":
                print(f"ERROR: step receipt {step_file.name} profile {step_data.get('profile')} does not match 'thesis-full'")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to read/validate step receipt {step_file}: {e}")
            return False
            
    # Everything checks out
    return True
