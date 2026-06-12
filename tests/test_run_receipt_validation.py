import json
import pytest
from pathlib import Path

from aac_adoption.run_receipt_validation import validate_run_receipts

@pytest.fixture
def mock_workspace(tmp_path, monkeypatch):
    # Mock __file__ resolution inside validate_run_receipts
    # validate_run_receipts uses: Path(__file__).resolve().parents[2]
    # We can patch Path(__file__) by monkeypatching, but maybe it's easier to mock project_root directly,
    # or just create the exact structure relative to a fake root.
    
    # We will mock the `project_root` inside the module
    import aac_adoption.run_receipt_validation
    monkeypatch.setattr(aac_adoption.run_receipt_validation, "__file__", str(tmp_path / "src" / "aac_adoption" / "run_receipt_validation.py"))
    
    # Create necessary dirs
    (tmp_path / "src" / "aac_adoption").mkdir(parents=True, exist_ok=True)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "run_receipts").mkdir()
    
    return tmp_path

def create_overall_receipt(workspace, run_id, sha="sha123", profile="thesis-full", status="ok", skipped=None, failed=None):
    reports = workspace / "reports"
    data = {
        "run_id": run_id,
        "profile": profile,
        "producer_source_sha": sha,
        "status": status,
        "skipped_steps": skipped or [],
        "failed_step": failed
    }
    with open(reports / "run_receipt.json", "w") as f:
        json.dump(data, f)
        
def create_step_receipt(workspace, run_id, step_name, sha="sha123", profile="thesis-full", status="ok"):
    run_dir = workspace / "reports" / "run_receipts" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "run_id": run_id,
        "profile": profile,
        "producer_source_sha": sha,
        "status": status,
    }
    with open(run_dir / f"{step_name}.json", "w") as f:
        json.dump(data, f)

def test_valid_thesis_full(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id)
    create_step_receipt(mock_workspace, run_id, "step_1")
    create_step_receipt(mock_workspace, run_id, "step_2")
    
    assert validate_run_receipts() is True

def test_partial_receipts(mock_workspace):
    # Overall is OK, but one step receipt is missing or status is not ok? Wait, partial means missing.
    # The logic requires at least one step receipt. What if there's no step receipts?
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id)
    # No step receipts created
    assert validate_run_receipts() is False

def test_skip_heavy(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, skipped=["step_2"])
    create_step_receipt(mock_workspace, run_id, "step_1")
    assert validate_run_receipts() is False

def test_continue_on_error(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, failed="step_1", status="error")
    create_step_receipt(mock_workspace, run_id, "step_1", status="failed")
    create_step_receipt(mock_workspace, run_id, "step_2")
    assert validate_run_receipts() is False

def test_failed_overall(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, status="failed")
    create_step_receipt(mock_workspace, run_id, "step_1")
    assert validate_run_receipts() is False

def test_running_overall_requires_explicit_internal_mode(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, status="running")
    create_step_receipt(mock_workspace, run_id, "step_1")

    assert validate_run_receipts() is False
    assert validate_run_receipts(allow_running=True) is True

def test_mixed_sha(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, sha="sha123")
    create_step_receipt(mock_workspace, run_id, "step_1", sha="sha123")
    create_step_receipt(mock_workspace, run_id, "step_2", sha="sha456") # different SHA
    assert validate_run_receipts() is False

def test_wrong_profile(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id, profile="thesis-quick")
    create_step_receipt(mock_workspace, run_id, "step_1", profile="thesis-quick")
    assert validate_run_receipts() is False

def test_incomplete_step_receipt(mock_workspace):
    # Step receipt has "status": "running" or something
    run_id = "run-123"
    create_overall_receipt(mock_workspace, run_id)
    create_step_receipt(mock_workspace, run_id, "step_1", status="running")
    assert validate_run_receipts() is False

def test_explicit_run_id_mismatch(mock_workspace):
    run_id = "run-123"
    create_overall_receipt(mock_workspace, "run-456") # latest is 456
    create_step_receipt(mock_workspace, "run-456", "step_1")
    
    # Try to validate run-123, which does not match run_receipt.json
    assert validate_run_receipts(run_id="run-123") is False
