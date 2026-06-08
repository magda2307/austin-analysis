import os
import json
from pathlib import Path
from aac_adoption.provenance import get_current_run_context, write_producer_receipt

def test_write_producer_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("AAC_RUN_ID", "test-run-123")
    monkeypatch.setenv("AAC_PRODUCER_SOURCE_SHA", "test-sha")
    monkeypatch.setenv("AAC_RUN_PROFILE", "thesis-full")
    monkeypatch.setenv("AAC_RECEIPTS_DIR", str(tmp_path / "receipts"))
    
    # Create mock output
    out_file = tmp_path / "output.csv"
    out_file.write_text("test")
    
    context = get_current_run_context(command=["test", "cmd"], inputs=[])
    
    receipt_path = write_producer_receipt("test_step", context, [out_file])
    assert receipt_path.exists()
    
    with open(receipt_path) as f:
        data = json.load(f)
        
    assert data["run_id"] == "test-run-123"
    assert data["producer_source_sha"] == "test-sha"
    assert data["profile"] == "thesis-full"
    assert str(out_file) in data["output_hashes"]
    assert data["command"] == ["test", "cmd"]
    assert data["status"] == "ok"
