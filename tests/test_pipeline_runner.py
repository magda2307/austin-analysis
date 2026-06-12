import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to sys.path to import scripts
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_full_pipeline as runner


@pytest.fixture
def mock_subprocess(monkeypatch):
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.returncode = 0
    mock.return_value = mock_run
    monkeypatch.setattr(subprocess, "run", mock)
    
    mock_check_output = MagicMock(return_value="deadbeef")
    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    return mock


@pytest.fixture
def mock_write_receipt(monkeypatch):
    mock_prov = MagicMock()
    monkeypatch.setitem(sys.modules, "aac_adoption.provenance", mock_prov)
    return mock_prov


@pytest.fixture
def mock_file_io(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    return tmp_path


def test_pipeline_stops_on_first_failure(mock_subprocess, mock_write_receipt, mock_file_io, monkeypatch):
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.returncode = 1 if call_count == 2 else 0
        return m
        
    mock_subprocess.side_effect = side_effect
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py"])
    
    with pytest.raises(SystemExit) as exc:
        runner.main()
        
    assert exc.value.code == 1
    # Stopped on 2nd step
    assert mock_subprocess.call_count == 2


def test_pipeline_quick_mode_skips_correct_steps(mock_subprocess, mock_write_receipt, mock_file_io, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--quick"])
    
    runner.main()
        
    executed_commands = []
    for call in mock_subprocess.call_args_list:
        cmd = call.args[0] if call.args else call.kwargs.get("args", [])
        executed_commands.append(" ".join([str(x) for x in cmd]))
        
    executed_joined = "\n".join(executed_commands)
    
    # Should skip download (step 1), shap (step 11, 15), tests (step 18)
    assert "download_raw_data.py" not in executed_joined
    assert "generate_diagnostics.py" not in executed_joined
    assert "generate_feature_family_importance.py" not in executed_joined
    assert "pytest" not in executed_joined
    
    # Should not skip backtesting (step 16) or validate run receipts (step 17)
    assert "evaluate_backtesting.py" in executed_joined
    assert "validate_run_receipts.py" in executed_joined


def test_pipeline_has_no_final_manifest_step():
    for step_num, name, cmd, tag in runner.STEPS:
        cmd_str = " ".join([str(x) for x in cmd])
        assert "generate_artifact_manifest.py" not in cmd_str


def test_pipeline_download_step_is_repeatable():
    download_step = next(step for step in runner.STEPS if step[0] == 1)

    assert "--overwrite" in download_step[2]


def test_pipeline_receipt_validation_allows_in_progress_overall_receipt():
    validation_step = next(step for step in runner.STEPS if step[0] == 17)

    assert "--allow-running" in validation_step[2]


def test_pipeline_steps_declares_partial_run(mock_subprocess, mock_write_receipt, mock_file_io, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--steps", "4,5"])
    
    runner.main()
        
    assert mock_subprocess.call_count == 2
    
    logs_dir = mock_file_io / "logs"
    log_files = list(logs_dir.glob("*.log"))
    assert len(log_files) == 1
    
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "WARNING:  Running partial pipeline with steps [4, 5]" in log_content


def test_pipeline_propagates_complete_run_context(mock_subprocess, mock_write_receipt, mock_file_io, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--steps", "9"])

    runner.main()

    env = mock_subprocess.call_args.kwargs["env"]
    assert env["AAC_RUN_ID"]
    assert env["AAC_PRODUCER_SOURCE_SHA"] == "deadbeef"
    assert env["AAC_RUN_PROFILE"] == "thesis-full"
    assert env["AAC_RECEIPTS_DIR"] == str(mock_file_io / "reports" / "run_receipts")
