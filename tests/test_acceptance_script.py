import os
import subprocess
import sys
from pathlib import Path
import tempfile
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_final_acceptance.ps1"

@pytest.fixture
def mock_python(tmp_path):
    # Create a mock python executable using a batch script or a python script wrapper.
    # We will create a small python script and a .bat wrapper to intercept calls.
    # Actually, simpler: create `python.bat` in a temp dir and prepend to PATH.
    
    log_file = tmp_path / "calls.log"
    
    py_script = tmp_path / "mock.py"
    with open(py_script, "w") as f:
        f.write(f"""import sys
with open(r'{log_file}', 'a') as lf:
    lf.write(' '.join(sys.argv[1:]) + '\\n')
sys.exit(0)
""")
    bat_content = f'@echo off\n"{sys.executable}" "{py_script}" %*\nexit /b %errorlevel%\n'
    bat_path = tmp_path / "python.bat"
    with open(bat_path, "w") as f:
        f.write(bat_content)
        
    old_path = os.environ.get("PATH", "")
    new_path = f"{tmp_path};{old_path}"
    
    env = os.environ.copy()
    env["PATH"] = new_path
    
    yield env, log_file

def run_ps1(args, env):
    cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT_PATH)] + args
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result

def test_long_and_skip_pytest_fails(mock_python):
    env, log_file = mock_python
    result = run_ps1(["-Long", "-SkipPytest"], env)
    
    assert result.returncode != 0
    assert "Invalid flags" in result.stderr or "Invalid flags" in result.stdout

def test_long_runs_only_verification(mock_python):
    env, log_file = mock_python
    result = run_ps1(["-Long"], env)
    
    assert result.returncode == 0
    
    if log_file.exists():
        with open(log_file, "r") as f:
            calls = f.read().strip().splitlines()
    else:
        calls = []
        
    # We should see:
    # 1. python scripts/validate_run_receipts.py
    # 2. python -m pytest -q
    # And absolutely NO producer scripts like evaluate_backtesting.py
    
    executed_commands = [c.strip() for c in calls]
    
    assert len(executed_commands) == 2
    assert "scripts/validate_run_receipts.py" in executed_commands[0]
    assert "-m pytest -q" in executed_commands[1]
    
    assert "evaluate_backtesting.py" not in "".join(executed_commands)
    
    # Check that it sets AAC_ACCEPTANCE=1
    assert "AAC_ACCEPTANCE=1" in result.stdout

def test_short_smoke_cleanup(mock_python):
    env, log_file = mock_python
    
    # We want to test that it cleans up the temporary smoke root.
    # If the script finishes, it should print "Cleaned up temporary smoke root"
    result = run_ps1([], env)
    
    assert result.returncode == 0
    assert "Created temporary smoke root" in result.stdout
    assert "Cleaned up temporary smoke root" in result.stdout
    
    if log_file.exists():
        with open(log_file, "r") as f:
            calls = f.read().strip().splitlines()
    else:
        calls = []
        
    executed_commands = "".join(calls)
    assert "evaluate_backtesting.py" in executed_commands
    assert "compare_recency.py" in executed_commands
    
def test_short_smoke_cleanup_on_failure(mock_python, tmp_path):
    env, log_file = mock_python
    
    py_script = tmp_path / "mock.py"
    with open(py_script, "w") as f:
        f.write(f"""import sys
with open(r'{log_file}', 'a') as lf:
    lf.write(' '.join(sys.argv[1:]) + '\\n')
if '--quick' in sys.argv:
    sys.exit(1)
sys.exit(0)
""")
    bat_content = f'@echo off\n"{sys.executable}" "{py_script}" %*\nexit /b %errorlevel%\n'
    bat_path = tmp_path / "python.bat"
    with open(bat_path, "w") as f:
        f.write(bat_content)

    result = run_ps1([], env)
    
    assert result.returncode != 0
    assert "Created temporary smoke root" in result.stdout
    assert "Cleaned up temporary smoke root" in result.stdout
