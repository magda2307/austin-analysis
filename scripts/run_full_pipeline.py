"""One-command thesis pipeline runner.

Usage examples:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-download --quick
    python scripts/run_full_pipeline.py --skip-shap
    python scripts/run_full_pipeline.py --steps 4,5,6
    python scripts/run_full_pipeline.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
DATA_ARG = "data/processed/modeling_dataset.csv"

# -------------------------------------------------------------------------
# Pipeline step definitions
# Each step: (step_number, name, command_parts, tag)
# tag can be: "download", "shap", "expensive", or None (always runs)
# -------------------------------------------------------------------------
STEPS = [
    (
        0,
        "Environment snapshot",
        [sys.executable, "scripts/generate_environment_snapshot.py"],
        None,
    ),
    (
        1,
        "Download raw data",
        [sys.executable, "scripts/download_raw_data.py"],
        "download",
    ),
    (
        2,
        "Build dataset",
        [
            sys.executable,
            "scripts/build_dataset.py",
            "--intakes", "data/raw/intakes.csv",
            "--outcomes", "data/raw/outcomes.csv",
            "--output", DATA_ARG,
            "--context-data-dir", "data/raw",
        ],
        None,
    ),
    (
        3,
        "Run EDA",
        [sys.executable, "scripts/run_eda.py", "--data", DATA_ARG],
        None,
    ),
    (
        4,
        "Train baseline models",
        [sys.executable, "scripts/train_baselines.py", "--data", DATA_ARG],
        None,
    ),
    (
        5,
        "Train adopted animals regression",
        [sys.executable, "scripts/train_adopted_regression.py", "--data", DATA_ARG],
        None,
    ),
    (
        6,
        "Tune hyperparameters",
        [sys.executable, "scripts/tune_models.py", "--data-path", DATA_ARG],
        "expensive",
    ),
    (
        7,
        "Train boosting models",
        [sys.executable, "scripts/train_boosting.py", "--data", DATA_ARG, "--tuned-params-path", "models/tuning/best_params.json"],
        None,
    ),
    (
        8,
        "Train advanced models (CatBoost)",
        [sys.executable, "scripts/train_advanced.py", "--data", DATA_ARG, "--tuned-params-path", "models/tuning/best_params.json"],
        "expensive",
    ),
    (
        9,
        "Calibrate classifiers",
        [sys.executable, "scripts/calibrate_classifiers.py", "--data-path", DATA_ARG],
        None,
    ),
    (
        10,
        "Run analysis",
        [sys.executable, "scripts/run_analysis.py", "--data", DATA_ARG],
        None,
    ),
    (
        11,
        "Generate diagnostics (with SHAP)",
        [
            sys.executable,
            "scripts/generate_diagnostics.py",
            "--data", DATA_ARG,
            "--include-shap",
        ],
        "shap",
    ),
    (
        12,
        "Generate animal research",
        [sys.executable, "scripts/generate_animal_research.py", "--data", DATA_ARG],
        None,
    ),
    (
        13,
        "Generate evidence pack",
        [sys.executable, "scripts/generate_evidence_pack.py", "--data", DATA_ARG],
        None,
    ),
    (
        14,
        "Generate report outputs",
        [sys.executable, "scripts/generate_report_outputs.py"],
        None,
    ),
    (
        15,
        "Generate feature family importance",
        [sys.executable, "scripts/generate_feature_family_importance.py"],
        "shap",
    ),
    (
        16,
        "Evaluate backtesting",
        [sys.executable, "scripts/evaluate_backtesting.py"],
        "expensive",
    ),
    (
        17,
        "Run test suite",
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        None,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command thesis pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run the full pipeline:
    python scripts/run_full_pipeline.py

  Skip download, skip SHAP-heavy steps:
    python scripts/run_full_pipeline.py --skip-download --skip-shap

  Quick iteration (skip download + SHAP + tests):
    python scripts/run_full_pipeline.py --quick

  Run only specific steps by number:
    python scripts/run_full_pipeline.py --steps 4,5,6
""",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip step 1 (data download).",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP-heavy steps (11, 15).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip download (1), SHAP (11, 15), and tests (17).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated list of step numbers to run (e.g. 4,5,6). Overrides skip flags.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Path to write the run log. Defaults to logs/pipeline_TIMESTAMP.log.",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        dest="run_id",
        help="Resume an existing run ID instead of starting a new one.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining steps even if one fails.",
    )
    return parser.parse_args()


def should_skip(step_number: int, tag: str | None, args: argparse.Namespace, only_steps: set[int]) -> bool:
    if only_steps:
        return step_number not in only_steps
    if tag == "download" and (args.skip_download or args.quick):
        return True
    if tag == "shap" and (args.skip_shap or args.quick):
        return True
    if step_number == 17 and args.quick:
        return True
    return False


def main() -> None:
    args = parse_args()

    only_steps: set[int] = set()
    if args.steps:
        only_steps = {int(s.strip()) for s in args.steps.split(",") if s.strip()}

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Run ID setup
    try:
        shortsha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        shortsha = "unknown"

    run_id = getattr(args, "run_id", None)
    if not run_id:
        run_id = f"{timestamp}-{shortsha}"
    
    receipt_path = ROOT / "reports" / "run_receipt.json"
    receipt_data = {
        "run_id": run_id,
        "producer_source_sha": shortsha,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "completed_at": None,
        "status": "running",
        "executed_steps": [],
        "skipped_steps": [],
        "failed_step": None,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2)

    log_path = Path(args.log_file) if args.log_file else ROOT / "logs" / f"pipeline_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        def _log(msg: str) -> None:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        def update_receipt(failed: bool = False):
            if failed:
                receipt_data["status"] = "failed"
                failed_items = [r["step"] for r in results if r["status"] == "failed"]
                if failed_items:
                    receipt_data["failed_step"] = failed_items[0]
            else:
                receipt_data["status"] = "ok"
                receipt_data["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
            
            receipt_data["executed_steps"] = [r["step"] for r in results if r["status"] == "ok"]
            receipt_data["skipped_steps"] = [r["step"] for r in results if r["status"] == "skipped"]
            
            if not failed:
                tmp_receipt = receipt_path.with_suffix(".tmp")
                with tmp_receipt.open("w", encoding="utf-8") as f:
                    json.dump(receipt_data, f, indent=2)
                tmp_receipt.replace(receipt_path)
            else:
                with receipt_path.open("w", encoding="utf-8") as f:
                    json.dump(receipt_data, f, indent=2)

        _log(f"=== AAC Thesis Pipeline Run ===")
        _log(f"Started:  {timestamp}")
        _log(f"Root:     {ROOT}")
        _log(f"Run ID:   {run_id}")
        _log(f"Log file: {log_path}")
        if only_steps:
            _log(f"WARNING:  Running partial pipeline with steps {sorted(list(only_steps))}. This may skip required dependencies.")
        _log("")

        for step_number, name, cmd, tag in STEPS:
            if should_skip(step_number, tag, args, only_steps):
                msg = f"[STEP {step_number:02d}] SKIPPED  {name}"
                _log(msg)
                results.append({"step": step_number, "name": name, "status": "skipped"})
                continue

            _log(f"[STEP {step_number:02d}] RUNNING  {name}")
            _log(f"          cmd: {' '.join(cmd)}")

            # Pass run context via environment variables
            run_env = os.environ.copy()
            run_env["AAC_RUN_ID"] = run_id

            # Discover inputs from command arguments
            inputs = []
            for arg in cmd:
                if str(arg).endswith(".csv") or str(arg).endswith(".json"):
                    if (ROOT / arg).exists():
                        inputs.append(ROOT / arg)

            # Scan state before
            def scan_state():
                state = {}
                for d in ["data", "models", "reports"]:
                    p = ROOT / d
                    if p.exists():
                        for f in p.rglob("*"):
                            if f.is_file() and ".tmp" not in f.name and not f.name.endswith(".log") and "run_receipts" not in str(f) and f.name != "run_receipt.json":
                                try:
                                    state[str(f.relative_to(ROOT))] = f.stat().st_mtime
                                except Exception:
                                    pass
                return state

            before_state = scan_state()

            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=run_env,
                capture_output=False,
                text=True,
            )

            after_state = scan_state()

            # Find changed or new files
            outputs = []
            for path, mtime in after_state.items():
                if path not in before_state or before_state[path] != mtime:
                    outputs.append(ROOT / path)

            # Write receipt
            import sys
            sys.path.insert(0, str(ROOT / "src"))
            from aac_adoption.provenance import get_current_run_context, write_producer_receipt
            
            ctx = get_current_run_context(command=cmd, inputs=inputs)
            safe_name = Path(cmd[1]).stem if len(cmd) > 1 else f"step_{step_number}"
            status_str = "ok" if proc.returncode == 0 else "error"
            err_msg = f"Exit code {proc.returncode}" if proc.returncode != 0 else None
            write_producer_receipt(f"{step_number:02d}-{safe_name}", ctx, outputs, status=status_str, error_message=err_msg)

            if proc.returncode == 0:
                status = "ok"
                _log(f"[STEP {step_number:02d}] ✓ OK      {name}")
            else:
                status = "failed"
                _log(f"[STEP {step_number:02d}] ✗ FAILED  {name} (exit code {proc.returncode})")

            results.append({"step": step_number, "name": name, "status": status})
            _log("")

            if proc.returncode != 0 and not args.continue_on_error:
                _log(f"Aborting pipeline on step {step_number} failure.")
                break

        failed = [r for r in results if r["status"] == "failed"]
        update_receipt(failed=bool(failed))

        _log("=== Pipeline Summary ===")
        for r in results:
            icon = {"ok": "✓", "failed": "✗", "skipped": "—"}.get(r["status"], "?")
            _log(f"  {icon} Step {r['step']:02d}: {r['name']} [{r['status']}]")

        _log("")
        if failed:
            _log(f"RESULT: {len(failed)} step(s) failed.")
            _log(f"Log written to: {log_path}")
            sys.exit(1)
        else:
            _log("RESULT: All executed steps completed successfully.")
            _log(f"Log written to: {log_path}")


if __name__ == "__main__":
    main()
