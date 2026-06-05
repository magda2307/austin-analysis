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
        "Train boosting models",
        [sys.executable, "scripts/train_boosting.py", "--data", DATA_ARG],
        None,
    ),
    (
        6,
        "Train advanced models (CatBoost)",
        [sys.executable, "scripts/train_advanced.py", "--data", DATA_ARG],
        "expensive",
    ),
    (
        7,
        "Run analysis",
        [sys.executable, "scripts/run_analysis.py", "--data", DATA_ARG],
        None,
    ),
    (
        8,
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
        9,
        "Generate animal research",
        [sys.executable, "scripts/generate_animal_research.py", "--data", DATA_ARG],
        None,
    ),
    (
        10,
        "Generate evidence pack",
        [sys.executable, "scripts/generate_evidence_pack.py", "--data", DATA_ARG],
        None,
    ),
    (
        11,
        "Generate report outputs",
        [sys.executable, "scripts/generate_report_outputs.py"],
        None,
    ),
    (
        12,
        "Generate feature family importance",
        [sys.executable, "scripts/generate_feature_family_importance.py"],
        "shap",
    ),
    (
        13,
        "Generate artifact manifest",
        [sys.executable, "scripts/generate_artifact_manifest.py"],
        None,
    ),
    (
        14,
        "Run test suite",
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        None,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command thesis pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""
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
        help="Skip SHAP-heavy steps (8, 12).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip download (1), SHAP (8, 12), and tests (14).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated list of step numbers to run (e.g. --steps 4,5,6). Overrides skip flags.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Path to write the run log. Defaults to logs/pipeline_TIMESTAMP.log.",
    )
    return parser.parse_args()


def should_skip(step_number: int, tag: str | None, args: argparse.Namespace, only_steps: set[int]) -> bool:
    if only_steps:
        return step_number not in only_steps
    if tag == "download" and (args.skip_download or args.quick):
        return True
    if tag == "shap" and (args.skip_shap or args.quick):
        return True
    if step_number == 14 and args.quick:
        return True
    return False


def main() -> None:
    args = parse_args()

    only_steps: set[int] = set()
    if args.steps:
        only_steps = {int(s.strip()) for s in args.steps.split(",") if s.strip()}

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(args.log_file) if args.log_file else ROOT / "logs" / f"pipeline_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        def _log(msg: str) -> None:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        _log(f"=== AAC Thesis Pipeline Run ===")
        _log(f"Started:  {timestamp}")
        _log(f"Root:     {ROOT}")
        _log(f"Log file: {log_path}")
        _log("")

        for step_number, name, cmd, tag in STEPS:
            if should_skip(step_number, tag, args, only_steps):
                msg = f"[STEP {step_number:02d}] SKIPPED  {name}"
                _log(msg)
                results.append({"step": step_number, "name": name, "status": "skipped"})
                continue

            _log(f"[STEP {step_number:02d}] RUNNING  {name}")
            _log(f"          cmd: {' '.join(cmd)}")

            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=False,
                text=True,
            )

            if proc.returncode == 0:
                status = "ok"
                _log(f"[STEP {step_number:02d}] ✓ OK      {name}")
            else:
                status = "failed"
                _log(f"[STEP {step_number:02d}] ✗ FAILED  {name} (exit code {proc.returncode})")

            results.append({"step": step_number, "name": name, "status": status})
            _log("")

        _log("=== Pipeline Summary ===")
        for r in results:
            icon = {"ok": "✓", "failed": "✗", "skipped": "—"}.get(r["status"], "?")
            _log(f"  {icon} Step {r['step']:02d}: {r['name']} [{r['status']}]")

        failed = [r for r in results if r["status"] == "failed"]
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
