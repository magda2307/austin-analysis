"""Save Python environment version metadata for thesis reproducibility appendix.

Usage:
    python scripts/generate_environment_snapshot.py

Outputs:
    reports/tables/environment_snapshot.csv
    reports/summary/environment_snapshot.md
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TABLES_DIR = ROOT / "reports" / "tables"
SUMMARY_DIR = ROOT / "reports" / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "catboost",
    "shap",
    "streamlit",
    "matplotlib",
    "altair",
    "joblib",
]


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    random_state: int | str = "N/A"
    try:
        from aac_adoption.config import RANDOM_STATE
        random_state = RANDOM_STATE
    except ImportError:
        pass

    row = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "random_state": str(random_state),
    }
    for pkg in PACKAGES:
        key = pkg.replace("-", "_") + "_version"
        row[key] = _pkg_version(pkg)

    df = pd.DataFrame([row])
    csv_out = TABLES_DIR / "environment_snapshot.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # Build Markdown table for thesis appendix
    md_lines = [
        "# Environment Snapshot",
        "",
        f"Generated: {row['run_timestamp']}",
        "",
        "## Python Runtime",
        "",
        f"- **Python version:** {row['python_version']}",
        f"- **Platform:** {row['platform']}",
        f"- **Random state:** {row['random_state']}",
        "",
        "## Key Library Versions",
        "",
        "| Library | Version |",
        "|---------|---------|" ,
    ]
    for pkg in PACKAGES:
        key = pkg.replace("-", "_") + "_version"
        md_lines.append(f"| {pkg} | {row[key]} |")

    md_lines += [
        "",
        "> This snapshot was generated automatically by `scripts/generate_environment_snapshot.py`.",
        "> Include in the thesis appendix to document the computational environment.",
        "> Saved models and SHAP outputs were generated in this environment.",
    ]

    md_out = SUMMARY_DIR / "environment_snapshot.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_out}")

    # Also regenerate requirements-lock.txt in workspace root
    lockfile_path = ROOT / "requirements-lock.txt"
    try:
        import subprocess
        with open(lockfile_path, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
        print(f"Regenerated lockfile: {lockfile_path}")
    except Exception as e:
        print(f"Warning: Failed to regenerate requirements-lock.txt: {e}")


if __name__ == "__main__":
    main()
