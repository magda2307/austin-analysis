"""Generate thesis-ready Markdown summaries and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.reporting.report import create_report_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AAC thesis report outputs")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--summary-dir", default="reports/summary")
    parser.add_argument("--diagnostics-dir", default="reports/diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = create_report_outputs(args.tables_dir, args.figures_dir, args.summary_dir, args.diagnostics_dir)
    print(f"Wrote report summary to {summary_path}")
    print(f"Wrote report figures to {args.figures_dir}")


if __name__ == "__main__":
    main()

