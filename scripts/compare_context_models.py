"""CLI for comparing base and context-enriched model runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.analysis.model_comparison import create_context_model_comparison_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and context-enriched AAC model metrics")
    parser.add_argument("--base-metrics-dir", required=True)
    parser.add_argument("--context-metrics-dir", required=True)
    parser.add_argument("--tables-dir", default="reports/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = create_context_model_comparison_table(
        args.base_metrics_dir,
        args.context_metrics_dir,
        args.tables_dir,
    )
    print(f"Wrote context comparison to {Path(args.tables_dir) / 'context_model_comparison.csv'}")
    if comparison.empty:
        print("No matching base/context model rows found.")
    else:
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
