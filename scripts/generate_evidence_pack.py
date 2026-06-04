"""Generate model evidence pack artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.reporting.evidence_pack import create_evidence_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AAC model evidence pack")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--diagnostics-dir", default="reports/diagnostics")
    parser.add_argument("--summary-dir", default="reports/summary")
    parser.add_argument("--models-dir", default="models/advanced")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--min-cohort-records", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_evidence_pack(
        data_path=args.data,
        tables_dir=args.tables_dir,
        diagnostics_dir=args.diagnostics_dir,
        summary_dir=args.summary_dir,
        models_dir=args.models_dir,
        bootstrap_samples=args.bootstrap_samples,
        min_cohort_records=args.min_cohort_records,
    )
    print(f"Wrote evidence pack table to {paths['evidence']}")
    print(f"Wrote evidence summary to {paths['summary']}")


if __name__ == "__main__":
    main()
