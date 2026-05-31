"""Generate animal-centered exploratory research outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.analysis.animal_profiles import create_animal_profile_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate animal-centered AAC exploratory outputs")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--diagnostics-dir", default="reports/diagnostics")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--min-records", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_animal_profile_outputs(
        data_path=args.data,
        diagnostics_dir=args.diagnostics_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        min_records=args.min_records,
    )
    print(f"Wrote animal research tables to {args.tables_dir}")
    print(f"Wrote animal research figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
