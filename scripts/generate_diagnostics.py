"""Generate advanced model diagnostics and interpretation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aac_adoption.diagnostics.model_diagnostics import generate_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AAC advanced model diagnostics")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--models-dir", default="models/advanced")
    parser.add_argument("--diagnostics-dir", default="reports/diagnostics")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--subset", default="combined", choices=["combined", "dogs", "cats"])
    parser.add_argument("--include-shap", action="store_true")
    parser.add_argument("--shap-max-rows", type=int, default=5000)
    parser.add_argument("--min-slice-records", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_diagnostics(
        data_path=args.data,
        models_dir=args.models_dir,
        diagnostics_dir=args.diagnostics_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        subset=args.subset,
        include_shap=args.include_shap,
        shap_max_rows=args.shap_max_rows,
        min_slice_records=args.min_slice_records,
    )
    print(f"Wrote diagnostics to {args.diagnostics_dir}")
    print(f"Wrote diagnostic figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
