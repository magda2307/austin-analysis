"""Yearly temporal backtesting script.

Reads modeling dataset and runs rolling window evaluation across years.
For each year, trains on 2013-(year-1) and tests on that year.

Usage:
    python -m scripts.evaluate_backtesting
    python -m scripts.evaluate_backtesting --output reports/tables/yearly_backtesting.csv
    python -m scripts.evaluate_backtesting --target classification_target --subset combined
"""

import argparse
import pandas as pd
from pathlib import Path

from aac_adoption.models.yearly_backtesting import run_yearly_backtesting


DEFAULT_OUTPUT = "reports/tables/yearly_backtesting.csv"
DEFAULT_TARGET = "classification_target"
DEFAULT_SUBSET = "combined"


def main():
    parser = argparse.ArgumentParser(description="Run yearly temporal backtesting")
    parser.add_argument("--data_path", type=str, default="data/processed/modeling_dataset.csv",
                        help="Path to modeling dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output CSV path")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET,
                        help="Target column name")
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET,
                        choices=["combined", "dogs", "cats"],
                        help="Animal subset")
    parser.add_argument("--n_bootstraps", type=int, default=100,
                        help="Number of bootstrap iterations for CI")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    print(f"Running yearly backtesting on {len(df)} records...")
    print(f"Target: {args.target}, Subset: {args.subset}")
    
    results = run_yearly_backtesting(
        df,
        target_column=args.target,
        animal_subset=args.subset,
        output_path=args.output,
        compute_ci=True,
        bootstrap_n=args.n_bootstraps,
    )
    
    print(f"Generated {len(results)} backtesting rows")
    print(f"Saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
