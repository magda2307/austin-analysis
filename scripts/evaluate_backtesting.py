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
    parser.add_argument("--target", type=str, default=None,
                        help="Target column name (if None, runs both classification and regression)")
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET,
                        choices=["combined", "dogs", "cats"],
                        help="Animal subset")
    parser.add_argument("--n_bootstraps", type=int, default=None,
                        help="Number of bootstrap iterations for CI (default: 5 for quick, 100 otherwise)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: run only 2 windows")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of iterations for CatBoost/HGB (default: 20 for quick, 100 otherwise)")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    print(f"Running yearly backtesting on {len(df)} records...")
    print(f"Subset: {args.subset}, Quick mode: {args.quick}")
    
    if args.target:
        targets = [args.target]
    else:
        targets = ["classification_target", "regression_target_days"]
    
    all_results = []
    for target in targets:
        print(f"Running backtesting for target: {target}")
        if args.iterations is None:
            iterations = 20 if args.quick else 100
        else:
            iterations = args.iterations
        if args.n_bootstraps is None:
            n_bootstraps = 5 if args.quick else 100
        else:
            n_bootstraps = args.n_bootstraps
        results = run_yearly_backtesting(
            df,
            target_column=target,
            animal_subset=args.subset,
            output_path=None,
            compute_ci=True,
            bootstrap_n=n_bootstraps,
            quick=args.quick,
            strict=True,
            iterations=iterations,
        )
        if not results.empty:
            all_results.append(results)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame()
        
    if args.output and not final_df.empty:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Generated {len(final_df)} backtesting rows")
        print(f"Saved to: {args.output}")
        
    return final_df


if __name__ == "__main__":
    main()
