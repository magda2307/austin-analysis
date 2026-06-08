"""Lightweight CLI wrapper for comparing recency strategies."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Compare recency strategies for model training with bootstrap CI and subgroup analysis."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/modeling_dataset.csv",
        help="Path to the modeling dataset CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/tables/recency_strategy_comparison.csv",
        help="Path to save the output CSV results.",
    )
    parser.add_argument(
        "--figure-output",
        type=str,
        default="reports/figures/recency_strategy_comparison.png",
        help="Path to save the output comparison figure.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Number of bootstrap iterations. Defaults to 5 in quick mode, 100 otherwise.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of CatBoost iterations. Defaults to 20 in quick mode, 300 otherwise.",
    )
    parser.add_argument(
        "--test-period",
        type=str,
        default="2024-2025",
        help="Test period range (e.g. 2024-2025).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable quick mode with reduced bootstrap iterations and model complexity.",
    )
    parser.add_argument(
        "--validation-gap-years",
        type=int,
        default=4,
        help="Gap years between training end and test start (default: 4).",
    )

    # Parse args immediately so --help exits instantly without imports
    args = parser.parse_args()

    # Heavy imports deferred until after arguments are validated/parsed
    import logging
    import pandas as pd
    from pathlib import Path
    from aac_adoption.analysis.recency_comparison import run_recency_comparison, plot_performance_comparison

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Modeling dataset file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Set parameters with quick defaults if not specified
    n_bootstraps = args.n_bootstraps if args.n_bootstraps is not None else (100 if not args.quick else 5)
    iterations = args.iterations if args.iterations is not None else (300 if not args.quick else 20)

    # Run the comparison
    results_df = run_recency_comparison(
        df=df,
        n_bootstraps=n_bootstraps,
        iterations=iterations,
        test_period=args.test_period,
        quick=args.quick,
        validation_gap_years=args.validation_gap_years,
    )

    if results_df.empty:
        logger.warning("No results were generated.")
        sys.exit(0)

    # Save CSV output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved recency strategy comparison to {output_path}")

    # Generate and save figure
    figure_path = Path(args.figure_output)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_performance_comparison(results_df, figure_path)
    logger.info(f"Saved performance comparison plot to {figure_path}")

if __name__ == "__main__":
    main()
