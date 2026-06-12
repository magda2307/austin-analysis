"""Run thesis analysis table generation from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Existing analysis functions
from aac_adoption.analysis.hypothesis_tables import (
    create_adopted_only_timing_tables,
    create_hypothesis_support_tables,
    create_survival_descriptive,
    create_h2_seasonality_outputs,
    create_h4_dark_color_outputs,
)
from aac_adoption.analysis.model_comparison import create_model_comparison_tables
from aac_adoption.visualization.plots import create_eda_outputs

# New analysis functions
from aac_adoption.analysis.hypothesis_evidence import create_hypothesis_evidence_matrix
from aac_adoption.analysis.h1_feature_family import (
    create_h1_feature_family_importance,
    create_h1_ablation_table,
    create_h1_interpretation_md,
)
from aac_adoption.analysis.h3_age_evidence import create_h3_age_evidence
from aac_adoption.analysis.h5_covid_evidence import create_h5_covid_evidence
from aac_adoption.analysis.model_selection import create_final_model_selection
from aac_adoption.analysis.threshold_analysis import create_threshold_analysis
from aac_adoption.analysis.calibration_summary import create_calibration_summary
from aac_adoption.analysis.reliability_red_flags import create_reliability_red_flags
from aac_adoption.reporting.report import create_report_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAC thesis analysis outputs")
    parser.add_argument("--data", default="data/processed/modeling_dataset.csv")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--summary-dir", default="reports/summary")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--skip-survival",
        action="store_true",
        help="Skip KM descriptive survival curve generation",
    )
    parser.add_argument(
        "--h1-ablation",
        action="store_true",
        help="Run H1 feature-family ablation study training (slow)",
    )
    parser.add_argument(
        "--skip-report-outputs",
        action="store_true",
        help="Leave final report generation to a later pipeline step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 1. Run basic EDA and model comparison
    print("Running EDA and model comparison...")
    create_eda_outputs(args.data, args.tables_dir, args.figures_dir)
    create_model_comparison_tables(args.metrics_dir, args.tables_dir)

    # 2. Primary H1/H3/H5 tables
    print("Generating primary hypothesis support tables...")
    create_hypothesis_support_tables(args.data, args.tables_dir)
    print("Generating H2 seasonality outputs...")
    create_h2_seasonality_outputs(args.data, args.tables_dir, args.figures_dir, args.summary_dir)
    print("Generating H4 dark colour outputs...")
    create_h4_dark_color_outputs(args.data, args.tables_dir, args.figures_dir, args.summary_dir)

    # 3. Adopted-only timing tables for H3 descriptive speed analysis
    print("Generating adopted-only timing tables...")
    create_adopted_only_timing_tables(args.data, args.tables_dir, args.figures_dir)

    # 4. KM-style descriptive survival curves
    if not args.skip_survival:
        print("Generating descriptive survival curves...")
        create_survival_descriptive(
            args.data,
            args.tables_dir,
            args.figures_dir,
            args.summary_dir,
        )

    # 5. Model selection and threshold analysis
    print("Performing final model selection...")
    create_final_model_selection(args.tables_dir, args.summary_dir)

    print("Running threshold analysis...")
    create_threshold_analysis(
        data_path=args.data,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        summary_dir=args.summary_dir,
        models_dir=args.models_dir,
    )

    # 6. Deep-dives for H1, H3, H5
    print("Generating H1 feature-family importance and interpretation...")
    create_h1_feature_family_importance(args.tables_dir, args.figures_dir)
    if args.h1_ablation:
        print("Running optional H1 feature-family ablation training...")
        create_h1_ablation_table(args.data, args.tables_dir)
    create_h1_interpretation_md(args.tables_dir, args.summary_dir)

    print("Generating H3 age evidence and interpretation...")
    create_h3_age_evidence(args.data, args.tables_dir, args.figures_dir, args.summary_dir)

    print("Generating H5 COVID evidence and interpretation...")
    create_h5_covid_evidence(args.data, args.tables_dir, args.figures_dir, args.summary_dir)

    # 7. Reliability and calibration diagnostics
    print("Generating calibration summary...")
    create_calibration_summary(args.tables_dir, args.summary_dir)

    print("Identifying model reliability red flags...")
    create_reliability_red_flags(args.tables_dir, args.summary_dir)

    # 8. Hypothesis evidence matrix (must run after individual evidence files exist)
    print("Assembling final hypothesis evidence matrix...")
    create_hypothesis_evidence_matrix(args.tables_dir, args.figures_dir, args.summary_dir)

    # 9. Regenerate final report current results summary markdown and figures
    if not args.skip_report_outputs:
        print("Regenerating final report summaries...")
        create_report_outputs(args.tables_dir, args.figures_dir, args.summary_dir)

    print(f"\nSuccessfully wrote all analysis tables to {args.tables_dir}")
    print(f"Wrote all figures to {args.figures_dir}")
    print(f"Wrote summary interpretations to {args.summary_dir}")


if __name__ == "__main__":
    main()
