#!/usr/bin/env bash
# One-command thesis pipeline runner (Unix shell wrapper).
# Usage: bash scripts/run_full_pipeline.sh [--skip-download] [--skip-shap] [--quick]
set -euo pipefail
python scripts/run_full_pipeline.py "$@"
