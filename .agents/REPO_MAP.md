# Repository Map

## Data And Feature Layer

| Area | Primary files | Responsibility |
|---|---|---|
| Paths/constants | `src/aac_adoption/config.py` | Project-root paths, seed, supported animals |
| Loading/cleaning | `src/aac_adoption/data/load_data.py`, `src/aac_adoption/data/clean_data.py` | Normalize columns, parse dates, filter cats/dogs |
| Episode matching | `src/aac_adoption/data/match_records.py` | Nearest unused future outcome, re-intakes, censoring |
| Dataset build | `src/aac_adoption/data/build_dataset.py` | Targets, horizon labels, context merge, validation |
| Context data | `src/aac_adoption/data/context_data.py` | Weather, 311, prior-window volume features |
| Feature creation | `src/aac_adoption/features/feature_engineering.py` | Age, breed, color, location, calendar features |
| Feature contract | `src/aac_adoption/features/feature_sets.py` | Intake features, targets, leakage checks, labels |
| Target encoding | `src/aac_adoption/features/target_encoder.py` | Out-of-fold Bayesian categorical encoding |

Critical chain:

```text
match_records.py -> build_dataset.py -> feature_sets.py -> training modules
```

Changes to matching, targets, censoring, or feature names usually require tests
across all four layers.

## Modeling Layer

| Area | Primary files | Responsibility |
|---|---|---|
| Splitting | `src/aac_adoption/models/split.py` | Chronological train/validation/test contract |
| Metrics | `src/aac_adoption/models/evaluate.py`, `src/aac_adoption/models/bootstrap.py` | Classification, regression, calibration, CIs |
| Baselines | `src/aac_adoption/models/train_baseline.py` | Dummy, linear, random forest |
| Boosting | `src/aac_adoption/models/train_boosting.py` | Histogram gradient boosting |
| Advanced | `src/aac_adoption/models/train_advanced.py` | CatBoost classification/regression |
| Calibration | `src/aac_adoption/models/calibrate.py` | Platt/isotonic post-hoc calibration |
| Tuning | `src/aac_adoption/models/tune.py` | Time-aware Optuna tuning |
| Backtesting | `src/aac_adoption/models/yearly_backtesting.py` | Expanding yearly windows and horizon targets |
| Survival | `src/aac_adoption/models/train_survival.py` | Cox/competing-risk artifact path |
| Artifacts | `src/aac_adoption/models/artifacts.py`, `src/aac_adoption/models/metadata.py` | Stable paths and training metadata |

Training generally runs for `combined`, `dogs`, and `cats` where data permits.
Tuning currently uses combined data and shares selected parameters across subsets.
Classification ranking is intended to prioritize PR-AUC; regression minimizes MAE.

Canonical backtesting CLI is `scripts/evaluate_backtesting.py`; legacy standalone
yearly/horizon implementations and debug backtest scripts were removed.

## Analysis And Presentation

| Area | Primary files | Responsibility |
|---|---|---|
| Comparison/selection | `src/aac_adoption/analysis/model_comparison.py`, `src/aac_adoption/analysis/model_selection.py` | Consolidate metrics, select final models |
| Thresholds/calibration | `src/aac_adoption/analysis/threshold_analysis.py`, `src/aac_adoption/analysis/calibration_summary.py` | Validation-selected operating points |
| Hypotheses | `src/aac_adoption/analysis/hypothesis_*.py`, `h1_*`, `h3_*`, `h5_*` | Thesis evidence tables and prose |
| Survival analysis | `src/aac_adoption/analysis/survival_analysis.py` | KM, Cox, PH, hazards, competing risks |
| Diagnostics | `src/aac_adoption/diagnostics/`, `src/aac_adoption/analysis/reliability_red_flags.py` | Error, calibration, SHAP, subgroup risks |
| Reports | `src/aac_adoption/reporting/report.py`, `src/aac_adoption/reporting/evidence_pack.py` | Summary Markdown, figures, evidence pack |
| Dashboard helpers | `src/aac_adoption/dashboard/data.py`, `src/aac_adoption/dashboard/story.py` | Artifact loading and prediction helpers |
| Dashboard UI | `streamlit_app.py` | Read-only thesis application |

## Pipeline Entry Points

`scripts/run_full_pipeline.py` defines 19 steps numbered 0-18:

```text
environment -> download -> dataset -> EDA -> baseline -> adopted regression
-> tuning -> boosting -> CatBoost -> calibration -> analysis -> diagnostics
-> animal research -> evidence pack -> report -> feature-family importance
-> manifest -> backtesting -> pytest
```

Scripts should remain thin CLI wrappers. Put reusable logic in `src/aac_adoption/`.

Operational caveats:

- Runner continues after a failed step, then exits nonzero at end. Downstream
  outputs may mix new and stale artifacts. Inspect run summary before trusting them.
- Canonical step 2 currently does not pass optional context data to dataset build.
- Quick mode skips step 18 tests, despite stale help text saying step 17.

## Artifact Ownership

| Artifact | Producer |
|---|---|
| `data/processed/modeling_dataset.csv` + feature/target JSON | dataset build |
| `models/**` + model JSON sidecars | training/calibration modules |
| `reports/metrics/**` | trainers, calibration, survival |
| `reports/tables/**` | analysis, diagnostics, audits, evidence pack |
| `reports/figures/**` | EDA, analysis, diagnostics, reporting |
| `reports/summary/**` | audits, selection, reporting, evidence pack |
| `reports/artifact_manifest.csv` | `scripts/generate_artifact_manifest.py` |

Fix producer first. Regenerate artifact. Model sidecars currently record
`git_sha: unknown`; do not claim artifact-to-commit traceability.

## Documentation Ownership

- Canonical project docs: root `README.md` plus `docs/ARCHITECTURE.md`,
  `docs/METHODOLOGY.md`, `docs/RESULTS.md`, `docs/ROADMAP.md`, and
  `docs/target_definitions.md`.
- Current closeout state: `.agents/CLOSEOUT.md` and
  `docs/PROJECT_CLOSEOUT_TASKS.md`.
- Generated narrative evidence: `reports/summary/`; fix its producer rather than
  editing generated Markdown.
- Historical `docs/old`, `docs/internal`, root request/delegation notes, and the
  compiled LLM megadocument were removed. Recover history with git only.

## Coupling Rules

- Target changes: dataset build, feature leakage lists, training, dashboard labels,
  methodology, results, and tests.
- Feature changes: feature engineering, registry, model metadata, dashboard record
  construction, diagnostics, and leakage tests.
- Metric/schema changes: trainers, model comparison, selection, reports, dashboard,
  artifact manifest, and acceptance alias tests.
- Artifact path changes: producer, dashboard loader, report loader, manifest, tests.
- Survival scope changes: survival code/tests plus README, METHODOLOGY, ROADMAP,
  RESULTS, dashboard copy, and generated report language.
