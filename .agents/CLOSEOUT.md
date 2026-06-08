# Closeout Router

Detailed evidence is in `docs/PROJECT_CLOSEOUT_TASKS.md`. This file is the compact
working index; update the detailed checklist when acceptance state changes.

## Dated Snapshot

Verified 2026-06-08:

- 311 tests collect.
- Closeout audit recorded 288 passed, 23 failed; not re-run during this docs pass.
- `scripts/compare_recency.py --help` fails from duplicate `--quick`.
- Pipeline quick-help says tests are step 17; implementation skips step 18.

Re-run before quoting current state.

## Execution Order

1. Matching and censoring propagation.
2. Decide survival scope and align code, tests, and docs.
3. Resolve yearly backtesting first-year semantics.
4. Fix report output generation.
5. Address feature leakage and methodology correctness risks.
6. Harden dashboard failure and schema handling.
7. Strengthen acceptance tests.
8. Regenerate artifacts, then reconcile final documentation.

## P0 Routing

| Work item | Implementation | Tests | Coupled outputs |
|---|---|---|---|
| Matching/censoring | `src/aac_adoption/data/match_records.py`, `src/aac_adoption/data/build_dataset.py` | `tests/test_match_records.py`, `tests/test_build_dataset.py`, `tests/test_integration_survival.py` | data audit, matching ambiguity |
| Survival decision | `src/aac_adoption/analysis/survival_analysis.py`, `src/aac_adoption/models/train_survival.py` | `tests/test_survival_analysis.py`, `tests/test_survival_analysis_new.py`, `tests/test_integration_survival.py` | README, METHODOLOGY, ROADMAP, RESULTS, dashboard |
| Yearly backtesting | `src/aac_adoption/models/yearly_backtesting.py`, backtesting scripts | `tests/test_yearly_backtesting.py` | yearly table, ROADMAP, RESULTS |
| Report output | `src/aac_adoption/reporting/report.py`, `scripts/generate_report_outputs.py` | `tests/test_report_outputs.py` | report Markdown and figures |

## High-Risk P1 Routing

| Risk | Primary location |
|---|---|
| Targets inside generic numeric features | `src/aac_adoption/features/feature_sets.py` |
| Adopted-only regression claim mismatch | `src/aac_adoption/models/train_advanced.py` |
| Feature-set label mismatch | `src/aac_adoption/features/feature_sets.py`, `src/aac_adoption/analysis/model_comparison.py` |
| Calibrated models omitted from selection | `src/aac_adoption/analysis/model_comparison.py`, `src/aac_adoption/analysis/model_selection.py` |
| Cluster bootstrap duplicate loss | `src/aac_adoption/models/evaluate.py`, `src/aac_adoption/models/bootstrap.py` |
| Intake-volume threshold applied too early | `src/aac_adoption/data/build_dataset.py` |
| Standalone helper defects | `src/aac_adoption/models/evaluate.py`, yearly backtesting script, pipeline help |

## Verified Contract Violations

- `build_dataset.py` overwrites matcher censoring fields.
- Final model selection currently ranks test metrics, violating test-only final
  evaluation intent.
- Dashboard prediction errors currently return fake `0.5` probability and `15.0`
  LOS defaults.
- Advanced regression constructs an adopted-only filter but does not apply it.
- Intake-volume threshold runs before context features exist, making it inert.
- Baseline/boosting metadata may omit exact feature columns; dashboard can fall
  back to current registry.
- Pipeline may continue after failure and create mixed-freshness outputs.

## Dashboard Routing

- Prediction failures: `src/aac_adoption/dashboard/data.py`
- Rendering/schema assumptions: `streamlit_app.py`
- Artifact path robustness: dashboard helpers plus `src/aac_adoption/config.py`
- Tests: `tests/test_dashboard_data.py`, `tests/test_dashboard_story.py`
- Add Streamlit `AppTest` coverage before calling dashboard acceptance complete.

## Handoff Template

Keep handoffs short:

```text
Scope:
Files changed:
Behavioral decision:
Tests run and result:
Artifacts regenerated:
Remaining risk/blocker:
```
