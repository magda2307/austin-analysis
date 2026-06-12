# Closeout Router

Detailed evidence is in `docs/PROJECT_CLOSEOUT_TASKS.md`. This file is the compact
working index; update the detailed checklist when acceptance state changes.

## Dated Snapshot

Verified 2026-06-09:

- 209 tests collect after the closeout commit removed four test modules.
- Phase 1 focused gate passes: 28 tests covering matching, horizon cohorts,
  context data, rolling intake volume, and dataset construction.
- Calibration chronology gate passes: calibration uses 2022 only; 2023 remains
  the selection period.
- Full pytest and long acceptance were not run in this audit.
- Final generated model-selection prose is stale until regeneration.
- Acceptance remains blocked by deleted-coverage review, concurrent worktree
  changes, and unresolved phase-level audit findings.

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

- Generated `reports/summary/final_model_selection.md` still describes test-set
  selection; producer wording is fixed but artifacts need regeneration.
- Four test modules were deleted during closeout, including all former survival
  suites and one rolling-feature suite; replacement coverage is not yet mapped.
- Phase trackers contain stale `FULL PASS` and `Remaining risk: None` claims that
  conflict with later deep-audit reports.
- Phase 2, Phase 5, and Phase 6 lack equivalent deep-audit reports.

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
