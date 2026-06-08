# Closeout Architecture Audit

Read-only finish strategy for the AAC adoption thesis closeout.

## Evidence Base

- `.agents/CLOSEOUT.md`
- `.agents/COMMANDS.md`
- `.agents/REPO_MAP.md`
- `docs/PROJECT_CLOSEOUT_TASKS.md`
- `scripts/run_full_pipeline.py`
- `scripts/generate_report_outputs.py`
- `scripts/generate_artifact_manifest.py`
- `scripts/validate_final_acceptance.ps1`
- `docs/ARCHITECTURE.md`
- `docs/METHODOLOGY.md`
- `docs/RESULTS.md`
- `docs/ROADMAP.md`
- `docs/target_definitions.md`

## Finish Order

1. Lock the producer chain that still breaks acceptance: matching/censoring, yearly backtesting, report output generation.
2. Choose and freeze the survival stance before any final docs pass.
3. Fix remaining method drift: leakage, adopted-only timing wording, PR-AUC/ROC label drift, dashboard failure modes.
4. Regenerate artifacts in producer order.
5. Rebuild manifest and final-facing docs from the regenerated artifacts.
6. Run tiered acceptance from cheap regressions to the long canonical closeout command.

## Dependency Graph

| Tier | Owner files | Prereqs | Tests / checks | Done gate | Main risk |
|---|---|---|---|---|---|
| P0-1 | `src/aac_adoption/data/match_records.py`, `src/aac_adoption/data/build_dataset.py`, `scripts/generate_data_audit.py`, `scripts/generate_leakage_audit.py` | none | `tests/test_match_records.py`, `tests/test_build_dataset.py`, `tests/test_integration_survival.py` | unmatched intakes are either excluded or separately audited; matcher censoring fields survive build; audit counts agree | stale per-animal episode state, overwritten censoring, mixed-freshness if downstream runs first |
| P0-2 | `src/aac_adoption/models/yearly_backtesting.py`, `scripts/evaluate_backtesting.py`, `docs/ROADMAP.md`, `docs/RESULTS.md` | P0-1 stable | `tests/test_yearly_backtesting.py` | yearly windows and skipped-window reasons are explicit; docs and table agree on first-year semantics | 2019 start gap, stale help text, schema drift |
| P0-3 | `src/aac_adoption/reporting/report.py`, `scripts/generate_report_outputs.py` | comparison tables fresh | `tests/test_report_outputs.py` | summary markdown and figures regenerate from current tables, not stale artifacts | old files masking a broken producer |
| P1-4 | `src/aac_adoption/features/feature_sets.py`, `src/aac_adoption/models/train_advanced.py`, `src/aac_adoption/analysis/model_comparison.py`, `src/aac_adoption/analysis/model_selection.py` | P0-1 stable | `tests/test_feature_sets.py`, `tests/test_leakage_audit.py`, `tests/test_target_encoder.py`, `tests/test_acceptance_schema_aliases.py` | no target/outcome columns in predictors; adopted-only regression wording matches code; PR-AUC/ROC labeling is consistent | leakage, misleading metric selection, target conflation |
| P1-5 | `src/aac_adoption/dashboard/data.py`, `streamlit_app.py` | artifact paths and schema stable | `tests/test_dashboard_data.py`, `tests/test_dashboard_story.py` | missing-artifact or model errors fail fast; no fake `0.5` or `15.0` default predictions | plausible-looking UI fallback hiding broken models |
| P2-6 | `scripts/generate_artifact_manifest.py`, `README.md`, `docs/METHODOLOGY.md`, `docs/RESULTS.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/target_definitions.md` | regenerated reports and manifest | `tests/test_artifact_manifest.py`, `tests/test_data_audit_outputs.py`, `tests/test_acceptance_schema_aliases.py` | manifest matches disk; final-facing docs remove stale TODO/PARTIAL/stub language unless explicitly archived | docs claim artifacts that do not exist; mojibake in generated Markdown |
| Final | `scripts/run_full_pipeline.py`, `scripts/validate_final_acceptance.ps1` | P0-P2 complete | `python -m pytest -q`, `python scripts/run_full_pipeline.py --skip-download --skip-shap`, `python scripts/generate_artifact_manifest.py`, `powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long` | full suite and long acceptance both pass against fresh outputs | pipeline continues after failure, so downstream outputs can be stale even when later steps appear green |

## Survival Decision

Recommended finish path: Path A.

- Keep survival as descriptive or future work in final-facing docs and dashboard copy.
- Do not make final acceptance depend on thesis-grade survival modeling unless that scope is explicitly re-approved.
- If Path B is chosen later, promote `tests/test_survival_analysis.py`, `tests/test_survival_analysis_new.py`, and the survival sections of `tests/test_integration_survival.py` into the acceptance tier and fix the survival code path before docs regeneration.

## Acceptance Tiers

- Tier 0: P0 producer regressions only. Fast, local, no regeneration.
- Tier 1: feature/leakage, yearly backtesting, report output, dashboard regression suites.
- Tier 2: artifact regeneration checks. `run_full_pipeline.py --skip-download --skip-shap` and `generate_artifact_manifest.py`.
- Tier 3: full closeout. `pytest -q` plus `validate_final_acceptance.ps1 -Long`.

## Manual Long Runs

- `python scripts/compare_recency.py --help` is currently a known defect reproduction, not a healthy smoke check.
- `scripts/validate_final_acceptance.ps1 -Long` is the canonical long run, but it is only usable after the recency CLI help defect is removed or the helper is replaced.
- `scripts/run_full_pipeline.py` must be treated as fail-fast on the first broken producer, even though it currently continues after failures and can leave mixed-freshness outputs.

## Stop Rules

- Stop on the first producer failure and restart from the broken upstream file, not from downstream artifacts.
- Do not trust generated reports, manifest, or docs if the upstream pipeline step failed earlier in the same run.
- Do not call the closeout complete until the final docs, manifest, and long acceptance all reflect the same target wording and artifact set.
