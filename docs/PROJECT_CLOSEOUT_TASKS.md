# Project Closeout Tasks

> **Authoritative execution plan:** [`docs/closeout/2026-06-08-thesis-closeout-implementation-plan.md`](closeout/2026-06-08-thesis-closeout-implementation-plan.md)
>
> Supporting validation matrix and manual long-run commands live in
> [`docs/closeout/`](closeout/README.md). The implementation plan records the
> final cohort, survival-analysis, chronological-selection, provenance, and
> acceptance decisions that govern the tasks below.

Generated on 2026-06-08 after reading project Markdown, Python code, Streamlit app, tests, roadmap/status files, and running the test suite.

Goal: make the project complete, reproducible, methodologically consistent, and ready for final regeneration.

## Current Acceptance State

- Verified 2026-06-13.
- Canonical run `20260613T001946Z-0bb4ce5`, profile `thesis-full`,
  producer source `0bb4ce5`.
- All 20 requested pipeline steps completed with no skips or failures.
- `python -m pytest --collect-only -q`: 271 tests collected.
- Long acceptance: 268 passed, 3 expected skips, 45 warnings; receipts valid.
- Strict manifest: 48 required artifacts, all present and hash-matched.
- Closeout blockers are resolved.

## P0 - Must Fix Before Regeneration

### 1. Fix dataset matching and censoring propagation

Evidence:

- `src/aac_adoption/data/match_records.py:143` appends unmatched intakes back into the modeling dataset while also counting them as unmatched.
- `src/aac_adoption/data/match_records.py:157` uses stale per-animal episode state in unmatched handling.
- `src/aac_adoption/data/build_dataset.py:136` overwrites matcher censoring fields with default non-censored values.
- Failing tests include `tests/test_integration_survival.py::test_censoring_propagation`, `tests/test_match_records.py::test_match_records_unmatched_intakes`, and `tests/test_integration_survival.py::test_censoring_indicator_consistency`.

Tasks:

- Rewrite unmatched-intake handling so unmatched rows are either excluded from the supervised modeling frame or explicitly carried only in a separate audit table.
- Preserve matcher-provided `is_censored`, `censoring_reason`, `event_type`, and `followup_days_censored` in `build_modeling_dataset`.
- Rebuild per-animal episode lookup inside unmatched handling instead of reusing stale state.
- Add repeat-stay tests covering fewer outcomes than intakes, reused outcomes, and trailing unresolved intakes.

Done when:

- Censoring columns are internally consistent.
- Unmatched count equals actual excluded/unresolved intakes.
- Matching ambiguity and data audit reports agree on counts.

### 2. Decide survival scope, then make code/tests/docs agree

Evidence:

- `docs/METHODOLOGY.md` says full survival modeling is future work.
- `tests/test_survival_analysis.py`, `tests/test_survival_analysis_new.py`, and `tests/test_integration_survival.py` expect working Cox, cumulative hazard, competing-risk, censoring, and encoding behavior.
- Full suite has many survival failures:
  - `fit_cox_with_censoring` returns empty summaries or `None`.
  - `validate_proportional_hazards` requires `training_df`, but tests call it without that argument.
  - `add_censoring_indicators` crashes on empty input.
  - `compute_cumulative_hazard` returns empty outputs for valid censored data.
  - `compute_subDistribution_hazard` returns `None`.
  - `compute_LOS_quantiles` returns 6 rows where tests expect 8.
  - `encode_categorical_features` one-hot names differ from tests.
- `src/aac_adoption/models/train_survival.py:68` computes a mean over dict encodings and crashes with `TypeError: unsupported operand type(s) for +: 'dict' and 'dict'`.

Tasks:

- Choose one of two final paths:
  - Path A: survival remains descriptive/future work. Remove or mark model-level survival tests as non-acceptance, and ensure docs/dashboard do not present survival models as complete.
  - Path B: survival is thesis-grade. Fix all functions in `src/aac_adoption/analysis/survival_analysis.py` and `src/aac_adoption/models/train_survival.py`, including censoring, categorical encoding, Cox fitting, PH validation, cumulative hazard, competing risks, and artifact generation.
- Make `validate_proportional_hazards` API compatible with tests or update tests to the real API.
- Fix target encoding in `train_survival.py` so unknown-category fallback uses numeric encoding values, not nested dicts.
- Add explicit event/censoring counts to generated survival outputs.

Done when:

- Either survival model tests pass, or survival modeling is clearly removed from final acceptance.
- Methodology, roadmap, README, reports, and dashboard all use the same survival claim.

### 3. Fix yearly backtesting first-year behavior

Evidence:

- `tests/test_yearly_backtesting.py::test_yearly_backtesting_horizon_targets` expects 2019-2024 rows.
- Current run skips 2019 because train set is empty.
- `docs/ROADMAP.md` still lists yearly backtesting as TODO although `reports/tables/yearly_backtesting.csv` exists.

Tasks:

- Decide intended yearly windows. If test year 2019 requires training through 2018, fixture/data must include pre-2019 training rows or test expectations must start at 2020.
- Align code, tests, docs, and generated table.
- Ensure yearly backtesting reports classification, horizon targets, animal subsets, and skipped-window reasons.

Done when:

- Yearly backtesting tests pass.
- `docs/ROADMAP.md`, `docs/RESULTS.md`, and artifact manifest consistently state whether yearly backtesting is complete.

### 4. Fix report output generation failure

Evidence:

- `tests/test_report_outputs.py::test_create_report_outputs_writes_summary_and_figures` fails in full suite.

Tasks:

- Run that test alone with verbose output.
- Fix expected generated summary/figure contract.
- Confirm report generation does not depend on stale tracked artifacts.

Done when:

- `python -m pytest tests/test_report_outputs.py -q` passes.

## P1 - Code Correctness and Methodology Risks

### 5. Remove target/leakage columns from generic numeric feature lists

Evidence:

- `src/aac_adoption/features/feature_sets.py:75` includes target/analysis columns in `NUMERIC_FEATURES`.

Tasks:

- Separate model predictor numeric features from target/analysis columns.
- Verify `available_features_for_df` cannot accidentally admit `days_to_outcome`, `regression_target_days`, `classification_target`, `adopted`, or future-derived fields.
- Add regression tests around final feature lists.

### 6. Fix adopted-only regression mismatch

Evidence:

- `src/aac_adoption/models/train_advanced.py:245` docstring says adopted-only filtering.
- `src/aac_adoption/models/train_advanced.py:260` defines `filter_df` / `filter_col`, but code trains on all outcomes.

Tasks:

- Either actually filter adopted-only regression to adopted records or remove the adopted-only claim.
- Ensure all labels distinguish:
  - all-outcome LOS: `regression_target_days`, days to any matched outcome;
  - adopted-only timing: `days_to_adoption`, adopted subset only.

### 7. Fix context comparison feature-set labels

Evidence:

- `src/aac_adoption/analysis/model_comparison.py:24` expects `intake_time_v1` and `intake_time_context_v1`.
- `src/aac_adoption/features/feature_sets.py:150` emits `intake_time_v2`.

Tasks:

- Align metadata labels or allow both old and new labels.
- Regenerate `reports/tables/context_model_comparison.csv`.

### 8. Include calibrated classifiers in model comparison and final selection

Evidence:

- `src/aac_adoption/analysis/model_comparison.py:116` excludes `calibrated_classification_metrics.csv`.
- `src/aac_adoption/analysis/model_selection.py:217` says formal calibration not applied while pipeline now has calibration code/artifacts.

Tasks:

- Decide whether calibrated classifiers compete in final selection or remain diagnostic-only.
- Update comparison, selection, README, roadmap, and generated narratives accordingly.

### 9. Fix cluster bootstrap weighting

Evidence:

- `src/aac_adoption/models/evaluate.py:71` cluster bootstrap uses `np.unique` after sampling clusters, which removes duplicated sampled animals and breaks bootstrap weighting.

Tasks:

- Keep duplicate sampled clusters or implement proper cluster weights.
- Add a test that duplicate sampled clusters affect bootstrap sample size as expected.

### 10. Fix inert intake-volume threshold

Evidence:

- `src/aac_adoption/data/build_dataset.py:339` applies `max_intake_volume_threshold` before context features are added, so the filter is usually ineffective.

Tasks:

- Move thresholding after `add_context_features_from_dir`.
- Add a test with context features proving high-volume rows are filtered.

### 11. Fix standalone helper defects

Evidence:

- `src/aac_adoption/models/evaluate.py:189` uses `pd.DataFrame` without importing pandas.
- `scripts/horizon_yearly_backtesting.py:44` one-hot encodes train/test separately, risking schema mismatch.
- `scripts/horizon_yearly_backtesting.py:33` treats raw age strings as numeric.
- `scripts/run_full_pipeline.py:172` quick help says tests are step 17, but code skips step 18.

Tasks:

- Import pandas or remove dead helper.
- Fit preprocessing schema on train and reindex test to train columns.
- Parse age strings or exclude raw age before numeric scaling.
- Fix quick-mode help text.

## P2 - Streamlit Dashboard Readiness

### 12. Stop silent fake predictions

Evidence:

- `src/aac_adoption/dashboard/data.py:389` returns default adoption probability `0.5` on classifier errors.
- `src/aac_adoption/dashboard/data.py:404` returns default LOS `15.0` on regressor errors.
- `streamlit_app.py:589` and `streamlit_app.py:1028` broad exception handling can mislabel failures as missing advanced training.

Tasks:

- Replace fake prediction defaults with typed failure states.
- Surface expected missing-artifact errors separately from unexpected model/feature errors.
- Add dashboard tests proving missing models do not produce plausible-looking predictions.

### 13. Add schema guards before rendering artifact tables

Evidence:

- `streamlit_app.py:539`, `:581`, `:762`, `:803`, `:878`, `:950`, and `:1177` assume nonempty tables have required columns and parse CSV booleans with `bool(...)`.

Tasks:

- Add shared required-column guard for each dashboard table.
- Parse string booleans explicitly (`true/false/1/0/yes/no`), not with Python `bool`.
- Show clear missing-column messages instead of crashing.

### 14. Make dashboard paths and install flow robust

Evidence:

- `src/aac_adoption/dashboard/data.py:334` hardcodes `reports/tables` relative to current working directory.
- `pyproject.toml` keeps Streamlit/Altair only in `dev`; dashboard is top-level runnable.
- `Makefile` has no dashboard smoke target.

Tasks:

- Resolve report/model paths from project root or pass base dirs into app helpers.
- Add `dashboard` extra or document `pip install -e ".[dev]"` as required.
- Add a Streamlit smoke test using `streamlit.testing.v1.AppTest` or equivalent.

## P3 - Tests and Acceptance Hardening

### 15. Replace weak/pass/skip acceptance tests

Evidence:

- `tests/test_artifact_manifest.py:95` has a `pass` body for required thesis artifact existence.
- `tests/test_data_audit_outputs.py` skips generated-output checks when files are absent.
- `tests/test_hypothesis_evidence.py` skips or only checks Markdown if reports exist.
- `test_fixes.py` exists at repository root and looks like a stray patch/debug file.

Tasks:

- Split tests into unit tests and acceptance tests.
- Add an acceptance mode that requires generated artifacts after pipeline regeneration.
- Remove `test_fixes.py` or convert it into a real test under `tests/`.

### 16. Add final smoke commands

Required final commands:

```powershell
python -m pytest -q
python scripts/run_full_pipeline.py --skip-download --skip-shap
python scripts/generate_artifact_manifest.py
python scripts/validate_final_acceptance.ps1
```

If `validate_final_acceptance.ps1` is not intended as canonical, replace it with one documented final acceptance command.

## P4 - Documentation and Artifact Consistency

### 17. Fix README broken and stale references

Evidence:

- `README.md:28` points to missing `docs/methodology_notes.md`; current file is `docs/METHODOLOGY.md`.
- `README.md` has duplicated reproduction/EDA/test/dataset sections and stale `docs/results_summary.md` references.
- `README.md` says survival models beyond descriptive views are not implemented while tests/code contain survival modeling paths.

Tasks:

- Update all doc links to current files.
- Dedupe reproduction and current-status sections.
- Make README status match actual final scope after P0 fixes.

### 18. Reconcile roadmap contradictions

Evidence:

- `docs/ROADMAP.md:18` says calibration CLI broken, while later sections say calibration is DONE.
- `docs/ROADMAP.md:20` says horizon targets missing, while later sections say DONE.
- `docs/ROADMAP.md:105` says yearly backtesting TODO while generated table exists.
- `docs/ROADMAP.md:262` and `:279` say matching ambiguity needs regeneration while matching ambiguity report exists.
- `docs/ROADMAP.md:276` and README still flag duration/adoption-speed wording.

Tasks:

- Collapse each duplicated/contradictory item into one final status.
- Keep only unresolved work in immediate priorities.
- Move historical agent plans to `docs/internal` or `docs/old` if they are not final-facing.

### 19. Reconcile methodology with current code

Evidence:

- `docs/METHODOLOGY.md:157` says duplicate feature cleanup is planned, but roadmap says DONE.
- `docs/METHODOLOGY.md:176` says re-intake ambiguity detection is planned, but matching ambiguity audit exists.

Tasks:

- Update methodology to reflect cleaned feature registry and current matching ambiguity handling.
- Keep residual risk language where still true.

### 20. Regenerate artifact manifest and summaries

Status: complete 2026-06-13.

- Manifest regenerated for run `20260613T001946Z-0bb4ce5`.
- 48 required artifacts have current disk and receipt hashes.
- Data audit, matching evidence, model selection, diagnostics, and required
  methodology summaries were regenerated by their producers.
- Non-required ephemeral outputs no longer invalidate required artifact lineage;
  conflicting hashes for required outputs still fail generation.

Historical evidence before closeout:

- `reports/summary/artifact_manifest.md` marks absent docs like `docs/model_diagnostics.md` and `docs/results_summary.md` as present.
- `reports/summary/data_audit.md` says 0 ambiguous episodes, while `reports/summary/matching_ambiguity.md` says 36.
- `docs/RESULTS.md` says ROC-AUC primary and latest tests were 81 passed on 2026-05-31; current full run is 288 passed / 23 failed on 2026-06-08.

Completed tasks:

- Regenerate data audit, matching ambiguity, artifact manifest, and results after code fixes.
- Update classification metric framing to PR-AUC primary if final selection uses PR-AUC.
- Ensure generated Markdown has no mojibake artifacts such as `â€”`, `â‰`, `â€“`, or broken symbols.

## Final Closeout Order

1. Fix P0 code blockers: matching/censoring, survival decision, yearly backtesting, report output.
2. Run targeted failing tests until all P0 tests pass.
3. Fix P1 methodology/code risks.
4. Harden dashboard failure handling and smoke test.
5. Regenerate pipeline artifacts.
6. Regenerate manifest and docs.
7. Run full test suite and final acceptance commands.
8. Update README/RESULTS/ROADMAP with final test status and exact regeneration date.

## Final Definition of Done

- [x] `python -m pytest -q` passes with no unexpected skips.
- [x] Full pipeline reruns from local raw data without manual intervention.
- [x] Artifact manifest matches required files and run receipts.
- [x] Final methodology/results documents use canonical target definitions.
- [ ] ROADMAP cleanup and thesis-export packaging remain separate presentation
  work; they are not canonical model/artifact acceptance gates.
- [x] Dashboard never shows fake default predictions as real model output.
- [x] Survival modeling is explicitly outside final predictive scope.
- [x] LOS and adopted-only timing are not conflated.
