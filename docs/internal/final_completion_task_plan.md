# Final Completion Task Plan

Project: AAC Adoption ML thesis pipeline  
Workspace: `C:\Users\paula\Documents\mgr pjatk`  
Created: 2026-06-07  
Purpose: finish project to thesis-submission readiness without mixing quick code fixes with long training/report steps.

This plan assumes current worktree may already contain uncommitted user/agent changes. Do not revert unrelated changes. Work in small batches, verify each batch, then move to the next.

## Current Completion State

The project is close, but not done. Main blockers are acceptance and reproducibility issues, not missing basic ML structure.

Verified quick checks:

- `python -m py_compile src/aac_adoption/dashboard/data.py` passes.
- `python scripts/calibrate_classifiers.py --help` passes.
- `python scripts/evaluate_backtesting.py --help` passes.

Known failures/gaps:

- `tests/test_build_dataset.py` currently has 5 failures caused by missing/removed output columns in `build_modeling_dataset()`.
- `tests/test_backtesting.py tests/test_yearly_backtesting.py` timed out in focused run.
- `scripts/compare_recency.py --help` timed out because script has no lightweight CLI/help path.
- `src/aac_adoption/analysis/recency_comparison.py` is missing, although original acceptance requested it.
- Dashboard still selects best classifier by ROC-AUC in one helper, while project direction says PR-AUC primary.
- Stacked ensemble still trains meta-learner on in-sample base predictions.
- Hyperparameter tuning still has classification/regression feature-frame coupling risk.
- Docs disagree: README/RESULTS/ROADMAP differ on PR-AUC primary, survival state, yearly backtesting, and recency comparison.

## Difficulty And Model Guidance

Use stronger model for tasks that require cross-file reasoning, ML methodology, or subtle leakage checks. Use cheaper/faster model only for mechanical doc sync or simple test repair.

Difficulty scale:

- S: straightforward, 1 file, low risk.
- M: 2-4 files, normal tests, moderate risk.
- L: cross-cutting, ML logic, high regression risk.
- XL: long-running training/reporting or thesis-wide acceptance.

Recommended model scale:

- Small/fast model: doc copy edits, command wrappers, obvious one-file fixes.
- Good model: code touching data contracts, model training, dashboard prediction behavior.
- Strong model: yearly backtesting, tuning leakage, ensemble OOF, final acceptance review.

## Dependency Map

```text
Batch 0: Snapshot / baseline
  -> Batch 1: dataset builder contract
    -> Batch 2: backtesting + recency CLI
      -> Batch 3: model-method hardening
        -> Batch 4: dashboard alignment
          -> Batch 5: docs reconciliation
            -> Batch 6: long runs and generated artifacts
              -> Batch 7: final acceptance checklist
```

Do not run long training/report regeneration before Batch 1-5 pass targeted tests. Otherwise artifacts may be stale again after fixes.

## Batch 0 - Snapshot And Guardrails

Difficulty: S  
Recommended model: small/fast ok  
Owner: coordinator

Goal:

- Record current state before changing code.
- Separate user changes from new work.

Tasks:

1. Run `git status --short --branch`.
2. Run `git diff --stat`.
3. Save current failing/passing test facts in notes.
4. Do not stage or commit until batches are reviewed.

Validation:

```powershell
git status --short --branch
git diff --stat
```

Acceptance:

- Known dirty files understood.
- No unrelated files reverted.

## Batch 1 - Dataset Builder Contract

Difficulty: M  
Recommended model: good model  
Owner: data-contract engineer

Primary files:

- `src/aac_adoption/data/build_dataset.py`
- `tests/test_build_dataset.py`
- possibly `src/aac_adoption/features/feature_engineering.py` only if aliases are defined there

Problem:

`build_modeling_dataset()` creates some columns, narrows to `ordered_columns`, then selects a fixed final list containing columns that may not exist in small fixtures:

- `outcome_subtype`
- `sex_upon_outcome`
- `age_upon_outcome`
- `has_name`
- `age_in_days`
- `age_in_months`
- `age_in_years`
- `age_months`
- `age_years`
- `is_censored`
- `censoring_reason`
- `event_type`
- `followup_days_censored`

This causes `KeyError` in build dataset tests.

Expected finish:

- Ensure output schema is stable for both real data and small fixtures.
- Create/fill compatibility columns before final selection.
- Preserve raw optional outcome metadata when present; fill missing optional metadata with nulls or safe defaults.
- Keep censoring fields required by `validate_modeling_dataset()`.
- Keep train-only winsorization principle: do not globally cap target values in dataset builder.
- Avoid adding leakage predictors to model feature registry.

Implementation details:

- Before final selection, normalize aliases:
  - `has_name` should mirror `is_named` when absent.
  - `age_in_days` should mirror `age_days` when absent.
  - `age_in_months` should mirror `age_months` or derive from `age_days`.
  - `age_in_years` should mirror `age_years` or derive from `age_days`.
  - `age_months` and `age_years` should derive from `age_days` if absent.
- Optional outcome metadata can be created as `pd.NA` when absent.
- Censoring defaults for matched outcome dataset:
  - `is_censored = False`
  - `censoring_reason = ""`
  - `event_type` should reflect actual outcome type where useful, not blindly "adoption" for all rows. Safer: normalized `outcome_type`.
  - `followup_days_censored = days_to_outcome`
- Replace fixed final `dataset = dataset[[...]]` with a safer helper that fills missing expected columns first.

Tests:

```powershell
python -m pytest tests/test_build_dataset.py -q
python -m pytest tests/test_target_definitions.py tests/test_horizon_targets.py -q
```

Validation:

- `tests/test_build_dataset.py` passes.
- Raw LOS outlier test still proves no global winsorization.
- Produced dataset contains required columns from `REQUIRED_MODELING_COLUMNS`.
- `classification_target` equals `adopted.astype(int)`.
- `days_to_outcome` has no negative values.

Acceptance:

- No `KeyError` when fixtures omit optional AAC outcome columns.
- Dataset output contract is documented or obvious in code.

## Batch 2 - Backtesting And Recency Reproducibility

Difficulty: L  
Recommended model: strong model  
Owner: temporal-validation engineer

Primary files:

- `src/aac_adoption/models/yearly_backtesting.py`
- `scripts/evaluate_backtesting.py`
- `scripts/compare_recency.py`
- new `src/aac_adoption/analysis/recency_comparison.py`
- `tests/test_backtesting.py`
- `tests/test_yearly_backtesting.py`
- `tests/test_recency_comparison.py`

Problem:

Yearly backtesting code appears partly implemented but likely fragile:

- Missing metric imports are referenced in `yearly_backtesting.py`.
- Broad `except Exception: continue` can silently produce incomplete output.
- CLI lacks required `--quick`.
- Acceptance requested six windows: train 2013-2018/test 2019 through train 2013-2023/test 2024.
- Existing helper may run many years starting at 2014.

Recency comparison has artifact output but no requested module and no lightweight CLI:

- `scripts/compare_recency.py --help` timed out.
- `src/aac_adoption/analysis/recency_comparison.py` does not exist.

Expected finish:

Yearly backtesting:

- Add missing imports.
- Add `--quick` flag.
- Use exact accepted windows by default.
- Quick mode runs only 2 windows, e.g. 2018->2019 and 2022->2023.
- Output required columns:
  - `train_years`
  - `test_year`
  - `model_name`
  - `animal_subset`
  - `pr_auc`
  - `roc_auc`
  - `brier_score`
  - `mae`
  - `train_rows`
  - `test_rows`
- Include classification and regression rows clearly. If one task has not-applicable metrics, keep nulls.
- Do not silently swallow exceptions. At minimum collect an `error` column in failed rows or log and re-raise in strict mode.

Recency comparison:

- Move business logic to `src/aac_adoption/analysis/recency_comparison.py`.
- Keep `scripts/compare_recency.py` as argparse wrapper.
- Add `--quick`, `--data-path`, `--output`, `--figure-output`, `--n-bootstraps`, `--iterations`.
- `--help` must return quickly without loading dataset/training CatBoost.
- Compare strategies:
  - `full_history`
  - `recent_5yr`
  - `recent_3yr`
  - `recency_weighted`
- Test period should be 2024-2025 unless overridden.
- Use PR-AUC as primary classification comparison.

Tests:

```powershell
python scripts/evaluate_backtesting.py --help
python scripts/evaluate_backtesting.py --quick --n_bootstraps 5
python scripts/compare_recency.py --help
python scripts/compare_recency.py --quick --n-bootstraps 5 --iterations 20
python -m pytest tests/test_backtesting.py tests/test_yearly_backtesting.py tests/test_recency_comparison.py -q
```

Validation:

- Quick commands finish in reasonable time.
- Backtesting CSV exists and has required columns.
- Every `test_year` is greater than max training year.
- Recency output includes all four strategies for combined subset.
- No broad silent failure hides missing model rows.

Acceptance:

- Temporal validation can be reproduced without full pipeline.
- Quick mode is CI/development safe.

## Batch 3 - Model Method Hardening

Difficulty: L  
Recommended model: strong model  
Owner: ML-methodology engineer

Primary files:

- `src/aac_adoption/models/tune.py`
- `src/aac_adoption/models/ensemble.py`
- diagnostics/permutation code, likely `scripts/generate_diagnostics.py` or `src/aac_adoption/diagnostics/model_diagnostics.py`
- relevant tests:
  - `tests/test_hyperparam_tuning.py`
  - `tests/test_ensemble.py`
  - `tests/test_diagnostics_outputs.py`

Problem:

Acceptance slices mention:

- Hyperparameter tuning leakage cleanup.
- Permutation importance validation split, not test split.
- Ensemble OOF fix.

Current risk:

- `tune.py` defines `cat_X` from classification train frame and reuses it for CatBoost regression with `cat_y_reg` from `reg_train_df`. If row filters/order diverge, regression tuning can be wrong.
- Stacked ensembles train meta-learner on predictions from base models fitted on same rows. This overfits and is not thesis-defensible.
- Permutation importance table exists, but must verify it is computed on validation data.

Expected finish:

Tuning:

- Build separate `X_clf`, `y_clf`, `X_reg`, `y_reg`.
- Fit preprocessors inside each CV fold.
- Sort by `intake_datetime` before `TimeSeriesSplit`.
- Classification objective optimizes PR-AUC.
- Regression objective optimizes MAE.
- Document fold strategy in code comments or generated tuning report.

Ensemble:

- Use out-of-fold predictions for meta-learner training.
- Fit final base estimators on all training data after OOF meta-frame is built.
- Support classifier and regressor.
- Add tests proving meta-learner training predictions are OOF, not in-sample.

Permutation:

- Identify permutation importance function.
- Ensure it uses validation split, not final test split.
- Artifact should include `evaluation_period=validation` or similar.

Tests:

```powershell
python -m pytest tests/test_hyperparam_tuning.py tests/test_ensemble.py tests/test_diagnostics_outputs.py -q
```

Validation:

- No classification/regression target-frame mismatch.
- OOF stacking tests fail on old implementation and pass after fix.
- Permutation importance artifact clearly states validation period.

Acceptance:

- Tuning/ensemble/importance methods can be defended in methodology chapter.

## Batch 4 - Dashboard Prediction And Model Selection Alignment

Difficulty: M  
Recommended model: good model  
Owner: dashboard/model-alignment engineer

Primary files:

- `src/aac_adoption/dashboard/data.py`
- `streamlit_app.py`
- `tests/test_dashboard_data.py`

Problem:

Dashboard helper `best_model_rows()` still selects classification winners by ROC-AUC. Project direction says classification ranking is PR-AUC primary.

Acceptance also asks:

- Prefer calibrated classifier artifacts in `predict_from_record()`.
- Add duration uncertainty / LOS wait-time buckets.
- Avoid wording that sounds like deterministic adoption speed.

Expected finish:

- `best_model_rows()` sorts classification by `pr_auc` descending, tie-break `roc_auc`.
- `primary_metric` for classification becomes `pr_auc`.
- `predict_from_record()` loads calibrated classification artifact when available, otherwise selected base classifier.
- Add helper:
  - `los_days_to_bucket(days)`
  - buckets: `0-7d`, `8-30d`, `31-60d`, `61-90d`, `90+d`
- `predict_from_record()` returns:
  - `adoption_probability`
  - `predicted_days_to_outcome`
  - `los_bucket`
- Dashboard labels use:
  - "Predicted days to outcome"
  - "Length-of-stay bucket"
  - not "predicted days to adoption" for all animals.

Tests:

```powershell
python -m pytest tests/test_dashboard_data.py -q
python -m py_compile streamlit_app.py src/aac_adoption/dashboard/data.py
```

Manual validation:

```powershell
streamlit run streamlit_app.py
```

Inspect:

- Overview cards show PR-AUC primary.
- Prediction view shows bucket and does not imply exact wait time.
- App loads without missing artifact crash.

Acceptance:

- Dashboard matches model-selection methodology.
- User-facing language avoids target conflation.

## Batch 5 - Documentation Reconciliation

Difficulty: M  
Recommended model: good model for methodology text, small/fast ok for mechanical edits  
Owner: documentation/methodology engineer

Primary files:

- `README.md`
- `docs/ROADMAP.md`
- `docs/RESULTS.md`
- `docs/METHODOLOGY.md`
- possibly generated summaries under `reports/summary/`

Problem:

Docs currently conflict:

- `docs/RESULTS.md` says ROC-AUC primary.
- `docs/ROADMAP.md` says PR-AUC primary but also has stale TODO/PARTIAL sections.
- README says survival models beyond descriptive views not implemented, while new survival code/tests exist.
- Roadmap still has checklist items unticked after artifacts exist.

Expected finish:

- Classification primary metric consistently PR-AUC.
- ROC-AUC remains secondary threshold-independent ranking context.
- Regression target consistently "length of stay / days to outcome", not adoption speed.
- `days_to_adoption` used only for adopted-only analyses.
- Survival status has one consistent framing:
  - either "descriptive/future work" if not thesis-grade survival modeling, or
  - "implemented survival analysis" only if validated and documented.
- Roadmap statuses match code + tests + artifacts, not wishes.
- Long-running artifacts listed separately from code readiness.

Tests/validation:

```powershell
rg -n "Primary metric: \\*\\*ROC-AUC|Use \\*\\*ROC-AUC\\*\\* for ranking|adoption speed|predicted days to adoption" README.md docs streamlit_app.py src
python -m pytest tests/test_target_definitions.py tests/test_report_outputs.py -q
```

Acceptance:

- No stale ROC-AUC-primary claims in current docs.
- No unqualified "days to adoption" for all-animal regression.
- Roadmap reflects accepted state after batches 1-4.

## Batch 6 - Long Runs And Artifact Regeneration

Difficulty: XL  
Recommended model: human runs commands; agent reviews outputs after  
Owner: user/operator

Do this only after Batches 1-5 pass targeted tests.

Commands:

```powershell
python -m pytest -q
python scripts/run_full_pipeline.py --skip-download --skip-shap
python scripts/calibrate_classifiers.py --data-path data/processed/modeling_dataset.csv
python scripts/evaluate_backtesting.py --quick
python scripts/compare_recency.py --quick --n-bootstraps 20 --iterations 50
python scripts/generate_report_outputs.py
python scripts/generate_artifact_manifest.py
```

Optional full heavy commands:

```powershell
python scripts/run_full_pipeline.py --skip-download
python scripts/evaluate_backtesting.py --n_bootstraps 100
python scripts/compare_recency.py --n-bootstraps 100 --iterations 300
```

Expected outputs:

- `reports/metrics/calibrated_classification_metrics.csv`
- `reports/tables/yearly_backtesting.csv`
- `reports/tables/recency_strategy_comparison.csv`
- refreshed `reports/summary/current_results.md`
- refreshed `reports/artifact_manifest.csv`

Validation:

- Check command exit codes.
- Inspect generated CSV row counts.
- Confirm manifest has no missing required thesis artifacts unless intentionally optional.

Acceptance:

- Full suite passes or known skipped/heavy failures are documented.
- Generated reports match current code.

## Batch 7 - Final Acceptance And Commit

Difficulty: M  
Recommended model: strong model for review, caveman-commit for message  
Owner: coordinator/reviewer

Tasks:

1. Run final status:

```powershell
git status --short --branch
git diff --stat
```

2. Run final quick acceptance:

```powershell
python -m pytest tests/test_build_dataset.py tests/test_dashboard_data.py tests/test_backtesting.py tests/test_yearly_backtesting.py tests/test_recency_comparison.py -q
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
python scripts/compare_recency.py --help
```

3. Review changed files for unrelated churn.
4. Stage only intended files.
5. Commit with concise conventional commit message.

Suggested commit message:

```text
fix(ml): finish thesis acceptance blockers
```

Acceptance:

- Working tree contains only expected generated artifacts and code/doc changes.
- User knows which long artifacts were regenerated and which were not.

## Responsibility Split

Coordinator:

- Own dependency order.
- Prevent long runs before code fixes.
- Track acceptance checklist.
- Avoid revert of user changes.

Data-contract engineer:

- Batch 1 only.
- Must not touch model training unless needed by tests.

Temporal-validation engineer:

- Batch 2 only.
- Own backtesting and recency comparison reproducibility.

ML-methodology engineer:

- Batch 3 only.
- Own leakage, OOF, permutation validation.

Dashboard/model-alignment engineer:

- Batch 4 only.
- Own PR-AUC display, calibrated artifact preference, LOS bucket.

Documentation engineer:

- Batch 5 only.
- Own README/ROADMAP/RESULTS/METHODOLOGY consistency.

Operator/user:

- Batch 6 long commands.
- Decide whether to run full SHAP/heavy training.

Reviewer:

- Batch 7.
- Check diff and acceptance, write commit message.

## How To Work In Batches

Recommended workflow per batch:

1. Read only relevant files.
2. Make minimal scoped edits.
3. Run targeted tests for that batch.
4. Record pass/fail.
5. Stop if dependency breaks.
6. Move to next batch only after current batch passes.

Do not mix:

- Dataset builder fix with dashboard wording.
- Backtesting fix with ensemble OOF.
- Docs reconciliation before code behavior is settled.
- Long training with code edits.

## Agent Pickup Pack

Use these handoff prompts when splitting work across AI agents. Each agent should edit only owned files, run listed tests, and report exact commands plus pass/fail.

Data-contract agent prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Own only Batch 1 from docs/internal/final_completion_task_plan.md.
Fix build_modeling_dataset output schema. Do not touch model training.
Do not revert unrelated user changes. Run:
python -m pytest tests/test_build_dataset.py -q
python -m pytest tests/test_target_definitions.py tests/test_horizon_targets.py -q
Return changed files, test output summary, and any residual risk.
```

Temporal-validation agent prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Own only Batch 2 from docs/internal/final_completion_task_plan.md.
Fix yearly backtesting quick mode and recency comparison module/CLI.
Do not edit dashboard, tuning, ensemble, or docs except tests needed for Batch 2.
Run:
python scripts/evaluate_backtesting.py --help
python scripts/evaluate_backtesting.py --quick --n_bootstraps 5
python scripts/compare_recency.py --help
python scripts/compare_recency.py --quick --n-bootstraps 5 --iterations 20
python -m pytest tests/test_backtesting.py tests/test_yearly_backtesting.py tests/test_recency_comparison.py -q
Return changed files, output CSV columns, and timing.
```

ML-methodology agent prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Own only Batch 3 from docs/internal/final_completion_task_plan.md.
Fix tuning classifier/regression frame separation, stacked ensemble OOF, and validation-split permutation importance.
Do not regenerate reports. Do not run long tuning.
Run:
python -m pytest tests/test_hyperparam_tuning.py tests/test_ensemble.py tests/test_diagnostics_outputs.py -q
Return changed files and how each leakage/overfit risk is now prevented.
```

Dashboard agent prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Own only Batch 4 from docs/internal/final_completion_task_plan.md.
Align dashboard model selection with PR-AUC, prefer calibrated classifier if present, add LOS bucket.
Do not touch data builder or training code.
Run:
python -m pytest tests/test_dashboard_data.py -q
python -m py_compile streamlit_app.py src/aac_adoption/dashboard/data.py
Return changed files and screenshots/manual notes if app was opened.
```

Docs agent prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Own only Batch 5 from docs/internal/final_completion_task_plan.md.
Reconcile README, ROADMAP, RESULTS, METHODOLOGY after code batches pass.
No code changes. Do not alter old archived docs unless current docs link to them.
Run:
rg -n "Primary metric: \*\*ROC-AUC|Use \*\*ROC-AUC\*\* for ranking|adoption speed|predicted days to adoption" README.md docs streamlit_app.py src
python -m pytest tests/test_target_definitions.py tests/test_report_outputs.py -q
Return changed files and remaining intentional matches.
```

Reviewer prompt:

```text
Workspace: C:\Users\paula\Documents\mgr pjatk.
Review current diff against docs/internal/final_completion_task_plan.md.
Prioritize bugs, leakage risks, stale docs, missing tests, and generated artifact drift.
Do not edit unless asked. Run git diff --stat and focused tests if cheap.
Return findings first with file:line references.
```

## Validation Script

Helper script added for agents/operators:

```powershell
.\scripts\validate_final_acceptance.ps1
```

Default mode runs quick compile checks, CLI help, targeted pytest groups, quick backtesting, and quick recency comparison. It should be used after Batches 1-5 are implemented.

Useful variants:

```powershell
# Skip script commands when CLI is known broken and only tests are needed
.\scripts\validate_final_acceptance.ps1 -SkipScripts

# Skip pytest and only check CLIs/quick generated artifacts
.\scripts\validate_final_acceptance.ps1 -SkipPytest

# Full slower acceptance after code is stable
.\scripts\validate_final_acceptance.ps1 -Long
```

Expected default checks:

- dashboard helpers compile
- calibration/backtesting/recency CLI help returns 0
- dataset/dashboard/temporal/method/report targeted tests pass
- quick yearly backtesting exits 0
- quick recency comparison exits 0

The script intentionally does not stage, commit, download data, run SHAP, or run full training unless `-Long` is passed.

## Artifact Validation Matrix

Use this table after quick or long runs to verify generated outputs. Each artifact must either exist with expected columns or be documented as intentionally not regenerated.

| Artifact | Producer | Expected columns / content | Validate with |
|---|---|---|---|
| `reports/metrics/calibrated_classification_metrics.csv` | `scripts/calibrate_classifiers.py` | `animal_subset`, `model_name`, `base_model_name`, `calibration_method`, `pr_auc`, `roc_auc`, `brier_score`, `expected_calibration_error`, `train_rows`, `validation_rows`, `test_rows` | `python - <<'PY'` snippet below |
| `reports/tables/yearly_backtesting.csv` | `scripts/evaluate_backtesting.py` | `train_years`, `test_year`, `model_name`, `animal_subset`, `pr_auc`, `roc_auc`, `brier_score`, `mae`, `train_rows`, `test_rows` | same snippet |
| `reports/tables/recency_strategy_comparison.csv` | `scripts/compare_recency.py` | strategy rows for `full_history`, `recent_5yr`, `recent_3yr`, `recency_weighted`; PR-AUC primary | same snippet |
| `reports/tables/final_model_selection.csv` | analysis/model selection | classification selected by PR-AUC primary, ROC-AUC tie-break | inspect sorted rows |
| `reports/tables/permutation_importance_classification.csv` | diagnostics | includes or documents validation evaluation period | inspect `evaluation_period`/metadata |
| `reports/summary/current_results.md` | `scripts/generate_report_outputs.py` | PR-AUC primary language, no target conflation | `rg` terminology check |
| `reports/artifact_manifest.csv` | `scripts/generate_artifact_manifest.py` | required thesis artifacts present or explained | inspect `required_for_thesis` missing rows |

PowerShell-friendly CSV column check:

```powershell
@'
import pandas as pd
from pathlib import Path

checks = {
    "reports/metrics/calibrated_classification_metrics.csv": {
        "animal_subset", "model_name", "base_model_name", "calibration_method",
        "pr_auc", "roc_auc", "brier_score", "expected_calibration_error",
        "train_rows", "validation_rows", "test_rows",
    },
    "reports/tables/yearly_backtesting.csv": {
        "train_years", "test_year", "model_name", "animal_subset",
        "pr_auc", "roc_auc", "brier_score", "mae", "train_rows", "test_rows",
    },
    "reports/tables/recency_strategy_comparison.csv": {"strategy", "pr_auc", "roc_auc"},
}

for path, required in checks.items():
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing: {path}")
    df = pd.read_csv(p)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    if df.empty:
        raise SystemExit(f"{path} is empty")
    print(f"ok: {path} rows={len(df)}")
'@ | python -
```

Manifest missing-required check:

```powershell
@'
import pandas as pd
df = pd.read_csv("reports/artifact_manifest.csv")
required = df[df["required_for_thesis"].astype(str).str.lower().eq("true")]
missing = required[~required["exists_on_disk"].astype(str).str.lower().eq("true")]
print(missing[["artifact_path", "chapter", "source_script", "notes"]].to_string(index=False))
raise SystemExit(1 if len(missing) else 0)
'@ | python -
```

## Failure Triage Guide

If Batch 1 tests fail:

- First inspect missing column names in `KeyError`.
- Fix schema fill before final column selection.
- Avoid changing tests unless test expectation conflicts with documented target definitions.

If backtesting times out:

- Add/reduce `--quick`.
- Lower iterations/bootstrap.
- Ensure CLI help does not import/load/train heavy objects before parsing args.
- Run on synthetic/tiny fixture first.

If recency comparison times out:

- Make `argparse` parse before dataset loading.
- Add `--max-rows`, `--iterations`, and low default for quick.
- Move training code out of top-level import path.

If CatBoost fails on categorical columns:

- Check `cat_features` are names or indices matching transformed frame.
- Fill categorical nulls with `"Unknown"`.
- Do not pass CatBoost `cat_features` into HistGradientBoosting.

If dashboard prediction fails:

- Check selected model name maps to correct models dir.
- Check calibrated artifact path convention.
- Fall back to base classifier when calibrated artifact absent.
- Use metadata `feature_columns` when available.

If docs tests fail:

- Check current docs only, not archived `docs/old`.
- Keep forbidden terminology in methodology "avoid" tables if tests allow it.
- If `rg` finds old archived text, decide whether archived docs are excluded from acceptance.

## Suggested Test Additions

These are useful if missing. Add only when implementing corresponding batch.

Batch 1:

- Test tiny fixtures with no `outcome_subtype`, `sex_upon_outcome`, or `name`.
- Assert `has_name`, `age_months`, `age_years`, `is_censored`, `event_type`, `followup_days_censored` exist.
- Assert 1000-day LOS remains uncapped in built dataset.

Batch 2:

- Test `evaluate_backtesting.py --quick` writes exactly two test years on synthetic data.
- Test every `test_year` is greater than max parsed `train_years`.
- Test `compare_recency.py --help` exits 0 under 5 seconds.
- Test recency comparison output contains four strategies.

Batch 3:

- Test CatBoost regression tuning uses `reg_train_df` feature matrix, not classification frame.
- Test stacked ensemble meta-estimator receives OOF predictions by using base estimator that memorizes row IDs and would score perfectly if in-sample.
- Test permutation importance output has `evaluation_period == "validation"`.

Batch 4:

- Test `best_model_rows()` picks higher PR-AUC even if ROC-AUC lower.
- Test `los_days_to_bucket()` boundary values: `0`, `7`, `8`, `30`, `31`, `60`, `61`, `90`, `91`.
- Test `predict_from_record()` returns `los_bucket`.
- Test calibrated model path is tried before base path, with fallback.

Batch 5:

- Test generated/current docs contain "PR-AUC primary" or equivalent.
- Test no unqualified "predicted days to adoption" appears outside methodology forbidden-label table.

## Acceptance Checklist

Batch 1:

- [ ] `python -m pytest tests/test_build_dataset.py -q` passes.
- [ ] No global target winsorization in dataset builder.

Batch 2:

- [ ] `python scripts/evaluate_backtesting.py --quick` exits 0.
- [ ] `python scripts/compare_recency.py --help` exits 0 quickly.
- [ ] Backtesting output has required acceptance columns.
- [ ] Recency comparison has all four strategies.

Batch 3:

- [ ] Tuning uses separate classifier/regression frames.
- [ ] Stacked ensemble meta-learner uses OOF predictions.
- [ ] Permutation importance uses validation split.

Batch 4:

- [ ] Dashboard classification winner uses PR-AUC primary.
- [ ] `predict_from_record()` returns LOS bucket.
- [ ] Calibrated classifier preferred when available.

Batch 5:

- [ ] Current docs no longer claim ROC-AUC primary.
- [ ] LOS/adoption timing terminology is consistent.
- [ ] Roadmap statuses match code/tests/artifacts.

Batch 6:

- [ ] Full test suite passes or exceptions are documented.
- [ ] Reports regenerated after final code changes.
- [ ] Artifact manifest refreshed.

Batch 7:

- [ ] Diff reviewed.
- [ ] Intended files staged.
- [ ] Commit message prepared.
