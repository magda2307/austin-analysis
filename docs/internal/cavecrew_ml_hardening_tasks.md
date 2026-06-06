# Cavecrew ML Hardening Tasks

Date: 2026-06-06

Goal: turn `docs/ml_code_review.md` bottom-line risks into agent-sized implementation tasks. Keep tasks small, mostly 1-2 files each, so `cavecrew-builder` can execute without stepping on other work.

## Ground Rules

- Do not revert existing user or agent edits.
- Before editing, run `git status --short`.
- Prefer one task per branch/commit.
- Each builder owns only listed files.
- Each task final output must use cavecrew-builder format:

```text
<path:line-range> -- <change <=10 words>.
verified: <command or re-read OK>
```

- Each review final output must use cavecrew-reviewer format:

```text
path:line: severity: problem. fix.
totals: N bug N risk N nit N q
```

## Task 1 - Repair Calibration CLI

Use: `cavecrew-builder`

Priority: P0

Own files:

- `src/aac_adoption/models/calibrate.py`
- `scripts/calibrate_classifiers.py`
- tests only if needed: `tests/test_calibration.py`

Problem:

- `python scripts/calibrate_classifiers.py --help` fails because `calibrate_classifiers` is imported but missing.
- `scripts/run_full_pipeline.py` calls this broken step.

Acceptance:

- `python scripts/calibrate_classifiers.py --help` exits 0.
- Missing source artifacts produce a clear "no models calibrated" result, not import crash.
- Add or update test covering CLI import/help path if local pattern exists.

Suggested fix:

- Implement `calibrate_classifiers(...)` API returning an output dataclass with `classification_metrics`.
- Or update CLI to call existing calibration helpers directly.
- Keep source artifact paths explicit.
- Do not redesign all calibration math here; Task 2 owns that.

Commit:

```text
fix(calibration): restore classifier CLI
```

Review focus:

```text
scripts/calibrate_classifiers.py:L7: bug: CLI imports missing symbol. Restore API or update import.
```

## Task 2 - Make Calibration Methodologically Clean

Use: `cavecrew-builder`

Priority: P0

Own files:

- `src/aac_adoption/models/calibrate.py`
- `src/aac_adoption/models/train_advanced.py`
- `tests/test_calibration.py`

Problem:

- Isotonic helper clones fitted model and refits only on calibration data.
- Platt path silently becomes isotonic.
- `post_hoc_calibration_pipeline()` uses `train+val` CV instead of true holdout calibration.
- CatBoost validation used for early stopping and calibration.
- Advanced metrics mix uncalibrated discrimination with calibrated Brier/ECE.

Acceptance:

- `calibrate_with_platt()` uses sigmoid.
- Prefit/FrozenEstimator calibration uses already fitted model and calibration split only.
- Advanced classification emits coherent metrics: either all calibrated or separate `catboost` and `catboost_calibrated` rows.
- Test asserts method mapping and no metric mixing.

Suggested fix:

- Keep `apply_calibration_to_predictions()` as canonical post-hoc path.
- For CatBoost, split existing validation into early-stop/calibration halves by time order, or disable reuse and document behavior.
- Save calibrated metadata with method and calibration rows.

Commit:

```text
fix(calibration): separate calibrated metrics
```

Review focus:

```text
src/aac_adoption/models/train_advanced.py:L170: risk: row mixes calibrated and uncalibrated metrics. Emit separate rows.
```

## Task 3 - Handle LOS Target Transform Without Test Leakage

Use: `cavecrew-builder`

Priority: P0

Own files:

- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/models/train_advanced.py`
- tests if needed: `tests/test_build_dataset.py` or new focused test

Problem:

- `length_of_stay`, `days_to_outcome`, and `regression_target_days` are winsorized in dataset build using full dataset quantiles before split.

Acceptance:

- Dataset build preserves raw observed LOS targets.
- Any winsor/cap transform is fit on train only and applied with stored caps.
- Model metadata records target transform and cap values when used.
- Test proves extreme test value does not change train cap.

Suggested fix:

- Remove pre-split winsorization from `build_modeling_dataset()`.
- Add helper in model training layer: fit train caps, transform train/validation/test copies.
- Or choose no target winsorization and rely on MAE/log transform.

Commit:

```text
fix(regression): remove target cap leakage
```

Review focus:

```text
src/aac_adoption/data/build_dataset.py:L101: bug: target caps use full dataset. Fit caps on train only.
```

## Task 4 - Define Censoring and Safe Horizon Labels

Use: `cavecrew-investigator` first, then `cavecrew-builder`

Priority: P0

Investigator question:

```text
Find all target definitions, horizon targets, censoring flags, survival outputs, and report/dashboard text that mention days-to-adoption, time-to-outcome, LOS, or censoring. Return path:line list only.
```

Builder own files after investigation:

- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/models/split.py`
- possibly `docs/target_definitions.md`

Problem:

- Unmatched/current intakes vanish from modeling frame.
- `censoring_flag` exists but has no effect.
- Horizon labels do not prove enough follow-up exists for late intake dates.

Acceptance:

- Code has explicit censor date / data extract date.
- Horizon labels include observed-safe masks or late-tail exclusion.
- Split/evaluation excludes unsafe tail rows or marks them as censored.
- Documentation states open/intake-tail limitation.

Suggested fix:

- Add `extract_end_date` parameter where data is built from files.
- Add `followup_days_available`.
- For `adopted_in_30d`, only evaluate rows with `followup_days_available >= 30` unless outcome observed sooner.
- Keep survival/descriptive views separate from supervised classification.

Commit:

```text
fix(targets): add censoring safeguards
```

Review focus:

```text
src/aac_adoption/models/split.py:L66: risk: `censoring_flag` unused. Enforce or remove claim.
```

## Task 5 - Wire Recency Sample Weights

Use: `cavecrew-builder`

Priority: P1

Own files:

- `src/aac_adoption/models/split.py`
- `src/aac_adoption/models/train_baseline.py`
- `src/aac_adoption/models/train_boosting.py`
- `src/aac_adoption/models/train_advanced.py`

This may exceed 2 files. Split if needed:

- 5A: fix formula/tests in `split.py`.
- 5B: pass weights in sklearn trainers.
- 5C: pass weights in CatBoost trainer.

Problem:

- Formula weights older rows more.
- `sample_weight` column is not passed to learners.

Acceptance:

- 2021 train rows get higher weight than 2013 rows.
- Sklearn `Pipeline.fit()` receives `model__sample_weight` when supported.
- CatBoost fit receives `sample_weight`.
- Metadata records whether recency weights used.

Suggested fix:

- Formula: normalize year progress from min train year to max train year.
- Helper: `training_sample_weight(split.train) -> Series | None`.
- Avoid passing weights to `Dummy*` if unsupported or pointless.

Commit:

```text
fix(training): apply recency weights
```

Review focus:

```text
src/aac_adoption/models/split.py:L78: bug: recency formula favors old rows. Reverse weighting.
```

## Task 6 - Fix Non-Leaky Tuning

Use: `cavecrew-builder`

Priority: P1

Own files:

- `src/aac_adoption/models/tune.py`
- `tests/test_hyperparam_tuning.py` or new test near tuning tests

Problem:

- Tuning uses only last `TimeSeriesSplit` fold despite CV wording.
- Regression labels can misalign with classification feature indices.
- HGB preprocessor fits before CV split.

Acceptance:

- Classification/regression tuning build independent frames and CV indices.
- Objective averages across chronological folds, or names single holdout clearly.
- Preprocessor lives inside fold loop/Pipeline and fits only on fold train.
- Test catches regression frame/index alignment.

Suggested fix:

- Factor `_time_cv_splits(frame)`.
- Factor `_evaluate_hist_trial(params, train_idx, val_idx)`.
- Build `reg_X` from `reg_train_df` and separate `reg_splits`.

Commit:

```text
fix(tuning): avoid fold leakage
```

Review focus:

```text
src/aac_adoption/models/tune.py:L146: bug: preprocessor fit before CV split leaks validation categories. Fit inside fold.
```

## Task 7 - Keep Test Set Final-Only

Use: `cavecrew-builder`

Priority: P1

Own files:

- `src/aac_adoption/models/train_boosting.py`
- tests if available: `tests/test_train_boosting_outputs.py`

Problem:

- Permutation importance samples from `split.test`.

Acceptance:

- Permutation importance uses validation when nonempty.
- Falls back to test only for tiny/no-validation synthetic cases, with metadata field `importance_split`.
- Test asserts validation path when validation exists.

Commit:

```text
fix(interpretation): use validation importance
```

Review focus:

```text
src/aac_adoption/models/train_boosting.py:L117: risk: test split used for interpretation. Use validation by default.
```

## Task 8 - Dashboard Uses Final Selected/Calibrated Model

Use: `cavecrew-builder`

Priority: P2

Own files:

- `src/aac_adoption/dashboard/data.py`
- `streamlit_app.py` only if needed

Problem:

- Dashboard ranks classification by ROC-AUC while thesis selection may use PR-AUC.
- Sensitivity demo hardcodes advanced CatBoost path instead of final selected/calibrated artifact.

Acceptance:

- Dashboard reads `reports/tables/final_model_selection.csv` when present.
- Sensitivity demo uses selected artifact path and calibrated classifier when available.
- Fallback to advanced CatBoost remains for missing selection artifact.

Commit:

```text
fix(dashboard): follow selected model
```

Review focus:

```text
src/aac_adoption/dashboard/data.py:L119: risk: dashboard picks ROC-AUC best, not final selected model. Read selection artifact.
```

## Task 9 - Add Reproducible Run Metadata

Use: `cavecrew-builder`

Priority: P2

Own files:

- `src/aac_adoption/models/artifacts.py`
- `src/aac_adoption/models/metadata.py`
- possibly `scripts/generate_environment_snapshot.py`

Problem:

- Sidecar JSON can miss own `artifact_path`.
- Calibrated artifacts lack standard sidecars.
- Data hash/package versions/git SHA not consistently attached to model runs.

Acceptance:

- `save_model_artifact()` injects `artifact_path` before writing JSON.
- Metadata includes feature list, params, split rows, timestamp, random state, data path/hash where available.
- Calibrated artifacts use same sidecar convention.

Commit:

```text
fix(artifacts): store run metadata
```

Review focus:

```text
src/aac_adoption/models/artifacts.py:L35: risk: JSON written before path injected. Add `artifact_path` in helper.
```

## Task 10 - Add CI-Safe Tiny Fixtures

Use: `cavecrew-builder`

Priority: P2

Own files:

- `tests/fixtures/` new fixture files
- `tests/models/test_reproducibility.py`
- `tests/test_artifact_manifest.py`
- `tests/test_calibration.py`

Problem:

- Several tests skip when generated local artifacts/data are absent.
- Clean clone can pass without proving full core ML semantics.

Acceptance:

- Tiny fixture dataset supports deterministic split, one small model train, calibration CLI import, artifact manifest smoke.
- Existing local-data skips remain only for expensive integration tests.
- CI can catch broken calibration import and model semantics drift.

Commit:

```text
test(ml): add tiny pipeline fixtures
```

Review focus:

```text
tests/models/test_reproducibility.py:L15: risk: test skips without local dataset. Add versioned tiny fixture.
```

## Suggested Parallel Work Plan

Round 1:

- Agent A: Task 1.
- Agent B: Task 3.
- Agent C: Task 6.
- Main thread: Task 4 investigator, then decide censoring design.

Round 2:

- Agent A: Task 2 after Task 1 merges.
- Agent B: Task 5 split into 5A/5B/5C.
- Agent C: Task 7.

Round 3:

- Agent A: Task 8.
- Agent B: Task 9.
- Agent C: Task 10.

Final review:

- Run `pytest`.
- Run `python scripts/calibrate_classifiers.py --help`.
- Run tiny full-pipeline smoke if Task 10 adds one.
- Use `cavecrew-reviewer` on diff before commit.

## Commit Batch Suggestions

Use these after each task, not one giant commit:

```text
fix(calibration): restore classifier CLI
fix(calibration): separate calibrated metrics
fix(regression): remove target cap leakage
fix(targets): add censoring safeguards
fix(training): apply recency weights
fix(tuning): avoid fold leakage
fix(interpretation): use validation importance
fix(dashboard): follow selected model
fix(artifacts): store run metadata
test(ml): add tiny pipeline fixtures
```

