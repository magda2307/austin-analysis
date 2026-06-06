# Original User Request

## Initial Request — 2026-06-06T21:25:07+02:00

Harden the AAC Adoption ML pipeline (`c:\Users\paula\Documents\mgr pjatk`) across 5 targeted slices:
fix immediate blockers, restore end-to-end calibration, move target winsorization to train-only,
fix and wire recency sample weights, and implement yearly temporal backtesting — making the
thesis pipeline methodologically defensible and fully reproducible.

Working directory: `c:\Users\paula\Documents\mgr pjatk`
Integrity mode: development

---

## Requirements

### R1. Slice 0 — Fix blockers and commit staged work

All previously staged work (HANDOFF_0606 slices 1–5) must be committed and pushed.
Before committing:
- Confirm `src/aac_adoption/dashboard/data.py` compiles without error (`python -m py_compile`).
- Confirm `python scripts/calibrate_classifiers.py --help` exits 0 (no ImportError).
- Fix any tests that reference moved stub paths (`docs/target_definitions.md`, `docs/methodology_notes.md`, etc.) — stubs were moved to `docs/old/_stub_*.md`. Tests should be updated to reference `docs/METHODOLOGY.md` or the old path.
- Run `python -m pytest -q` and ensure it passes (was 26 targeted tests passing before doc move).
- Commit message: `"Strengthen ML validation pipeline (slices 1-5)"` then push to `origin main`.

### R2. Slice 6 — Restore end-to-end calibration pipeline

`scripts/calibrate_classifiers.py` must run successfully end-to-end on the real dataset:
```
python scripts/calibrate_classifiers.py --data-path data/processed/modeling_dataset.csv
```
It must produce `reports/metrics/calibrated_classification_metrics.csv` containing rows with columns:
`animal_subset`, `model_name`, `base_model_name`, `calibration_method`, `pr_auc`, `roc_auc`, `brier_score`, `expected_calibration_error`, `train_rows`, `validation_rows`, `test_rows`.
The calibration design must be clean: base model trained on 2013–2021, calibrator fitted on 2022–2023 validation (second half only via `_split_frame_for_calibration`), evaluated on untouched 2024–2025 test.
Add or extend `tests/test_calibration.py` with:
- A test that `scripts/calibrate_classifiers.py --help` exits 0.
- A tiny fixture end-to-end test: synthetic classifier → calibrate → verify output CSV has required columns.
- A test that Platt calibration uses `method="sigmoid"` (not overridden to isotonic).
Update `docs/ROADMAP.md` to mark Slice 3b as DONE.

### R3. Slice 7 — Move target winsorization to train-only

Currently `src/aac_adoption/data/build_dataset.py` winsorizes `regression_target_days` /
`length_of_stay` using percentiles from the **full** dataset — this leaks test-period distribution
into training labels.
Remove global target winsorization from `build_dataset.py`.
In each training module (`train_baseline.py`, `train_boosting.py`, `train_advanced.py`):
fit 1st/99th percentile caps on **train split only**, store cap values in model artifact metadata,
then apply the same frozen caps to validation and test labels.
Add tests asserting:
- `build_dataset()` does NOT winsorize targets globally.
- Train-only cap values are saved in artifact metadata.

### R4. Slice 8 — Fix recency weights formula and wire to model fitting

The recency weight formula in `src/aac_adoption/models/split.py` is **inverted** — it currently
assigns higher weight to older rows. Fix to:
`weight = 1.0 + 0.5 * (year - min_year) / (max_year - min_year)`
so that recent rows receive weight up to 1.5 and oldest rows receive weight 1.0.
Then wire the computed `sample_weight` column to each model's `.fit()` call:
- `train_baseline.py`: pass via `Pipeline.fit(X, y, <step_name>__sample_weight=weights)` where the step supports it (LogisticRegression, RandomForest, Ridge).
- `train_boosting.py`: pass via pipeline fit kwargs.
- `train_advanced.py`: pass via CatBoost `fit(X, y, sample_weight=weights)`.
Add tests asserting:
- 2021 row weight > 2013 row weight.
- `sample_weight` column exists on the returned split result.

### R5. Slice 9 — Implement yearly temporal backtesting

Implement `src/aac_adoption/analysis/backtesting.py` with a function that runs 6 rolling
training windows and evaluates the best selected classifier and regressor on each:
- train 2013–2018 → test 2019
- train 2013–2019 → test 2020
- train 2013–2020 → test 2021
- train 2013–2021 → test 2022
- train 2013–2022 → test 2023
- train 2013–2023 → test 2024

Wire to `scripts/evaluate_backtesting.py` (currently a stub). Must produce
`reports/tables/yearly_backtesting.csv` with columns:
`train_years`, `test_year`, `model_name`, `animal_subset`, `pr_auc`, `roc_auc`, `brier_score`, `mae`, `train_rows`, `test_rows`.
Add `--quick` flag that runs only 2 windows (2018→2019, 2022→2023) for fast CI validation.
Extend `tests/test_backtesting.py`:
- Test each test year is strictly after the training window.
- Test output CSV has all required columns.
- Test with 2 tiny windows on synthetic data (no real data required).
Update `docs/ROADMAP.md` to mark item 10 DONE.

---

## Acceptance Criteria

### R1 — Blockers Fixed
- [ ] `python -m py_compile src/aac_adoption/dashboard/data.py` exits 0
- [ ] `python scripts/calibrate_classifiers.py --help` exits 0
- [ ] `python -m pytest -q` passes (no collection errors, no failures)
- [ ] `git log --oneline -1` shows the new commit on `main`
- [ ] `git status` shows clean working tree after push

### R2 — Calibration Pipeline
- [ ] `python scripts/calibrate_classifiers.py --data-path data/processed/modeling_dataset.csv` exits 0
- [ ] `reports/metrics/calibrated_classification_metrics.csv` exists and contains all required columns
- [ ] `python -m pytest tests/test_calibration.py tests/test_calibration_advanced.py -q` passes
- [ ] New tests cover: `--help` exits 0, end-to-end fixture, Platt keeps sigmoid

### R3 — Winsorization Train-Only
- [ ] `build_dataset.py` contains no global percentile capping of regression targets
- [ ] Artifact metadata JSON for at least one trained regression model contains `target_cap_p1` and `target_cap_p99` keys
- [ ] `python -m pytest tests/test_build_dataset.py -q` passes including new no-global-winsorization test

### R4 — Recency Weights
- [ ] `python -m pytest tests/test_split.py -q` passes including new direction assertion
- [ ] Weights column exists in split result and 2021 row weight > 2013 row weight
- [ ] At least one training module (`train_boosting.py` or `train_advanced.py`) passes `sample_weight` to model fit — verifiable by code inspection or by adding a tiny integration test

### R5 — Yearly Backtesting
- [ ] `python scripts/evaluate_backtesting.py --quick` exits 0
- [ ] `reports/tables/yearly_backtesting.csv` exists with all required columns
- [ ] `python -m pytest tests/test_backtesting.py -q` passes including new tests
- [ ] Each row's `test_year` > max year in `train_years`

### Overall
- [ ] `python -m pytest -q` full suite passes after all slices
- [ ] `docs/ROADMAP.md` updated to reflect completed slices

## Follow-up — 2026-06-06T21:25:07+02:00

Complete the final 8 slices of the AAC Adoption ML thesis pipeline
(`c:\Users\paula\Documents\mgr pjatk`), bringing the project to full thesis-submission
readiness: recency strategy comparison, duration uncertainty outputs, tuning leakage cleanup,
permutation importance fix, ensemble OOF fix, dashboard model alignment, full report
regeneration, and a final acceptance checklist pass.

Working directory: `c:\Users\paula\Documents\mgr pjatk`
Integrity mode: development

**Prerequisite:** Slices 0, 6, 7, 8, 9 are assumed complete (blockers fixed, calibration
pipeline working, winsorization train-only, recency weights wired, backtesting implemented).
Verify with `python -m pytest -q` before starting any slice here.

### R1. Slice 10 — Recency Strategy Comparison
Compare four training strategies on 2024–2025 test period: `full_history`, `recent_5yr`, `recent_3yr`, `recency_weighted`.
Create `src/aac_adoption/analysis/recency_comparison.py` with `compare_recency_strategies()`.
Output: `reports/tables/recency_strategy_comparison.csv`.

### R2. Slice 11 — Duration Uncertainty / LOS Wait-Time Buckets
Add `los_days_to_bucket()` helper, add `los_bucket` to `predict_from_record()`, update dashboard phrasing.
Bucket definitions: `0–7d`, `8–30d`, `31–60d`, `61–90d`, `90+d`.

### R3. Slice 12 — Hyperparameter Tuning Leakage Fix
Separate `X_clf` and `X_reg` feature matrices, in-fold preprocessor fitting, clarify fold strategy in tune.py.

### R4. Slice 13 — Permutation Importance to Validation + Ensemble OOF Fix
Move permutation importance from test→validation. Fix ensemble meta-learner to use OOF predictions.

### R5. Slice 14 — Dashboard Model Selection Alignment
Fix `best_model_rows()` to sort by PR-AUC primary. Prefer calibrated artifacts in `predict_from_record()`.

### R6. Slice 15 — Report Regeneration + RESULTS.md Update
Regenerate all stale reports; update RESULTS.md primary metric to PR-AUC.

### R7. Slice 16 — Context Pipeline Integration + Intake Volume Fix
Add context steps to `run_full_pipeline.py`. Fix intake volume aggregation in `context_data.py`.

### R8. Slice 17 — Final Acceptance Checklist Pass
Fix terminology ("days to adoption" vs "days to outcome"), add bootstrap limitation note, tick all ROADMAP checklist boxes.
