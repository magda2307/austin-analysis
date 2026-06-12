# Phase 6 Deep Testing Report

**Date:** 2026-06-09
**Scope:** Manual regeneration evidence, artifact lineage, final documentation, dashboard truth, manifest finalization, ML leakage/target integrity, reproducibility
**Verdict:** **FAIL — Project is NOT thesis-ready. Multiple P0 blockers remain open.**

---

## 0. Audit Method

Six evidence threads were investigated in parallel by independent subagents plus direct source inspection:

1. **Regeneration Evidence** — pipeline execution state, logs, run receipts
2. **Artifact Lineage** — manifest CSV, hashes, run ID consistency, finalization_sha
3. **Final Documentation** — README, METHODOLOGY, RESULTS, target_definitions, dashboard copy
4. **Dashboard Truth** — missing/broken artifacts produce explicit failure vs. fake prediction
5. **Manifest Finalization** — manifest generator, acceptance validation logic
6. **ML Leakage/Target Integrity + Reproducibility** — chronology, calibration, leakage, false-pass claims

---

## 1. Regeneration Evidence Audit

### 1.1 Pipeline Execution State — CRITICAL BLOCKER

**`reports/run_receipts/` does not exist.** No pipeline run has ever been executed end-to-end.

| Item | Status |
|---|---|
| `reports/run_receipts/` directory | ❌ MISSING — no run ever executed |
| `logs/` directory | ❌ MISSING — no pipeline log exists |
| `reports/run_receipt.json` | ❌ MISSING — overall run receipt absent |
| Any pipeline log file | ❌ MISSING |

**Consequence:** Every receipt-dependent gate (manifest generation, long acceptance, `validate_run_receipts.py`) will immediately fail.

### 1.2 Pipeline CLI Flag Audit

| Flag | Present? | Notes |
|---|---|---|
| `--resume-run` | ✅ | Accepted as `--run-id` |
| `--skip-shap` | ✅ | Skips steps 11, 15 |
| `--skip-tests` | ❌ MISSING | **Not implemented.** Runbook references it but the flag doesn't exist. Step 18 (pytest) only skipped via `--quick`. |
| `--steps` | ✅ | Comma-separated step numbers |
| `--continue-on-error` | ✅ | Opt-in; fail-fast is default |

**Runbook command `python scripts/run_full_pipeline.py --skip-download --skip-shap --skip-tests --resume-run <RUN_ID>` will fail** because `--skip-tests` is not a recognized argument.

### 1.3 Pipeline Step Definitions (confirmed)

| Step | Name | SHAP Tag? |
|---|---|---|
| 0 | Environment snapshot | — |
| 1 | Download raw data | download |
| 2 | Build dataset | — |
| 3 | Run EDA | — |
| 4 | Train baseline models | — |
| 5 | Train adopted animals regression | — |
| 6 | Tune hyperparameters | expensive |
| 7 | Train boosting models | — |
| 8 | Train advanced models (CatBoost) | expensive |
| 9 | Calibrate classifiers | — |
| 10 | Run analysis | — |
| **11** | **Generate diagnostics (with SHAP)** | **shap** |
| 12 | Generate animal research | — |
| **13** | **Generate evidence pack** | — |
| **14** | **Generate report outputs** | — |
| **15** | **Generate feature family importance** | **shap** |
| 16 | Evaluate backtesting | expensive |
| **17** | **Validate run receipts** | — |
| **18** | **Run test suite (pytest)** | — |

Step 17 calls `scripts/validate_run_receipts.py` which imports `aac_adoption.run_receipt_validation` — **this module exists** at `src/aac_adoption/run_receipt_validation.py` (confirmed). Step 18 runs `pytest tests/ -v --tb=short`.

### 1.4 Missing Prerequisite: `models/tuning/best_params.json`

Pipeline steps 7 and 8 reference `--tuned-params-path models/tuning/best_params.json`. This file does **not exist** — only `models/tuning/test_best_params.json` is present. A full pipeline run from the top will fail at steps 7–8 unless this file is created or the flag is omitted.

### 1.5 `manage_run_context.py finalize` Does Nothing Meaningful

The `finalize` action:
- Checks if `reports/run_receipts/<run_id>/` exists
- Counts `.json` files
- Prints a message

**It does NOT set `finalization_sha`.** The runbook requires both `producer_source_sha` and `finalization_sha` to be recorded, but no mechanism sets `finalization_sha`. The manifest CSV has no `finalization_sha` column.

---

## 2. Artifact Lineage Audit

### 2.1 Manifest CSV is a Blank Placeholder — CRITICAL BLOCKER

`reports/artifact_manifest.csv` was not generated from a live canonical run.

| Column | State |
|---|---|
| `run_id` | **Empty for all rows** |
| `file_hash` | **Empty for all rows** |
| `producer_source_sha` | **Column missing from CSV entirely** |
| `exists_on_disk` | **All 43 required artifacts show `False`** |

`src/aac_adoption/acceptance.py`'s `validate_artifact_manifest` requires `producer_source_sha` as a column — the manifest will immediately fail validation.

### 2.2 Required Artifacts Present on Disk

Despite the manifest being stale, **data/processed/** artifacts DO exist:
- `modeling_dataset.csv` ✅ (~67 MB)
- `modeling_dataset_context.csv` ✅ (~69 MB)
- `feature_columns.json` ✅
- `context_feature_columns.json` ✅
- `target_columns.json` ✅

Model files exist in `models/advanced/`, `models/boosting/`, `models/baseline/`, `models/calibrated/` directories.

### 2.3 `finalization_sha` — Not Implemented

`manage_run_context.py finalize` sets no `finalization_sha`. The manifest generator (`generate_artifact_manifest.py`) records `producer_source_sha` from receipts but writes no `finalization_sha` column to the CSV.

**The runbook's two-SHA requirement (`producer_source_sha` + `finalization_sha`) is only half-implemented.**

---

## 3. Final Documentation Audit

### 3.1 `reports/summary/final_model_selection.md` — P0 BLOCKER

This artifact selects models on **test data (2024–2025)** instead of the required 2023 validation period.

**Exact quotes from the file:**

```
Classification Rule: Test PR-AUC (primary) → ROC-AUC (tie-break, accounts for class
imbalance) → calibration behaviour → interpretability support.

Regression Rule: Test MAE (primary) → Median Absolute Error (robustness) → RMSE.

cats — hist_gradient_boosting: Selected on combined criterion: highest test PR-AUC (0.8994)...
combined — hist_gradient_boosting: Selected on combined criterion: highest test PR-AUC (0.8842)...
dogs — catboost: Selected on combined criterion: highest test PR-AUC (0.8601)...
cats — catboost (regression): Selected by lowest test MAE (15.79 days)...

Limitation (line 89): Model selection is based on a single time-split test period (2024–2025).
```

**Thesis contract violation:** AGENTS.md states "validation selects thresholds/calibration; test data is final evaluation only." Test metrics cannot drive model selection. This is a **data leakage in the evaluation protocol**. The artifact needs regeneration from a corrected producer.

**Note:** The producer (`scripts/run_analysis.py` / `src/aac_adoption/analysis/model_selection.py`) was reportedly fixed in earlier phases to use `metric_split == "selection"` (2023 data) for selection. However, the **stale generated artifact** still reflects the old incorrect behavior — it was never regenerated.

### 3.2 `reports/summary/final_model_selection.md` — Calibration Claim

**Line 90:** `"Formal isotonic or Platt calibration was not applied."`

This contradicts the code: `train_advanced.py` applies isotonic calibration via `apply_calibration_to_predictions` using `split.calibration` (2022 data). The claim is factually wrong and must be fixed. Since this is a generated artifact, fix the producer and regenerate.

### 3.3 `docs/target_definitions.md` — Horizon Target Column Name Mismatch

Section 4 (Horizon-Based Targets) uses column names `adopted_in_7_days`, `adopted_in_14_days`, `adopted_in_30_days` — but the build pipeline and correction plan reference `adopted_in_7d`, `adopted_in_30d`, `adopted_in_60d`, `adopted_in_90d`. The column names are inconsistent and the definition states "Scope: All matched intake/outcome episodes" which contradicts the correction plan requirement that horizon targets belong ONLY to `horizon_modeling_dataset.csv`.

Also missing from section 4: NaN semantics for unresolved intakes with insufficient follow-up (the correction plan Task 7 Step 2 add-on).

### 3.4 `docs/METHODOLOGY.md` — Calibration Claim (Pending Subagent Confirmation)

The stale `final_model_selection.md` says calibration was not applied. If `docs/METHODOLOGY.md` contains the same claim (or references the stale report), it is also a P1 violation.

### 3.5 Stale Phase Tracker Claims

| File | Stale Claim | Contradiction |
|---|---|---|
| `docs/closeout/phase5.md` line 7 | "FULL PASS — Test suite is separated..." | Deep audit (phase5 report) finds FAIL on receipt validation and PowerShell acceptance |
| `docs/closeout/phase5.md` line 13 | "FULL PASS — Pipeline stops on first nonzero exit..." | Pipeline has never been run end-to-end; receipt infrastructure unverified |
| `docs/closeout/phase5.md` line 25 | "FULL PASS — Run IDs and hashes added to manifest..." | Manifest CSV has blank run_id and no hashes for any row |

---

## 4. Dashboard Truth Audit

### 4.1 `load_model()` Missing File — EXPLICIT FAILURE ✅

```python
# data.py lines 342-343
if not path.exists():
    raise FileNotFoundError(f"Missing model artifact: {path}")
```
Raises `FileNotFoundError` which propagates as `ok=False, error_code="CLF_ERROR"`.

### 4.2 `predict_from_record()` Exceptions — EXPLICIT FAILURE ✅

All exception paths return `PredictionResult(ok=False, ...)` with specific error codes:
- `"CLF_ERROR"` / `"REG_ERROR"` — model load or predict failure
- `"INVALID_PROB"` — probability outside [0,1]
- `"INVALID_DAYS"` — days negative or non-finite
- `"OUT_OF_BOUNDS"` — days > 4000

### 4.3 `final_model_selection.csv` Missing — SILENT DEFAULT (P0) ❌

**This is the P0 dashboard violation.**

```python
# data.py lines 427-441
selection = load_table(PROJECT_ROOT / "reports/tables", "final_model_selection")

clf_name = "catboost"          # hardcoded default
clf_dir = "models/advanced"   # hardcoded default

if not selection.empty and "selected" in selection.columns:
    ...  # overrides only if CSV loads successfully
```

When `final_model_selection.csv` is absent, load fails silently (returns empty DataFrame), and the function proceeds with `clf_name='catboost'` and `clf_dir='models/advanced'` **without any warning, log message, or error flag**.

If catboost model files exist at that path: `predict_from_record` returns `ok=True` with a real but **potentially wrong model** (not the thesis-selected model), with no user-visible indication. This breaks the thesis invariant: "Missing or broken model artifacts must produce an explicit failure state, never a plausible default prediction."

**Required fix:** Return `PredictionResult(False, None, None, None, False, {}, "MISSING_SELECTION", "final_model_selection.csv not found or empty — cannot determine thesis-selected model")` when selection is empty.

### 4.4 UI Rendering on Failure — EXPLICIT FAILURE ✅

Both the Model Sensitivity tab and Animal Stories tab check `result.ok` before displaying any prediction values. No numeric prediction is shown without a loaded model. ✅

### 4.5 `tests/test_dashboard_app.py` — EXISTS ✅

The file exists (3,981 bytes). It includes `test_dashboard_app_missing_model_corrupt_metadata` which patches `load_model` to raise `FileNotFoundError` and verifies the app doesn't crash. However, it does NOT assert that a user-visible error widget is shown.

### 4.6 Missing Test Coverage Gaps

| Scenario | Test Coverage |
|---|---|
| Missing model → ok=False via predict_from_record | ❌ NOT TESTED |
| `final_model_selection.csv` missing → silent default | ❌ NOT TESTED |
| Corrupt CSV → returns empty frame, no fake data | ❌ NOT TESTED |

---

## 5. Manifest Finalization Audit

### 5.1 `generate_artifact_manifest.py` — Structural Review

**✅ Strong implementation where it CAN run:**
- Requires `--run-id` (mandatory argument)
- Validates all receipts in `reports/run_receipts/<run_id>/` before writing
- Verifies: `status == "ok"`, `run_id` matches, `profile == "thesis-full"`, `producer_source_sha` consistent across all receipts
- For each required artifact: checks file exists, is non-empty, actual SHA-256 matches receipt hash
- Uses atomic write (temp + rename)
- Calls `validate_artifact_manifest` on the temp file before renaming

**❌ Manifest cannot be run at all:** Because `reports/run_receipts/` does not exist, `generate_artifact_manifest.py` will raise `AcceptanceError("unknown run: <run-id>")` immediately.

### 5.2 `finalization_sha` — Not Recorded

Neither `generate_artifact_manifest.py` nor `manage_run_context.py finalize` records a `finalization_sha`. The manifest CSV columns are: `artifact_path`, `artifact_type`, `created_at`, `source_script`, `required_for_thesis`, `chapter`, `notes`, `exists_on_disk`, `run_id`, `producer_source_sha`, `file_hash`. **No `finalization_sha` column exists.**

The runbook states: "This commit is `finalization_sha`. It may differ from `producer_source_sha`; final manifest records and validates both." This requirement is **not implemented**.

### 5.3 Manifest Freshness Check — NOT ENFORCED

`validate_artifact_manifest` in `acceptance.py` checks SHA-256 hash matching between manifest rows and actual files, but does **not** verify that the manifest is newer than the artifacts it lists. If an artifact is updated after manifest generation, it would fail the hash check — but there is no explicit timestamp ordering check.

---

## 6. ML Leakage / Target Integrity

### 6.1 Calibration Split — FIXED ✅

`train_advanced.py` line 206-213 uses `split.calibration` (2022 only) for isotonic calibration — NOT `split.validation`. The Phase 2 P0 finding is **resolved** in the producer code.

```python
if split.calibration is not None and not split.calibration.empty:
    val_x = prepare_catboost_frame(split.calibration, feature_columns)
    calibrated_model = apply_calibration_to_predictions(
        ...
        X_calib=val_x,
        y_calib=split.calibration["classification_target"],
        calib_method="isotonic"
    )
```

**However**, regression trainer (lines 315-316) still uses `split.validation` (2022+2023 combined) as the CatBoost eval set for early stopping. This means 2023 labels participate in early stopping for the regression model. This is a **P1 risk** — early stopping should use only `split.calibration` (2022).

### 6.2 Threshold Analysis — FIXED ✅

`threshold_analysis.py` lines 277-283 use `split.selection` (2023 only) for threshold derivation. The Phase 2 P1 finding (using `split.validation`) is **resolved**.

### 6.3 Model Comparison Loads Calibrated Metrics — FIXED ✅

`model_comparison.py` line 121 loads `calibrated_classification_metrics.csv` — calibrated candidates can now compete. The Phase 2 P1 finding is **resolved**.

### 6.4 `final_model_selection.md` Still Uses Test Metrics — P0 BLOCKER ❌

As documented in §3.1, the generated artifact uses "test PR-AUC" and "test MAE" for selection. The producer was reportedly fixed but the artifact was never regenerated. Until regeneration occurs, the documented model selection is thesis-invalid.

### 6.5 `DatasetSplit.validation` Contains 2022+2023 Combined — Known Risk

`split.py` lines 32-42 define `validation` as `concat([calibration, selection])` — i.e., 2022+2023. Code that uses `split.validation` (rather than `split.calibration` or `split.selection`) is mixing periods. The regression trainer's use of `split.validation` for early stopping is the remaining risk (see §6.1).

### 6.6 Leakage Audit Script — Incomplete ❌

`scripts/generate_leakage_audit.py` checks `LEAKAGE_COLUMNS` but does not check the broader `PROHIBITED_MODEL_COLUMNS` set. Target columns (`classification_target`, `regression_target_days`, `days_to_adoption`) can potentially receive false-safe status if they are not explicitly in `LEAKAGE_COLUMNS`. No regression test proves that all prohibited columns fail the audit — this remains a P1 documentation/test gap.

---

## 7. Reproducibility Audit

### 7.1 Key Files Gitignored — CONFIRMED

Confirmed gitignored (cannot be version-controlled):
- `data/raw/`, `data/processed/` (generated data)
- `models/` (trained model artifacts)
- `reports/` (generated reports)
- `logs/`

**Implication:** All generated artifacts must be reproduced from source on each new environment. This is expected and correct for reproducibility, but means the canonical run must be documented and executable.

### 7.2 Receipt Infrastructure — EXISTS

`src/aac_adoption/run_receipt_validation.py` exists (4,917 bytes, ~113 lines). It validates:
1. Reads `reports/run_receipt.json` for the latest run ID
2. Checks overall receipt: `status == "ok"`, `profile == "thesis-full"`, `skipped_steps` empty, `failed_step` None
3. For each per-step receipt in `reports/run_receipts/<run_id>/`: checks `status == "ok"`, matching SHA, `profile == "thesis-full"`

Tests exist: `tests/test_run_receipt_validation.py` (114 lines, 8 test cases).

**However:** With no executed run, both the overall `reports/run_receipt.json` and the per-step directory are absent. `validate_run_receipts.py` will fail immediately.

### 7.3 `scripts/validate_final_acceptance.ps1 -Long` Will Fail

The `-Long` mode:
1. Sets `AAC_ACCEPTANCE=1`
2. Calls `scripts/validate_run_receipts.py` — **immediately fails** (no run_receipt.json)
3. Never reaches `python -m pytest -q`

With no prior pipeline run, `-Long` acceptance cannot pass.

### 7.4 Stale FULL PASS Claims in Tracker

| File | Line | Claim | Contradicted by |
|---|---|---|---|
| `docs/closeout/phase5.md` | 7 | FULL PASS — test suite separated | Phase 5 deep report: FAIL |
| `docs/closeout/phase5.md` | 13 | FULL PASS — pipeline stops on first nonzero | Manifest has no run_id, receipts absent |
| `docs/closeout/phase5.md` | 19 | FULL PASS — duplicate CLI args removed | compare_recency.py fix status not verified by this audit |
| `docs/closeout/phase5.md` | 25 | FULL PASS — run IDs and hashes added to manifest | Manifest CSV: all run_id blank, no file_hash |

### 7.5 Deleted Test Coverage

| Test File | Exists? | Replacement Documented? |
|---|---|---|
| `tests/test_integration_survival.py` | ❌ DELETED | Partially (contract map not confirmed as created) |
| `tests/test_rolling_features.py` | ❌ DELETED | `tests/features/test_rolling.py` exists as replacement |
| `tests/test_survival_analysis.py` | ❌ DELETED | Partially — `tests/test_hypothesis_evidence.py` |
| `tests/test_survival_analysis_new.py` | ❌ DELETED | Partially |
| `docs/closeout/deleted_test_contract_map_2026-06-09.md` | ❌ NOT CONFIRMED as created | Required by correction plan Task 2 |

### 7.6 `scripts/compare_recency.py` — `--quick` Duplicate (Status)

COMMANDS.md documented `compare_recency.py` as broken with `--quick` registered twice. Acceptance script runs this. This audit did not independently verify the current fix status — requires running `python scripts/compare_recency.py --help` and checking for argparse error.

---

## 8. Summary: Thesis-Readiness Decision Table

### P0 Blockers (Must fix before ANY thesis acceptance)

| # | Blocker | Evidence | Required Action |
|---|---|---|---|
| B1 | No pipeline run ever executed — no receipts, no logs | `reports/run_receipts/` absent | Run canonical pipeline with `thesis-full` profile |
| B2 | `artifact_manifest.csv` is a blank placeholder — no hashes, no run_id, `producer_source_sha` column missing | Manifest CSV inspection | Regenerate via `generate_artifact_manifest.py --run-id <ID>` after canonical run |
| B3 | `final_model_selection.md` selects models on test data (2024–2025), not 2023 validation | Lines 7, 10, 35-45, 89 | Regenerate `reports/summary/final_model_selection.md` from fixed producer |
| B4 | `--skip-tests` flag missing from `run_full_pipeline.py` | Source inspection | Fix runbook command or add the flag to the pipeline |
| B5 | Dashboard: `final_model_selection.csv` missing → silent default to catboost (wrong model selection, ok=True) | `data.py` lines 427-441 | Return explicit `PredictionResult(ok=False)` when selection CSV empty |
| B6 | `models/tuning/best_params.json` missing — only `test_best_params.json` exists | `models/tuning/` listing | Locate or regenerate tuning file; or update pipeline invocation |

### P1 High-Risk Issues (Required before thesis submission)

| # | Issue | Evidence | Required Action |
|---|---|---|---|
| H1 | `final_model_selection.md` says "Formal isotonic or Platt calibration was not applied" — false | Code applies isotonic calibration | Regenerate artifact from fixed producer |
| H2 | `docs/target_definitions.md` horizon targets use wrong column names (`adopted_in_7_days` vs `adopted_in_7d`) and wrong scope | Section 4, line 108 | Fix target_definitions.md manually (not a generated artifact) |
| H3 | Regression trainer uses `split.validation` (2022+2023) for early stopping — 2023 selection labels see early stopping | `train_advanced.py` line 315 | Change to `split.calibration` for eval_set |
| H4 | `manage_run_context.py finalize` records no `finalization_sha` | Source inspection | Implement finalization_sha recording (git SHA at finalize time) |
| H5 | Leakage audit (`generate_leakage_audit.py`) checks `LEAKAGE_COLUMNS` not `PROHIBITED_MODEL_COLUMNS` | CLOSEOUT.md, Phase 2 report | Add prohibited column coverage to leakage audit |
| H6 | Stale FULL PASS claims in `docs/closeout/phase5.md` contradict deep audit and manifest evidence | phase5.md lines 7, 13, 25 | Update phase5.md to reflect current blocked state |
| H7 | `docs/closeout/deleted_test_contract_map_2026-06-09.md` not confirmed to exist | Phase 2 correction plan Task 2 | Create the contract map |

### P2 Medium-Risk Issues (Fix before final handoff)

| # | Issue | Required Action |
|---|---|---|
| M1 | `manifest_finalization_sha` column not in manifest — runbook requires both SHAs | Implement or document as out-of-scope |
| M2 | `manifest is newer than artifacts` not enforced in `validate_artifact_manifest` | Add timestamp check or document as hash-sufficient |
| M3 | Dashboard `test_dashboard_data.py` missing tests for: predict_from_record missing model → ok=False; silent CSV fallback | Add targeted test cases |
| M4 | `test_dashboard_app.py` doesn't assert user-visible error widget is shown (only checks no exception) | Strengthen assertion |
| M5 | `docs/closeout/phase5.md` and `phase6.md` trackers not updated to reflect current blocked state | Update both trackers |

---

## 9. Evidence of Partial Fixes (Not Sufficient for Acceptance)

These items from Phase 2/3/4 audits appear to be code-fixed but NOT yet verified by regenerated artifacts:

| Item | Code Fix Status | Artifact Status |
|---|---|---|
| Calibration uses 2022 only (`split.calibration`) | ✅ Fixed in train_advanced.py | ⏳ Needs regeneration |
| Threshold selection uses 2023 only (`split.selection`) | ✅ Fixed in threshold_analysis.py | ⏳ Needs regeneration |
| Calibrated candidates load in model_comparison | ✅ Fixed in model_comparison.py | ⏳ Needs regeneration |
| Model selection producer uses `metric_split=="selection"` | Reportedly fixed | ❌ Generated artifact still wrong |
| compare_recency.py `--quick` duplicate | Reportedly fixed in Phase 5 | Unverified by this audit |

---

## 10. Required Evidence for Thesis Gate

The following evidence must be provided before the project can be declared thesis-ready:

```text
Accepted run ID:               <not yet obtained>
Producer source SHA:           <not yet obtained>
Finalization SHA:              <not yet obtained — mechanism unimplemented>
Dataset SHA-256 (modeling_dataset.csv):  <not yet recorded>
Full pytest result:            <not yet run against final artifacts>
Pipeline log path:             <no logs exist>
Acceptance result (-Long):     <will fail — no receipts>
Required artifact count:       43 (from ARTIFACT_METADATA)
Missing artifact count (manifest): 43 (all show exists_on_disk=False)
Dashboard AppTest result:      <not run against final artifacts>
Known non-blocking limitations: none declared
```

---

## 11. Stop Conditions for Thesis-Ready Declaration

Per `docs/closeout/correction_plan_2026-06-09.md`:

- [ ] Full short suite passes or each non-pass has documented accepted reason
- [ ] Deleted test coverage has contract-by-contract replacement map
- [ ] Manual regeneration completes without unexplained failure or stale outputs
- [ ] `scripts/validate_final_acceptance.ps1 -Long` passes after regeneration
- [ ] Final docs, reports, dashboard text, and manifest agree on target semantics and chronology

**Current state: 0 of 5 stop conditions met.**

---

## 12. Recommended Next Steps (User-Run)

Before running anything long, fix these SHORT-GATE items first:

1. **Fix `--skip-tests` flag** in `run_full_pipeline.py` or update the runbook to use `--quick` instead
2. **Fix `predict_from_record` silent default** (data.py lines 427-441) — add explicit error when `selection` is empty
3. **Fix `docs/target_definitions.md`** Section 4 — correct column names and add NaN semantics
4. **Fix regression trainer eval_set** — use `split.calibration` not `split.validation` for early stopping
5. **Run `python -m pytest -q`** — full collection count and any failures
6. **Check `python scripts/compare_recency.py --help`** — verify `--quick` not duplicate

Then run the canonical sequence from `docs/closeout/manual-regeneration-runbook.md`:

```powershell
# After fixing all P0 code issues:
python scripts/manage_run_context.py start --profile thesis-full
$env:AAC_RUN_ID = "<emitted-run-id>"
python scripts/run_full_pipeline.py --skip-download --skip-shap --resume-run $env:AAC_RUN_ID
python scripts/run_full_pipeline.py --steps 11,15 --resume-run $env:AAC_RUN_ID
python scripts/run_full_pipeline.py --steps 13,14 --resume-run $env:AAC_RUN_ID
python scripts/manage_run_context.py finalize --run-id $env:AAC_RUN_ID
python scripts/generate_artifact_manifest.py --run-id $env:AAC_RUN_ID
$env:AAC_ACCEPTANCE = "1"
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
Remove-Item Env:AAC_ACCEPTANCE
```

**Note:** The runbook's `--skip-tests` flag must be replaced with omitting step 18 from `--steps` or using `--quick` (if SHAP is also acceptable to skip). Fix the runbook to reflect actual CLI flags.
