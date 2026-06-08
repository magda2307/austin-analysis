# Phase 4 Deep Testing Report

**Date:** 2026-06-09  
**Auditor:** ML Systems Auditor (Phase 4 Subagent)  
**Scope:** Tasks 14, 14A, 15, 16, 17  
**Project root:** `c:\Users\paula\Documents\mgr pjatk`

---

## Executive Summary

| Item | Status | Notes |
|------|--------|-------|
| Task 14: `ModelMetadata` required fields present in schema | PASS | All 21 required fields declared in `REQUIRED_MODEL_METADATA` |
| Task 14: `base_training_metadata()` populates all required fields | PASS | All fields generated in `metadata.py:base_training_metadata` |
| Task 14: `calibrate_classifiers()` passes all required metadata fields | FAIL (BUG) | Sidecar metadata built from raw `metadata` dict (not `base_training_metadata`); misses 17+ required fields |
| Task 14: `save_model_artifact()` validates before writing (atomic) | PASS | `validate_model_metadata()` called before `.write_text()` |
| Task 14A: Run context populated (`run_id`, `run_timestamp`, `producer_source_sha`) | PASS in `train_advanced.py`; FAIL in `calibrate.py` | `calibrate_classifiers()` does not call `base_training_metadata()` |
| Task 15: `predict_from_record()` returns `PredictionResult` with explicit error states | PASS | Returns `PredictionResult(ok=False, ...)` on error; no fake defaults |
| Task 15: No fake probability=0.5 or LOS=15.0 defaults on failure | PASS | All error paths set `adoption_probability=None`, `predicted_days_to_outcome=None` |
| Task 16: Schema guards in `load_table()` | PASS | Missing columns return empty DataFrame |
| Task 16: `_cached_load_metadata()` silently swallows validation failure | WARNING | Returns `{}` on any exception including `ValueError` from `validate_model_metadata` |
| Task 16: Dashboard fallback to `INTAKE_TIME_FEATURES` when metadata missing | WARNING | `model_feature_columns()` falls back to unvalidated feature list |
| Task 17: Prediction state tied to input hash (stale cache prevention) | PASS | `prediction_hash` guards stale display |
| Task 17: `PredictionResult` is a frozen dataclass | PASS | `@dataclass(frozen=True)` |
| Task 17: AppTest catches prediction failures without crash | PASS | 5/5 `test_dashboard_app.py` tests pass |
| `test_calibration.py::test_calibrate_classifiers_csv_columns_format` | FAIL | `ValueError: Missing required model metadata fields` |
| `test_dashboard_data.py::test_model_feature_columns_uses_artifact_metadata` | FAIL | Extra `is_extreme_heat` in result vs. expected 3-col list |
| `test_dashboard_data.py::test_predict_from_record_handles_calibration` | FAIL | `TypeError: 'PredictionResult' object is not subscriptable` |
| `test_dashboard_story.py` | PASS | 2/2 pass |

**3 tests are currently FAILING. 2 are confirmed bugs; 1 is a test contract mismatch.**

---

## Task 14: Model Metadata Completeness

### 14.1 -- Required fields schema (`metadata.py:13-37`)

`REQUIRED_MODEL_METADATA` declares 21 fields:

```
schema_version, model_name, task, animal_subset, artifact_path, artifact_sha256,
dataset_path, dataset_sha256, feature_columns, target_column, target_transform,
prediction_inverse_transform, split_strategy, is_thesis_evaluation, train_period,
calibration_period, selection_period, test_period, random_state, run_id,
run_timestamp, producer_source_sha, packages
```

PASS -- Schema is comprehensive and covers all required invariants including split identity, run context, and artifact identity.

### 14.2 -- `base_training_metadata()` populates all required fields (`metadata.py:95-134`)

`base_training_metadata()` constructs a dictionary containing all 21 required fields **except** `artifact_sha256` and `artifact_path`. These two are intentionally deferred to `save_model_artifact()` in `artifacts.py`, which correctly:
1. Dumps the model (`joblib.dump`)
2. Computes `artifact_sha256` from the saved file
3. Calls `validate_model_metadata()` before writing the sidecar JSON

PASS -- The `train_advanced.py` path (`_fit_and_save()` -> `save_model_artifact()`) correctly generates all required fields.

### 14.3 -- `calibrate_classifiers()` metadata completeness (`calibrate.py:215-241`)

**BUG CONFIRMED**

In `calibrate.py:calibrate_classifiers()`, the calibrated model metadata is built as:

```python
# calibrate.py:216-224
calibrated_metadata = {
    **metadata,           # raw metadata dict loaded from the source .json sidecar
    "model_name": calibrated_model_name,
    "task": "classification_calibrated",
    "base_model_name": model_name,
    "base_artifact_path": str(source_path),
    "calibration_method": _calibration_method(calib_method),
    "calibration_rows": len(calibration_validation),
    "feature_columns": feature_columns,
}
```

The problem: `**metadata` is the raw JSON dict loaded from the *source* model's sidecar (line 192: `metadata = json.loads(metadata_path.read_text(...))`). If that source sidecar was created without `base_training_metadata()` (e.g., a test fixture or an older artifact), the required fields `run_id`, `split_strategy`, `producer_source_sha`, `run_timestamp`, `packages`, `schema_version`, etc. will all be missing.

The test at `test_calibration.py:175-236` creates a minimal sidecar containing only `{"feature_columns": [...]}`, then calls `calibrate_classifiers()`. When `save_model_artifact()` runs `validate_model_metadata()`, it raises:

```
ValueError: Missing required model metadata fields: {
  'run_timestamp', 'prediction_inverse_transform', 'selection_period', 'random_state',
  'split_strategy', 'producer_source_sha', 'dataset_sha256', 'target_column',
  'train_period', 'run_id', 'test_period', 'packages', 'schema_version',
  'calibration_period', 'dataset_path', 'animal_subset', 'target_transform',
  'is_thesis_evaluation'
}
```

**Root cause:** `calibrate_classifiers()` does not call `base_training_metadata()` to synthesize the missing fields for the calibrated artifact. It relies entirely on the source artifact's metadata being complete, which is not enforced at read time.

**Impact in production:** If the original model artifacts were trained with the current `train_advanced.py` (which calls `base_training_metadata()`), the fields will be present and the function works. But the function is not safe to use with any sidecar that was not produced by `base_training_metadata()`, and the test correctly exposes this.

### 14.4 -- Atomic metadata write (`artifacts.py:24-45`)

```python
# artifacts.py:34-44
path = artifact_path(base_dir, task, animal_subset, model_name)
path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, path)                          # model saved first
sidecar_metadata["artifact_path"] = str(path)
sidecar_metadata["artifact_sha256"] = compute_file_sha256(path)  # SHA computed after save
validate_model_metadata(sidecar_metadata)            # raises ValueError if incomplete
path.with_suffix(".json").write_text(...)            # JSON written only if valid
```

PASS -- Atomicity is preserved: validation failure prevents the JSON from being written. However, the model `.joblib` file is already saved at this point. A failed validation leaves an orphaned `.joblib` without a corresponding `.json`. This is the correct tradeoff (fail loudly before advertising an invalid artifact), but consumers should be aware.

---

## Task 14A: Run Context and Receipts

### 14A.1 -- `run_id` field

- `base_training_metadata()` accepts `run_id: str = "dev"` (line 104). Default is `"dev"` -- not a random UUID or timestamp.
- In `train_advanced.py`, `_fit_and_save()` calls `base_training_metadata()` without explicitly passing `run_id`, so all production artifacts default to `run_id="dev"`.

WARNING -- `run_id="dev"` is a static literal, not a per-run unique identifier. Two separate training runs will produce artifacts with identical `run_id` values, making it impossible to distinguish between them by `run_id` alone. `run_timestamp` does differ per run, but `run_id` provides no differentiation.

### 14A.2 -- `producer_source_sha`

`get_source_sha()` calls `git rev-parse HEAD` and falls back to a descriptive error string. PASS -- populated in all training artifacts.

### 14A.3 -- `calibrate_classifiers()` does NOT add run context

`calibrate_classifiers()` never calls `get_source_sha()`, never sets `run_timestamp`, and never sets `run_id` for the calibrated artifact. It inherits these from `**metadata` (source artifact's values), which means calibrated artifacts carry the *original model's* run context, not the calibration run's context.

WARNING -- This means there is no way to determine when calibration was run or which code version performed it, independently from the base model.

---

## Task 15: Dashboard Fake Prediction Guards

### 15.1 -- `predict_from_record()` return contract (`data.py:427-533`)

PASS -- No fake values on failure.

The function uses a `PredictionResult` frozen dataclass (lines 416-425):

```python
@dataclass(frozen=True)
class PredictionResult:
    ok: bool
    adoption_probability: float | None
    predicted_days_to_outcome: float | None
    los_bucket: str | None
    is_calibrated: bool
    model_artifacts: dict[str, str]
    error_code: str | None
    error_message: str | None
```

All error paths explicitly set `adoption_probability=None`, `predicted_days_to_outcome=None`:
- CLF error (line 493): `PredictionResult(False, None, None, None, False, artifacts, "CLF_ERROR", str(e))`
- Invalid probability (line 490): `PredictionResult(False, None, None, None, False, artifacts, "INVALID_PROB", ...)`
- REG error (line 522): `PredictionResult(False, None, None, None, False, artifacts, "REG_ERROR", str(e))`
- Invalid days (line 515): `PredictionResult(False, None, None, None, False, artifacts, "INVALID_DAYS", ...)`
- Out-of-bounds days (line 519): `PredictionResult(False, None, None, None, False, artifacts, "OUT_OF_BOUNDS", ...)`

There is **no path that returns probability=0.5 or LOS=15.0 as a silent default**. Every failure produces an explicit error code and `None` values.

### 15.2 -- `streamlit_app.py` prediction failure handling

In the Animal Stories tab (lines 594-599):
```python
try:
    profile_prediction = predict_from_record(profile_record, MODELS_DIR)
    if not profile_prediction.ok:
        profile_prediction = None
except Exception:
    profile_prediction = None
```

In the Model Sensitivity Demo tab (lines 1027-1034):
```python
if st.button(t("Run prediction"), ...):
    try:
        prediction = predict_from_record(record, MODELS_DIR)
        st.session_state["prediction_result"] = prediction
        st.session_state["prediction_hash"] = record_hash
    except Exception as error:
        st.error(str(error))
        st.info(t("Run `python scripts/train_advanced.py ...` first."))
```

PASS with WARNING -- Both call sites handle exceptions gracefully. However, there is a subtle asymmetry:

- Animal Stories tab: checks `profile_prediction.ok` and nullifies silently, then shows an info message. Correct.
- Model Sensitivity Demo tab: `predict_from_record()` almost never raises exceptions (it catches all internal errors and returns `PredictionResult(ok=False, ...)`). The outer `try/except` only catches unexpected exceptions outside the function. The `ok=False` case is **not explicitly checked** at lines 1036-1054 -- the code proceeds to display `prediction.adoption_probability * 100` which would raise `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`.

**BUG-4 (WARNING)** -- In the Model Sensitivity Demo tab, a `PredictionResult(ok=False)` stored in session_state would cause a `TypeError` crash on the next render at line 1038.

---

## Task 16: Dashboard Schema Guards

### 16.1 -- `load_table()` schema validation (`data.py:150-176`)

PASS -- `load_table()` returns `pd.DataFrame()` (empty) if:
- File not found
- File is empty
- Required schema columns are missing
- A boolean column cannot be parsed strictly

### 16.2 -- `_cached_load_metadata()` silently swallows validation errors (`data.py:348-356`)

```python
@st.cache_data(show_spinner=False)
def _cached_load_metadata(path: Path, fingerprint: tuple[str, int, int]) -> dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        from aac_adoption.models.metadata import validate_model_metadata
        validate_model_metadata(meta)
        return meta
    except Exception:
        return {}    # swallows BOTH I/O errors AND validation failures
```

WARNING -- A sidecar JSON that fails `validate_model_metadata()` silently returns `{}`. This empty dict is then used in `model_feature_columns()`:

```python
# data.py:379-388
def model_feature_columns(record, models_dir, task, subset, model_name):
    metadata = load_model_metadata(...)
    expected = metadata.get("feature_columns")
    if expected is not None:
        ...
        return expected
    fallback = [col for col in INTAKE_TIME_FEATURES if col in record.columns]
    return fallback   # fallback to unvalidated features
```

If metadata returns `{}`, the fallback silently uses `INTAKE_TIME_FEATURES`, which may not match the feature list the model was actually trained on. This is a silent feature-mismatch risk, not a hard failure. This also causes BUG-3 in tests.

### 16.3 -- `DASHBOARD_TABLE_SCHEMAS` coverage

`DASHBOARD_TABLE_SCHEMAS` covers 7 of the 19 tables in `TABLE_FILES`. The 12 uncovered tables receive no column-level validation.

WARNING -- Partial schema coverage. A corrupt or schema-drifted CSV in an uncovered table would not be caught at load time.

---

## Task 17: Dashboard Cache & Prediction State

### 17.1 -- Prediction state not cached across inputs (`streamlit_app.py:1024-1054`)

```python
record_hash = hashlib.md5(str(record.to_dict()).encode()).hexdigest()

if st.button(t("Run prediction"), ...):
    ...
    st.session_state["prediction_hash"] = record_hash

if st.session_state.get("prediction_hash") == record_hash and "prediction_result" in st.session_state:
    # display
elif "prediction_hash" in st.session_state and st.session_state["prediction_hash"] != record_hash:
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_hash", None)
```

PASS -- The MD5 hash of the full record dict guards against showing stale predictions. When inputs change, the hash mismatches and the session state is cleared before display.

### 17.2 -- `PredictionResult` frozen dataclass prevents mutation

PASS -- `@dataclass(frozen=True)` on `PredictionResult` prevents any post-hoc mutation of prediction results stored in session state.

### 17.3 -- AppTest catches prediction failures without crash

All 5 `test_dashboard_app.py` tests pass, including `test_dashboard_app_missing_model_corrupt_metadata` which mocks `load_model` to raise `FileNotFoundError`.

---

## Test Run Results

### Run 1: `tests/test_calibration.py tests/test_dashboard_data.py tests/test_dashboard_story.py`

```
3 failed, 18 passed in 19.15s
```

| Test | Result | Category |
|------|--------|----------|
| `test_calibrate_with_isotonic` | PASS | |
| `test_calibrate_with_platt` | PASS | |
| `test_post_hoc_calibration_pipeline` | PASS | |
| `test_apply_calibration_to_predictions` | PASS | |
| `test_apply_calibration_preserves_platt_method` | PASS | |
| `test_calibrate_classifiers_help_exits_0` | PASS | |
| `test_calibrate_classifiers_end_to_end_fixture` | PASS | |
| **`test_calibrate_classifiers_csv_columns_format`** | FAIL | BUG in `calibrate.py` |
| `test_calibrate_platt_uses_sigmoid_method` | PASS | |
| `test_calibrate_classifiers_handles_missing_artifacts` | PASS | |
| `test_best_model_rows_selects_expected_metrics` | PASS | |
| `test_build_prediction_record_creates_model_features` | PASS | |
| `test_build_profile_prediction_record_uses_representative_values` | PASS | |
| **`test_model_feature_columns_uses_artifact_metadata`** | FAIL | Test contract mismatch (cascades from silent metadata swallowing) |
| `test_similar_historical_cases_returns_outcome_mix` | PASS | |
| `test_profile_global_shap_reasons_maps_profile_values` | PASS | |
| `test_visibility_need_from_prediction_labels_quadrants` | PASS | |
| `test_los_days_to_bucket` | PASS | |
| **`test_predict_from_record_handles_calibration`** | FAIL | Test uses dict subscript on dataclass |
| `test_story_helpers_return_expected_content` | PASS | |
| `test_streamlit_report_allowlist_is_thesis_only` | PASS | |

### Run 2: `tests/test_dashboard_app.py`

```
5 passed in 17.77s
```

| Test | Result |
|------|--------|
| `test_dashboard_app_current_artifacts_empty` | PASS |
| `test_dashboard_app_current_artifacts_with_data` | PASS |
| `test_dashboard_app_missing_columns` | PASS |
| `test_dashboard_app_string_booleans` | PASS |
| `test_dashboard_app_missing_model_corrupt_metadata` | PASS |

---

## Bugs Found

### BUG-1 (HIGH): `calibrate_classifiers()` does not call `base_training_metadata()`

**File:** `src/aac_adoption/models/calibrate.py:215-241`  
**Test exposing it:** `tests/test_calibration.py::test_calibrate_classifiers_csv_columns_format`  
**Exact error:**

```
ValueError: Missing required model metadata fields: {
  'run_timestamp', 'prediction_inverse_transform', 'selection_period', 'random_state',
  'split_strategy', 'producer_source_sha', 'dataset_sha256', 'target_column',
  'train_period', 'run_id', 'test_period', 'packages', 'schema_version',
  'calibration_period', 'dataset_path', 'animal_subset', 'target_transform',
  'is_thesis_evaluation'
}
```

**Root cause:** `calibrated_metadata` is built from `{**metadata, ...}` where `metadata` is the raw JSON from the source artifact's sidecar. When the source sidecar does not contain all `REQUIRED_MODEL_METADATA` fields (any sidecar not produced by `base_training_metadata()`), `save_model_artifact()` raises `ValueError` before writing.

**Fix:** Call `base_training_metadata()` inside `calibrate_classifiers()` to generate the required fields for the calibrated artifact. The `split` object and `data_path` are already available in scope. Suggested change:

```python
# Add at top of calibrate.py:
from datetime import datetime, timezone
from aac_adoption.models.metadata import base_training_metadata

# Inside calibrate_classifiers(), after split is computed, replace the metadata dict:
calib_run_timestamp = datetime.now(timezone.utc).isoformat()
calib_base = base_training_metadata(
    model_name=calibrated_model_name,
    task="classification_calibrated",
    split=split,
    feature_columns=feature_columns,
    run_timestamp=calib_run_timestamp,
    target_column="classification_target",
    dataset_path=str(data_path),
)
calibrated_metadata = {
    **calib_base,
    "base_model_name": model_name,
    "base_artifact_path": str(source_path),
    "calibration_method": _calibration_method(calib_method),
    "calibration_rows": len(calibration_validation),
    "feature_columns": feature_columns,
}
```

### BUG-2 (MEDIUM): `test_predict_from_record_handles_calibration` uses dict subscript on frozen dataclass

**File:** `tests/test_dashboard_data.py:294-307`  
**Exact error:** `TypeError: 'PredictionResult' object is not subscriptable`  
**Root cause:** The test accesses prediction results using dict syntax (`res_calibrated["adoption_probability"]`), but `predict_from_record()` returns a `PredictionResult` frozen dataclass, not a dict. Dataclass attributes require dot notation.

This is a **test defect, not a source code defect**. The source behavior is correct.

**Fix:** Change all subscript accesses to attribute access:
```python
# Before (wrong):
assert res_calibrated["adoption_probability"] == 0.9
assert res_calibrated["predicted_days_to_outcome"] == pytest.approx(15.0)
assert res_calibrated["los_bucket"] == "8-30d"
assert res_fallback["adoption_probability"] == 0.7
assert res_fallback["predicted_days_to_outcome"] == pytest.approx(15.0)
assert res_fallback["los_bucket"] == "8-30d"

# After (correct):
assert res_calibrated.adoption_probability == 0.9
assert res_calibrated.predicted_days_to_outcome == pytest.approx(15.0)
assert res_calibrated.los_bucket == "8-30d"
assert res_fallback.adoption_probability == 0.7
assert res_fallback.predicted_days_to_outcome == pytest.approx(15.0)
assert res_fallback.los_bucket == "8-30d"
```

### BUG-3 (MEDIUM): `test_model_feature_columns_uses_artifact_metadata` gets 4 features not 3

**File:** `tests/test_dashboard_data.py:109-132`  
**Exact error:**
```
AssertionError: assert ['animal_type', 'intake_type', 'age_days', 'is_extreme_heat']
                    == ['animal_type', 'intake_type', 'age_days']
Left contains one more item: 'is_extreme_heat'
```

**Root cause (cascading):** The test writes a minimal sidecar `{"feature_columns": ["animal_type", "intake_type", "age_days"]}`. When `_cached_load_metadata()` reads it, `validate_model_metadata()` raises `ValueError` (21 required fields missing). The `except Exception: return {}` catches this and returns empty dict. Then `model_feature_columns()` falls through to the `INTAKE_TIME_FEATURES` fallback, which includes `is_extreme_heat` (it is in the record returned by `build_prediction_record()`).

**Fix options:**
1. Write a complete valid metadata sidecar in the test (all 21 required `REQUIRED_MODEL_METADATA` fields).
2. Change `_cached_load_metadata()` to distinguish I/O errors from validation errors -- return the raw dict even if validation fails, since the caller only needs `feature_columns` and doesn't need the full contract.

### BUG-4 (MEDIUM): `ok=False` result not checked before display in Model Sensitivity Demo tab

**File:** `streamlit_app.py:1036-1047`  
**Root cause:** When `predict_from_record()` returns `PredictionResult(ok=False)`, it is stored in `st.session_state["prediction_result"]`. On the next render, lines 1038-1040 unconditionally access `prediction.adoption_probability * 100` and `prediction.predicted_days_to_outcome`, which raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` since both fields are `None` for `ok=False`.

**Fix:** Add an `ok` check before display:
```python
if st.session_state.get("prediction_hash") == record_hash and "prediction_result" in st.session_state:
    prediction = st.session_state["prediction_result"]
    if not prediction.ok:
        st.error(f"Prediction failed: [{prediction.error_code}] {prediction.error_message}")
        st.info(t("Run `python scripts/train_advanced.py --data data/processed/modeling_dataset.csv` first."))
    else:
        probability_pct = prediction.adoption_probability * 100
        days = prediction.predicted_days_to_outcome
        wait_bucket = prediction.los_bucket
        # ... rest of display code
```

This bug is not caught by `test_dashboard_app.py` because:
- The test uses `mock_load_model.side_effect = FileNotFoundError("Missing")`
- `predict_from_record()` catches this internally and returns `PredictionResult(ok=False)`
- The test's `.run()` does not click the button, so the result is never stored in session_state

---

## Recommendations

### Priority 1 -- Fix immediately (blocking tests)

1. **`calibrate.py:calibrate_classifiers()`**: Call `base_training_metadata()` to generate the full required metadata for calibrated artifacts. Do not rely on the source artifact's metadata being complete.

2. **`test_dashboard_data.py::test_predict_from_record_handles_calibration`**: Change all `res[\"field\"]` dict accesses to `res.field` attribute accesses.

3. **`test_dashboard_data.py::test_model_feature_columns_uses_artifact_metadata`**: Either write a complete metadata sidecar (all 21 required fields) in the test fixture, or create a separate test that explicitly validates the fallback behavior.

### Priority 2 -- Fix before production use

4. **`streamlit_app.py:1036-1047`**: Add `if not prediction.ok:` guard before accessing `adoption_probability` and `predicted_days_to_outcome` in the Model Sensitivity Demo tab.

5. **`data.py:_cached_load_metadata()`**: Separate I/O errors (return `{}`) from validation errors (log warning and return the raw dict anyway, so `feature_columns` is still accessible even if other fields are missing).

### Priority 3 -- Improvement

6. **`run_id` is always `"dev"`**: Pass a per-run unique identifier (timestamp or UUID4) from training scripts to `base_training_metadata()`. Without this, multiple training runs cannot be distinguished by `run_id`.

7. **Calibrated artifact provenance**: Even after fixing BUG-1, the calibrated artifact should record calibration-specific `run_timestamp` and `producer_source_sha`, not inherited values from the base model.

8. **`DASHBOARD_TABLE_SCHEMAS` coverage**: Extend schema validation to the remaining 12 tables in `TABLE_FILES` (especially `h1`, `h3`, `h5`).

9. **AppTest for `ok=False` state**: Add a test in `test_dashboard_app.py` that simulates a button click with a failed prediction result and verifies an error message is shown rather than a crash.

---

## Code Logic Analysis: Actual vs. Intended

### `calibrate_classifiers()` -- actual vs. intended

| Aspect | Actual | Should be |
|--------|--------|-----------|
| Metadata construction | `{**metadata, "model_name": ..., "task": ...}` -- inherits from source | Call `base_training_metadata()` for the calibrated artifact |
| `run_id` | Inherits source model's `run_id="dev"` | Generate a new `run_id` per calibration run |
| `run_timestamp` | Inherits source model's training timestamp | Set to current time at calibration run start |
| `producer_source_sha` | Inherits source model's git SHA | Set to git SHA at calibration time |
| Validation | `save_model_artifact()` validates before write | Same -- correct |
| Failure mode | Raises `ValueError` if source sidecar incomplete | Should be robust regardless of source sidecar completeness |

### `_cached_load_metadata()` -- actual vs. intended

| Aspect | Actual | Should be |
|--------|--------|-----------|
| I/O error | Returns `{}` | Return `{}` -- correct |
| Validation failure | Returns `{}` silently | Log warning and return partial dict, or raise explicitly |
| Effect of empty `{}` | `model_feature_columns()` falls back to `INTAKE_TIME_FEATURES` | Should surface the schema mismatch to the caller |

### `predict_from_record()` -- actual vs. intended

| Aspect | Actual | Should be |
|--------|--------|-----------|
| CLF failure | Returns `PredictionResult(ok=False, error_code="CLF_ERROR")` | Correct |
| REG failure | Returns `PredictionResult(ok=False, error_code="REG_ERROR")` | Correct |
| Invalid probability | Returns `PredictionResult(ok=False, error_code="INVALID_PROB")` | Correct |
| Out-of-bounds days | Returns `PredictionResult(ok=False, error_code="OUT_OF_BOUNDS")` | Correct |
| Fake defaults | None | Correct -- no fake defaults |

---

*Report generated: 2026-06-09T00:32 CEST*
