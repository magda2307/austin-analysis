# Phase 4 Deep Testing Report

## Executive Summary

This testing report presents the findings of a comprehensive code audit and logical correctness validation conducted on the system components associated with Phase 4 (Tasks 14, 14A, 15, 16, and 17) of the Austin Animal Center (AAC) adoption outcomes thesis project.

The scope of this audit covers model metadata persistence, runtime context propagation, prediction error path robustness, dashboard schema validation, and cache stale state prevention.

### Quick Status Summary

| Audit Item | Status | Key Evidence / Observations |
| :--- | :--- | :--- |
| **Task 14: Model Metadata Schema** | **PASS** | 21 required fields declared in `metadata.py:REQUIRED_MODEL_METADATA`. |
| **Task 14: Metadata Generation** | **PASS** | `base_training_metadata()` correctly populates metadata. |
| **Task 14: Calibration Sidecar Creation** | **FAIL (BUG-1)** | `calibrate_classifiers()` in `calibrate.py` constructs metadata by spreading the raw source metadata instead of synthesizing missing fields, which causes validation failures during test execution. |
| **Task 14: Atomic Saving** | **PASS** | Model artifacts and JSON sidecars are saved sequentially with sidecar validation checks guarding the final write. |
| **Task 14A: Run Context & Git SHA** | **PASS** | Git HEAD SHA is successfully fetched or gracefully handled with warning strings; timestamps and IDs are populated. |
| **Task 15: No Fake Predictions** | **PASS** | `predict_from_record()` returns a `PredictionResult` with explicit error codes; there are no silent fallbacks to probability `0.5` or `15.0` days on failure. |
| **Task 16: Dashboard Schema Guards** | **PASS** | `load_table()` enforces columns and types on loaded tables. |
| **Task 16: Silent Validation Swallowing** | **WARNING** | `_cached_load_metadata()` catches all exceptions and silently returns `{}` on metadata validation errors, causing downstream feature mismatch risks. |
| **Task 17: Cache & Input State Hashing** | **PASS** | Input MD5 hashing in `streamlit_app.py` successfully prevents stale cache prediction displays. |
| **Task 17: Model Sensitivity Failures** | **FAIL (BUG-4)** | The Model Sensitivity tab does not check `prediction.ok` before referencing properties, risking `TypeError` crashes on failed predictions. |

---

## Task 14: Model Metadata Completeness

### 14.1 Required Fields Schema (`src/aac_adoption/models/metadata.py`)
The `REQUIRED_MODEL_METADATA` set (lines 13–37) defines the following 21 fields:
* `schema_version`, `model_name`, `task`, `animal_subset`, `artifact_path`, `artifact_sha256`
* `dataset_path`, `dataset_sha256`, `feature_columns`, `target_column`, `target_transform`, `prediction_inverse_transform`
* `split_strategy`, `is_thesis_evaluation`, `train_period`, `calibration_period`, `selection_period`, `test_period`
* `random_state`, `run_id`, `run_timestamp`, `producer_source_sha`, `packages`

The metadata schema successfully matches the required invariants.

### 14.2 Sidecar Metadata Instantiation & Call Chain
During primary model training (`train_advanced.py` and other estimators), metadata is generated using `base_training_metadata()` inside `src/aac_adoption/models/metadata.py` (lines 95–134).
* The actual saving is performed by `save_model_artifact()` inside `src/aac_adoption/models/artifacts.py` (lines 24–45).
* Atomicity check: `save_model_artifact()` saves the `.joblib` model file first, then computes its SHA256 hash, updates the metadata dictionary, runs `validate_model_metadata()`, and finally writes the `.json` sidecar. If metadata validation fails, `ValueError` is raised, preventing the `.json` file from being created. This ensures corrupted metadata sidecars are never serialized.

### 14.3 Trace: Calibration Validation Failure
In `src/aac_adoption/models/calibrate.py:calibrate_classifiers()` (lines 216–225), the metadata for a calibrated model is built as:
```python
calibrated_metadata = {
    **metadata,
    "model_name": calibrated_model_name,
    "task": "classification_calibrated",
    "base_model_name": model_name,
    "base_artifact_path": str(source_path),
    "calibration_method": _calibration_method(calib_method),
    "calibration_rows": len(calibration_validation),
    "feature_columns": feature_columns,
}
```
* **Failure Mechanism**: `metadata` is loaded directly from the source model's sidecar JSON (line 192). In tests or older deployments where the source metadata is minimal (e.g. `{"feature_columns": ["feature1", "feature2"]}`), the spread operator `**metadata` fails to introduce the remaining 17+ required fields.
* When `save_model_artifact()` is called, it triggers `validate_model_metadata(calibrated_metadata)`, raising:
  `ValueError: Missing required model metadata fields: {'animal_subset', 'packages', 'split_strategy', ...}`
* This bug is verified by the failure of `test_calibration.py::test_calibrate_classifiers_csv_columns_format`.

---

## Task 14A: Run Context and Receipts

### 14A.1 Run Identifiers and Timestamps
* The function `base_training_metadata()` accepts a `run_id` which defaults to `"dev"`.
* The `run_timestamp` is passed from the calling runner script as an ISO-formatted string.
* In the current training scripts, `run_id` is not dynamically overridden with UUIDs or unique task execution receipts. Consequently, separate training runs will share `run_id: "dev"`.

### 14A.2 Git HEAD SHA Resolution
Git revision identification is resolved through `get_source_sha()` inside `metadata.py` (lines 75–87) using `git rev-parse HEAD`.
* The subprocess call specifies a 5-second timeout and catches common operational failures (such as git missing or running outside of a repository), returning a fallback string: `f"unavailable (git failed: {e})"`.

### 14A.3 Calibration Provenance Loss
Because `calibrate_classifiers()` in `calibrate.py` spreads the original metadata directly (`**metadata`), the calibrated JSON sidecar inherits the original base model's `run_timestamp`, `run_id`, and `producer_source_sha` instead of generating its own execution receipt. This obscures the provenance of when calibration was executed.

---

## Task 15: Dashboard Fake Prediction Guards

### 15.1 Prediction Results Contract (`src/aac_adoption/dashboard/data.py`)
`predict_from_record()` (lines 427–533) ensures that on model prediction failure, no fake probability (e.g., `0.5`) or default Length of Stay (e.g., `15.0` days) is returned.
* The function returns a frozen `PredictionResult` dataclass (lines 416–425) with `ok=False` and explicit internal error codes:
  * `"CLF_ERROR"`: Classifier prediction failed.
  * `"REG_ERROR"`: Regressor prediction failed.
  * `"INVALID_PROB"`: Probabilities out of `[0, 1]` bounds.
  * `"INVALID_DAYS"`: Predicted days to outcome are negative or infinite.
  * `"OUT_OF_BOUNDS"`: Predicted days exceed operational bound (`4000` days).
* For all such failure modes, `adoption_probability` and `predicted_days_to_outcome` are explicitly set to `None`.

### 15.2 Streamlit Error Handling
* **Animal Stories Tab** (`streamlit_app.py:594–599`):
  Checks `not profile_prediction.ok` and sets `profile_prediction = None` to gracefully hide predictions and show an informational box.
* **Model Sensitivity Demo Tab** (`streamlit_app.py:1027–1047`):
  When a prediction fails, `predict_from_record` returns `PredictionResult(ok=False)`.
  * **Warning**: While the button click catches external exceptions, the UI display code at lines 1036–1047 does not verify `prediction.ok`. It attempts to multiply `prediction.adoption_probability * 100`. Since this field is `None`, a `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` crash will occur.

---

## Task 16: Dashboard Schema Guards

### 16.1 Table Schema Validation
`load_table()` in `src/aac_adoption/dashboard/data.py` (lines 150–176) validates structured files using schemas defined in `DASHBOARD_TABLE_SCHEMAS`.
* A target table is rejected and an empty `pd.DataFrame()` is returned if:
  1. Any required column in the schema is missing.
  2. A boolean column contains values that cannot be strictly parsed to `True`/`False`/`None` via `parse_strict_boolean()`.

### 16.2 Silent Failure Swallowing
`_cached_load_metadata()` in `src/aac_adoption/dashboard/data.py` (lines 348–356) silently catches all exceptions and returns an empty dictionary `{}` on validation failures.
* If a metadata sidecar JSON exists but lacks required fields, `validate_model_metadata()` raises a `ValueError`. The function swallows it, returning `{}`.
* Consequently, `model_feature_columns()` falls back to `INTAKE_TIME_FEATURES` without raising a validation failure, masking underlying metadata integration issues.

---

## Task 17: Dashboard Cache & State

### 17.1 Stale Cache Prevention
In `streamlit_app.py` (lines 1024–1054), prediction state is safely guarded using an MD5 hash of the input fields:
```python
record_hash = hashlib.md5(str(record.to_dict()).encode()).hexdigest()
```
* The prediction is only displayed if `prediction_hash` matches `record_hash`.
* If user inputs are changed, the hash mismatch is detected, and the stale predictions are immediately evicted from the session state.

### 17.2 Frozen Dataclass Mutability Protection
The `PredictionResult` dataclass is marked as `@dataclass(frozen=True)` which prevents downstream elements or UI callbacks from modifying the prediction values during rendering cycles.

### 17.3 Automated Integration Testing
`tests/test_dashboard_app.py` utilizes Streamlit's `AppTest` interface to verify the app layout and ensures the UI degrades gracefully under missing datasets or simulated model loading exceptions.

---

## Test Run Results

The targeted tests were run from the repository root:
```bash
python -m pytest tests/test_calibration.py tests/test_dashboard_data.py tests/test_dashboard_app.py tests/test_dashboard_story.py -q
```

### Summary of Results
* **Passed**: 23 tests
* **Failed**: 3 tests
* **Total Time**: ~16 seconds

### Failing Tests Detailed Trace
1. `tests/test_calibration.py::test_calibrate_classifiers_csv_columns_format`
   * **Result**: **FAIL**
   * **Error**: `ValueError: Missing required model metadata fields: {'animal_subset', 'packages', 'split_strategy', ...}`
   * **Root Cause**: The source mock metadata lacks required schema fields, and the calibration function does not synthesize them using `base_training_metadata()`.

2. `tests/test_dashboard_data.py::test_model_feature_columns_uses_artifact_metadata`
   * **Result**: **FAIL**
   * **Error**: `AssertionError: assert ['animal_type', 'intake_type', 'age_days', 'is_extreme_heat'] == ['animal_type', 'intake_type', 'age_days']`
   * **Root Cause**: The mock metadata contains only `{"feature_columns": [...]}`. It fails validation, returns `{}`, and triggers the unvalidated fallback `INTAKE_TIME_FEATURES` list, which includes `is_extreme_heat`.

3. `tests/test_dashboard_data.py::test_predict_from_record_handles_calibration`
   * **Result**: **FAIL**
   * **Error**: `TypeError: 'PredictionResult' object is not subscriptable`
   * **Root Cause**: The test attempts to access prediction properties using dict key indexing (`res_calibrated["adoption_probability"]`) instead of attribute dot notation (`res_calibrated.adoption_probability`).

---

## Bugs Found

### BUG-1: Metadata Schema Validation Failure in Calibration Pipeline
* **File**: `src/aac_adoption/models/calibrate.py` (lines 216–225)
* **Description**: `calibrate_classifiers()` spreads the source model metadata directly instead of calling `base_training_metadata()` to synthesize required fields for the calibrated artifact.

### BUG-2: Dict Indexing in Calibration Test Assertions
* **File**: `tests/test_dashboard_data.py` (lines 294–308)
* **Description**: The test uses dictionary subscripts (`res_calibrated["adoption_probability"]`) on the `PredictionResult` object, which is a dataclass.

### BUG-3: Mock Metadata Swallowing in Feature Resolution Test
* **File**: `tests/test_dashboard_data.py` (lines 109–132)
* **Description**: The mock metadata sidecar written by the test lacks required fields, causing validation to fail silently and return `{}`, which forces a fallback to unvalidated features.

### BUG-4: Missing UI Prediction Failure Check
* **File**: `streamlit_app.py` (lines 1036–1047)
* **Description**: The Model Sensitivity tab does not check if `prediction.ok` is `True` before displaying values, leading to a `TypeError` crash when prediction fails.

---

## Recommendations

1. **Synthesize Provenance in Calibration**:
   Update `calibrate_classifiers()` in `src/aac_adoption/models/calibrate.py` to construct metadata using `base_training_metadata()`, tracking the new calibration run details rather than inheriting stale source metadata.
2. **Correct Dataclass Access in Tests**:
   Update `tests/test_dashboard_data.py` to use dot notation (`res_calibrated.adoption_probability`) instead of subscript syntax.
3. **Correct Mock Metadata in Tests**:
   Modify `test_model_feature_columns_uses_artifact_metadata` to build a complete, valid metadata dictionary to satisfy schema requirements.
4. **Enforce UI Prediction Guards**:
   Add a conditional block `if prediction.ok:` in the Model Sensitivity tab of `streamlit_app.py` before accessing fields like `adoption_probability`.
5. **Separate I/O Errors from Validation Failures**:
   Refactor `_cached_load_metadata()` to distinguish between file reading issues (return `{}`) and format validation errors (raise or log a warning), rather than silently swallowing both.
