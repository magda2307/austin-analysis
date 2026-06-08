# Closeout Validation Matrix

## Rules

- Agent runs fast and medium checks.
- User runs commands marked `USER-RUN`.
- Quick/smoke runs must use temporary output roots.
- Stop after first failed producer. Do not inspect downstream artifact quality
  until upstream failure is fixed and regeneration reruns.
- Capture command, exit code, duration, log path, and generated run ID.

## Per-Task Gates

| Area | Fast gate | Medium gate | Required evidence |
|---|---|---|---|
| Matching | `pytest tests/test_match_records.py -q` | matching + build tests | intake conservation, no cross-intake match |
| Dataset targets | `pytest tests/test_build_dataset.py -q` | matching/build integration | no null target/outcome in supervised frame |
| Context | context + rolling tests | build from files with raw history | calendar windows, prior-day weather, outcome independence |
| Leakage | feature/encoder tests | leakage audit fixture | targets, IDs, raw timestamps rejected |
| Hypothesis targets | hypothesis/evidence tests | generated table schema | adopted-only timing unaffected by non-adopted LOS |
| Split | split tests | trainer output tests | train 2013-2021, calibration 2022, selection 2023, test 2024-2025 |
| Horizon cohort | horizon tests | build integration | eligible unresolved negatives; short follow-up null |
| Metrics | trainer output tests | comparison/selection tests | 2023 selection and test rows separated |
| Selection | acceptance schema tests | analysis outputs | 2023 winner unchanged by 2022 fit/test metrics |
| Calibration | calibration tests | calibrated winner resolution | exact artifact path, subset-correct summaries |
| Bootstrap | focused deterministic test | evidence-pack tests | repeated cluster multiplicity preserved |
| Backtesting | helper/schema subset | `USER-RUN` full file | no impossible windows; horizon targets use horizon dataset; no failed rows |
| Reports | report output test | analysis/report routed set | PR-AUC/MAE 2023 selection wording |
| Metadata | trainer metadata tests | dashboard load tests | hashes, features, transforms, run ID |
| Dashboard data | dashboard data tests | AppTest suite | no fake values, path/schema failures explicit |
| Provenance | provenance tests | receipt audit | clean git SHA, one full-profile run, output hashes |
| Pipeline | runner unit tests | temp-root smoke | fail-fast, no manifest mutation, step numbering |
| Manifest | fixture manifest tests | acceptance mode | required files present, fresh, hashed |
| Docs | target/report/story tests | terminology scan | no target conflation or stale claims |

## Fast Gate A: Producer Contracts

```powershell
python -m pytest `
  tests/test_match_records.py `
  tests/test_build_dataset.py `
  tests/test_context_data.py `
  tests/test_rolling_features.py `
  tests/features/test_rolling.py `
  tests/test_feature_sets.py `
  tests/test_target_encoder.py `
  tests/test_split.py `
  -q
```

Expected: zero failures. No canonical artifact writes.

## Fast Gate B: Modeling and Selection

```powershell
python -m pytest `
  tests/test_train_baseline_outputs.py `
  tests/test_train_boosting_outputs.py `
  tests/test_train_advanced_outputs.py `
  tests/test_hyperparam_tuning.py `
  tests/test_calibration.py `
  tests/test_calibration_advanced.py `
  tests/test_analysis_outputs.py `
  tests/test_acceptance_schema_aliases.py `
  tests/test_evidence_pack.py `
  -q
```

Expected:

- 2023 selection/test metrics separate;
- no test-selected model;
- exact calibrated artifact identity;
- bootstrap cluster duplicates preserved.

## Fast Gate C: Dashboard and Reporting

```powershell
python -m pytest `
  tests/test_report_outputs.py `
  tests/test_dashboard_data.py `
  tests/test_dashboard_story.py `
  tests/test_dashboard_app.py `
  tests/test_artifact_manifest.py `
  -q
```

Expected:

- AppTest zero uncaught exceptions;
- missing/corrupt artifacts render explicit errors;
- no prediction values on failure;
- manifest fixture detects missing required files.

## CLI Gate

```powershell
python -m py_compile src/aac_adoption/dashboard/data.py streamlit_app.py
python scripts/run_full_pipeline.py --help
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
python scripts/compare_recency.py --help
```

Expected: every command exits 0 and writes no artifacts.

## Medium Integration Gate

```powershell
python -m pytest `
  tests/test_integration_survival.py `
  tests/test_survival_analysis.py `
  tests/test_survival_analysis_new.py `
  tests/test_data_audit_outputs.py `
  tests/test_hypothesis_evidence.py `
  -m "not slow and not acceptance" `
  -q
```

Interpretation:

- Survival tests cover only descriptive scope.
- Artifact-dependent tests use temporary fixtures.
- Missing canonical artifacts are not silently accepted as success.

## Fixture Acceptance Behavior Gate

```powershell
$env:AAC_ACCEPTANCE_FIXTURE_ROOT = "<temporary fixture root>"
python -m pytest -m acceptance -q
Remove-Item Env:AAC_ACCEPTANCE_FIXTURE_ROOT
```

Expected: fixture cases prove missing, stale, cross-run, and hash-mismatched
required artifacts fail before canonical regeneration exists.

## USER-RUN Long Gates

### Full yearly backtesting tests

```powershell
python -m pytest tests/test_yearly_backtesting.py -q
```

### Full suite

```powershell
python -m pytest -q
```

### Canonical pipeline

```powershell
python scripts/manage_run_context.py start --profile thesis-full
$env:AAC_RUN_ID = "<emitted-run-id>"
python scripts/run_full_pipeline.py --skip-download --skip-shap --skip-tests --resume-run $env:AAC_RUN_ID
python scripts/run_full_pipeline.py --steps 11,15 --resume-run $env:AAC_RUN_ID
python scripts/run_full_pipeline.py --steps 13,14 --resume-run $env:AAC_RUN_ID
python scripts/manage_run_context.py finalize --run-id $env:AAC_RUN_ID
```

### Manifest

```powershell
python scripts/generate_artifact_manifest.py --run-id $env:AAC_RUN_ID
```

### Canonical acceptance

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
```

Canonical order:

```text
implementation -> tests -> clean commit -> artifact producers -> final docs ->
clean commit -> manifest -> verification-only acceptance -> dashboard smoke ->
final review
```

Provenance records both:

```text
producer_source_sha
finalization_sha
```

## Evidence Review

After USER-RUN commands, verify:

```powershell
git status --short
Get-ChildItem logs -Filter "pipeline_*.log" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 FullName, LastWriteTime, Length
rg -n "FAILED|ERROR|SKIPPED|RESULT:" logs/pipeline_*.log
```

Required closeout record:

```text
Full pytest:
  command:
  exit code:
  passed/failed/skipped:

Pipeline:
  command:
  exit code:
  log:
  run ID:
  skipped steps:

Manifest:
  command:
  exit code:
  required artifacts missing:
  run ID consistency:

Acceptance:
  command:
  exit code:
```
