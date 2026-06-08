# Commands And Test Matrix

Run commands from the repository root with the project environment active.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

`requirements.txt` is minimal; `pyproject.toml` is the package contract. Streamlit
and pytest are currently in the `dev` extra.

## Fast Checks

```powershell
python -m pytest --collect-only -q
python -m py_compile src/aac_adoption/dashboard/data.py streamlit_app.py
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
```

Known broken as of 2026-06-08:

```powershell
python scripts/compare_recency.py --help
```

It raises an argparse conflict because `--quick` is registered twice. Use this as
a defect reproduction, not a healthy smoke check.

## Stale-File Proof

```powershell
git ls-files -- <path>
rg -n --fixed-strings "<file-name>" . --glob "!agentsbatch*/**"
git log --oneline --all -- <path>
```

Delete only after live references are absent or redirected to a canonical file.
Do not search or clean `agentsbatch*/` during normal work; those directories can
contain concurrent or historical agent state.

## Test Routing

| Change area | Start with |
|---|---|
| Matching/censoring | `python -m pytest tests/test_match_records.py tests/test_build_dataset.py tests/test_integration_survival.py -q` |
| Feature registry/leakage | `python -m pytest tests/test_feature_sets.py tests/test_leakage_audit.py tests/test_target_encoder.py -q` |
| Split/training | `python -m pytest tests/test_split.py tests/test_train_baseline_outputs.py tests/test_train_boosting_outputs.py tests/test_train_advanced_outputs.py -q` |
| Survival | `python -m pytest tests/test_survival_analysis.py tests/test_survival_analysis_new.py tests/test_integration_survival.py -q` |
| Backtesting | `python -m pytest tests/test_yearly_backtesting.py tests/test_backtesting.py tests/test_recency_comparison.py -q` |
| Analysis/reporting | `python -m pytest tests/test_analysis_outputs.py tests/test_report_outputs.py tests/test_hypothesis_evidence.py -q` |
| Dashboard | `python -m pytest tests/test_dashboard_data.py tests/test_dashboard_story.py -q` |
| Artifacts/acceptance | `python -m pytest tests/test_artifact_manifest.py tests/test_data_audit_outputs.py tests/test_acceptance_schema_aliases.py -q` |

## Pipeline

```powershell
python scripts/run_full_pipeline.py --skip-download --skip-shap
python scripts/run_full_pipeline.py --steps 2,10,14,16
python scripts/generate_artifact_manifest.py
```

Use `--quick` for development only. It skips expensive work and step 18 tests, so
it is not final acceptance. Runner continues after failures; review final summary
and never assume later artifacts are fresh when an earlier step failed.

## Acceptance

```powershell
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
```

The long acceptance command can be expensive and requires local raw/processed data.
It currently reaches the broken recency CLI help check and fails. Do not report
acceptance success until that defect is fixed. Report skipped commands and missing
prerequisites explicitly.

Full pytest currently contains artifact-dependent skips. A green unit run alone
does not prove regenerated-artifact acceptance; run pipeline/manifest checks too.

## Dashboard

```powershell
streamlit run streamlit_app.py
```

The app expects generated tables/models. A UI smoke test should verify missing
artifacts show honest errors rather than realistic-looking fallback predictions.
