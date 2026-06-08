# Manual Thesis Regeneration Runbook

Long tasks are user-run. Execute only after implementation plan Tasks 1-21 pass
their non-long gates.

## 1. Preflight

From repository root:

```powershell
git status --short --branch
python --version
python -m pip install -e ".[dev,dashboard]"
python scripts/run_full_pipeline.py --help
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
python scripts/compare_recency.py --help
```

Stop if any help command fails.

Canonical regeneration requires:

```powershell
git status --porcelain
```

to return no output. Commit approved code before continuing.

Confirm required raw inputs:

```powershell
Get-Item data/raw/intakes.csv, data/raw/outcomes.csv
```

If context features are accepted in final scope, confirm configured context files
also exist. Do not run a context-labeled model with missing context sources.

## 2. Full Test Baseline

```powershell
python -m pytest -q
```

Record:

- exit code;
- passed/failed/skipped counts;
- duration;
- first failure if nonzero.

Stop on failure.

## 3. Start Canonical Run Context

```powershell
python scripts/manage_run_context.py start --profile thesis-full
$env:AAC_RUN_ID = "<emitted-run-id>"
```

Context records current clean commit as `producer_source_sha` and remains
`incomplete`.

## 4. Canonical Non-SHAP Regeneration

```powershell
python scripts/run_full_pipeline.py --skip-download --skip-shap --skip-tests --resume-run $env:AAC_RUN_ID
```

Expected:

- fail-fast behavior;
- one run ID;
- `thesis-full` profile;
- no failed producer;
- successful receipt for every output;
- no final manifest yet.

Record newest log:

```powershell
$log = Get-ChildItem logs -Filter "pipeline_*.log" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$log.FullName
Get-Content $log.FullName
```

Stop if log contains `FAILED`, `ERROR`, or an unexplained `SKIPPED`.

## 5. Required SHAP Regeneration

SHAP is long. Run after non-SHAP pipeline passes when SHAP/feature-family outputs
remain marked `required_for_thesis`:

```powershell
python scripts/run_full_pipeline.py --steps 11,15 --resume-run $env:AAC_RUN_ID
```

Then regenerate any dependent evidence/report outputs:

```powershell
python scripts/run_full_pipeline.py --steps 13,14 --resume-run $env:AAC_RUN_ID
```

If SHAP is removed from thesis-required scope instead, update the manifest
producer and final documentation before acceptance. Never accept stale SHAP files
from an older run.

Finalize only after all required receipts exist:

```powershell
python scripts/manage_run_context.py finalize --run-id $env:AAC_RUN_ID
```

## 6. Reconcile and Commit Final Documentation

Update README, methodology, results, roadmap, target definitions, architecture,
and dashboard copy from accepted artifacts. Commit them. Confirm:

```powershell
git status --porcelain
```

returns no output.

This commit is `finalization_sha`. It may differ from `producer_source_sha`; final
manifest records and validates both.

## 7. Generate Manifest Last

```powershell
python scripts/generate_artifact_manifest.py --run-id <accepted-run-id>
```

Manifest must:

- contain one row per artifact path;
- mark every required artifact present;
- carry current run ID;
- carry matching hashes;
- include final documentation hashes;
- be newer than required artifacts.

Do not edit code, docs, or generated artifacts after this step without
regenerating the manifest.

## 8. Acceptance Mode

```powershell
$env:AAC_ACCEPTANCE = "1"
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
Remove-Item Env:AAC_ACCEPTANCE
```

Stop on any missing, stale, schema-invalid, or hash-invalid artifact.
Acceptance is verification-only and must not invoke producers.

## 9. Dashboard Smoke

```powershell
python -m pytest tests/test_dashboard_app.py -q
streamlit run streamlit_app.py
```

Manual checks:

1. Home page loads without exception.
2. Model selection states 2023 selection source.
3. Regression wording says days to any matched outcome.
4. Missing model simulation shows error, not `0.5` or `15 days`.
5. Prediction values are finite and operationally plausible.
6. Refresh displays regenerated artifact timestamps.
7. Changing inputs invalidates old prediction.

Stop if any prediction appears without validated artifact metadata.

## 10. Final Documentation Evidence

Run:

```powershell
rg -n "adoption speed|time to adoption|caus|test PR-AUC|test MAE|TODO|PARTIAL|stub" `
  README.md docs reports/summary streamlit_app.py src/aac_adoption/dashboard
python scripts/check_text_encoding.py README.md docs reports/summary streamlit_app.py src/aac_adoption/dashboard
```

Review every hit. Valid adopted-only timing references must explicitly say
"among adopted animals." Remove stale status text and encoding corruption.

## 11. Final Handoff

Provide:

```text
Accepted run ID:
Producer source SHA:
Finalization SHA:
Dataset SHA-256:
Full pytest result:
Pipeline log:
Acceptance result:
Required artifact count:
Missing artifact count:
Dashboard AppTest result:
Known non-blocking limitations:
```

Project is thesis-ready only when every field is populated and no blocking
limitation remains.
