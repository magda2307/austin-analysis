# Thesis Closeout Correction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the remaining methodological, validation, artifact, and documentation blockers so the project can be regenerated and defended for thesis submission.

**Architecture:** Treat generated artifacts as outputs, not hand-edited sources. Fix producers and tests first, then run manual regeneration, then reconcile final-facing docs from fresh evidence. Keep chronological ML contracts strict: train 2013-2021, calibration 2022, selection 2023, test 2024-2025.

**Tech Stack:** Python, pandas, scikit-learn, pytest, Streamlit, Markdown closeout docs, PowerShell acceptance scripts.

---

## Current State

Fresh evidence from 2026-06-09:

- `python -m pytest --collect-only -q`: 209 tests collected.
- Focused correction suite: 82 passed, 23 warnings.
- Syntax check passed for touched producers.
- Long acceptance and regeneration were not run because those are manual tasks.
- Generated artifacts in `reports/`, `models/`, and `data/processed/` may be stale.

Important already-fixed items:

- Matching no longer assigns outcomes after `extract_end_date`.
- Matching no longer assigns an outcome across a later intake boundary.
- Unresolved follow-up is truncated at the next intake boundary.
- Negative unresolved follow-up is rejected.
- Episode numbering is based on intake sequence, not duplicate prior outcome rows.
- Horizon builder rejects invalid cross-boundary rows and post-extraction intakes.
- Horizon target metadata contains only `adopted_in_*` columns.
- Intake-volume windows exclude simultaneous intakes.
- Duplicate raw export rows do not inflate intake-volume counts.
- Missing raw-intake joins fail explicitly instead of becoming zero.
- Weather context uses prior completed calendar-day weather.
- Calibration uses 2022 only; 2023 remains selection.
- Model-selection producer wording now says 2023 selects and 2024-2025 tests.

## Stop Conditions

Do not call the project thesis-ready until all are true:

- Full short suite passes or each non-pass has documented, accepted reason.
- Deleted test coverage has a contract-by-contract replacement map.
- Manual regeneration completes without unexplained failure or stale outputs.
- `scripts/validate_final_acceptance.ps1 -Long` passes after regeneration.
- Final docs, reports, dashboard text, and manifest agree on target semantics and chronology.

---

## File Responsibility Map

Primary planning and audit docs:

- `docs/closeout/correction_plan_2026-06-09.md`: this action plan.
- `docs/closeout/harsh_audit_2026-06-09.md`: summarized audit verdict and blockers.
- `.agents/CLOSEOUT.md`: compact active closeout router.
- `docs/PROJECT_CLOSEOUT_TASKS.md`: detailed closeout task register.
- `docs/closeout/validation-matrix.md`: command and coverage matrix.

Core producers to verify or modify:

- `src/aac_adoption/data/match_records.py`: intake/outcome matching.
- `src/aac_adoption/data/build_dataset.py`: matched dataset, unresolved audit, horizon dataset.
- `src/aac_adoption/data/context_data.py`: weather, 311, and raw-intake context features.
- `src/aac_adoption/features/rolling_features_cache.py`: strict prior intake-volume windows.
- `src/aac_adoption/models/split.py`: chronological split contract.
- `src/aac_adoption/models/calibrate.py`: post-hoc calibration on 2022 only.
- `src/aac_adoption/analysis/model_selection.py`: 2023 selection, test-only reporting.
- `src/aac_adoption/analysis/hypothesis_tables.py`: target alias normalization for reporting.
- `src/aac_adoption/models/yearly_backtesting.py`: yearly windows and skip semantics.
- `src/aac_adoption/reporting/report.py`: final report tables and figures.
- `src/aac_adoption/dashboard/data.py`, `streamlit_app.py`: dashboard artifact failures and rendering.

Critical tests:

- `tests/test_match_records.py`
- `tests/test_build_dataset.py`
- `tests/test_horizon_targets.py`
- `tests/test_context_data.py`
- `tests/features/test_rolling.py`
- `tests/test_calibration_chronology.py`
- `tests/test_calibration.py`
- `tests/test_split.py`
- `tests/test_acceptance_schema_aliases.py`
- `tests/test_yearly_backtesting.py`
- `tests/test_report_outputs.py`
- `tests/test_dashboard_data.py`
- `tests/test_dashboard_story.py`
- Replacement coverage for deleted survival and rolling suites.

---

## Task 1: Freeze and Review the Current Worktree

**Why:** Multiple agents modified source, tests, and docs. No phase should be accepted from an unstable diff.

**Files:**

- Read: all changed files from `git status --short`.
- Modify: none unless review finds correction needs.
- Create: optional `docs/closeout/diff_review_YYYY-MM-DD.md`.

- [ ] **Step 1: Capture status**

Run:

```powershell
git status --short
git diff --stat
```

Expected:

- `.omx/` changes ignored as local runtime state.
- Source/test/doc changes are reviewed by owner.

- [ ] **Step 2: Split changed files into buckets**

Use these buckets:

```text
owned-by-current-correction
concurrent-agent-change
generated-artifact
local-runtime-state
unknown
```

- [ ] **Step 3: Review all `unknown` and `concurrent-agent-change` files**

Run file-specific diffs:

```powershell
git diff -- <path>
```

Expected:

- Every source/test/doc change has a known task or a blocker note.
- No generated artifact is hand-edited as a substitute for producer fixes.

- [ ] **Step 4: Record review result**

Create or update:

```text
docs/closeout/diff_review_2026-06-09.md
```

Required fields:

```markdown
# Diff Review - 2026-06-09

## Accepted Changes

| File | Reason | Verification |
|---|---|---|

## Needs Follow-Up

| File | Risk | Owner | Next test |
|---|---|---|---|

## Ignored Local State

| File | Reason |
|---|---|
```

---

## Task 2: Replace Deleted Coverage With Explicit Contracts

**Why:** Collection dropped from documented 311 to 209 tests. Some deleted tests may be invalid because full survival modeling is out of scope, but deletion must be replaced by documented coverage.

**Files:**

- Modify: `docs/closeout/validation-matrix.md`
- Modify: `.agents/CLOSEOUT.md`
- Create: `docs/closeout/deleted_test_contract_map_2026-06-09.md`
- Possibly create tests under `tests/`

- [ ] **Step 1: List deleted test modules**

Run:

```powershell
git diff --name-status f9da5d7^ f9da5d7 -- tests
```

Expected deleted modules include:

```text
tests/test_integration_survival.py
tests/test_rolling_features.py
tests/test_survival_analysis.py
tests/test_survival_analysis_new.py
```

- [ ] **Step 2: Build contract map**

Create:

```text
docs/closeout/deleted_test_contract_map_2026-06-09.md
```

Required table:

```markdown
# Deleted Test Contract Map - 2026-06-09

| Deleted test file | Original contract | Keep, replace, or retire | Replacement test | Rationale |
|---|---|---|---|---|
| `tests/test_integration_survival.py` | matching/censoring integration | replace | `tests/test_match_records.py`, `tests/test_horizon_targets.py`, new integration test if needed | full Cox scope retired, but matching boundaries stay required |
| `tests/test_rolling_features.py` | rolling leakage guards | replace | `tests/features/test_rolling.py` | strict `[t-window,t)` still required |
| `tests/test_survival_analysis.py` | survival model behavior | replace with descriptive timing scope | `tests/test_hypothesis_evidence.py`, new `tests/test_survival_descriptive_scope.py` | Cox/Fine-Gray not thesis acceptance, but adopted-only timing language remains required |
| `tests/test_survival_analysis_new.py` | survival model behavior | replace with descriptive timing scope | `tests/test_hypothesis_evidence.py`, new `tests/test_survival_descriptive_scope.py` | Cox/Fine-Gray not thesis acceptance, but descriptive survival/retention claims still need guardrails |
```

- [ ] **Step 3: Add missing replacement tests**

Minimum replacement tests:

```python
def test_descriptive_survival_uses_adopted_only_days_to_adoption():
    # verifies descriptive timing only, not Cox/Fine-Gray model claims
    ...

def test_matching_and_horizon_outputs_are_consistent_for_unresolved_tail():
    # verifies unresolved intake excluded from supervised dataset but included in horizon cohort
    ...
```

- [ ] **Step 4: Verify collection count is documented**

Run:

```powershell
python -m pytest --collect-only -q
```

Expected:

- Updated collection count recorded in `.agents/CLOSEOUT.md`.
- No doc still claims 311 tests as current.

---

## Task 3: Deep-Audit Phase 2, Phase 5, and Phase 6

**Why:** Phase 1, 3, and 4 have deep reports. Phase 2, 5, and 6 do not. Phase trackers contain optimistic `FULL PASS` claims without equivalent harsh audit.

**Files:**

- Create: `docs/closeout/deep_testing_report_phase2.md`
- Create: `docs/closeout/deep_testing_report_phase5.md`
- Create: `docs/closeout/deep_testing_report_phase6.md`
- Modify: `docs/closeout/phase2.md`
- Modify: `docs/closeout/phase5.md`
- Modify: `docs/closeout/phase6.md`
- Modify: `.agents/CLOSEOUT.md`

- [ ] **Step 1: Audit Phase 2**

Scope:

```text
feature registry, leakage, target encoder, split chronology, metrics, model selection, bootstrap, tuning failure
```

Run:

```powershell
python -m pytest tests/test_feature_sets.py tests/test_leakage_audit.py tests/test_target_encoder.py tests/test_split.py tests/test_bootstrap.py -q
```

Report:

```text
docs/closeout/deep_testing_report_phase2.md
```

- [ ] **Step 2: Audit Phase 5**

Scope:

```text
test separation, acceptance markers, pipeline fail-fast, recency CLI, manifest freshness
```

Run:

```powershell
python -m pytest tests/test_pipeline_runner.py tests/test_recency_comparison.py tests/test_artifact_manifest.py -q
```

Report:

```text
docs/closeout/deep_testing_report_phase5.md
```

- [ ] **Step 3: Audit Phase 6**

Scope:

```text
manual regeneration, final docs, manifest, generated reports, stale text, stale artifacts
```

Run after manual regeneration only:

```powershell
python scripts/generate_artifact_manifest.py
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
```

Report:

```text
docs/closeout/deep_testing_report_phase6.md
```

- [ ] **Step 4: Remove false pass claims**

Search:

```powershell
rg -n "FULL PASS|Remaining risk: None|202 passed|311 tests|288 passed" docs/closeout .agents docs/PROJECT_CLOSEOUT_TASKS.md
```

Expected:

- Every stale claim either updated with fresh evidence or marked historical.

---

## Task 4: Finish External Context Correctness

**Why:** Weather lag and intake-volume issues are fixed, but timezone-aware dates and missing 311 coverage still carry ML leakage/meaning risk.

**Files:**

- Modify: `src/aac_adoption/data/context_data.py`
- Test: `tests/test_context_data.py`
- Maybe modify: `docs/METHODOLOGY.md`

- [ ] **Step 1: Add failing timezone tests**

Add tests:

```python
def test_weather_dates_with_austin_offsets_preserve_local_calendar_day():
    weather = pd.DataFrame(
        {
            "DATE": ["2024-07-01T23:30:00-05:00"],
            "TMAX": [99],
            "TMIN": [80],
            "PRCP": [0.0],
        }
    )
    normalized = normalize_weather_daily(weather)
    assert normalized["context_date"].tolist() == [pd.Timestamp("2024-07-01")]


def test_311_dates_with_offsets_preserve_local_calendar_day():
    requests = pd.DataFrame(
        {
            "request_date": ["2024-07-01T23:30:00-05:00"],
            "animal_311_requests": [3],
        }
    )
    normalized = normalize_311_animal_requests(requests)
    assert normalized["context_date"].tolist() == [pd.Timestamp("2024-07-01")]
```

- [ ] **Step 2: Add failing 311 coverage test**

Add:

```python
def test_311_context_outside_source_coverage_is_not_observed_zero():
    modeling = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "intake_datetime": pd.to_datetime(["2024-01-10"]),
        }
    )
    requests = pd.DataFrame(
        {
            "request_date": ["2024-02-01"],
            "animal_311_requests": [5],
        }
    )
    result = add_context_features(
        modeling,
        raw_intakes=modeling,
        weather_daily=None,
        requests_311=requests,
    )
    assert "animal_311_context_available" in result.columns
    assert bool(result["animal_311_context_available"].item()) is False
```

- [ ] **Step 3: Implement local-date normalization helper**

Add helper in `context_data.py`:

```python
def _parse_local_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed.map(lambda value: pd.NaT if pd.isna(value) else pd.Timestamp(value).tz_localize(None).normalize())
```

Use it in weather and 311 normalization.

- [ ] **Step 4: Add 311 availability flag**

Add to `CONTEXT_FEATURES`:

```python
"animal_311_context_available",
```

Implementation rule:

```text
available = target context date is between min and max observed 311 source dates
```

Keep observed missing within coverage as zero after rollup. Keep outside coverage flagged unavailable.

- [ ] **Step 5: Verify**

Run:

```powershell
python -m pytest tests/test_context_data.py tests/features/test_rolling.py -q
```

Expected:

- No same-day weather leakage.
- Offset timestamps preserve local calendar dates.
- Missing 311 coverage is distinguishable from observed zero demand.

---

## Task 5: Verify Chronological ML Contracts End-to-End

**Why:** Thesis methodology depends on strict split roles. Calibration and model selection were risky and must stay locked.

**Files:**

- Modify if needed: `src/aac_adoption/models/split.py`
- Modify if needed: `src/aac_adoption/models/calibrate.py`
- Modify if needed: `src/aac_adoption/analysis/model_selection.py`
- Test: `tests/test_calibration_chronology.py`
- Test: `tests/test_split.py`
- Test: `tests/test_acceptance_schema_aliases.py`

- [ ] **Step 1: Confirm split roles**

Run:

```powershell
python -m pytest tests/test_split.py tests/test_calibration_chronology.py -q
```

Expected:

- Train years: 2013-2021.
- Calibration year: 2022.
- Selection year: 2023.
- Test years: 2024-2025.

- [ ] **Step 2: Add model-selection anti-test-leak regression**

Add a test where:

```text
Model A wins 2023 selection but loses 2024-2025 test.
Model B loses 2023 selection but wins 2024-2025 test.
Expected selected model: A.
```

Minimum fixture columns:

```python
pd.DataFrame(
    [
        {"model_name": "A", "animal_subset": "combined", "metric_split": "selection", "pr_auc": 0.80, "roc_auc": 0.70, "brier_score": 0.20, "expected_calibration_error": 0.08, "split_strategy": "time", "is_thesis_evaluation": True, "selection_eligible": True, "artifact_path": "models/A.joblib"},
        {"model_name": "B", "animal_subset": "combined", "metric_split": "selection", "pr_auc": 0.70, "roc_auc": 0.95, "brier_score": 0.10, "expected_calibration_error": 0.02, "split_strategy": "time", "is_thesis_evaluation": True, "selection_eligible": True, "artifact_path": "models/B.joblib"},
        {"model_name": "A", "animal_subset": "combined", "metric_split": "test", "pr_auc": 0.60, "roc_auc": 0.60, "artifact_path": "models/A.joblib"},
        {"model_name": "B", "animal_subset": "combined", "metric_split": "test", "pr_auc": 0.95, "roc_auc": 0.95, "artifact_path": "models/B.joblib"},
    ]
)
```

- [ ] **Step 3: Verify model-selection text**

Run:

```powershell
python -m pytest tests/test_acceptance_schema_aliases.py -q
```

Expected:

- Summary text says 2023 selection chooses.
- Summary text says 2024-2025 test reports final performance only.
- Calibrated candidates mention 2022 calibration.

---

## Task 6: Regenerate Artifacts Manually

**Why:** Producers are changed, but generated files remain stale until pipeline/manual regeneration runs.

**Files written by producers:**

- `data/processed/modeling_dataset.csv`
- `data/processed/horizon_modeling_dataset.csv`
- `data/processed/unresolved_intakes.csv`
- `models/**`
- `reports/tables/**`
- `reports/figures/**`
- `reports/summary/**`
- `reports/artifact_manifest.csv`

- [ ] **Step 1: Run pipeline**

User-owned long command:

```powershell
python scripts/run_full_pipeline.py --skip-download --skip-shap
```

Expected:

- Pipeline stops at first failure.
- Run receipt exists.
- No later artifact produced after earlier failed step.

- [ ] **Step 2: Generate manifest**

Run:

```powershell
python scripts/generate_artifact_manifest.py
```

Expected:

- Manifest includes run IDs and hashes.
- No required thesis artifact path is null or missing.

- [ ] **Step 3: Run long acceptance**

User-owned long command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
```

Expected:

- Pass.
- If fail: record exact failing command, exit code, and artifact/log path in `docs/closeout/phase6.md`.

---

## Task 7: Reconcile Final-Facing Documents

**Why:** Thesis docs must not contradict code or regenerated artifacts.

**Files:**

- Modify: `README.md`
- Modify: `docs/METHODOLOGY.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/RESULTS.md`
- Modify: `docs/target_definitions.md`
- Modify: `streamlit_app.py` if final-facing dashboard copy is stale
- Modify: generated summaries only through producers when possible

- [ ] **Step 1: Search stale claims**

Run:

```powershell
rg -n "test PR-AUC|highest test|Selected by lowest test|time-split test period|Formal isotonic or Platt calibration was not applied|caus|TODO|PARTIAL|stub" README.md docs reports/summary streamlit_app.py src/aac_adoption/dashboard
```

Expected:

- No final-facing text says selection used 2024-2025 test metrics.
- No final-facing text claims causal effects.
- No final-facing text claims full Cox/Fine-Gray survival modeling is accepted.

- [ ] **Step 2: Add horizon target definition**

Update `docs/target_definitions.md` with:

```markdown
### Horizon Targets

`adopted_in_7d`, `adopted_in_30d`, `adopted_in_60d`, and `adopted_in_90d`
belong only to `horizon_modeling_dataset.csv`.

Values:

- `1.0`: matched adoption occurs within the horizon and before the next intake.
- `0.0`: matched non-adoption occurs within the horizon, matched outcome occurs after the horizon, or unresolved intake has at least that many observable follow-up days.
- `NaN`: unresolved intake has less than the required observable follow-up.
```

- [ ] **Step 3: Verify docs**

Run:

```powershell
python -m pytest tests/test_target_definitions.py tests/test_artifact_manifest.py -q
```

Expected:

- Target docs mention leakage.
- Artifact manifest wording matches adopted-only timing and final regenerated outputs.

---

## Task 8: Final Smoke and Handoff

**Why:** Human thesis review needs one clean status page with evidence and remaining caveats.

**Files:**

- Modify: `docs/closeout/phase6.md`
- Modify: `.agents/CLOSEOUT.md`
- Create: `docs/closeout/final_acceptance_evidence_YYYY-MM-DD.md`

- [ ] **Step 1: Run short suite**

Run:

```powershell
python -m pytest -q
```

Expected:

- Pass, or every fail is documented and intentionally out of thesis scope.

- [ ] **Step 2: Run final smoke commands**

Run:

```powershell
python -m pytest --collect-only -q
python -m py_compile src/aac_adoption/dashboard/data.py streamlit_app.py
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
python scripts/compare_recency.py --help
```

Expected:

- All exit 0.
- No duplicate CLI args.

- [ ] **Step 3: Write final evidence file**

Create:

```text
docs/closeout/final_acceptance_evidence_2026-06-09.md
```

Required content:

```markdown
# Final Acceptance Evidence - 2026-06-09

## Commands Run

| Command | Result | Notes |
|---|---|---|

## Artifact Regeneration

| Artifact group | Fresh run ID | Status |
|---|---|---|

## Remaining Caveats

| Caveat | Thesis risk | Mitigation |
|---|---|---|

## Final Verdict

Ready / Not ready.
```

---

## Execution Order

1. Task 1: Freeze and review current diff.
2. Task 2: Replace deleted coverage map.
3. Task 3: Deep-audit missing phases.
4. Task 4: Finish external context correctness.
5. Task 5: Verify chronological ML contracts.
6. Task 6: Regenerate artifacts manually.
7. Task 7: Reconcile final-facing docs.
8. Task 8: Final smoke and handoff.

## Current Blocker Summary

P0:

- Deleted coverage is not mapped.
- Generated artifacts are stale.
- Missing phase deep audits.

P1:

- Timezone context and 311 coverage semantics need tests.
- Trackers still contain historical overconfident pass claims.
- Full suite and long acceptance are not yet fresh.

## Self-Review

- Spec coverage: covers code correctness, ML chronology, deleted coverage, docs, regeneration, acceptance.
- Placeholder scan: no unresolved placeholder language remains.
- Type consistency: uses current names `classification_target`, `regression_target_days`, `adopted_in_*`, `horizon_modeling_dataset.csv`, `DatasetSplit.calibration`, and `metric_split == "selection"`.
