# AAC Thesis Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` task-by-task. Use a fresh implementer,
> then specification reviewer, then code-quality reviewer for every task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AAC adoption project reproducible, methodologically defensible,
artifact-consistent, dashboard-safe, and ready for thesis submission.

**Architecture:** Repair producer contracts from raw episode matching upward.
Freeze methodological decisions in machine-readable metadata. Separate cheap
isolated validation from canonical artifact regeneration. Make every downstream
consumer reject missing, stale, incompatible, or non-thesis artifacts.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, CatBoost, lifelines
(descriptive survival helpers only), Streamlit, Plotly, pytest, PowerShell.

---

## 0. Execution Contract

### Worktree state

The worktree is already dirty. Existing edits include:

- `scripts/compare_recency.py`
- `src/aac_adoption/models/train_boosting.py`
- `src/aac_adoption/models/tune.py`
- `src/aac_adoption/models/yearly_backtesting.py`
- `tests/test_hyperparam_tuning.py`
- `tests/test_yearly_backtesting.py`

Workers must inspect current diffs before editing these files. Never revert
unrelated user or agent changes.

### Execution waves

Tasks sharing files run sequentially. Safe cavecrew/subagent waves:

1. Task 1 then Task 2, same matching owner.
2. Task 3 survival-scope owner may run after Task 1 target schema is stable.
3. Task 4 then Task 5, same context owner.
4. Task 6 and Task 7 may run in parallel after Phase 1.
5. Task 8 then Task 9, same metric/selection contract chain.
6. Tasks 10, 11, 12, and 13 may run in parallel after Task 8 interfaces settle,
   provided their write sets remain disjoint.
7. Task 14 precedes Tasks 15-17.
8. Tasks 18-21 run after producer and dashboard contracts are stable.
9. Tasks 22-24 are strictly sequential.

Each task uses one implementer at a time. Review agents are read-only. When work
is executed in an isolated clean branch, make one focused checkpoint commit per
completed task after both reviews pass. Do not create commits from the current
dirty main worktree unless explicitly requested.

### Global invariants

- Supervised rows require a valid matched outcome.
- Horizon targets use a separate all-intake cohort with outcome/follow-up-aware
  labels; unresolved rows never enter matched-episode classification/regression.
- No outcome may cross a later intake boundary for the same animal.
- No target, alias, post-outcome field, raw identifier, or raw timestamp may be a
  model predictor.
- Context features must be computable at intake time from raw chronological
  source data, independent of later outcome availability.
- Thesis evaluation uses training 2013-2021, calibration fitting 2022,
  selection/threshold choice 2023, and final test 2024-2025.
- Selection uses 2023 metrics. Test metrics are reporting only.
- Every saved model must carry exact feature order, target, transform, split,
  artifact identity, dataset fingerprint, package versions, and source revision.
- Dashboard must consume artifacts only. It must never train or silently invent
  predictions.
- Generated files are changed by producer regeneration, not hand editing.
- Canonical regeneration requires a clean committed tree. Every producer writes
  a receipt under one accepted run ID.

### Worker handoff format

```text
Scope:
Files changed:
Behavioral contract:
Tests run:
Test result:
Generated artifacts changed:
Remaining risk:
```

---

## Phase 1: Episode and Dataset Integrity

### Task 1: Separate matched supervised episodes from unresolved intake audit

**Ownership**

- Modify: `src/aac_adoption/data/match_records.py`
- Modify: `src/aac_adoption/data/build_dataset.py`
- Modify: `tests/test_match_records.py`
- Modify: `tests/test_build_dataset.py`
- Modify: `tests/test_integration_survival.py` only for shared matching/censoring
  integration cases, not model-level survival behavior

**Required interface**

Replace the ambiguous `(matched_df, unmatched_count)` contract with a result
object:

```python
@dataclass(frozen=True)
class MatchResult:
    matched_episodes: pd.DataFrame
    unresolved_intakes: pd.DataFrame
    unmatched_intakes: int
```

`matched_episodes` contains only rows with a valid `outcome_datetime` and
non-negative `days_to_outcome`. `unresolved_intakes` contains audit-only rows and
must not receive supervised targets.

- [ ] **Step 1: Add failing unresolved-row exclusion test**

```python
def test_unresolved_intake_never_enters_supervised_dataset():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A2"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "animal_type": ["Dog", "Cat"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "outcome_datetime": pd.to_datetime(["2024-01-05"]),
            "outcome_type": ["Adoption"],
        }
    )

    result = match_intakes_to_future_outcomes(intakes, outcomes)

    assert result.unmatched_intakes == 1
    assert result.matched_episodes["animal_id"].tolist() == ["A1"]
    assert result.unresolved_intakes["animal_id"].tolist() == ["A2"]
```

- [ ] **Step 2: Add failing supervised-target completeness test**

```python
def test_supervised_dataset_has_no_null_outcome_or_target():
    result = build_modeling_dataset(intakes_fixture, outcomes_fixture)
    required = [
        "outcome_datetime",
        "outcome_type",
        "days_to_outcome",
        "classification_target",
        "regression_target_days",
    ]
    assert result.dataset[required].notna().all().all()
```

- [ ] **Step 3: Run tests and confirm current failure**

Run:

```powershell
python -m pytest tests/test_match_records.py::test_unresolved_intake_never_enters_supervised_dataset tests/test_build_dataset.py::test_supervised_dataset_has_no_null_outcome_or_target -q
```

Expected: FAIL because unresolved rows are appended to the modeling frame.

- [ ] **Step 4: Implement `MatchResult`**

Build `unresolved_intakes` inside the same per-animal loop. Do not reconstruct it
after matching with stale `episodes_by_time` state. Preserve original intake
columns plus:

```python
{
    "unresolved_reason": "no_unused_future_outcome",
    "observation_end": extract_end_date,
    "followup_days_available": observation_end - intake_datetime,
}
```

Pass `extract_end_date` into the matcher. Reject a missing cutoff when unresolved
audit durations are requested.

- [ ] **Step 5: Build supervised targets from `matched_episodes` only**

`build_modeling_dataset()` must:

1. Read `match_result.matched_episodes`.
2. Compute targets only there.
3. Return unresolved audit data in `DatasetBuildResult`.
4. Set `matched_rows = len(dataset)`.
5. Set `unmatched_intakes = len(unresolved_intakes)`.

Extend the result:

```python
@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: pd.DataFrame
    unresolved_intakes: pd.DataFrame
    matched_rows: int
    unmatched_intakes: int
```

`build_modeling_dataset_from_files()` writes unresolved audit rows to
`data/processed/unresolved_intakes.csv` by default, or to a caller-supplied path.
The file records source intake identity, observation cutoff, available follow-up,
and unresolved reason. It contains no classification or regression target.

- [ ] **Step 6: Validate count conservation**

Add:

```python
assert result.matched_rows + result.unmatched_intakes == len(clean_intake_df)
```

For animals with unusable historical outcomes before intake, ensure those
outcomes do not alter the intake conservation equation.

- [ ] **Step 7: Run focused contract suite**

```powershell
python -m pytest tests/test_match_records.py tests/test_build_dataset.py -q
```

Done gate: all pass; unresolved rows never contain supervised targets.
`reports/summary/data_audit.md` and
`reports/summary/matching_ambiguity.md` must derive matched/unresolved counts from
the new result object and agree exactly.

---

### Task 1A: Build a separate all-intake horizon cohort

**Ownership**

- Modify: `src/aac_adoption/data/build_dataset.py`
- Modify: `scripts/build_dataset.py`
- Modify: `tests/test_horizon_targets.py`
- Modify: `tests/test_build_dataset.py`

**Methodological contract**

Matched-episode data remains source for `classification_target` and
`regression_target_days`. Horizon targets use one row per intake. For horizon H:

```text
1 = matched adoption occurs within H days and before next intake boundary
0 = matched non-adoption occurs within H days
0 = matched outcome occurs after H days
0 = unresolved intake has at least H days observable follow-up
NaN = unresolved intake has less than H days observable follow-up
```

A later intake truncates prior-episode follow-up. Prior episode is eligible only
for horizons fully observed before that boundary.

- [ ] **Step 1: Add result type**

```python
@dataclass(frozen=True)
class HorizonDatasetBuildResult:
    dataset: pd.DataFrame
    observation_end: pd.Timestamp
    horizon_days: tuple[int, ...]
```

- [ ] **Step 2: Add failing truth-table tests**

For 7, 30, 60, and 90 days cover adoption before/after H, non-adoption before H,
unresolved follow-up exactly H, unresolved follow-up shorter than H, and later
intake truncation.

- [ ] **Step 3: Add cohort separation test**

```python
assert unresolved_id not in matched_result.dataset["animal_id"].tolist()
assert unresolved_id in horizon_result.dataset["animal_id"].tolist()
```

- [ ] **Step 4: Write separate artifacts**

```text
data/processed/modeling_dataset.csv
data/processed/horizon_modeling_dataset.csv
data/processed/unresolved_intakes.csv
```

Each receives separate schema/target metadata. Horizon trainers read only
`horizon_modeling_dataset.csv`.

- [ ] **Step 5: Verify**

```powershell
python -m pytest tests/test_horizon_targets.py tests/test_build_dataset.py tests/test_match_records.py -q
```

Done gate: eligible unresolved negatives support horizon evaluation without
entering matched-only classification/regression.

---

### Task 2: Prevent outcome assignment across a later intake boundary

**Ownership**

- Modify: `src/aac_adoption/data/match_records.py`
- Modify: `tests/test_match_records.py`

**Behavioral decision**

For intake `i`, candidate outcome must satisfy:

```text
intake_i <= outcome < intake_(i+1)
```

for any later intake of the same animal. An outcome after the next intake belongs
to the later episode candidate set, not the earlier one. Boundary equality must be
documented and tested; use strict `< next_intake` to avoid assigning an outcome at
the exact next-intake timestamp to the prior stay.

- [ ] **Step 1: Add failing boundary test**

```python
def test_outcome_after_next_intake_not_assigned_to_prior_episode():
    intakes = pd.DataFrame(
        {
            "animal_id": ["A1", "A1"],
            "intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "animal_type": ["Dog", "Dog"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "animal_id": ["A1"],
            "outcome_datetime": pd.to_datetime(["2024-01-10"]),
            "outcome_type": ["Adoption"],
        }
    )

    result = match_intakes_to_future_outcomes(
        intakes, outcomes, extract_end_date=pd.Timestamp("2024-01-15")
    )

    assert result.matched_episodes["intake_datetime"].tolist() == [
        pd.Timestamp("2024-01-05")
    ]
    assert result.unresolved_intakes["intake_datetime"].tolist() == [
        pd.Timestamp("2024-01-01")
    ]
```

- [ ] **Step 2: Add repeat-stay property cases**

Cover:

- equal numbers of intakes/outcomes;
- fewer outcomes than intakes;
- outcomes before first intake;
- exact same timestamp boundary;
- unsorted input;
- duplicate timestamps with stable original-row tie-break;
- outcome reuse prohibited.

- [ ] **Step 3: Implement bounded candidate selection**

Within each animal:

1. Sort intakes and outcomes stably.
2. For each intake, derive `next_intake_datetime`.
3. Advance past outcomes earlier than current intake.
4. Match only if candidate is before next intake or no next intake exists.
5. Consume matched outcome once.
6. Send intake to unresolved audit otherwise.

- [ ] **Step 4: Remove `is_ambiguous_match` as a substitute for correctness**

Keep an ambiguity/audit field only for genuinely unresolved source ambiguity.
Never knowingly create a cross-episode match and label it ambiguous.

- [ ] **Step 5: Verify**

```powershell
python -m pytest tests/test_match_records.py tests/test_build_dataset.py::test_repeated_animal_matches_each_intake_to_next_unused_outcome -q
```

Done gate: no matched interval contains another intake for the same animal.

---

### Task 3: Remove invalid survival-model censoring from supervised dataset contract

**Ownership**

- Modify: `src/aac_adoption/data/build_dataset.py`
- Modify: `src/aac_adoption/analysis/survival_analysis.py`
- Modify: `src/aac_adoption/models/train_survival.py`
- Modify: `tests/test_integration_survival.py`
- Modify: `tests/test_survival_analysis.py`
- Modify: `tests/test_survival_analysis_new.py`
- Modify later: final-facing docs and dashboard wording

**Locked scope**

Full Cox and competing-risk modeling is not accepted thesis functionality.
Descriptive Kaplan-Meier analysis among adopted animals may remain if it uses
`days_to_adoption` and is clearly labeled descriptive.

- [ ] **Step 1: Classify survival tests**

Keep tests for:

- descriptive Kaplan-Meier output;
- empty-input handling;
- adopted-only durations;
- descriptive group curves;
- explicit labels and event counts.

Remove from acceptance or delete tests that require:

- Cox model artifacts;
- proportional-hazards validation;
- Brier scores from partial hazards;
- Fine-Gray/subdistribution claims;
- model-level censoring propagation into the supervised frame.

Do not mark broken tests `xfail`. Remove invalid scope and replace it with tests
for the locked descriptive contract.

- [ ] **Step 2: Remove fabricated censoring defaults**

Matched supervised episodes are resolved events. Do not overwrite or fabricate:

```python
dataset["is_censored"] = False
dataset["censoring_reason"] = ""
dataset["event_type"] = outcome_type_normalized
```

Either remove these fields from `REQUIRED_MODELING_COLUMNS`, or define them only
for a separate time-to-event audit dataset with a correct estimand. For thesis
closeout, prefer removal from the main modeling dataset.

- [ ] **Step 3: Disable unsupported training entrypoint**

`train_all_survival()` must fail explicitly:

```python
raise NotImplementedError(
    "Thesis scope includes descriptive Kaplan-Meier analysis only; "
    "Cox and competing-risk model training is not an accepted pipeline step."
)
```

If no canonical script calls this entrypoint, remove dead training code rather
than retaining misleading implementation.

- [ ] **Step 4: Ensure descriptive survival uses adopted-only timing**

Required input:

```python
adopted = df.loc[df["classification_target"].eq(1)].copy()
duration_col = "days_to_adoption"
event_observed = np.ones(len(adopted), dtype=int)
```

This is a distribution-of-time-among-observed-adoptions view, not a censor-aware
estimate of adoption incidence. Generated text must say that directly.

- [ ] **Step 5: Verify reduced scope**

```powershell
python -m pytest tests/test_survival_analysis.py tests/test_survival_analysis_new.py tests/test_integration_survival.py -q
```

Done gate: every retained test corresponds to descriptive functionality claimed
in methodology; no output calls a regular Cox model a competing-risk model.

---

### Task 4: Compute intake-volume context from raw intake history with calendar windows

**Ownership**

- Modify: `src/aac_adoption/data/context_data.py`
- Modify: `src/aac_adoption/features/rolling_features_cache.py`
- Modify: `src/aac_adoption/data/build_dataset.py`
- Modify: `scripts/build_dataset.py`
- Modify: `scripts/run_full_pipeline.py`
- Modify: `tests/test_context_data.py`
- Modify: `tests/test_rolling_features.py`
- Modify: `tests/features/test_rolling.py`
- Modify: `tests/test_build_dataset.py`

- [ ] **Step 1: Add outcome-independence test**

Build identical raw intake history with two different outcome files. Context
volume for the same focal intake must be identical.

```python
assert dataset_a.loc[focal, "intake_volume_7d"] == dataset_b.loc[focal, "intake_volume_7d"]
```

- [ ] **Step 2: Add sparse-date calendar test**

```python
def test_intake_volume_7d_uses_calendar_days_not_observed_rows():
    history = pd.DataFrame(
        {"intake_datetime": pd.to_datetime(["2024-01-01", "2024-01-10"])}
    )
    result = compute_prior_intake_counts(history)
    assert result.loc[result["intake_datetime"].eq("2024-01-10"), "intake_volume_7d"].item() == 0
```

- [ ] **Step 3: Refactor context API**

Use:

```python
def add_context_features(
    modeling_df: pd.DataFrame,
    *,
    raw_intakes: pd.DataFrame,
    weather_daily: pd.DataFrame | None,
    requests_311: pd.DataFrame | None,
) -> pd.DataFrame:
```

The rolling intake counts must derive from `raw_intakes`, not `modeling_df`.

- [ ] **Step 4: Implement true time-based windows**

For each focal intake timestamp `t`, count raw intakes in:

```text
[t - 7 days, t)
[t - 30 days, t)
```

Exclude the current intake and all same-time later rows. Use stable timestamp plus
source-row order if exact ties exist.

- [ ] **Step 5: Move volume threshold after context creation**

Remove the default row filter. Set `max_intake_volume_threshold=None` by default.
An intake-volume predictor is not a data-quality defect, and silently deleting
busy-day rows changes the estimand. If an explicit sensitivity run enables the
threshold, apply it only after `intake_volume_7d` exists and record:

- threshold value;
- rows before;
- rows removed;
- rows after.

Write these to dataset audit metadata.

- [ ] **Step 6: Wire canonical pipeline context inputs**

Pipeline step 2 must pass configured context paths. If files are absent, emit an
explicit context-disabled status and omit context feature-set claims. Do not fill
missing context and still label model `intake_time_context_v2`.

- [ ] **Step 7: Verify**

```powershell
python -m pytest tests/test_context_data.py tests/test_rolling_features.py tests/features/test_rolling.py tests/test_build_dataset.py -q
```

Done gate: outcome file changes cannot change intake-volume features.

---

### Task 5: Make weather features intake-time valid

**Ownership**

- Modify: `src/aac_adoption/data/context_data.py`
- Modify: `tests/test_context_data.py`
- Modify: `docs/METHODOLOGY.md` later

**Decision**

Daily completed weather is not known during a morning intake. For strict
intake-time prediction, use prior completed calendar day weather. Rename metadata
semantics, not necessarily columns, but document the lag.

- [ ] **Step 1: Add failing morning-intake test**

An intake at `2024-07-02 08:00` must not receive the completed weather summary for
July 2.

- [ ] **Step 2: Lag daily weather by one day**

Join focal intake date `D` to weather date `D-1`. Add:

```python
context_weather_lag_days = 1
```

to dataset/build metadata.

- [ ] **Step 3: Preserve missingness**

Missing weather must remain nullable. `is_extreme_heat=False` must mean observed
weather below threshold, not unknown weather. Add:

- `weather_available`;
- nullable `is_extreme_heat`;
- nullable `is_rainy_day`.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_context_data.py -q
```

Done gate: no feature uses weather information completed after intake timestamp.

---

## Phase 2: Feature, Split, and Evaluation Correctness

### Task 6: Harden predictor registry against targets, identifiers, and timestamps

**Ownership**

- Modify: `src/aac_adoption/features/feature_sets.py`
- Modify: `src/aac_adoption/features/target_encoder.py`
- Modify: `tests/test_feature_sets.py`
- Modify: `tests/test_target_encoder.py`
- Modify: `tests/test_leakage_audit.py`

- [ ] **Step 1: Remove targets from `NUMERIC_FEATURES`**

Allowed numeric predictors:

```python
NUMERIC_FEATURES = [
    "age_days",
    "daily_temp_max",
    "daily_temp_min",
    "daily_precipitation",
    "animal_311_requests_7d",
    "animal_311_requests_30d",
    "intake_volume_7d",
    "intake_volume_30d",
]
```

- [ ] **Step 2: Define prohibited predictor classes**

```python
IDENTIFIER_COLUMNS = {"animal_id"}
RAW_TIME_COLUMNS = {"intake_datetime", "outcome_datetime"}
PROHIBITED_MODEL_COLUMNS = LEAKAGE_COLUMNS | IDENTIFIER_COLUMNS | RAW_TIME_COLUMNS
```

Derived intake calendar fields remain allowed.

- [ ] **Step 3: Reject prohibited fields in every feature API**

`available_features_for_df`, `model_feature_columns`, and target encoder column
selection must all call the same validator.

- [ ] **Step 4: Add exact tests**

```python
@pytest.mark.parametrize(
    "column",
    [
        "classification_target",
        "regression_target_days",
        "days_to_outcome",
        "days_to_adoption",
        "animal_id",
        "intake_datetime",
        "outcome_type",
    ],
)
def test_validate_no_leakage_rejects_prohibited_predictor(column):
    with pytest.raises(ValueError, match=column):
        validate_no_leakage([column])
```

- [ ] **Step 5: Verify feature artifacts**

```powershell
python -m pytest tests/test_feature_sets.py tests/test_target_encoder.py tests/test_leakage_audit.py -q
```

Done gate: feature list generation cannot silently retain prohibited columns.

---

### Task 6A: Enforce target-specific analysis tables and labels

**Ownership**

- Modify: `src/aac_adoption/analysis/h3_age_evidence.py`
- Modify: `src/aac_adoption/analysis/h5_covid_evidence.py`
- Modify: `src/aac_adoption/analysis/hypothesis_tables.py`
- Modify: direct hypothesis/evidence tests

- [ ] **Step 1: Remove semantic fallback chains**

Adopted-only timing code must require `days_to_adoption` or derive it only by:

```python
adopted = df.loc[df["classification_target"].eq(1)].copy()
adopted["days_to_adoption"] = adopted["regression_target_days"]
```

It must never silently select the first available column from
`days_to_adoption`, `days_to_outcome`, and `regression_target_days` without also
enforcing the adopted subset.

- [ ] **Step 2: Make each output declare its estimand**

Required columns:

```text
target_column
population_scope
estimand_label
```

Examples:

- all episodes + `classification_target`;
- all matched episodes + `regression_target_days`;
- adopted episodes only + `days_to_adoption`.

- [ ] **Step 3: Add adversarial non-adoption test**

Create non-adopted rows with extreme LOS. Assert adopted-only H3 medians are
unchanged when those rows are added.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_hypothesis_evidence.py tests/test_analysis_outputs.py tests/test_acceptance_schema_aliases.py -q
```

Done gate: every hypothesis table's population and target are machine-readable
and semantically correct.

---

### Task 7: Make chronological calibration, selection, and test periods mandatory

**Ownership**

- Modify: `src/aac_adoption/models/split.py`
- Modify: `tests/test_split.py`
- Modify: all trainer metadata tests as required

**Required API**

```python
def make_time_split(
    df: pd.DataFrame,
    target_column: str,
    animal_subset: str | None = None,
    recency_weighting: bool = True,
    *,
    allow_random_fallback: bool = False,
) -> DatasetSplit:
```

- [ ] **Step 1: Extend split result**

`DatasetSplit` contains separate `train`, `calibration`, `selection`, and `test`
frames.

- [ ] **Step 2: Require all periods**

Chronological thesis split requires nonempty:

- train: 2013-2021;
- calibration: 2022;
- selection: 2023;
- test: 2024-2025.

If any is empty and `allow_random_fallback=False`, raise:

```text
Thesis chronological split unavailable: missing calibration period 2022
```

- [ ] **Step 3: Mark development fallback clearly**

When explicitly enabled:

```python
strategy = "random_development_only"
is_thesis_evaluation = False
```

- [ ] **Step 4: Add tests**

Cover missing train, calibration, selection, and test periods separately. Assert
random fallback requires explicit opt-in.

- [ ] **Step 5: Update trainers**

Production scripts use default strict behavior. Small unit fixtures may pass
`allow_random_fallback=True`, and expected metadata must say development-only.

- [ ] **Step 6: Verify**

```powershell
python -m pytest tests/test_split.py tests/test_train_baseline_outputs.py tests/test_train_boosting_outputs.py tests/test_train_advanced_outputs.py -q
```

Done gate: no random-split result can be selected or described as thesis evidence.

---

### Task 8: Persist selection and test metrics separately

**Ownership**

- Modify: `src/aac_adoption/models/train_baseline.py`
- Modify: `src/aac_adoption/models/train_boosting.py`
- Modify: `src/aac_adoption/models/train_advanced.py`
- Modify: `src/aac_adoption/models/train_adopted_regression.py`
- Modify: direct output tests

**Required metric schema**

Every model row carries:

```text
metric_split
selection_eligible
split_strategy
is_thesis_evaluation
artifact_path
target_column
target_transform
feature_columns
```

Produce one selection row and one test row per fitted artifact. Calibration-period
metrics may be diagnostic but are never selection-eligible. Prefer long-form rows
because selection filters `metric_split == "selection"`.

- [ ] **Step 1: Add output-schema tests**

Assert selection and test rows exist and share the same artifact identity.

- [ ] **Step 2: Evaluate selection and test without refitting**

Train base model on 2013-2021. Fit calibration on 2022. Use untouched 2023 rows
for model family, calibration-method, and threshold selection. Evaluate frozen
artifact and threshold on 2024-2025.

- [ ] **Step 3: Correct regression transform metadata**

Persist:

```python
{
    "target_column": "regression_target_days",
    "target_transform": "log1p" | "identity",
    "prediction_inverse_transform": "expm1" | "identity",
    "training_target_min": ...,
    "training_target_max": ...,
}
```

Do not infer transform from model family.

- [ ] **Step 4: Resolve adopted-only regression ownership**

`train_advanced_regression()` is the main all-outcome LOS model. Remove its
adopted-only docstring, unused `filter_df`/`filter_col`, and any adopted-only
labels. `train_adopted_regression.py` exclusively owns optional
`days_to_adoption` modeling and must filter before splitting:

```python
adopted_df = df.loc[
    df["classification_target"].eq(1) & df["days_to_adoption"].notna()
].copy()
split = make_time_split(adopted_df, "days_to_adoption", animal_subset=subset)
```

The two tasks use distinct artifact task names: `regression` and
`regression_adopted`.

- [ ] **Step 5: Verify**

```powershell
python -m pytest tests/test_train_baseline_outputs.py tests/test_train_boosting_outputs.py tests/test_train_advanced_outputs.py -q
```

Done gate: every candidate has 2023 selection metrics and frozen test metrics.
Main regression metadata says `regression_target_days`; adopted-only metadata
says `days_to_adoption`.

---

### Task 9: Select model, calibration, and threshold on 2023 only

**Ownership**

- Modify: `src/aac_adoption/analysis/model_comparison.py`
- Modify: `src/aac_adoption/analysis/model_selection.py`
- Modify: `src/aac_adoption/analysis/threshold_analysis.py`
- Modify: `src/aac_adoption/analysis/calibration_summary.py`
- Modify: `tests/test_analysis_outputs.py`
- Modify: `tests/test_acceptance_schema_aliases.py`
- Modify: `tests/test_calibration.py`

- [ ] **Step 1: Add adversarial model-selection test**

Create model A that wins 2023 and loses test, model B that loses 2023 and wins
test. Assert model A is selected.

```python
assert selected["model_name"].item() == "selection_winner"
assert selected["selection_source"].item() == "selection_2023"
```

- [ ] **Step 2: Filter selection candidates**

Candidates require:

```python
metric_split == "selection"
split_strategy == "time"
is_thesis_evaluation == True
selection_eligible == True
artifact_path.notna()
```

- [ ] **Step 3: Include calibrated classifiers**

Calibrated artifacts compete only when fitted on 2022 and evaluated on untouched
2023 rows. Persist `calibration_period=2022` and `selection_period=2023`. Rank
classification by:

1. 2023 PR-AUC;
2. 2023 Brier score;
3. 2023 ECE;
4. 2023 ROC-AUC.

Document deterministic tie tolerances.

Also align feature-set labels. Current producers emit `intake_time_v2` and
`intake_time_context_v2`; comparison code must not require v1 labels. Support
legacy labels only as explicit migration aliases, never as canonical new output.

- [ ] **Step 4: Freeze exact artifact identity**

Final selection must preserve:

- `artifact_path`;
- `model_name`;
- `artifact_task`;
- `calibration_method`;
- `feature_set`;
- `feature_columns`;
- `target_column`;
- `target_transform`;
- `selection_source`;
- `selection_metric`;
- `selection_value`.

Remove arbitrary artifact-search fallback from threshold analysis. Missing exact
artifact is an error.

- [ ] **Step 5: Keep test metrics reporting-only**

Join test metrics after winner selection using artifact identity. They may appear
in the final table but cannot affect `selected`.

- [ ] **Step 6: Fix calibration subset summary**

Filter reliability/calibration rows by actual animal subset. If source data is
combined-only, emit combined-only, not copied dogs/cats rows.

- [ ] **Step 7: Verify**

```powershell
python -m pytest tests/test_analysis_outputs.py tests/test_acceptance_schema_aliases.py tests/test_calibration.py tests/test_calibration_advanced.py -q
```

Done gate: changing 2022 fit diagnostics or test metrics cannot change selected
artifact; only untouched 2023 selection metrics can.

---

### Task 10: Fix cluster bootstrap multiplicity

**Ownership**

- Modify: `src/aac_adoption/models/bootstrap.py`
- Modify: `src/aac_adoption/models/evaluate.py`
- Modify: `tests/test_evidence_pack.py`
- Add or modify: focused bootstrap tests

- [ ] **Step 1: Add deterministic multiplicity test**

Expose or inject one sampled cluster vector such as `["A", "A", "B"]`. Assert
rows for A appear twice in the bootstrap sample.

- [ ] **Step 2: Remove `np.unique(sample_indices)`**

Concatenated cluster row indices must preserve repeated cluster draws. Validate
input lengths:

```python
len(y_true) == len(y_pred) == len(animal_ids)
```

- [ ] **Step 3: Deduplicate implementation**

Keep one canonical bootstrap implementation. Have `evaluate.py` import it rather
than maintaining a second copy.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_evidence_pack.py tests/test_diagnostics_outputs.py -q
```

Done gate: duplicate sampled animals alter sample weight as bootstrap requires.

---

### Task 11: Make tuning failure explicit and usable

**Ownership**

- Modify current dirty files carefully:
  `src/aac_adoption/models/tune.py`,
  `src/aac_adoption/models/train_boosting.py`,
  `src/aac_adoption/models/train_advanced.py`
- Modify: `tests/test_hyperparam_tuning.py`

- [ ] **Step 1: Add all-trials-pruned test**

Patch model fitting to raise a known prunable error. Assert tuning returns a
structured failed study status and does not write `null` parameters as usable
configuration.

- [ ] **Step 2: Define tuning result**

```python
{
    "status": "ok" | "failed",
    "best_params": dict | None,
    "best_value": float | None,
    "completed_trials": int,
    "pruned_trials": int,
    "failure_reason": str | None,
}
```

- [ ] **Step 3: Trainer behavior**

If tuned params status is failed:

- fail in strict thesis pipeline mode; or
- use declared built-in defaults only when an explicit development flag permits.

Never call `dict.update(None)`.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_hyperparam_tuning.py -q
```

Done gate: tuning cannot silently manufacture a successful configuration.

---

## Phase 3: Backtesting and Reporting

### Task 12: Define yearly backtesting windows and failure rows

**Ownership**

- Modify current dirty files carefully:
  `src/aac_adoption/models/yearly_backtesting.py`,
  `tests/test_yearly_backtesting.py`
- Modify: `scripts/evaluate_backtesting.py`

**Locked window rule**

For a test year `Y`, training uses all available years from 2013 through `Y-1`.
A row is eligible only if at least one pre-`Y` year exists. Therefore, a fixture
whose earliest year is 2019 cannot produce a valid 2019 test row.

- [ ] **Step 1: Correct fixture or expected years**

Preferred: extend six-year fixture with 2018 training rows if 2019 must be tested.
Otherwise expect 2020-2024. Do not mutate code to create `(2019, 2018)`.

- [ ] **Step 2: Add explicit window status**

Every requested year/subset/target/model yields:

```text
status = ok | skipped | failed
skip_reason
error_type
error_message
train_start_year
train_end_year
test_year
```

Failed rows are not plausible metric rows. Acceptance requires no `failed` rows.

- [ ] **Step 3: Align preprocessing**

Fit preprocessing on training only. Apply same fitted schema to test. Raw age
strings must be parsed through shared feature engineering or excluded.

- [ ] **Step 4: Test horizon targets**

For each horizon target, ensure rows with insufficient follow-up are absent from
that year's evaluation.

`scripts/evaluate_backtesting.py` and pipeline routing must select:

```text
classification_target, regression_target_days -> modeling_dataset.csv
adopted_in_*d -> horizon_modeling_dataset.csv
```

Passing `modeling_dataset.csv` for a horizon target must fail with a schema/cohort
error. Add an integration test asserting source path, all-intake row count, and
eligible target count in backtesting output metadata.

- [ ] **Step 5: Verify short cases**

```powershell
python -m pytest tests/test_yearly_backtesting.py -k "get_test_years or get_train_years or empty_splits or output_schema" -q
```

- [ ] **Step 6: Mark full suite USER-RUN**

The full yearly backtesting test file may exceed two minutes. User executes it at
the manual long-run gate.

Done gate: no impossible first-year window; every skip/failure is explicit.

---

### Task 13: Repair report generation around PR-AUC and current schemas

**Ownership**

- Modify: `src/aac_adoption/reporting/report.py`
- Modify: `tests/test_report_outputs.py`
- Modify: `scripts/generate_report_outputs.py` only if CLI/path behavior changes

- [ ] **Step 1: Decide contract in test**

Classification summary heading must be:

```text
Best classification models by 2023 selection PR-AUC
```

Regression heading:

```text
Best regression models by 2023 selection MAE
```

- [ ] **Step 2: Require selection source columns**

Report generation must reject model comparison data without split/source metadata
once Task 8 is complete. During migration, update fixture rows to include it.

- [ ] **Step 3: Generate from passed directories only**

Tests use temporary tables/figures/summary directories. Producer must not read
canonical reports as fallback.

- [ ] **Step 4: Add missing-table behavior**

Required input missing: explicit failure.
Optional hypothesis table missing: section states unavailable and names producer.

- [ ] **Step 5: Verify**

```powershell
python -m pytest tests/test_report_outputs.py -q
```

Done gate: report text uses 2023 selection and current target terminology.

---

## Phase 4: Artifact Metadata and Dashboard Safety

### Task 14: Make model sidecar metadata complete and verifiable

**Ownership**

- Modify: `src/aac_adoption/models/metadata.py`
- Modify: `src/aac_adoption/models/artifacts.py`
- Modify: all trainers as needed
- Modify: model output tests

- [ ] **Step 1: Define metadata schema**

Required:

```python
REQUIRED_MODEL_METADATA = {
    "schema_version",
    "model_name",
    "task",
    "animal_subset",
    "artifact_path",
    "artifact_sha256",
    "dataset_path",
    "dataset_sha256",
    "feature_columns",
    "target_column",
    "target_transform",
    "prediction_inverse_transform",
    "split_strategy",
    "is_thesis_evaluation",
    "train_period",
    "calibration_period",
    "selection_period",
    "test_period",
    "random_state",
    "run_id",
    "run_timestamp",
    "producer_source_sha",
    "packages",
}
```

- [ ] **Step 2: Compute real hashes and revision**

Use `hashlib.sha256` streaming reads. Resolve source SHA with a short subprocess call;
if repository state prevents it, record explicit `unavailable` plus reason, not
`unknown`.

- [ ] **Step 3: Validate before write and load**

Implement shared `validate_model_metadata(metadata)` and use it in dashboard
loading.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_train_baseline_outputs.py tests/test_train_boosting_outputs.py tests/test_train_advanced_outputs.py tests/test_dashboard_data.py -q
```

Done gate: dashboard never guesses feature order or target transform.

---

### Task 14A: Add run context and per-producer artifact receipts

**Ownership**

- Add: `src/aac_adoption/provenance.py`
- Modify: canonical producer scripts under `scripts/`
- Modify: `scripts/generate_artifact_manifest.py`
- Add: `tests/test_provenance.py`

**Canonical source requirement**

`git status --porcelain` must return no output before canonical regeneration.
Development runs may use dirty trees but are never acceptance-eligible.

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str
    producer_source_sha: str
    profile: str
    started_at: str
    command: list[str]
    input_hashes: dict[str, str]
```

Profiles:

- `thesis-full`: all required artifacts, including SHAP;
- `development-no-shap`: never acceptance-eligible.

- [ ] **Step 1: Create producer receipt schema**

Every producer writes
`reports/run_receipts/<run_id>/<step>-<producer>.json` with command, inputs and
hashes, outputs and hashes, timestamps, status, and error. Write temporary file
then rename atomically.

- [ ] **Step 2: Share context through environment**

```text
AAC_RUN_ID
AAC_RUN_PROFILE
AAC_PRODUCER_SOURCE_SHA
AAC_RECEIPTS_DIR
```

Add explicit lifecycle:

```powershell
python scripts/manage_run_context.py start --profile thesis-full
python scripts/run_full_pipeline.py ... --resume-run <run-id>
python scripts/manage_run_context.py finalize --run-id <run-id>
```

Every canonical producer invocation requires `--resume-run`. Standalone calls
create development context and are never acceptance-eligible. Full run remains
`incomplete` until required SHAP and dependent producers have successful receipts.

- [ ] **Step 3: Receipt every artifact class**

Cover datasets, schema JSON, models, sidecars, tables, Markdown, and figures.
Formats need not embed run IDs because receipts map path to run and hash.

- [ ] **Step 4: Build manifest only from receipts**

Accept successful receipts from one `thesis-full` run. Re-hash disk outputs.
Unreceipted, cross-run, no-SHAP-profile, or hash-mismatched required files fail.

- [ ] **Step 5: Verify**

Test dirty-tree rejection, failed receipt, cross-run file, hash mismatch, and
no-SHAP profile.

```powershell
python -m pytest tests/test_provenance.py tests/test_artifact_manifest.py -q
```

Done gate: every accepted artifact traces to one clean committed revision and one
successful full-profile run.
Receipts preserve `producer_source_sha`; final manifest separately records
`finalization_sha` after final documentation is committed.

---

### Task 15: Replace dashboard fake predictions with typed results

**Ownership**

- Modify: `src/aac_adoption/dashboard/data.py`
- Modify: `tests/test_dashboard_data.py`

**Required return type**

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

- [ ] **Step 1: Add missing/corrupt/incompatible artifact tests**

All must return `ok=False` and no numeric prediction.

- [ ] **Step 2: Resolve paths from `PROJECT_ROOT`**

Default tables/models paths use `config.PROJECT_ROOT`. Supplied absolute paths
remain absolute. Test from an unrelated current working directory.

- [ ] **Step 3: Require exact metadata feature contract**

Missing feature columns are errors. Do not intersect expected columns with record
columns and continue.

- [ ] **Step 4: Apply declared inverse transform**

Use metadata. Reject:

- nonfinite prediction;
- negative days;
- output beyond declared operational bound.

Operational bound should derive from training target metadata plus documented
margin, not a hidden magic constant.

- [ ] **Step 5: Remove broad exception defaults**

Delete `probability = 0.5` and `days = 15.0`.

- [ ] **Step 6: Verify**

```powershell
python -m pytest tests/test_dashboard_data.py -q
```

Done gate: any inference failure produces no plausible numeric output.

---

### Task 16: Add artifact schema guards and strict boolean parsing

**Ownership**

- Modify: `src/aac_adoption/dashboard/data.py`
- Modify: `streamlit_app.py`
- Add: `tests/test_dashboard_app.py`

- [ ] **Step 1: Create schema registry**

```python
DASHBOARD_TABLE_SCHEMAS = {
    "final_model_selection": {...},
    "subgroup_reliability": {...},
    "context_model_comparison": {...},
    ...
}
```

Loader returns a typed unavailable state with missing columns.

- [ ] **Step 2: Add strict boolean parser**

Accept case-insensitive `true/false/1/0/yes/no`, booleans, and nullable values.
Invalid values return an error, not truthiness.

- [ ] **Step 3: Update every rendering block**

Before column access, check table availability and schema. Render a clear producer
command when unavailable.

- [ ] **Step 4: Fix trust-page target wording**

Main regression predicts days to any matched outcome. Adopted-only timing is a
separate artifact.

- [ ] **Step 5: Verify with AppTest**

Cases:

- current artifacts;
- empty artifacts;
- missing columns;
- string booleans;
- missing model;
- corrupt metadata.

```powershell
python -m pytest tests/test_dashboard_app.py tests/test_dashboard_data.py tests/test_dashboard_story.py -q
```

Done gate: AppTest reports zero uncaught exceptions.

---

### Task 17: Make dashboard cache and prediction state truthful

**Ownership**

- Modify: `streamlit_app.py`
- Modify: dashboard tests

- [ ] **Step 1: Cache by file fingerprint**

Use resolved path, size, and `st_mtime_ns` or content hash as cache inputs.

- [ ] **Step 2: Add refresh control**

Refresh clears data/resource caches and prediction session state.

- [ ] **Step 3: Persist prediction with input hash**

Store result in `st.session_state`. Invalidate when any input changes.

- [ ] **Step 4: Add AppTest lifecycle**

Click predict, rerun, assert result persists. Change input, assert old result is
removed before new prediction.

- [ ] **Step 5: Add dashboard dependency extra**

```toml
dashboard = ["streamlit>=1.35", "altair>=5"]
dev = ["pytest>=7.0", "streamlit>=1.35", "altair>=5"]
```

Done gate: regenerated artifacts appear after refresh and stale prediction never
survives changed inputs.

---

## Phase 5: Acceptance Architecture

### Task 18: Separate unit, integration, acceptance, and slow tests

**Ownership**

- Modify: `pyproject.toml`
- Modify markers in affected tests
- Modify: `scripts/validate_final_acceptance.ps1`

- [ ] **Step 1: Register markers**

```toml
markers = [
  "integration: multi-module tests using temporary artifacts",
  "acceptance: requires canonical generated thesis artifacts",
  "slow: model training, backtesting, or full pipeline checks",
]
addopts = "--strict-markers"
```

- [ ] **Step 2: Remove unconditional passes and artifact skips**

Replace `tests/test_artifact_manifest.py` pass body with assertions against a
fixture manifest that contains one present and one missing required artifact.

Acceptance tests may skip only when not in acceptance mode. In acceptance mode,
missing artifacts fail.

- [ ] **Step 3: Add acceptance switch**

Use environment variable:

```text
AAC_ACCEPTANCE=1
```

Artifact-required tests fail when this variable is set.

- [ ] **Step 4: Split fixture behavior from canonical acceptance**

Before regeneration, acceptance tests run against temporary fixtures via:

```text
AAC_ACCEPTANCE_FIXTURE_ROOT=<temporary path>
```

They prove missing, stale, cross-run, and hash-invalid artifacts fail. Canonical
`AAC_ACCEPTANCE=1` runs only after final documentation and final manifest.

- [ ] **Step 5: Move timing benchmarks out of default suite**

Mark wall-clock tests `slow` and avoid brittle fixed thresholds in correctness
tests.

- [ ] **Step 6: Verify fixture behavior**

```powershell
python -m pytest -m "not slow and not acceptance" -q
$env:AAC_ACCEPTANCE_FIXTURE_ROOT="<temporary fixture root>"
python -m pytest -m acceptance -q
Remove-Item Env:AAC_ACCEPTANCE_FIXTURE_ROOT
```

Done gate: default tests do not falsely pass due to missing outputs; acceptance
failure behavior is proven without requiring canonical artifacts.

---

### Task 19: Make pipeline fail-fast, ordered, and run-identifiable

**Ownership**

- Modify: `scripts/run_full_pipeline.py`
- Add: `tests/test_pipeline_runner.py`

- [ ] **Step 1: Add orchestration unit tests**

Mock `subprocess.run`. Assert:

- first failed producer stops later steps;
- quick mode skips step 18, not 17;
- final manifest is absent from producer pipeline;
- requested `--steps` respects dependencies or clearly declares partial run.

- [ ] **Step 2: Fail immediately**

Default behavior stops on first nonzero exit. Optional `--continue-on-error` may
exist for diagnosis but must mark outputs untrusted and must not be used by
acceptance.

- [ ] **Step 3: Generate run ID**

Pass one run ID to producers via environment:

```text
AAC_RUN_ID=YYYYMMDDTHHMMSSZ-<shortsha>
```

Metadata and manifest record it.

Write `reports/run_receipt.json` at start and finalize it atomically at success:

```json
{
  "run_id": "...",
  "producer_source_sha": "...",
  "started_at": "...",
  "completed_at": "...",
  "status": "ok",
  "executed_steps": [],
  "skipped_steps": [],
  "failed_step": null
}
```

On failure, status is `failed`; finalization is not executed.

- [ ] **Step 4: Remove final manifest from normal producer pipeline**

Pipeline produces artifacts and receipts but not final manifest. Final docs are
written after producer completion. Manifest is a separate finalization command.
Producer order:

```text
... report -> feature importance -> backtesting -> receipt validation
```

- [ ] **Step 5: Fix help text and mojibake**

Quick mode says tests step 18. Console symbols must be valid UTF-8 or ASCII.

- [ ] **Step 6: Verify**

```powershell
python -m pytest tests/test_pipeline_runner.py -q
python scripts/run_full_pipeline.py --help
```

Done gate: failed producer cannot leave later newly generated outputs.

---

### Task 20: Make quick acceptance isolated and canonical acceptance non-duplicative

**Ownership**

- Modify: `scripts/validate_final_acceptance.ps1`
- Modify: CLI scripts to accept output roots as needed
- Add orchestration tests where practical

- [ ] **Step 1: Fix recency CLI duplicate arguments**

Inspect current dirty diff. Keep one `--quick` and one
`--validation-gap-years`.

- [ ] **Step 2: Add temporary smoke root**

Default non-long acceptance creates a temporary output root and passes it to quick
backtesting/recency commands. It must not touch `reports/` or `models/`.

- [ ] **Step 3: Avoid duplicate full pytest**

`-Long` executes full pytest once. Canonical finalization invokes only
`validate_final_acceptance.ps1 -Long`; it does not run a separate preceding full
pytest. If pipeline includes tests during regeneration, call it with `--skip-tests`.

Canonical `-Long` becomes verification-only after final manifest. It must not run
training, reports, backtesting, or manifest generation.

- [ ] **Step 4: Add canonical acceptance environment**

Set:

```powershell
$env:AAC_ACCEPTANCE = "1"
```

for artifact-required tests.

- [ ] **Step 5: Verify CLI smoke**

```powershell
python scripts/calibrate_classifiers.py --help
python scripts/evaluate_backtesting.py --help
python scripts/compare_recency.py --help
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -SkipPytest
```

Done gate: smoke acceptance cannot overwrite canonical thesis artifacts.

---

### Task 21: Strengthen artifact manifest and freshness checks

**Ownership**

- Modify: `scripts/generate_artifact_manifest.py`
- Modify: `tests/test_artifact_manifest.py`

- [ ] **Step 1: Remove duplicated schema entries**

Define required columns once.

- [ ] **Step 2: Add manifest invariants**

Assert:

- unique artifact paths;
- every required artifact exists and is nonempty in acceptance mode;
- source script exists;
- run ID matches final pipeline run;
- artifact hash matches disk;
- manifest generation time is newer than required artifacts;
- final documentation hashes are included;
- every required generated artifact has one successful receipt from the same
  `thesis-full` run;
- no path claims deleted docs are present.

- [ ] **Step 3: Add expected target wording checks**

Artifact notes must distinguish LOS from adopted-only timing.

- [ ] **Step 4: Verify**

```powershell
python -m pytest tests/test_artifact_manifest.py -q
```

Done gate: missing/stale required artifacts make acceptance fail.

---

## Phase 6: Final Documentation and Regeneration

### Task 22: Regenerate producers in dependency order

**Ownership**

- No hand edits to generated artifacts
- User runs long commands from `manual-regeneration-runbook.md`

- [ ] Code and fast/medium tests all pass.
- [ ] Source tree is clean and committed.
- [ ] User runs canonical pipeline regeneration.
- [ ] User provides exit code and log path.
- [ ] Agent reviews log for skipped/failed steps.
- [ ] User runs SHAP and feature-family producers when those artifacts remain
  required for thesis.
- [ ] Agent checks producer receipts, run profile, hashes, and schemas.

Done gate: all required generated artifacts share one successful `thesis-full`
run ID. Final manifest does not exist yet.
No required SHAP artifact may be carried forward from an older run.

---

### Task 23: Reconcile final-facing documentation from accepted artifacts

**Ownership**

- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/METHODOLOGY.md`
- Modify: `docs/RESULTS.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/target_definitions.md`
- Modify: dashboard copy
- Add: `scripts/check_text_encoding.py`
- Add: `tests/test_text_encoding.py`

- [ ] **Step 1: Update methodology**

Document:

- episode boundary rule;
- unresolved intake audit exclusion;
- prior-day weather;
- raw-intake calendar volume;
- strict train/calibration/selection/test chronology;
- 2023-only selection;
- descriptive-only survival scope.

- [ ] **Step 2: Rewrite results from accepted artifacts**

Use exact accepted run ID/date, dataset counts, 2023 selection criteria, and
frozen test metrics. Never quote stale test counts.

- [ ] **Step 3: Collapse roadmap contradictions**

Keep unresolved future work only. Do not preserve duplicate DONE/PARTIAL/TODO
entries.

- [ ] **Step 4: Shrink README**

Remove repeated reproduction, EDA, dataset, and critique sections. Link canonical
docs and exact commands.

- [ ] **Step 5: Add deterministic encoding scanner**

`check_text_encoding.py` recursively reads supplied UTF-8 text files/directories,
skips binary/generated cache directories, and exits nonzero for decode errors or
known mojibake markers. Test clean UTF-8, invalid UTF-8 bytes, and representative
mojibake.

- [ ] **Step 6: Scan terminology and mojibake**

```powershell
rg -n "adoption speed|time to adoption|caus|test PR-AUC|test MAE|TODO|PARTIAL|stub" README.md docs reports/summary streamlit_app.py src/aac_adoption/dashboard
python scripts/check_text_encoding.py README.md docs reports/summary streamlit_app.py src/aac_adoption/dashboard
```

Review each hit; do not blindly replace legitimate adopted-only wording.

- [ ] **Step 7: Verify documentation contracts**

Run target-definition, report, artifact, dashboard story, and acceptance schema
tests plus `tests/test_text_encoding.py`.

Done gate: all final-facing surfaces tell the same methodological story.

---

### Task 24: Finalize manifest, verify acceptance, and perform harsh review

- [ ] **Step 1: Freeze and commit documentation**

Confirm clean tree. Any code/doc/generated-artifact change after final manifest
requires manifest regeneration. This commit becomes `finalization_sha`. Producer
receipts retain the earlier `producer_source_sha`; both are required.

- [ ] **Step 2: Generate final manifest exactly once**

```powershell
python scripts/generate_artifact_manifest.py --run-id <accepted-run-id>
```

- [ ] **Step 3: Run canonical verification-only acceptance**

```powershell
$env:AAC_ACCEPTANCE="1"
powershell -ExecutionPolicy Bypass -File scripts/validate_final_acceptance.ps1 -Long
Remove-Item Env:AAC_ACCEPTANCE
```

These commands must not mutate manifest-tracked files.

- [ ] **Step 4: Run dashboard smoke**

Run AppTest and manual Streamlit smoke. Cache/runtime files are not
manifest-tracked outputs.

- [ ] **Step 5: Dispatch fresh reviewer**

**Reviewer scope**

Use a fresh high-capability reviewer. Review code and accepted artifacts against:

1. target correctness;
2. temporal leakage;
3. selection leakage;
4. artifact lineage;
5. dashboard truthful failure;
6. final documentation consistency;
7. reproducibility.

- [ ] No supervised unresolved rows.
- [ ] No cross-intake outcome matches.
- [ ] No target/identifier/raw timestamp predictors.
- [ ] Context independent of outcome availability.
- [ ] No random fallback called thesis evaluation.
- [ ] No test-selected winner.
- [ ] No invalid cluster bootstrap.
- [ ] No fake dashboard prediction.
- [ ] No guessed transform or feature schema.
- [ ] No quick run overwrites canonical artifacts.
- [ ] No required artifact test can pass with missing files.
- [ ] No survival-model claim beyond descriptive scope.

Done gate: verification-only acceptance passes, manifest still matches disk, and
reviewer returns no P0/P1 findings. P2 findings are fixed or documented as
explicit non-blocking limitations.

---

## Final Definition of Done

- Fast and medium validation gates pass.
- User-run full pytest passes with no unexpected skips.
- User-run canonical pipeline completes from local raw data.
- All required artifacts come from one accepted run.
- Manifest matches disk and is newer than all required outputs.
- Final model choice is selected on untouched 2023 data.
- Test metrics are frozen final evaluation only.
- Dashboard AppTest has zero uncaught exceptions.
- Dashboard never emits invented predictions.
- README, methodology, results, roadmap, dashboard, and generated summaries agree.
- No final-facing stale TODO/PARTIAL/stub or mojibake remains.
