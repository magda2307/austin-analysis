# Slice 12 Implementation Plan: Survival Analysis Section

**Status**: PARTIAL → TODO  
**Last Updated**: 2026-06-06  
**Primary Goal**: Transform descriptive survival analysis into thesis-grade censored survival modeling

---

## Current Status Analysis

### From ROADMAP.md

**Current State**: PARTIAL  
**Required State**: Full implementation for thesis submission

### What Exists
- Descriptive Kaplan-Meier curves for adopted-only subset (`reports/tables/adoption_survival_curves.csv`)
- Survival utilities in `src/aac_adoption/analysis/survival_analysis.py`
- Basic tests in `tests/test_survival_analysis.py`
- Descriptive note in `reports/summary/survival_descriptive_note.md`

### What's Missing
- Native censoring support (censoring only added post-hoc)
- Handling of unresolved episodes (currently silently excluded)
- Competing-risk framework (adoption vs transfer vs euthanasia vs return-to-owner)
- Integration with main training pipeline
- Horizon-based survival analysis
- Proper categorical encoding for survival models

### Key Gap: Censoring Handling
The dataset currently excludes animals without outcomes. For proper survival analysis, unresolved episodes must be:
1. Marked as censored (not discarded)
2. Include `is_censored` flag
3. Record `censoring_date = extract_end_date` for right-censoring
4. Document reasons: `no_outcome`, `end_of_extract`, `ambiguous_match`

---

## Implementation Roadmap

### Phase 1: Data Preparation and Censoring Integration (Deps: None)

#### Task 1.1: Add native censored status to build_dataset.py
**File**: `src/aac_adoption/data/build_dataset.py`  
**Complexity**: Medium  
**Dependencies**: None

**Action Items**:
- Add `is_censored`, `censoring_reason`, and `event_type` columns to dataset
- Define event types: `adoption`, `transfer`, `euthanasia`, `return_to_owner`, `censored`
- Add `followup_days_censored` for time-to-event analysis (min of observed time or censoring time)
- Track unresolved episodes explicitly in the dataset

**Acceptance Criteria**:
- CSV output includes `is_censored`, `censoring_reason`, `event_type` columns
- Tests verify censoring logic: unresolved episodes marked as censored
- Dataset size unchanged (no episodes dropped)

---

#### Task 1.2: Extend match_records.py with censoring metadata
**File**: `src/aac_adoption/data/match_records.py`  
**Complexity**: Low  
**Dependencies**: None

**Action Items**:
- Add censoring flags during matching phase
- Mark intake as censored if no future outcome exists (`outcome_index >= len(outcome_records)`)
- Record `censoring_date = extract_end_date` for right-censoring
- Document reasons: `no_outcome`, `end_of_extract`, `ambiguous_match`

**Acceptance Criteria**:
- Matching reports include censoring counts per reason
- Output DataFrame includes `is_censored`, `censoring_reason` columns
- Unmatched intakes tracked but not silently dropped

---

#### Task 1.3: Update survival_analysis.py with native censoring support
**File**: `src/aac_adoption/analysis/survival_analysis.py`  
**Complexity**: High  
**Dependencies**: Tasks 1.1, 1.2

**Action Items**:
- Revise `compute_kaplan_meier_survival()` to use `event_observed` parameter properly
- Add `fit_cox_with_censoring()` that handles censored observations natively
- Implement competing-risk survival curves (Fine-Gray or multi-state)
- Add diagnostic functions for proportional hazards assumption

**Acceptance Criteria**:
- All Kaplan-Meier functions document censored vs observed events
- Cox model accepts censored data without errors
- Functions return event counts and censoring percentages

---

### Phase 2: Survival Model Implementations (Deps: Phase 1)

#### Task 2.1: Cox Proportional Hazards with Proper Encoding
**File**: `src/aac_adoption/analysis/survival_analysis.py`  
**Complexity**: High  
**Dependencies**: Tasks 1.1, 1.2, 1.3

**Action Items**:
- One-hot encode categorical features before Cox fitting
- Handle missing values explicitly (not silent dropna)
- Validate proportional hazards assumption (Schoenfeld residuals)
- Add `cox_diagnostic_report()` function with PH test results
- Store both full and simplified Cox models for comparison

**Acceptance Criteria**:
- Cox model trains without errors on censored data
- Diagnostic report generated with PH assumption tests
- Model artifacts saved with metadata

---

#### Task 2.2: Accelerated Failure Time (AFT) Model
**File**: `src/aac_adoption/analysis/survival_analysis.py`  
**Complexity**: Medium  
**Dependencies**: Task 2.1

**Action Items**:
- Fit AFT model (Weibull, exponential) as alternative to Cox
- Compare AFT vs Cox using AIC/BIC
- Extract predicted survival functions
- Store both models in `models/survival/` artifacts

**Acceptance Criteria**:
- AFT model artifacts saved to `models/survival/`
- AIC/BIC comparison report generated
- Both models produce consistent conclusions

---

#### Task 2.3: Competing-Risk Framework
**File**: `src/aac_adoption/analysis/survival_analysis.py`  
**Complexity**: High  
**Dependencies**: Tasks 2.1, 2.2

**Action Items**:
- Fine-Gray model for subdistribution hazards (competing risks)
- Or multi-state model with transition intensities
- Event-specific hazard ratios
- Cumulative incidence functions (CIF) for each event type

**Acceptance Criteria**:
- CIF curves generated for each event type (adoption, transfer, euthanasia, return)
- Competing-risk model artifacts saved
- Event-specific hazard ratios documented

---

### Phase 3: Integration with Training Pipeline (Deps: Phases 1 & 2)

#### Task 3.1: Create survival model training script
**File**: `scripts/train_survival.py` **(NEW)**  
**Complexity**: Medium  
**Dependencies**: Tasks 1.1, 1.2, 1.3, 2.1, 2.2

**Action Items**:
- Load dataset with censoring columns from Task 1
- Split by time: 2013-2021 train, 2022-2023 val, 2024-2025 test
- Train Cox + AFT on train set
- Evaluate concordance index, Brier score for survival on validation
- Save model artifacts and evaluation metrics to `reports/survival/`

**Pipeline Structure**:
```python
# scripts/train_survival.py
def main():
    df = load_dataset()
    df = add_censoring_columns(df)
    split = make_time_split(df, "survival_time", animal_subset="combined")
    
    # Train survival models
    cox_model = fit_cox_with_censoring(split.train, ...)
    aft_model = fit_aft_model(split.train, ...)
    
    # Evaluate on validation
    metrics = survival_metrics(split.validation, cox_model, aft_model)
    
    # Save artifacts
    save_model_artifact(cox_model, "cox_censored")
    save_model_artifact(aft_model, "aft_weibull")
    metrics.to_csv("reports/survival/survival_metrics.csv")
```

**Acceptance Criteria**:
- Script runs end-to-end without errors
- All artifacts saved to `models/survival/` and `reports/survival/`
- Metrics include concordance index, Brier score, AIC/BIC

---

#### Task 3.2: Integrate survival metrics into evaluation
**File**: `src/aac_adoption/models/evaluate.py`  
**Complexity**: Medium  
**Dependencies**: Task 3.1

**Action Items**:
- Add `survival_concordance_index()` function
- Add `survival_time_dependent_roc_auc()` function
- Add `survival_integrated_brier_score()` function
- Update `classification_metrics()` to handle survival-specific metrics

**Acceptance Criteria**:
- `survival_metrics()` function exists with all metrics
- Metrics appear in `survival_classification_metrics.csv`
- Tests verify metric calculations

---

#### Task 3.3: Horizon-based survival targets
**File**: `src/aac_adoption/features/target_encoder.py` or **NEW FILE**  
**Complexity**: Low  
**Dependencies**: Task 1.1

**Action Items**:
- Add survival targets for 7/30/60/90 day horizons
- Binary event: adopted within horizon (1) or censored (0)
- Time-to-event: min(days_to_outcome, horizon) with event indicator
- Similar to existing `adopted_in_7d/30d/60d/90d` but survival-ready

**Output Schema**:
| Column | Description |
|--------|-------------|
| `survival_7d_time` | min(days_to_outcome, 7) or censoring time |
| `survival_7d_event` | 1 if adopted within 7d, 0 if censored |
| `survival_30d_time` | min(days_to_outcome, 30) or censoring time |
| `survival_30d_event` | 1 if adopted within 30d, 0 if censored |
| `survival_60d_time` | min(days_to_outcome, 60) or censoring time |
| `survival_60d_event` | 1 if adopted within 60d, 0 if censored |
| `survival_90d_time` | min(days_to_outcome, 90) or censoring time |
| `survival_90d_event` | 1 if adopted within 90d, 0 if censored |

**Acceptance Criteria**:
- Horizon survival targets in dataset
- Tests verify horizon logic

---

### Phase 4: Validation and Testing (Deps: All phases)

#### Task 4.1: Unit tests for survival functions with censoring
**File**: `tests/test_survival_analysis.py`  
**Complexity**: Low  
**Dependencies**: Task 2.1

**Action Items**:
- Test censored data handling (Kaplan-Meier, Cox)
- Test competing-risk curves (if implemented)
- Test Cox model with missing values
- Test concordance index with censored data
- Increase coverage to >80%

**Acceptance Criteria**:
- All tests pass
- Coverage >80% for survival_analysis.py
- Tests cover edge cases (empty data, all censored, no events)

---

#### Task 4.2: Integration test for full survival pipeline
**File**: `tests/test_survival_integration.py` **(NEW)**  
**Complexity**: Medium  
**Dependencies**: Task 3.1

**Action Items**:
- Load dataset, add censoring, split, train survival models
- Verify artifacts created
- Verify metrics computed
- Test reproducibility (same seed = same results)

**Test Cases**:
- Full pipeline execution (end-to-end)
- Artifact existence checks
- Metric computation validation
- Reproducibility verification

**Acceptance Criteria**:
- Integration test passes
- Reproducibility test confirms deterministic results

---

#### Task 4.3: Acceptance tests for survival outputs
**File**: `tests/test_acceptance_survival.py` **(NEW)**  
**Complexity**: Low  
**Dependencies**: Task 5.1

**Action Items**:
- Validate CSV schema (censoring columns, event types)
- Validate model metadata (censored count, event counts)
- Validate diagnostic reports include PH assumption test
- Validate report formats match requirements

**Acceptance Criteria**:
- All schema contracts pass
- Artifacts match documentation

---

### Phase 5: Reporting and Documentation (Deps: Phases 1-4)

#### Task 5.1: Generate survival diagnostics report
**File**: `src/aac_adoption/diagnostics/survival_diagnostics.py` **(NEW)**  
**Complexity**: Medium  
**Dependencies**: Tasks 2.1, 3.1

**Action Items**:
- Event/censoring counts by animal type, age group, breed group
- Kaplan-Meier curves stratified by subgroup
- Cox coefficient plots
- Concordance index by subgroup
- PH assumption diagnostics (Schoenfeld tests)
- Competing-risk cumulative incidence functions (if implemented)

**Output Files**:
- `reports/diagnostics/survival_diagnostics.csv`
- `reports/diagnostics/survival_subgroup_curves.csv`
- `reports/diagnostics/survival_ph_assumption.csv`
- Survival curve PNGs in `reports/diagnostics/survival_curves/`

**Acceptance Criteria**:
- Diagnostics report generated
- Subgroup analysis available
- PH assumption tests documented

---

#### Task 5.2: Update survival descriptive note
**File**: `reports/summary/survival_descriptive_note.md`  
**Complexity**: Low  
**Dependencies**: Tasks 5.1

**Action Items**:
- Document censoring handling methodology
- Explain competing-risk framework
- State limitations (observational data, no unmeasured confounders)
- Update thesis defense statement to reflect actual implementation

**Key Sections**:
1. **What These Curves Are** (updated for censoring)
2. **What These Curves Are NOT** (clarify model scope)
3. **Methodology** (censoring, competing risks, model choices)
4. **Limitations** (observational, confounding)
5. **Thesis Defense Statement** (updated)

**Acceptance Criteria**:
- Note updated and committed
- Reflects implemented methodology

---

#### Task 5.3: Add survival section to evidence pack
**File**: `src/aac_adoption/reporting/evidence_pack.py`  
**Complexity**: Low  
**Dependencies**: Tasks 3.1, 3.3

**Action Items**:
- Survival model selection table (Cox vs AFT)
- Event counts and censoring percentages
- Subgroup survival curves
- Model comparison metrics (AIC, BIC, concordance)
- Horizon survival performance

**Acceptance Criteria**:
- Evidence pack includes survival section
- All required metrics present

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Data Preparation                     │
├─────────────────────────────────────────────────────────────────┤
│ Task 1.1: build_dataset.py (native censoring)                   │
│   └─ No dependencies                                            │
│                                                                 │
│ Task 1.2: match_records.py (censoring metadata)                 │
│   └─ No dependencies                                            │
│                                                                 │
│ Task 1.3: survival_analysis.py (censoring support)              │
│   └─ Depends on: 1.1, 1.2                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Survival Models                      │
├─────────────────────────────────────────────────────────────────┤
│ Task 2.1: Cox with encoding                                     │
│   └─ Depends on: 1.1, 1.2, 1.3                                  │
│                                                                 │
│ Task 2.2: AFT model                                             │
│   └─ Depends on: 2.1                                            │
│                                                                 │
│ Task 2.3: Competing-risk framework                              │
│   └─ Depends on: 2.1, 2.2                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 3: Pipeline Integration                   │
├─────────────────────────────────────────────────────────────────┤
│ Task 3.1: train_survival.py (NEW)                               │
│   └─ Depends on: 1.1, 1.2, 1.3, 2.1, 2.2                       │
│                                                                 │
│ Task 3.2: evaluate.py extensions                                │
│   └─ Depends on: 3.1                                            │
│                                                                 │
│ Task 3.3: Horizon survival targets                              │
│   └─ Depends on: 1.1                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 4: Testing                            │
├─────────────────────────────────────────────────────────────────┤
│ Task 4.1: test_survival_analysis.py (extend)                   │
│   └─ Depends on: 2.1                                            │
│                                                                 │
│ Task 4.2: test_survival_integration.py (NEW)                   │
│   └─ Depends on: 3.1                                            │
│                                                                 │
│ Task 4.3: test_acceptance_survival.py (NEW)                    │
│   └─ Depends on: 5.1 (diagnostics schema)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 5: Reporting & Docs                      │
├─────────────────────────────────────────────────────────────────┤
│ Task 5.1: survival_diagnostics.py (NEW)                         │
│   └─ Depends on: 2.1, 3.1                                       │
│                                                                 │
│ Task 5.2: Update descriptive note                               │
│   └─ Depends on: 5.1                                            │
│                                                                 │
│ Task 5.3: Evidence pack updates                                 │
│   └─ Depends on: 3.1, 3.3                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sub-Agent Distribution Plan

### Agent 1: Data Pipeline Engineer
**Focus**: Tasks 1.1, 1.2  
**Goal**: Native censoring in dataset  
**Files to Modify**:
- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/data/match_records.py`

**Expected Output**:
- Dataset with columns: `is_censored`, `censoring_reason`, `event_type`
- Unresolved episodes not dropped, clearly marked

**Validation**:
```bash
python -m pytest tests/test_build_dataset.py::test_censoring_columns -v
python -m pytest tests/test_build_dataset.py::test_unresolved_not_dropped -v
```

---

### Agent 2: Survival Modeling Expert
**Focus**: Tasks 1.3, 2.1, 2.2  
**Goal**: Working Cox and AFT models with censoring  
**Files to Modify**:
- `src/aac_adoption/analysis/survival_analysis.py`

**Expected Output**:
- `fit_cox_with_censoring()` function
- `fit_aft_model()` function  
- Survival model artifacts in `models/survival/`

**Validation**:
```bash
python -m pytest tests/test_survival_analysis.py -v --cov=src/aac_adoption/analysis/survival_analysis.py
```

---

### Agent 3: Pipeline Integration Specialist
**Focus**: Task 3.1, 3.2, 3.3  
**Goal**: End-to-end survival training pipeline  
**Files to Create/Modify**:
- `scripts/train_survival.py` **(NEW)**
- `src/aac_adoption/models/evaluate.py`
- **NEW**: `src/aac_adoption/features/survival_targets.py`

**Expected Output**:
- `scripts/train_survival.py` runs end-to-end
- Metrics saved to `reports/survival/survival_metrics.csv`
- Model artifacts in `models/survival/`

**Validation**:
```bash
python scripts/train_survival.py
python -m pytest tests/test_survival_integration.py -v
```

---

### Agent 4: Testing Specialist
**Focus**: Tasks 4.1, 4.2, 4.3  
**Goal**: >80% coverage, all acceptance criteria  
**Files to Modify/Create**:
- `tests/test_survival_analysis.py` (extend)
- `tests/test_survival_integration.py` **(NEW)**
- `tests/test_acceptance_survival.py` **(NEW)**

**Expected Output**:
- All tests pass with >80% coverage
- Integration test validates full pipeline
- Acceptance tests validate artifacts

**Validation**:
```bash
python -m pytest tests/test_survival*.py -v --cov=...
```

---

### Agent 5: Reporting & Documentation Specialist
**Focus**: Tasks 5.1, 5.2, 5.3  
**Goal**: Complete reporting package  
**Files to Create/Modify**:
- `src/aac_adoption/diagnostics/survival_diagnostics.py` **(NEW)**
- `reports/summary/survival_descriptive_note.md`
- `src/aac_adoption/reporting/evidence_pack.py`

**Expected Output**:
- `reports/diagnostics/survival_diagnostics.csv`
- `reports/diagnostics/survival_subgroup_curves.csv`
- Updated `survival_descriptive_note.md`
- Evidence pack includes survival section

**Validation**:
```bash
python -c "from aac_adoption.diagnostics.survival_diagnostics import generate_report; generate_report()"
```

---

## Critical Dependencies Summary

### Sequential Dependencies (Must Wait)
1. **Task 1.3** waits for Tasks 1.1, 1.2 to provide dataset with censoring columns
2. **Task 2.1** waits for Task 1.3 for censoring support
3. **Task 2.2** waits for Task 2.1 to establish Cox baseline
4. **Task 2.3** waits for Tasks 2.1, 2.2 for model comparison
5. **Task 3.1** waits for Tasks 1.1-1.3 (data) and 2.1-2.2 (models)
6. **Task 3.2** waits for Task 3.1 pipeline existence
7. **Task 4.3** waits for Task 5.1 diagnostic schema to be stable
8. **Task 5.2** waits for Task 5.1 diagnostics to document

### Parallelizable (Can Work Together)
- Tasks 1.1, 1.2 (both data preparation)
- Tasks 4.1, 4.2 (both testing)
- Tasks 5.1, 5.3 (both reporting, though 5.1 should complete first)

---

## Acceptance Criteria for Slice 12

### Must-Have (Thesis Minimum)
1. ✅ Censoring is native: Dataset includes `is_censored`, `censoring_reason`, `event_type`
2. ✅ Survival models train: Cox and AFT models train without data errors
3. ✅ Unresolved episodes not dropped: Censored observations documented and included
4. ✅ Integration complete: `scripts/train_survival.py` runs end-to-end
5. ✅ Reports complete: Diagnostics and descriptive note updated
6. ❓ Competing risks: At minimum, event-type stratified analysis exists

### Nice-to-Have (Publication Quality)
7. ✓ Full competing-risk framework (Fine-Gray or multi-state)
8. ✓ Subgroup survival curves with confidence intervals
9. ✓ Horizon-based survival models for 7/30/60/90 days

---

## Estimated Effort

| Phase | Tasks | Total Complexity | Estimated Time |
|-------|-------|------------------|----------------|
| Phase 1 | 3 | Medium-High | 2-3 days |
| Phase 2 | 3 | High | 3-4 days |
| Phase 3 | 3 | Medium | 2-3 days |
| Phase 4 | 3 | Low-Medium | 1-2 days |
| Phase 5 | 3 | Low-Medium | 1-2 days |
| **Total** | **15** | **High** | **9-14 days** |

---

## Recommendation

**Recommended Path for Thesis Submission**: Execute all phases sequentially, prioritizing:
1. Phase 1 (data) - foundation for everything
2. Phase 2 (models) - core survival analysis
3. Phase 3 (integration) - reproducibility
4. Phase 5 (reporting) - documentation

**Testing (Phase 4)** should run in parallel with reporting to catch issues early.

**Sub-agent approach** allows parallel execution with clear handoff points between phases.

---

*End of Slice 12 Implementation Plan*
