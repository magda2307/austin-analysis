# Slice 12: Survival Analysis Implementation - Summary & Deployment Guide

**Date**: 2026-06-06  
**Status**: PARTIAL → TODO  
**Target**: Complete implementation for thesis submission

---

## What is Slice 12?

Slice 12 transforms the descriptive survival analysis into a thesis-grade censored survival modeling system. It addresses the key gaps identified in the roadmap:

| Gap | Current State | Target State |
|-----|---------------|--------------|
| Censoring | Not implemented (episodes silently dropped) | Native censoring columns in dataset |
| Event types | Adopted-only KM curves | Full event taxonomy (adoption, transfer, euthanasia, return, censored) |
| Unresolved Episodes | Dropped silently | Preserved as censored with reason codes |
| Competing Risks | Not addressed | Event-stratified analysis and CIF curves |
| Training Integration | None (descriptive only) | End-to-end training pipeline |

---

## Five-Step Deployment Plan

### Overview

```
Phase 1: Data Pipeline (Agent 1) ──┐
Phase 2: Survival Models (Agent 2) ├─> Sequential dependencies
Phase 3: Integration (Agent 3) ────┘
Phase 4: Testing (Agent 4) ────────┐
Phase 5: Reporting (Agent 5) ──────┘
```

### Agent 1: Data Pipeline Engineer
**Focus**: Tasks 1.1, 1.2  
**Start Time**: Day 1  
**Duration**: 2-3 days  

**Tasks**:
1. Add `is_censored`, `event_type`, `censoring_reason`, `followup_days_censored` columns to `build_dataset.py`
2. Modify `match_records.py` to preserve unmatched intakes as censored records
3. Ensure all intakes stay in dataset (none dropped)

**Files to Modify**:
- `src/aac_adoption/data/build_dataset.py` (lines ~98-113)
- `src/aac_adoption/data/match_records.py` (lines ~112-115)

**Validation**:
```bash
python -m pytest tests/test_build_dataset.py::test_censoring_columns -v
python -m pytest tests/test_build_dataset.py::test_unresolved_not_dropped -v
```

**Deliverable**: Dataset with censoring columns

---

### Agent 2: Survival Modeling Expert
**Focus**: Tasks 1.3, 2.1, 2.2  
**Start Time**: Day 3 (after Agent 1)  
**Duration**: 3-4 days  

**Tasks**:
1. Enhance `compute_kaplan_meier_survival()` to use `event_observed` parameter
2. Fix `fit_cox_proportional_hazards()` to handle censored data (no silent dropna)
3. Add `fit_aft_model()` for Weibull and exponential distributions
4. Add proportional hazards assumption testing

**Files to Modify**:
- `src/aac_adoption/analysis/survival_analysis.py` (complete rewrite of key functions)

**Validation**:
```bash
python -m pytest tests/test_survival_analysis.py::test_kaplan_meier_with_censored_data -v
python -m pytest tests/test_survival_analysis.py::test_cox_with_censored_data -v
python -m pytest tests/test_survival_analysis.py::test_aft_model_weibull -v
```

**Deliverable**: Working Cox and AFT models with censoring support

---

### Agent 3: Pipeline Integration Specialist
**Focus**: Tasks 3.1, 3.2, 3.3  
**Start Time**: Day 7 (after Agents 1-2)  
**Duration**: 2-3 days  

**Tasks**:
1. Create `scripts/train_survival.py` (end-to-end training pipeline)
2. Add survival metrics to `src/aac_adoption/models/evaluate.py`
3. Create `src/aac_adoption/features/survival_targets.py` for horizon targets

**Files to Create**:
- `scripts/train_survival.py` (new)
- `src/aac_adoption/features/survival_targets.py` (new)

**Files to Modify**:
- `src/aac_adoption/models/evaluate.py` (add metrics)

**Validation**:
```bash
python scripts/train_survival.py --animal-subset combined
python -m pytest tests/test_survival_integration.py -v
```

**Deliverable**: Working training pipeline with artifacts in `models/survival/` and `reports/survival/`

---

### Agent 4: Testing Specialist
**Focus**: Tasks 4.1, 4.2, 4.3  
**Start Time**: Day 9 (parallel with Agent 3)  
**Duration**: 1-2 days  

**Tasks**:
1. Extend `tests/test_survival_analysis.py` with censoring tests
2. Create `tests/test_survival_integration.py` for pipeline validation
3. Create `tests/test_acceptance_survival.py` for artifact contracts

**Files to Create/Modify**:
- `tests/test_survival_analysis.py` (extend)
- `tests/test_survival_integration.py` (new)
- `tests/test_acceptance_survival.py` (new)

**Validation**:
```bash
python -m pytest tests/test_survival*.py -v --cov=... --cov-report=term-missing
```

**Deliverable**: >80% coverage, all tests passing

---

### Agent 5: Reporting & Documentation Specialist
**Focus**: Tasks 5.1, 5.2, 5.3  
**Start Time**: Day 10 (after Agents 1-3)  
**Duration**: 1-2 days  

**Tasks**:
1. Create `src/aac_adoption/diagnostics/survival_diagnostics.py`
2. Update `reports/summary/survival_descriptive_note.md` with final methodology
3. Integrate survival section into `src/aac_adoption/reporting/evidence_pack.py`

**Files to Create**:
- `src/aac_adoption/diagnostics/survival_diagnostics.py` (new)

**Files to Modify**:
- `reports/summary/survival_descriptive_note.md` (rewrite)
- `src/aac_adoption/reporting/evidence_pack.py` (add survival section)

**Validation**:
```bash
python -c "from aac_adoption.diagnostics.survival_diagnostics import generate_survival_diagnostics_report; generate_survival_diagnostics_report(df, Path('reports/diagnostics'))"
cat reports/survival/survival_metrics.csv
```

**Deliverable**: Complete reporting package with diagnostics and documentation

---

## Dependency Map

```
Day 1-3:  Agent 1 (Data) ──────────────────────────┐
         ↓                                         │
Day 3-6:  Agent 2 (Models) ────────────────────────┤
         ↓                                         │
Day 7-9:  Agent 3 (Integration) ───────────────────┤
         ↓                                         │
         ├── Agent 4 (Testing) ←───────────────────┤
         │                                         │
         └── Agent 5 (Reporting) ←─────────────────┘
```

**Sequential Flow**: Agent 1 → Agent 2 → Agent 3 (must wait)  
**Parallel Flow**: Agent 4 and Agent 5 can work in parallel with Agent 3

---

## Acceptance Criteria Checklist

### Must-Have for Thesis Submission
- [ ] Censoring columns in dataset (`is_censored`, `event_type`, `censoring_reason`)
- [ ] Unresolved episodes not dropped
- [ ] Cox model handles censored data properly
- [ ] AFT model implemented
- [ ] `scripts/train_survival.py` runs end-to-end
- [ ] Survival metrics in `reports/survival/survival_metrics.csv`
- [ ] Survival diagnostics in `reports/diagnostics/survival_diagnostics.csv`
- [ ] `survival_descriptive_note.md` updated with methodology
- [ ] Evidence pack includes survival section
- [ ] Tests pass with >80% coverage

### Nice-to-Have for Publication
- [ ] Competing-risk framework (Fine-Gray or multi-state)
- [ ] Subgroup survival curves with confidence intervals
- [ ] Horizon-based survival models (7/30/60/90 days)

---

## Quick Start Commands

### Run All Agents Sequentially
```bash
# Agent 1: Data pipeline
cd src/aac_adoption/data
# Modify build_dataset.py and match_records.py (see Agent 1 tasks)

# Agent 2: Survival models
cd src/aac_adoption/analysis
# Modify survival_analysis.py (see Agent 2 tasks)

# Agent 3: Integration
# Create scripts/train_survival.py (see Agent 3 tasks)
python scripts/train_survival.py --animal-subset combined

# Agent 4: Testing
python -m pytest tests/test_survival*.py -v

# Agent 5: Reporting
python -c "from aac_adoption.diagnostics.survival_diagnostics import generate_survival_diagnostics_report; generate_survival_diagnostics_report(df, Path('reports/diagnostics'))"
```

### Individual Agent Validation
```bash
# Agent 1
python -m pytest tests/test_build_dataset.py::test_censoring_columns -v

# Agent 2
python -m pytest tests/test_survival_analysis.py::test_cox_with_censored_data -v

# Agent 3
python scripts/train_survival.py --animal-subset combined && ls models/survival/

# Agent 4
python -m pytest tests/test_survival*.py -v --cov=... --cov-report=term-missing

# Agent 5
python -c "import json; pack=json.load(open('reports/evidence_pack.json')); assert 'survival_analysis' in pack"
```

---

## Estimated Timeline

| Agent | Duration | Key Milestone |
|-------|----------|---------------|
| Agent 1 | 2-3 days | Censoring columns in dataset |
| Agent 2 | 3-4 days | Working Cox/AFT models |
| Agent 3 | 2-3 days | Training pipeline working |
| Agent 4 | 1-2 days | All tests passing |
| Agent 5 | 1-2 days | Documentation complete |
| **Total** | **9-14 days** | Slice 12 complete |

---

## Critical Dependencies Summary

1. **Agent 2** must wait for **Agent 1** to provide dataset with censoring columns
2. **Agent 3** must wait for **Agents 1-2** for both data and models
3. **Agent 4** and **Agent 5** can work in parallel once **Agent 3** has artifacts

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Agent 1 changes break other modules | Test `build_dataset.py` before and after changes |
| Cox model assumes PH assumption fails | Provide AFT fallback, document limitation |
| Missing features cause errors | Handle missing explicitly in feature engineering |
| Tests fail due to missing data | Mock datasets in tests, document dependencies |

---

## Handoff Points

**Between Agents**:
- Agent 1 → Agent 2: Dataset with `is_censored`, `event_type` columns
- Agent 2 → Agent 3: Working Cox and AFT model functions
- Agent 3 → Agent 5: Model artifacts in `models/survival/`

**Final Deliverables**:
- Modified source files
- New scripts and features
- Diagnostic output files
- Documentation updates
- Test coverage >80%

---

## Next Steps

1. **Review this plan** with your team
2. **Assign agents** (1-5) or work sequentially
3. **Set up daily checkpoints** for progress tracking
4. **Run validation commands** after each agent completion
5. **Generate final report** when all agents complete

---

## Questions & Support

**For Agent-specific questions**:  
- Agent 1: `docs/internal/slice_12_agent_tasks_agent1_data.md`  
- Agent 2: `docs/internal/slice_12_agent_tasks_agent2_survival.md`  
- Agent 3: `docs/internal/slice_12_agent_tasks_agent3_pipeline.md`  
- Agent 4: `docs/internal/slice_12_agent_tasks_agent4_testing.md`  
- Agent 5: `docs/internal/slice_12_agent_tasks_agent5_reporting.md`  

**General questions**:  
- `docs/internal/slice_12_implementation_plan.md` (full roadmap)

---

*End of Slice 12 Deployment Guide*
