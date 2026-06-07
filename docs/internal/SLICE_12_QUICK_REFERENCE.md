# Slice 12 Agent Quick Reference Card

---

## Agent 1: Data Pipeline Engineer

**Tasks**: 1.1, 1.2  
**Start**: Day 1  
**Duration**: 2-3 days  

**Key Files**:
- `src/aac_adoption/data/build_dataset.py` (add censoring columns)
- `src/aac_adoption/data/match_records.py` (preserve unmatched intakes)

**Key Output**: Dataset with `is_censored`, `event_type`, `censoring_reason` columns

**Validation**:
```bash
python -m pytest tests/test_build_dataset.py::test_censoring_columns -v
python -m pytest tests/test_build_dataset.py::test_unresolved_not_dropped -v
```

**Critical Note**: NO episodes dropped (all intakes remain in dataset)

---

## Agent 2: Survival Modeling Expert

**Tasks**: 1.3, 2.1, 2.2  
**Start**: Day 3 (after Agent 1)  
**Duration**: 3-4 days  

**Key File**: `src/aac_adoption/analysis/survival_analysis.py`  

**Tasks**:
- Enhance `compute_kaplan_meier_survival()` with censoring
- Fix `fit_cox_proportional_hazards()` for censored data
- Add `fit_aft_model()` for Weibull/Exponential
- Add proportional hazards testing

**Validation**:
```bash
python -m pytest tests/test_survival_analysis.py::test_cox_with_censored_data -v
python -m pytest tests/test_survival_analysis.py::test_aft_model_weibull -v
```

**Critical Note**: Use `event_observed` parameter, no silent dropna

---

## Agent 3: Pipeline Integration Specialist

**Tasks**: 3.1, 3.2, 3.3  
**Start**: Day 7 (after Agents 1-2)  
**Duration**: 2-3 days  

**Key Files**:
- `scripts/train_survival.py` (NEW - create this!)
- `src/aac_adoption/models/evaluate.py` (add metrics)
- `src/aac_adoption/features/survival_targets.py` (NEW - create this!)

**Keys**:
- Use `followup_days_censored` as survival time
- Event indicator: `~is_censored`
- Time-based split: 2013-2021 train, 2022-2023 val, 2024-2025 test

**Validation**:
```bash
python scripts/train_survival.py --animal-subset combined
ls models/survival/
```

**Critical Note**: Train Cox + AFT, output metrics to `reports/survival/`

---

## Agent 4: Testing Specialist

**Tasks**: 4.1, 4.2, 4.3  
**Start**: Day 9 (parallel with Agent 3)  
**Duration**: 1-2 days  

**Key Files**:
- `tests/test_survival_analysis.py` (extend with censoring tests)
- `tests/test_survival_integration.py` (NEW - pipeline tests)
- `tests/test_acceptance_survival.py` (NEW - artifact contracts)

**Coverage Target**: >80% for survival_analysis.py

**Validation**:
```bash
python -m pytest tests/test_survival*.py -v --cov=... --cov-report=term-missing
```

**Critical Note**: Test censoring handling explicitly, not just existing tests

---

## Agent 5: Reporting & Documentation Specialist

**Tasks**: 5.1, 5.2, 5.3  
**Start**: Day 10 (parallel with Agent 3)  
**Duration**: 1-2 days  

**Key Files**:
- `src/aac_adoption/diagnostics/survival_diagnostics.py` (NEW)
- `reports/summary/survival_descriptive_note.md` (rewrite)
- `src/aac_adoption/reporting/evidence_pack.py` (add survival section)

**Output Files**:
- `reports/diagnostics/survival_diagnostics.csv`
- `reports/diagnostics/survival_curves_by_*.csv`
- `reports/diagnostics/event_type_distribution.csv`
- `reports/diagnostics/censoring_reason_distribution.csv`

**Validation**:
```bash
python -c "from aac_adoption.diagnostics.survival_diagnostics import generate_survival_diagnostics_report; generate_survival_diagnostics_report(df, Path('reports/diagnostics'))"
cat reports/survival/survival_metrics.csv
```

**Critical Note**: Evidence pack MUST include survival section

---

## Dependency Flow

```
Agent 1 (Data) ──► Agent 2 (Models) ──► Agent 3 (Integration)
                                             ├─► Agent 4 (Testing)
                                             └─► Agent 5 (Reporting)
```

**Sequential**: Agents 1 → 2 → 3  
**Parallel**: Agents 4 & 5 work with Agent 3

---

## Quick Validation Summary

| Agent | Command | Success Indicator |
|-------|---------|-------------------|
| 1 | `test_censoring_columns` | Dataset has all 4 censoring columns |
| 2 | `test_cox_with_censored_data` | Cox model trains on censored data |
| 3 | `train_survival.py` | Files in `models/survival/` and `reports/survival/` |
| 4 | `pytest test_survival*.py` | All tests pass, >80% coverage |
| 5 | `generate_survival_diagnostics_report` | Diagnostics CSV generated |

---

## Priority Order

1. **Agent 1 first** (censoring is foundational)
2. **Agent 2 second** (needs censoring data)
3. **Agent 3 third** (needs models and data)
4. **Agents 4 & 5 in parallel** (can work once Agent 3 has artifacts)

---

## Common Pitfalls to Avoid

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Silent dropna() in Cox model | Censored data silently dropped | Use `event_observed` parameter |
| Dropping unmatched intakes | Survival analysis biased | Keep all intakes as censored |
| Not generating diagnostics | Can't verify model assumptions | Run `generate_survival_diagnostics_report()` |
| Missing acceptance tests | Changes break without detection | Write contracts before implementation |
| Not updating documentation | Methodology unclear in thesis | Write descriptive note before final |

---

## Handoff Checklist

- [ ] Agent 1 provides dataset with censoring columns
- [ ] Agent 2 provides working Cox/AFT functions  
- [ ] Agent 3 provides artifacts in `models/survival/`
- [ ] Agent 4 provides test suite with >80% coverage
- [ ] Agent 5 provides evidence pack with survival section

---

## Files to Monitor

### Modified Files (4 files)
- `src/aac_adoption/data/build_dataset.py`
- `src/aac_adoption/data/match_records.py`
- `src/aac_adoption/analysis/survival_analysis.py`
- `src/aac_adoption/reporting/evidence_pack.py`

### New Files (4 files)
- `scripts/train_survival.py`
- `src/aac_adoption/features/survival_targets.py`
- `src/aac_adoption/diagnostics/survival_diagnostics.py`
- `tests/test_survival_integration.py`
- `tests/test_acceptance_survival.py`

### New Directories (2 folders)
- `models/survival/` (model artifacts)
- `reports/survival/` (metrics outputs)

---

## Final Checklist

- [ ] Censoring columns in dataset
- [ ] Unresolved episodes not dropped
- [ ] Cox model handles censored data
- [ ] AFT model implemented
- [ ] Training pipeline runs end-to-end
- [ ] Survival metrics generated
- [ ] Diagnostics report generated
- [ ] Documentation updated
- [ ] Evidence pack includes survival
- [ ] All tests passing
- [ ] Coverage >80%

---

*End of Quick Reference Card*
