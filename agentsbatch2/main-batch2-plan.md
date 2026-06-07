# Batch 2 Main Plan - Orchestrator

**Date:** 2026-06-07  
**Batch:** 2  
**Goal:** Fix yearly_backtesting.py quick mode iterations issue + recency comparison module

---

## P1: Yearly Backtesting Quick Mode Issue

**Problem:** Quick mode reduces test windows but not model iterations (100 iterations hardcoded)

**Files to fix:**
- `src/aac_adoption/models/yearly_backtesting.py`
- `scripts/evaluate_backtesting.py`
- `tests/test_yearly_backtesting.py`

**Fix:**
1. Add `iterations` parameter to `run_yearly_backtesting()`
2. Add `--iterations` CLI flag to evaluate_backtesting.py
3. Use iterations parameter in CatBoost model constructors
4. Default to 100 for normal mode, 20 for quick mode

---

## P2: Recency Comparison Issue

**Problem:** `scripts/compare_recency.py` exists but `src/aac_adoption/analysis/recency_comparison.py` missing

**Files to fix:**
- Create `src/aac_adoption/analysis/recency_comparison.py`
- Ensure CLI `--help` works instantly

**Fix:**
1. Create recency_comparison.py module with business logic
2. CLI should parse args before heavy imports

---

## Agent Roles

1. **Implementer Agent** - Fix P1 quick mode iterations
2. **Implementer Agent** - Create recency_comparison module
3. **Validator Agent** - Verify fixes
4. **Reviewer Agent** - Review changes
5. **Test Agent** - Run tests

---

## Tasks

### Task 1: Quick Mode Iterations Fix
- Edit `yearly_backtesting.py` to add iterations parameter
- Edit `evaluate_backtesting.py` to add --iterations flag
- Test quick mode works

### Task 2: Recency Module Creation
- Create `src/aac_adoption/analysis/recency_comparison.py`
- Ensure CLI help works without loading data

### Task 3: Testing
- Run pytest on target tests
- Verify quick mode behavior

### Task 4: Review
- Check changes against plan
- Verify acceptance criteria

---

## Communication Protocol

 Agents write to shared file: `agentsbatch2\communication.md`

 Format:
 - Agent: <name>
 - Task: <task_id>
 - Status: <status>
 - Output: <results>

---

## Exit Criteria

- [ ] P1: quick mode uses reduced iterations
- [ ] P2: recency module exists and CLI works
- [ ] All tests pass
- [ ] Review approved
