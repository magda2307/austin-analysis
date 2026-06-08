# Batch 3 Orchestrator - P1/P2/P3 Task Plan

**Date:** 2026-06-07  
**Batch:** 3  
**Priority:** Model Method Hardening  
**Status:** Orchestrating P1, P2, P3 tasks

## Task Breakdown

### P1: tune.py:L104 - Exception Handling
- **Severity:** HIGH
- **Issue:** Broad except turns tuning failures into valid trials (return 0.0)
- **Fix:** Re-raise unexpected errors or fail trial cleanly
- **Agent Pattern:** Validator → Fix → Review → Pytest

### P2: train_boosting.py:L218 - Permutation Scoring
- **Severity:** MEDIUM
- **Issue:** Permutation importance uses ROC-AUC scoring
- **Fix:** Change to PR-AUC: scoring="average_precision"
- **Agent Pattern:** Validator → Fix → Review → Pytest

### P3: test_hyperparam_tuning.py:L164 - Weak Spy Test
- **Severity:** HIGH
- **Issue:**Spy test weak; objective can die before all folds and still pass
- **Fix:** Assert fold calls or fake fit+predict
- **Agent Pattern:** Validator → Fix → Review → Pytest

## Communication Protocol

- **Main communication:** `agentsbatch3/communication.md`
- **Status tracker:** `agentsbatch3/status_tracker.md`
- **Validator findings:** `agentsbatch3/p1_validator_findings.md`, `p2_validator_findings.md`, `p3_validator_findings.md`
- **Fix summary:** `agentsbatch3/p1_fix_summary.md`, `p2_fix_summary.md`, `p3_fix_summary.md`
- **Review report:** `agentsbatch3/p1_review_report.md`, `p2_review_report.md`, `p3_review_report.md`
- **Test results:** `agentsbatch3/p1_pytest_results.md`, `p2_pytest_results.md`, `p3_pytest_results.md`
- **Final summary:** `agentsbatch3/final_summary_batch3_P.md`

## Agent Roles

- **Validator Agent:** Verify issues, check code paths, identify edge cases
- **Fix Agent:** Implement fixes for identified risks
- **Reviewer Agent:** Review changes, verify thesis-safe implementation
- **Pytest Agent:** Run focused tests, report pass/fail

## Timeline

16:14 - Batch 3 P1/P2/P3 started
16:14 - Dispatching P1, P2, P3 in parallel (3 separate agents)
