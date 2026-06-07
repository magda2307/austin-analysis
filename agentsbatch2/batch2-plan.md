# Agent Batch 2 - Risk Validation and Fix Plan

## Task
Address risk: `default validation_gap_years=3` on line 37 means test period 2024-2025 trains only through 2021.
- If intended: document the rationale
- If not intended: update default to train through 2023 or make gap explicit

## Agent Structure

### 1. Validation Agent
**Purpose**: Analyze the validation gap logic and determine intent
- Read `recency_comparison.py` lines 37-80
- Analyze how `validation_gap_years=3` affects training end date
- Check if this is intentional (temporal leakage prevention) or oversight
- Read any related documentation
- Report intent findings

**Output**: `agentsbatch2/validation_findings.md`

### 2. Documentation Agent
**Purpose**: Add appropriate documentation based on validation findings
- If intentional: add rationale comment explaining why 3-year gap is appropriate
- If unintentional: add warning comment and propose default change
- Update docstring in `run_recency_comparison()` function
- Add typehint for explicitness

**Output**: `agentsbatch2/documentation_update.md`

### 3. Review Agent
**Purpose**: Review the changes for correctness and completeness
- Verify logic is sound
- Check for edge cases
- Ensure documentation is clear
- Confirm no side effects on existing functionality

**Output**: `agentsbatch2/review_findings.md`

### 4. Validator Agent (Cross-check)
**Purpose**: Validate the validation agent's findings independently
- Re-analyze the gap calculation logic
- Verify against test period 2024-2025
- Cross-check with code execution if possible
- Confirm no other related risks exist

**Output**: `agentsbatch2/validator_findings.md`

## Communication Protocol

All agents write to `agentsbatch2/communication.md` for:
- Task completion status
- Findings summary
- Recommendations
- Any blocking issues

## Execution Order

1. Validation Agent →
2. Documentation Agent (if fix needed) →
3. Review Agent →
4. Validator Agent (cross-check)

## Success Criteria

- [ ] Intent determined and documented
- [ ] Code updated with appropriate comments/docstring
- [ ] All agents agree on approach
- [ ] Risk resolved or properly documented
