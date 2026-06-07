## Documentation Review

### Current State:
- Does docstring explain validation_gap_years impact? [Partially]
- Does it document temporal implications? [Partially]

### Issues Found:
- Line 53-54: Default explained but no rationale for "3 years" — why 3 not 2 or 5?
- Line 60-63: Temporal logic described (train_end = test_start - gap) but no concrete example with actual dates
- Line 63: "provides approximately 2 years of validation data" is vague — how is this calculated?
- Missing: clear explanation of what happens if user changes gap to different values
- Missing: warning about temporal leakage prevention mechanics

### Recommendations:
- Add concrete example: "For test_period='2024-2025', gap=3 → train_end=2021, training uses 2013-2021"
- Explain calculation: "gap=3 years with test_start=2024 means validation window 2022-2023 (2 years)"
- Add rationale: "3-year gap balances sufficient validation data vs. acceptable training sample size"
- Document impact: " smaller gap → less buffer, larger gap → less training data"
- clarify "validation data" claim at line 63
