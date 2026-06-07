# P3 Validation Gap Review

## FINDING: UNINTENDED
Docstring lie: "3-year validation window" claims 3 years but actually provides 2 years.

## MATH VERIFICATION
- test_start=2024, validation_gap_years=3
- train_end=2021 (line 84: test_start - validation_gap_years)
- validation window: train_end+1=2022 to test_start-1=2023
- validation years: 2022, 2023 → **2 years**

## GAP PARAMETER SEMANTICS
Parameter name `validation_gap_years` represents year difference (test_start - train_end), NOT validation window count.

- gap=3 → train_end=2021 → 2-year validation window
- gap=2 → train_end=2022 → 1-year validation window  
- gap=4 → train_end=2020 → 3-year validation window

## DEFAULT DECISION

### Option A: Keep gap=3, fix docstring
- Pros: No code change
- Cons: Confusing semantics (gap=3 ≠ 3-year window)

### Option B: Change default to gap=4
- Pros: gap=N gives N-year validation window (clear semantics)
- Cons: Potential behavior change (train_end=2020 instead of 2021)

### Option C: Change default to gap=2
- Pros: Matches 2-year window implied by current docstring
- Cons: Only 1-year validation (too short)

## RECOMMENDATION

**CHANGE DEFAULT TO gap=4** for 3-year validation window.

### Why:
- Users interpret "gap=3" as "3-year validation" (intuitive)
- Current gap=3 only gives 2-year window (confusing)
- 3-year validation window provides more robust model evaluation
- gap=N semantics become: gap equals validation window years

### CHANGES NEEDED:
1. Line 37: `validation_gap_years: int = 4` (was 3)
2. Line 63: Docstring fix "3-year validation window"
3. Line 83: Update inline comment to show gap=4 example

### DOCUMENTATION:
Add explicit formula:
```
validation_gap_years = gap parameter
train_end = test_start - validation_gap_years
validation_window_years = validation_gap_years - 1
```
