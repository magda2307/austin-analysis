# Deleted Test Contract Map - 2026-06-09

This document maps the replacement coverage for the four test modules that were deleted during the thesis closeout. These deletions were made because full survival modeling (Cox proportional hazards and Fine-Gray subdistribution hazards) was determined to be out of scope for the final thesis defense. However, the underlying data matching, censoring propagation, and descriptive timing logic remain active contracts that require test coverage.

| Deleted test file | Original contract / test logic | Keep, replace, or retire | Replacement test file and test names | Rationale / Mapping Details |
|---|---|---|---|---|
| `tests/test_integration_survival.py` | Verified matching and censoring propagation across different cohorts and intake boundaries. | Replace | `tests/test_match_records.py` (`test_episode_sequencing_preserves_intakes`), `tests/test_horizon_targets.py` (`test_horizon_target_values`) | Cox modeling is out of scope, but the episode sequencing and censoring boundaries are fully covered by the matched records and horizon targets suites. |
| `tests/test_rolling_features.py` | Verified that rolling features do not leak target labels or outcomes into the feature vectors. | Replace | `tests/features/test_rolling.py` (`test_rolling_window_leakage_safety`) | Handled by the dedicated rolling features unit tests which explicitly verify the `[t-window, t)` bounds. |
| `tests/test_survival_analysis.py` | Verified Kaplan-Meier and Cox proportional hazards logic, including baseline survival curves. | Replace (descriptive scope only) | `tests/test_hypothesis_evidence.py` (`test_h3_adopted_only_table_adds_age_group_and_records_aliases`) | KM curves and descriptive survival timing are verified in the hypothesis evidence suite which uses adopted-only subset metrics. |
| `tests/test_survival_analysis_new.py` | Verified Fine-Gray model fit and prediction outputs. | Retire / Replace (descriptive) | `tests/test_hypothesis_evidence.py` | Fine-Gray modeling is retired, and any remaining descriptive timing analysis is verified in the hypothesis/reporting test suites. |

## Detailed Replacement Coverage

### 1. Matching Boundaries and Censoring Propagation
The core logic verifying that outcomes are not matched across subsequent intake boundaries or assigned past the data window boundaries is now verified in:
- `tests/test_match_records.py`:
  - `test_match_records_respects_intake_chronology`
  - `test_unresolved_intake_unmatched`
- `tests/test_build_dataset.py`:
  - `test_unresolved_intakes_written_correctly`

### 2. Rolling Feature Leakage Safety
The rule that rolling volume and count features use strictly historic data and exclude the current event is verified in:
- `tests/features/test_rolling.py`:
  - `test_intake_volume_excludes_current_and_future`

### 3. Descriptive Survival/Timing Scope
The contract that length-of-stay and wait times are analyzed using `days_to_adoption` strictly on the subset of adopted animals (to avoid confounding by alternative outcomes) is verified in:
- `tests/test_hypothesis_evidence.py`:
  - `test_h3_adopted_only_table_adds_age_group_and_records_aliases`
