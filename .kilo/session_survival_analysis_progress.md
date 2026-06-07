## Goal
- Implement survival analysis section for Slice 12, integrating censoring handling, competing-risk framework, and end-to-end training pipeline for ML thesis preparation

## Constraints & Preferences
- Survival analysis must be publication-quality before thesis submission
- Deadline-driven: prioritize pragmatically-executable tasks (Option B recommended)
- Must preserve reproducibility with fixed seed and deterministic behavior
- Integration with existing pipeline: build_dataset.py, match_records.py, train_*.py, evaluate.py

## Progress
### Done
- Descriptive Kaplan-Meier curves implemented for adopted-only subset (no censoring)
- Survival analysis module exists at `src/aac_adoption/analysis/survival_analysis.py`
- Unit tests exist at `tests/test_survival_analysis.py`
- Evidence pack docs at `reports/summary/survival_descriptive_note.md`
- Survival curves CSV at `reports/tables/adoption_survival_curves.csv`
- Basic integration: `log_transform_LOS` used in `train_advanced.py` line 258

### In Progress
- Slice 12 implementation planning (options A/B/C defined)
- Gap analysis completed: no native censoring, no competing risks, no horizon-based survival

### Blocked
- (none)

## Key Decisions
- Roadmap prioritizes survival analysis integration before thesis submission
- Recommended implementation path: Option B (pragmaic 8 tasks: 1.1, 1.2, 1.3, 2.1, 3.1, 4.1, 5.1, 5.2)
- Data-first approach: add censoring indicators at dataset building stage before analysis/modeling

## Next Steps
1. Add `is_censored`, `censoring_reason`, `event_type`, `followup_days_censored` to `build_dataset.py`
2. Extend `match_records.py` with censoring flags and `censoring_date`
3. Revise `survival_analysis.py` to use `event_observed` parameter, implement `fit_cox_with_censoring()`
4. Add categorical encoding and proportional hazards validation to Cox model
5. Create `scripts/train_survival.py` for end-to-end training
6. Extend `tests/test_survival_analysis.py` with censoring tests
7. Create `survival_diagnostics.py` for diagnostics
8. Update `survival_descriptive_note.md` with methodology

## Critical Context
- Current censoring only added post-hoc in `survival_analysis.py` via `add_censoring_indicators()` (not natively in dataset)
- Existing survival analysis limited to adopted animals (no censoring, no competing risks)
- Horizon targets [7, 30, 60, 90] already defined in `build_dataset.py` for regression
- Training pipeline uses CatBoost most frequently (`train_advanced.py`, `train_boosting.py`)
- `OOFBayesianTargetEncoder` exists for target encoding but not yet used in survival context

## Relevant Files
- `src/aac_adoption/data/build_dataset.py`: needs censoring columns addition
- `src/aac_adoption/data/match_records.py`: needs censoring date/flag integration
- `src/aac_adoption/analysis/survival_analysis.py`: main analysis module needing censoring support
- `src/aac_adoption/models/train_advanced.py`: line 258 uses `log_transform_LOS`
- `src/aac_adoption/reporting/evidence_pack.py`: needs survival section addition
- `tests/test_survival_analysis.py`: unit tests needing extension
- `reports/tables/adoption_survival_curves.csv`: current survival output
