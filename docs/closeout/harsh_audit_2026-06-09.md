# Harsh Closeout Audit - 2026-06-09

## Verdict

Not thesis-ready yet. Phase 1 focused behavior is now validated, but project-wide
acceptance remains blocked.

## P0 Findings

1. Calibration used the combined 2022-2023 validation alias, contaminating the
   2023 selection period. Fixed: calibration now uses only the dedicated 2022
   frame. Regression coverage added.
2. Test collection fell from the documented 311 tests to 209 after four test
   modules were deleted. Deletion may be valid for unsupported survival scope,
   but every retained contract needs replacement coverage and matrix updates.
3. Generated final-model-selection prose still says winners were chosen on the
   2024-2025 test set. Producer wording is fixed; artifact remains stale until
   regeneration.

## Phase 1 Defects Fixed

- Matched outcomes are bounded by both next intake and extraction cutoff.
- Unresolved follow-up is bounded by the next intake and cannot be negative.
- Episode numbering follows intake sequence, not count of prior outcome rows.
- Horizon builder rejects cross-episode matches and post-extraction intakes.
- Horizon target metadata contains only `adopted_in_*` targets.
- Intake-volume windows exclude all simultaneous rows.
- Duplicate raw export rows do not inflate intake-volume counts.
- Missing focal intake-history joins fail explicitly.
- Weather lag and nullable flags are tested.

## Remaining High-Risk Work

- Review deleted test suites and create a contract-by-contract replacement map.
- Deep-audit Phase 2, Phase 5, and Phase 6.
- Reconcile false `FULL PASS` and `Remaining risk: None` tracker claims.
- Validate timezone-aware weather/311 dates using Austin local calendar days.
- Represent missing 311 source coverage separately from observed zero requests.
- Freeze concurrent edits, review the complete diff, then run full pytest.
- Run manual regeneration and long acceptance only after all short gates pass.
