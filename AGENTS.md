# Agent Guide

This repository is an artifact-first Python ML thesis project for Austin Animal
Center adoption outcomes. Keep changes reproducible, leakage-safe, chronological,
and consistent with the written methodology.

## Task-First Routing

Read `.agents/README.md`, then load one task packet:

- Known file/bug: target module + direct tests from `.agents/COMMANDS.md`.
- Unknown ownership: `.agents/REPO_MAP.md`, then target module.
- Active closeout task: `.agents/CLOSEOUT.md`, then only its numbered section in
  `docs/PROJECT_CLOSEOUT_TASKS.md`.
- Target, leakage, or thesis wording: relevant heading in
  `docs/target_definitions.md`; use `rg -n "^##|^###"`.
- Verification only: `.agents/COMMANDS.md`.

Never recursively search `agentsbatch*/` or load all of
`THESIS_CONTEXT_FOR_LLM.md`, `README.md`, or historical agent state unless the
user explicitly resumes that work. Use heading search and bounded reads.

## Required End-State Contracts

- Model predictors must be available at intake time. Run leakage checks after
  changing feature lists.
- `classification_target` means adoption vs non-adoption for a matched episode.
- `regression_target_days` means days to any matched outcome, not days to adoption.
- `days_to_adoption` is valid only for adopted-only timing analysis.
- Thesis evaluation is chronological: train 2013-2021, validation 2022-2023,
  test 2024-2025. Current code falls back to deterministic random splitting when
  those years are unavailable; do not silently treat fallback results as thesis
  evaluation.
- Validation selects thresholds/calibration; test data is final evaluation only.
- The dashboard consumes generated artifacts and must not retrain models.
- Missing or broken model artifacts must produce an explicit failure state, never
  a plausible default prediction.
- Generated outputs live under `data/processed`, `models`, and `reports`. Avoid
  hand-editing generated artifacts; fix the producer and regenerate.
- Use predictive/descriptive language, not causal claims.

## Working Rules

- The worktree may contain concurrent agent changes. Inspect `git status` first
  and never revert unrelated edits.
- Prefer focused tests during implementation, then broaden according to risk.
- Preserve CSV schemas and documented alias columns used by tests and dashboard.
- Use `src/aac_adoption/config.py` paths or project-root-derived paths instead of
  current-working-directory assumptions.
- Keep random behavior deterministic with `RANDOM_STATE = 42`.
- Treat existing violations of required contracts as bugs, not documentation
  ambiguity. See `.agents/CLOSEOUT.md`.

Current acceptance state belongs in `.agents/CLOSEOUT.md`. Re-run commands before
reporting counts; dated snapshots become stale.
