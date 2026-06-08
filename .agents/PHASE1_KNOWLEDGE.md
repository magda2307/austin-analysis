# Phase 1 Knowledge

You are working on Phase 1 of the AAC Adoption project closeout.

**Important Files:**
- Master Implementation Plan: `docs/closeout/2026-06-08-thesis-closeout-implementation-plan.md`
- Status Tracking: `docs/closeout/phase1.md`
- Global Rules: `.agents/AGENTS.md` and `.agents/CLOSEOUT.md`

**Worker Guidelines:**
1. Read the section for your assigned task in the Master Implementation Plan.
2. Modify the necessary files to fulfill your task's "Required interface" and "Behavioral contract".
3. Ensure you run the exact pytest commands listed in your task's verification step to ensure they pass.
4. Once completed and passing locally, append the standard handoff template to `docs/closeout/phase1.md` and update the checkbox for your task.
5. The handoff template must be exactly:
```text
Scope: [Task Name]
Files changed:
Behavioral contract:
Tests run:
Test result:
Generated artifacts changed:
Remaining risk:
```
6. Send a message to the orchestrator confirming that you have finished your task.

**Reviewer Guidelines:**
1. Review the worker's changes against the exact requirements in the Master Implementation Plan.
2. Check that the required tests pass locally.
3. Verify that no global invariants were violated (see Master Implementation Plan).
4. If approved, notify the orchestrator. If changes are needed, list them out clearly so the orchestrator can re-assign the worker to fix them.
