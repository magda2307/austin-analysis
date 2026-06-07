# P1 Findings - Missing target_definitions.md

**Date:** 2026-06-07  
**Issue:** P1 - tests/test_target_definitions.py (line 12) fails because docs/target_definitions.md is missing

## Analysis

The test `test_target_definitions_doc_exists` in `tests/test_target_definitions.py:12` expects `docs/target_definitions.md` to exist.

### Investigation

- File was found in `docs/old/target_definitions.md` instead of `docs/target_definitions.md`
- The file in `docs/old/` is an outdated location
- Test expects file in root `docs/` folder

## Required Action

Move `docs/old/target_definitions.md` to `docs/target_definitions.md` to restore compatibility.

## Test Requirements

File must contain:
1. Binary adoption outcome definition
2. Length-of-stay / days target definition  
3. Discussion about intake-time features and leakage

---

*Documented by: Agent Manager Subagent*  
*Date: 2026-06-07*
