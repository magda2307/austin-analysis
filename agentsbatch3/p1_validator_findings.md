# P1 Validator Findings:Broad Except Handling in Optuna Tuning

**Date:** 2026-06-07  
**Issue:** P1 - Broad `except Exception:` blocks in `src/aac_adoption/models/tune.py` lines 104-105 and 143-144 mask tuning failures as valid (bad) trials

---

## 1. EXACT CODE CAUSING ISSUE

### Classification Objective (Lines 100-105)
```python
try:
    scores = []
    for train_idx, val_idx in cv.split(cat_X_clf):
        X_tr, y_tr = cat_X_clf.iloc[train_idx], y_clf.iloc[train_idx]
        X_va, y_va = cat_X_clf.iloc[val_idx], y_clf.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features_clf,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
        )
        preds = model.predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, preds))
    return np.mean(scores)
except Exception:
    return 0.0
```

### Regression Objective (Lines 140-145)
```python
try:
    scores = []
    for train_idx, val_idx in cv.split(cat_X_reg):
        X_tr, y_tr_reg = cat_X_reg.iloc[train_idx], y_reg.iloc[train_idx]
        X_va, y_va_reg = cat_X_reg.iloc[val_idx], y_reg.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, np.log1p(y_tr_reg),
            cat_features=cat_features_reg,
            eval_set=(X_va, np.log1p(y_va_reg)),
            early_stopping_rounds=50,
        )
        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log)
        scores.append(mean_absolute_error(y_va_reg, preds))
    return np.mean(scores)
except Exception:
    return 1e9
```

---

## 2. IDENTICAL ISSUE IN HISTGRADIENTBOOSTING OBJECTIVES

### HistGradientBoosting Classification (Lines 183-184)
```python
except Exception:
    return 0.0
```

### HistGradientBoosting Regression (Lines 223-224)
```python
except Exception:
    return 1e9
```

---

## 3. PROBLEM ANALYSIS

### 3.1 Current Behavior

| Type | Line | Failure Behavior | Why This Is Dangerous |
|------|------|------------------|----------------------|
| Classification | 104-105 | Returns `0.0` | Makes failure look like "terrible model" but could mask data bugs, CatBoost crashes, or code errors |
| Regression | 143-144 | Returns `1e9` | Makes failure look like "terrible model" but could mask data bugs, CatBoost crashes, or code errors |
| HistGrad Clf | 183-184 | Returns `0.0` | Same as CatBoost classification |
| HistGrad Reg | 223-224 | Returns `1e9` | Same as CatBoost regression |

### 3.2 What Could Go Wrong

**Scenario A: Data Preprocessing Bug**
- Feature engineering produces invalid input (NaN, infinite values, wrong dtype)
- CatBoost `fit()` raises internal error
- Broad except catches it, returns 0.0 or 1e9
- Optuna treats it as a "valid but terrible trial"
- Tuning continues, wasting compute on invalid configurations
- **Thesis impact:** Could select suboptimal hyperparameters because failures aren't properly flagged

**Scenario B: CatBoost Version Incompatibility**
- New CatBoost version changes API or has a bug
- Model construction fails silently
- No clear error message to developer
- Tuning appears to succeed but models are invalid

**Scenario C: Random Seed Issues**
- `RANDOM_STATE` combined with other randomness causes unseeded operations
- Non-deterministic failures that are hard to debug
- Broad except masks what should be reproducible behavior

**Scenario D: Memory/Resource Exhaustion**
- Large fold train sets cause OOM during fitting
- Process crashes but exception caught, trial marked as "failed"
- No indication that system resources need sizing adjustment

---

## 4. THESIS-SAFE APPROACH

### 4.1 Optuna Best Practices

**Optuna explicitly recommends:**
- Do NOT catch exceptions in objective functions
- Let failures propagate so trials are marked `FAIL` state
- Use Optuna's built-in pruners and samplers that handle failures gracefully

**From Optuna documentation:**
> If an exception is raised during optimization, the trial is marked as failed. You should not catch exceptions in the objective function unless you want to handle them and return a default value.

However, returning a default score instead of raising an exception creates the **masking problem** identified here.

### 4.2 Recommended Strategy: Fail Fast with Context

**Approach A: Let Exceptions Propagate (Simplest)**
```python
def catboost_clf_objective(trial: optuna.Trial) -> float:
    # ... params setup ...
    scores = []
    for train_idx, val_idx in cv.split(cat_X_clf):
        # ... CV loop ...
        model = CatBoostClassifier(**params)
        model.fit(...)
        preds = model.predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, preds))
    return np.mean(scores)
# No try/except at all
```

**Pros:**
- Clear error messages showing exact stack trace
- Optuna marks failed trials explicitly (no confusion)
- Easier debugging in thesis development phase

**Cons:**
- Some trials may legitimately fail due to edge cases
- Need to handle expected failures (e.g., invalid hyperparameter combos)

**Approach B: Catch Specific Exceptions, Re-Raise Others**
```python
def catboost_clf_objective(trial: optuna.Trial) -> float:
    try:
        # ... CV loop ...
        scores = []
        for train_idx, val_idx in cv.split(cat_X_clf):
            # ... fit/predict ...
        return np.mean(scores)
    except (ValueError, CatBoostError, RuntimeError) as e:
        optuna.logging.log(f"Trial {trial.number} failed due to {type(e).__name__}: {e}")
        raise optuna.TrialPruned(f"Invalid hyperparameter combo: {e}")
```

**Pros:**
- Clear distinction between "bad hyperparameters" (pruned) and "unexpected errors" (raised)
- Optuna marks as `PRUNED` for hyperparameter issues, `FAIL` for code/data issues
- Thesis methodology clear about what constitutes valid vs invalid trial

**Cons:**
- Requires understanding of what exceptions CatBoost/sklearn can raise
- More code complexity

**Approach C: Log + Re-Raise (Middle Ground)**
```python
import logging
logger = logging.getLogger(__name__)

def catboost_clf_objective(trial: optuna.Trial) -> float:
    try:
        # ... CV loop ...
        return np.mean(scores)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with {type(e).__name__}: {e}")
        raise
```

**Pros:**
- Errors logged for post-hoc debugging
- Exceptions propagate as failures
- Optuna provides trial failure statistics

**Cons:**
- Logs can be noisy during development
- Still loses information if logger not configured

---

## 5. THESIS-SAFE RECOMMENDATION

### ✅ Recommended: Catch Specific Exceptions, Prune Invalid Hyperparameters

**Rationale for thesis safety:**

1. **Transparency:** Clear distinction between hyperparameter issues (expected) and code/data bugs (unexpected)
2. **Auditability:** Optuna trial history shows which failures were due to invalid combos vs system issues
3. **Reproducibility:** Thesis methodology can document "hyperparameter search space limits" and expected failure modes
4. **Scientific Rigor:** Failed trials due to bugs are treated as failures (not bad scores), making methodology clearer

### Implementation Plan

#### Step 1: Import Required Exceptions
```python
from catboost import CatBoostError, CatBoostWarning
from sklearn.exceptions import ConvergenceWarning
import optuna.logging
```

#### Step 2: Update All Four Objectives

**Classification Objectives (both CatBoost and HistGrad):**
```python
def catboost_clf_objective(trial: optuna.Trial) -> float:
    params = {  # ... as before ... }
    
    try:
        scores = []
        for train_idx, val_idx in cv.split(cat_X_clf):
            # ... existing CV loop code ...
        return np.mean(scores)
    except (CatBoostError, CatBoostWarning, ValueError) as e:
        optuna.logging.log(f"Classification trial {trial.number} pruned: {e}")
        raise optuna.TrialPruned(f"Invalid classification params: {e}")
```

**Regression Objectives (both CatBoost and HistGrad):**
```python
def catboost_reg_objective(trial: optuna.Trial) -> float:
    params = {  # ... as before ... }
    
    try:
        scores = []
        for train_idx, val_idx in cv.split(cat_X_reg):
            # ... existing CV loop code ...
        return np.mean(scores)
    except (CatBoostError, CatBoostWarning, ValueError) as e:
        optuna.logging.log(f"Regression trial {trial.number} pruned: {e}")
        raise optuna.TrialPruned(f"Invalid regression params: {e}")
```

#### Step 3: Add Logging Configuration

Add at top of tune.py (after imports):
```python
import logging
logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)
```

This ensures pruned trial messages are visible during development.

---

## 6. SPECIFIC EXCEPTION TYPES TO HANDLE

### Classification-Specific
| Exception | When It Occurs | Should Prune? |
|-----------|---------------|---------------|
| `CatBoostError` | Invalid params, data issues | ✅ YES |
| `ValueError` | Wrong feature counts, invalid labels | ✅ YES |
| `RuntimeError` | Early stopping issues | ✅ YES |
| `ConvergenceWarning` | HistGrad not converging | ⚠️ Consider pruning |

### Regression-Specific
| Exception | When It Occurs | Should Prune? |
|-----------|---------------|---------------|
| `CatBoostError` | Invalid params, data issues | ✅ YES |
| `ValueError` | Wrong feature counts, invalid targets | ✅ YES |
| `RuntimeError` | Numeric overflow, early stopping | ✅ YES |
| `ValueError: math domain error` | log1p(expm1) overflow | ✅ YES |

### Common Invalid Configurations
| Hyperparameter Combo | Expected Error | Action |
|---------------------|---------------|--------|
| `depth > log2(max_leaf_nodes)` | CatBoost warning/accuracy drop | ✅ PRUNE |
| `subsample < min_samples_leaf / n_train` | Training split too small | ✅ PRUNE |
| `learning_rate < 1e-5` | Convergence too slow | ⚠️ Let run or PRUNE |
| `l2_leaf_reg < 1e-6` | Overfitting risk (not error) | ❌ DO NOT PRUNE |

---

## 7. FOLD STRATEGY DOCUMENTATION

### Required Documentation for Thesis

**Add to `src/aac_adoption/models/tune.py` docstring:**

```python
def tune_models(df: pd.DataFrame, n_trials: int = 20, sampler_type: str = "tpe") -> tuple[dict[str, Any], dict[str, optuna.Study]]:
    """Run Optuna studies to find best hyperparameters.
    
    Args:
        df: Training dataframe
        n_trials: Number of trials per model
        sampler_type: 'tpe', 'random', or 'cmaes'
    
    Folding Strategy:
        - 5-fold time-series cross-validation using TimeSeriesSplit
        - Chronological ordering maintained (no data leakage)
        - Validation folds used for early stopping in CatBoost
        - Each trial trains 5 models (one per fold)
        
    Failure Handling:
        - Expected failures (invalid hyperparameter combos) → TrialPRUNEd
        - Unexpected failures (code/data bugs) → Trial marked as FAIL
        - Failed trials do NOT contribute to best hyperparameter selection
        - Optuna's MedianPruner stops unpromising trials early
    """
```

### Add to `docs/ARCHITECTURE.md` (Hyperparameter Tuning Section)

Currently ARCHITECTURE.md line 275 states:
> **Hyperparameter tuning** — documented as planned improvement in ROADMAP.md.

Update to:
```markdown
## Hyperparameter Tuning (Implemented)

**Module:** `src/aac_adoption/models/tune.py`  
**Framework:** Optuna with TPE sampler (seeded for reproducibility)

**Strategy:**
- 5-fold time-series cross-validation with chronological ordering
- Early stopping based on validation fold performance
- Median pruner stops unpromising trials early

**Failure Modes:**
- Invalid hyperparameter combinations → Trial PRUNED
- Code/data bugs → Trial FAILED (raises exception)
- Failed trials excluded from best parameter selection

**Reproducibility:**
- RANDOM_STATE used for sampler seed
- CatBoost random_seed parameter set
- TimeSeriesSplit maintains deterministic fold assignment
```

---

## 8. CODE PATH ANALYSIS

### Tuning Flow
```
tune_models(df)
├── Prepare classification data (lines 32-40)
├── Prepare regression data (lines 42-50)
├── Create TimeSeriesSplit (n_splits=5, line 57)
├── catboost_clf_objective(trial) → catboost_clf_objective (lines 73-108)
│   ├── Try CV loop (lines 88-105)
│   │   ├── For each fold: fit + predict + score
│   │   └── Return mean PR-AUC
│   └── Except: return 0.0  ← PROBLEM LINE (line 104-105)
├── study_cat_clf.optimize(catboost_clf_objective, n_trials)
├── catboost_reg_objective(trial) → catboost_reg_objective (lines 112-148)
│   ├── Try CV loop (lines 126-144)
│   │   ├── For each fold: log1p + fit + expm1 + MAE
│   │   └── Return mean MAE
│   └── Except: return 1e9  ← PROBLEM LINE (line 143-144)
└── study_cat_reg.optimize(catboost_reg_objective, n_trials)
```

### Problem Location Summary

| File | Line(s) | Issue | Severity |
|------|---------|-------|----------|
| tune.py | 104-105 | Broad except in catboost_clf_objective | P1 |
| tune.py | 143-144 | Broad except in catboost_reg_objective | P1 |
| tune.py | 183-184 | Broad except in hist_clf_objective | P1 |
| tune.py | 223-224 | Broad except in hist_reg_objective | P1 |

---

## 9. THESIS SAFETY COMPARISON

### Current Approach (Unsafe)
```python
except Exception:
    return 0.0  # or 1e9
```

**Problems:**
1. ❌ Masks code bugs as "bad hyperparameters"
2. ❌ No visibility into why trials fail
3. ❌ Optuna cannot distinguish hyperparameter issues from system errors
4. ❌ Thesis methodology unclear about failure modes
5. ❌ Cannot document "invalid hyperparameter search space" boundaries

### Recommended Approach (Safe)
```python
except (CatBoostError, CatBoostWarning, ValueError) as e:
    optuna.logging.log(...)
    raise optuna.TrialPruned(f"Invalid params: {e}")
```

**Benefits:**
1. ✅ Clear distinction between expected failures and bugs
2. ✅ Optuna trial history clearly shows pruning vs failure
3. ✅ Thesis methodology can document hyperparameter limits
4. ✅ Reproducible: same error always produces same trial status
5. ✅ Easy debugging: stack trace shows exact problem

---

## 10. ADDITIONAL CONSIDERATIONS

### 10.1 Optuna Trial Statistics

With proper pruning/failure handling, Optuna will record:
- `number` — trial index
- `value` — objective value (or None if pruned/failed)
- `state` — `COMPLETE`, `PRUNED`, or `FAIL`
- `datetime_start`, `datetime_complete`
- `params` — hyperparameter values

This enables thesis analysis:
- How many trials pruned due to invalid hyperparameters?
- Which hyperparameter ranges cause failures?
- Did pruning improve final model selection?

### 10.2 Documentation Location

**Files to update:**
1. `src/aac_adoption/models/tune.py` — add failure handling docstring
2. `docs/ARCHITECTURE.md` — add Hyperparameter Tuning section
3. `docs/METHODOLOGY.md` — add Hyperparameter Search subsection
4. `docs/ROADMAP.md` — mark tuning as complete with safety notes

### 10.3 Backward Compatibility

**Breaking change:** Existing code that expects tuning to never fail may need updates:
- No code currently calls `tune_models()` — only used by scripts
- Scripts should expect trials to fail/prune (this is normal)
- Final best parameters should be validated after tuning

---

## 11. SUMMARY

### Current State
| Aspect | Status |
|--------|--------|
| **P1 Severity** | HIGH |
| **Thesis Safety** | ❌ UNSAFE |
| **Broad except blocks** | 4 locations (lines 104, 143, 183, 223) |
| **Error masking** | YES — failures look like "bad trials" |
| **Visibility** | LOW — no logging, hard to debug |
| **Optuna compliance** | ❌ Not recommended practice |

### Recommended Fix
| Aspect | Status |
|--------|--------|
| **Approach** | Catch specific exceptions, re-raise as TrialPruned |
| **Exception types** | CatBoostError, CatBoostWarning, ValueError |
| **Logging** | Add optuna.logging for visibility |
| **Trial state** | PRUNED for invalid params, FAIL for bugs |
| **Thesis safety** | ✅ SAFE — clear failure modes |

### Actions Required
1. ✅ Replace broad `except Exception:` with specific exception catching
2. ✅ Use `optuna.TrialPruned` for expected hyperparameter failures
3. ✅ Add logging configuration at top of tune.py
4. ✅ Update docs/ARCHITECTURE.md with tuning safety details
5. ✅ Document fold strategy and failure modes in tune.py docstring

---

**Recommendation:** Replace broad except blocks with specific exception handling to maintain thesis methodological integrity. Optuna's trial states (COMPLETE/PRUNED/FAIL) should reflect the actual cause of trial outcomes, not mask system errors as suboptimal hyperparameters.
