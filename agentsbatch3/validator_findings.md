# Validator Findings: P2 Issues in Ensemble Stacking Fallback

## Overview
Analyzed `src/aac_adoption/models/ensemble.py` forthesis-unsafe behavior in ensemble stacking fallback mechanisms.

---

## 1. EXACT CODE CAUSING ISSUE

### Classifier Stacking Fallback (Lines 100-103)
```python
if actual_n_splits < 2 or not stratification_possible:
    self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
    base_predictions = np.column_stack([est.predict_proba(X)[:, 1] for est in self.base_estimators_])
    self.meta_estimator_ = clone(self.meta_estimator).fit(base_predictions, y)
```

### Regressor Stacking Fallback (Lines 154-157)
```python
if actual_n_splits < 2:
    self.base_estimators_ = [clone(est).fit(X, y) for est in self.base_estimators]
    base_predictions = np.column_stack([est.predict(X) for est in self.base_estimators_])
    self.meta_estimator_ = clone(self.meta_estimator).fit(base_predictions, y)
```

---

## 2. THESIS SAFETY CONCERN

**Meta-learner overfits when trained on in-sample predictions:**

1. Base models are trained on full training data `X, y`
2. Meta-learner receives predictions from these same(base models on same data)
3. Meta-learner cannot distinguish between generalizable patterns and memorization artifacts
4. Stacking loses its primary benefit (reducing bias/variance via OOF predictions)
5. Validation metrics become optimistically biased
6. Results poor generalization to unseen data

**Why this defeats stacking purpose:** Stacking relies on meta-learner learning to combine base model predictions *in a way that generalizes*. When meta-learner sees predictions from models that perfectly memorized the training data, it learns to trust these memorized patterns, leading to excessive overfitting.

---

## 3. THESIS-SAFE FIX OPTIONS

### Option A: ✅ RAISE EXCEPTION (RECOMMENDED)
```python
if actual_n_splits < 2 or not stratification_possible:
    raise ValueError(
        f"Stacking ensemble requires at least {actual_n_splits}-fold cross-validation. "
        f"Got {len(X)} samples with {len(np.unique(y))} classes. "
        "Increase dataset size or use simple ensemble instead."
    )
```

**Rationale:**
- Stacking fundamentally requires OOF predictions for thesis validity
- Silent fallback undermines methodological rigor
- Clear error forces proper data preparation or alternative approach
- Matches academic standards for ML methodology reporting

### Option B: Skip stacking, use weighted ensemble
**Pros:** Graceful degradation  
**Cons:** May mislead user about ensemble type; inconsistent behavior

### Option C: Mark as non-thesis with warning
**Pros:** Allows exploratory use  
**Cons:** Risk of accidental use in thesis without awareness; hard to enforce

**Final Recommendation:** Option A (raise exception)

---

## 4. CURRENT TEST COVERAGE

### Tests for Fallback Behavior

| Test Function | Lines | Purpose | Thesis-Safety Validation |
|--------------|-------|---------|-------------------------|
| `test_stacked_ensemble_classifier_fallback` | 359-382 | Tests classifier fallback when stratification impossible | ❌ Only verifies code path runs |
| `test_stacked_ensemble_regressor_fallback` | 384-406 | Tests regressor fallback when only 1 sample | ❌ Only verifies code path runs |

### Gaps in Test Coverage

1. **No OOF vs in-sample validation**: Tests don't verify whether meta-learner receives OOF predictions (safe) vs in-sample predictions (unsafe)
2. **No generalization assessment**: No tests checking if overfitting occurs in fallback mode
3. **Missing edge case coverage**: Limited testing for rare class scenarios, multi-class imbalanced data
4. **No integration tests**: No tests verifying fallback behavior affects final model performance negatively

---

## 5. EDGE CASES

| Edge Case | Classifier | Regressor | Current Handling |
|-----------|-----------|-----------|------------------|
| **n < n_splits** | ✅ `actual_n_splits = min(n_splits, len(X))` | ✅ Same | Falls back to in-sample |
| **Stratification impossible** | ❌ Class count < n_splits | N/A | Falls back to in-sample |
| **Single sample (n=1)** | ❌ actual_n_splits = 1 | ❌ actual_n_splits = 1 | Falls back to in-sample |
| **Rare class (binary)** | ❌ Minority class < n_splits | N/A | Falls back to in-sample |
| **Multi-class imbalance** | ❌ Any class < n_splits | N/A | Falls back to in-sample |
| **All same class** | ❌ Only 1 unique class | ✅ Works | Classifier: fallback; Regressor: works |

---

## SUMMARY

| Item | Status |
|------|--------|
| **P2 Severity** | HIGH |
| **Thesis Safety** | ❌ UNSAFE |
| **Risk** | Meta-learner overfits, optimistic validation, poor generalization |
| **Fix Required** | YES — raise exception |
| **Test Coverage** | INADEQUATE — verifies execution, not thesis safety |
| **Edge Cases** | HANDLED — triggers fallback, but fallback itself is unsafe |

---

**Recommendation:** Replace fallback logic with exception raising to maintain thesis methodological integrity.
