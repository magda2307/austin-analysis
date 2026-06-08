# P2 Risk Validator Findings

**Date:** 2026-06-07  
**Analyst:** Kilo (AI assistant)  
**File Analyzed:** `src/aac_adoption/models/train_boosting.py`  
**Line:** 218  
**Issue:** Permutation importance uses ROC-AUC (`scoring="roc_auc"`) instead of PR-AUC (`scoring="average_precision"`)

---

## Executive Summary

The permutation importance calculation for classification models at line 218 uses `scoring="roc_auc"`, which is inconsistent with the project's primary classification metric (PR-AUC). This creates a **methodological inconsistency** that could affect thesis validity, as permutation importance measures feature stability using a different evaluation metric than the one used for model optimization and final evaluation.

---

## Answer to Questions

### 1. Is using ROC-AUC for permutation importance a thesis safety violation?

**Severity: MEDIUM**  

Yes, this is a thesis methodology issue. The permutation importance should measure feature stability using the same validation-only metric that the model is optimized for. Since the project uses PR-AUC as the primary classification metric (per `tune.py:102`), permutation importance should also use PR-AUC to maintain methodological consistency.

**Reason:** Permutation importance evaluates how much feature shuffling degrades model performance. If the scoring metric doesn't match the optimization metric, the feature importance ranking may not reflect feature importance relative to the chosen optimization objective.

---

### 2. Should permutation importance use `average_precision` (PR-AUC) instead?

**Severity: HIGH**  

**Yes, absolutely.** The evidence shows:

1. **Training objective:** `tune.py:102` uses `average_precision_score` for CatBoost classification CV
2. **Training objective:** `tune.py:181` uses `average_precision_score` for HistGradientBoosting classification CV  
3. **Final evaluation:** `evaluate.py:118` computes PR-AUC as `average_precision_score` for classification metrics
4. **Model selection:** Studies maximize PR-AUC (`direction="maximize"` in `tune.py:107, 186`)

Permutation importance should use `scoring="average_precision"` to align with the PR-AUC optimization objective.

---

### 3. What's the impact of using different scoring for permutation vs model training?

**Severity: MEDIUM to HIGH**

The impact depends on the classification problem characteristics:

| Scenario | Impact |
|----------|--------|
| **Well-calibrated probabilities, balanced classes** | ROC-AUC and PR-AUC may rank features similarly |
| **Class imbalance (typical for AAC adoption)** | **HIGH RISK:** PR-AUC is more informative for imbalanced data; ROC-AUC can be misleading |
| **Threshold-sensitive decision making** | Different metrics may prioritize different features |

**Specific risk for this project:**

The AAC adoption dataset likely has imbalanced classes (few positive adoptions vs many negatives). PR-AUC is specifically designed for imbalanced scenarios and focuses on the positive class performance. Using ROC-AUC for permutation importance could:

- **Underestimate importance** of features that improve precision recalled tradeoff
- **Overestimate importance** of features that only improve TPR with minimal FPR improvement
- Create **inconsistent feature importance** relative to the model's optimization target

---

### 4. Are there other permutation importance calls that need fixing?

**Severity: LOW**  

Current codebase has **two** permutation importance calls:

| File | Line | Task | Current Scoring | Should Be | Status |
|------|------|------|-----------------|-----------|--------|
| `train_boosting.py` | 218 | Classification | `"roc_auc"` | `"average_precision"` | **FIX NEEDED** |
| `train_boosting.py` | 274 | Regression | `"neg_mean_absolute_error"` | No change | OK (already aligned with training objective in tune.py) |

The regression permutation importance uses `neg_mean_absolute_error`, which is consistent with the regression training objective in `tune.py:141` (MAE minimization). No change needed for regression.

---

## Code Evidence

### Current problematic code (train_boosting.py:218)
```python
append_table(
    _permutation_table(
        pipeline,
        split,
        feature_columns,
        "classification_target",
        metadata,
        scoring="roc_auc",  # ❌ INCONSISTENT with training
        repeats=permutation_repeats,
        max_rows=permutation_max_rows,
    ),
    tables_dir / "permutation_importance_classification.csv",
)
```

### Training objective evidence (tune.py:102, 181)
```python
# CatBoost classification CV
scores.append(average_precision_score(y_va, preds))  # PR-AUC used

# HistGradientBoosting classification CV
scores.append(average_precision_score(y_va, preds))  # PR-AUC used
```

### Validation metrics (evaluate.py:118)
```python
metrics["pr_auc"] = average_precision_score(y_true, y_score)  # PR-AUC in final metrics
```

---

## Recommendation

**Immediate fix required at `train_boosting.py:218`:**

Change:
```python
scoring="roc_auc",
```

To:
```python
scoring="average_precision",
```

This ensures permutation importance measures feature stability using the same PR-AUC metric that drives model training and evaluation, maintaining thesis methodological validity.

---

## Thesis Methodology Implications

**Risk: VALIDITY THREAT**

If permutation importance results are used to support thesis conclusions about feature importance:

1. **Inconsistent evaluation:** Feature importance is measured relative to ROC-AUC, but model performance is reported relative to PR-AUC
2. **Reduced reproducibility:** Future readers cannot easily verify that feature importance aligns with reported model metrics
3. **Methodological confusion:** Mixing metrics without justification weakens methodological rigor

**Solution:** Align permutation importance scoring with the project's primary classification metric (PR-AUC) as implemented in `tune.py`.

---

## Files Referenced

| File | Line | Purpose |
|------|------|---------|
| `src/aac_adoption/models/train_boosting.py` | 218 | **CURRENT ISSUE - permutation importance** |
| `src/aac_adoption/models/tune.py` | 102 | CatBoost classification PR-AUC objective |
| `src/aac_adoption/models/tune.py` | 181 | HistGradientBoosting classification PR-AUC objective |
| `src/aac_adoption/models/evaluate.py` | 118 | PR-AUC computation in metrics |
| `src/aac_adoption/models/tune.py` | 107, 186 | Study direction: maximize PR-AUC |

---

*End of findings*
