# Slice 12 Agent Tasks: Agent 1 - Data Pipeline Engineer

## Your Mission
Implement native censoring support in the dataset pipeline. Transform the current approach (where unresolved episodes are silently excluded) into a proper censored data structure.

## Files to Modify

### 1. src/aac_adoption/data/build_dataset.py
**Current Line**: ~98-113 (censoring comment but no implementation)

**Your Tasks**:

1. **Add `event_type` column definition**
   - Location: After `dataset["adopted_in_90d"]` assignment (around line 113)
   - Action: Add event type categorization
   ```python
   # Add event type for survival analysis
   dataset["event_type"] = dataset["outcome_type"].fillna("censored").apply(
       lambda x: "censored" if pd.isna(x) else 
                 "adoption" if str(x).lower() == "adoption" else
                 "transfer" if "transfer" in str(x).lower() else
                 "euthanasia" if "euthanasia" in str(x).lower() else
                 "return_to_owner" if "return" in str(x).lower() else "other"
   )
   ```

2. **Add `is_censored` column**
   - Location: After `event_type` assignment
   - Action: Mark unresolved episodes
   ```python
   dataset["is_censored"] = dataset["event_type"].eq("censored")
   ```

3. **Add `censoring_reason` column**
   - Location: After `is_censored` assignment
   - Action: Document why episodes are censored
   ```python
   dataset["censoring_reason"] = dataset.apply(
       lambda row: "no_outcome" if pd.isna(row["outcome_datetime"]) else
                   "end_of_extract" if row["is_ambiguous_match"] else
                   "ambiguous_match" if row["is_ambiguous_match"] else "unknown",
       axis=1
   )
   ```

4. **Add `followup_days_censored` column**
   - Location: After `censoring_reason` assignment
   - Action: Calculate time at risk (censored or observed)
   ```python
   dataset["followup_days_censored"] = dataset.apply(
       lambda row: row["days_to_outcome"] if not row["is_censored"] else
                   row["followup_days_available"],
       axis=1
   )
   ```

### 2. src/aac_adoption/data/match_records.py
**Current Line**: ~112-115 (unmatched_intakes counter)

**Your Tasks**:

1. **Modify unmatched intakes handling**
   - Location: `else` block around line 114
   - Change: Don't just increment counter, add censored record
   ```python
   else:
       # Add censored record for intakes without future outcome
       row = intake.copy()
       row["outcome_datetime"] = None
       row["outcome_type"] = None
       row["is_censored"] = True
       row["censoring_reason"] = "no_outcome"
       row["event_type"] = "censored"
       row["followup_days_censored"] = (max_date - intake["intake_datetime"]).days
       # Add episode info
       episode_info = episodes_by_time.get(intake["intake_datetime"], {})
       row["episode_number"] = episode_info.get("episode_number", 1)
       row["is_reintake"] = episode_info.get("is_reintake", False)
       row["days_since_last_stay"] = episode_info.get("days_since_last_stay")
       rows.append(row)
       unmatched_intakes += 1
   ```

2. **Ensure max_date is available**
   - Location: Need to pass `extract_end_date` parameter to `match_intakes_to_future_outcomes`
   - Action: Function signature change + max_date calculation

## Critical Acceptance Criteria

✅ **Dataset maintains same row count** (no episodes dropped)  
✅ **Censored episodes marked with `is_censored = True`**  
✅ **Unresolved episodes have `event_type = "censored"`**  
✅ **`censoring_reason` explains WHY each episode is censored**

## Validation Commands

```bash
# Test 1: Censoring columns exist
python -c "
import pandas as pd
df = pd.read_csv('data/modeling_dataset.csv')
assert 'is_censored' in df.columns, 'Missing is_censored column'
assert 'event_type' in df.columns, 'Missing event_type column'
assert 'censoring_reason' in df.columns, 'Missing censoring_reason column'
assert 'followup_days_censored' in df.columns, 'Missing followup_days_censored column'
print('✓ All censoring columns present')
"

# Test 2: Row count unchanged
python -c "
import pandas as pd
intakes = pd.read_csv('data/intakes.csv')
df = pd.read_csv('data/modeling_dataset.csv')
assert len(df) == len(intakes), f'Row count mismatch: {len(df)} vs {len(intakes)}'
print(f'✓ Row count maintained: {len(df)} rows')
"

# Test 3: Censored episodes not dropped
python -c "
import pandas as pd
df = pd.read_csv('data/modeling_dataset.csv')
censored_count = df['is_censored'].sum()
assert censored_count > 0, 'No censored episodes found - check matching logic'
print(f'✓ Censored episodes: {censored_count} ({100*censored_count/len(df):.1f}%)')
"
```

## Key Design Decisions

1. **No episodes dropped**: Unmatched intakes become censored records, not excluded
2. **Event type taxonomy**: Limited to 5 categories (adoption, transfer, euthanasia, return_to_owner, censored)
3. **Censoring reason**: Primary cause is `no_outcome`, secondary is `end_of_extract` for ambiguous matches

## Next Handoff Points

After completing these tasks, you will have produced:
- Modified `build_dataset.py` with censoring columns
- Modified `match_records.py` that preserves all intakes
- Dataset with 100% of original intakes (some censored)

**Hand to Agent 2**: Dataset with censoring columns ready for survival analysis functions.

---

*End of Agent 1 Tasks*
