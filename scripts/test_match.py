import time
import pandas as pd
from aac_adoption.data.load_data import load_intakes, load_outcomes
from aac_adoption.data.clean_data import clean_intakes, clean_outcomes

def log(msg):
    with open("test_match_debug.log", "a") as f:
        f.write(msg + "\n")
    print(msg)

def main():
    open("test_match_debug.log", "w").close()
    log("Loading data...")
    t0 = time.time()
    intakes = clean_intakes(load_intakes("data/raw/intakes.csv"))
    outcomes = clean_outcomes(load_outcomes("data/raw/outcomes.csv"))
    log(f"Loaded and cleaned in {time.time() - t0:.2f}s")
    
    INTAKE_COLS = ["animal_id", "name", "animal_type", "intake_datetime", "intake_type",
                   "intake_condition", "sex_upon_intake", "age_upon_intake", "breed", "color", "found_location"]
    OUTCOME_COLS = ["animal_id", "outcome_datetime", "outcome_type", "outcome_subtype",
                    "sex_upon_outcome", "age_upon_outcome"]

    intake_subset = intakes[[c for c in INTAKE_COLS if c in intakes.columns]]
    outcome_subset = outcomes[[c for c in OUTCOME_COLS if c in outcomes.columns]]
    
    log("Running verify_reintake_patterns...")
    from aac_adoption.data.match_records import verify_reintake_patterns
    t_v = time.time()
    episodes = verify_reintake_patterns(intake_subset, outcome_subset)
    log(f"verify_reintake_patterns took {time.time() - t_v:.2f}s")
    
    log("Running match_intakes_to_future_outcomes...")
    from aac_adoption.data.match_records import match_intakes_to_future_outcomes
    t1 = time.time()
    matched, unmatched = match_intakes_to_future_outcomes(intake_subset, outcome_subset)
    log(f"Matched in {time.time() - t1:.2f}s")
    log(f"Matched {len(matched)} rows, {unmatched} unmatched.")
    print(f"Matched {len(matched)} rows, {unmatched} unmatched.")

if __name__ == "__main__":
    main()
