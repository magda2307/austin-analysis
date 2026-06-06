import time
import pandas as pd
from aac_adoption.data.build_dataset import build_modeling_dataset_from_files

def main():
    t0 = time.time()
    print("Building dataset...")
    build_modeling_dataset_from_files(
        "data/raw/intakes.csv",
        "data/raw/outcomes.csv",
        "data/processed/modeling_dataset_test.csv"
    )
    print(f"Done in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
