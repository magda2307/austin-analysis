import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aac_adoption.run_receipt_validation import validate_run_receipts

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AAC pipeline run receipts.")
    parser.add_argument("--run-id", type=str, help="Specific run ID to validate")
    parser.add_argument("--receipts-dir", type=str, help="Base directory for receipts")
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Allow the in-progress overall receipt when validation runs inside the pipeline.",
    )
    
    args = parser.parse_args()
    
    is_valid = validate_run_receipts(
        args.run_id,
        args.receipts_dir,
        allow_running=args.allow_running,
    )
    
    if not is_valid:
        print("Run receipts validation failed.")
        sys.exit(1)
        
    print("Run receipts validation successful.")
    sys.exit(0)

if __name__ == "__main__":
    main()
