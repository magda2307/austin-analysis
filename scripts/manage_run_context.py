import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

def get_source_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--profile", choices=["thesis-full", "development-no-shap"], required=True)
    
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-id", required=True)
    
    args = parser.parse_args()
    
    if args.action == "start":
        # Check clean worktree for thesis-full
        if args.profile == "thesis-full":
            try:
                status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
                if status.stdout.strip():
                    print("ERROR: Cannot start thesis-full run with a dirty worktree.")
                    print(status.stdout)
                    sys.exit(1)
            except Exception as e:
                print(f"WARNING: Could not check git status: {e}")
                
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        sha = get_source_sha()
        print(f"AAC_RUN_ID={run_id}")
        print(f"AAC_RUN_PROFILE={args.profile}")
        print(f"AAC_PRODUCER_SOURCE_SHA={sha}")
        print(f"AAC_RECEIPTS_DIR=reports/run_receipts")
        
    elif args.action == "finalize":
        receipts_dir = Path("reports/run_receipts") / args.run_id
        if not receipts_dir.exists():
            print(f"ERROR: No receipts found for run {args.run_id}")
            sys.exit(1)
            
        receipts = list(receipts_dir.glob("*.json"))
        print(f"Run {args.run_id} finalized with {len(receipts)} receipts.")
        # We don't build the manifest here, just print info. The pipeline builds manifest in step 16.

if __name__ == "__main__":
    main()
