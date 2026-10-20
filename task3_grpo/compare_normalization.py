from __future__ import annotations
import argparse, os
from common.data import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    updates = cfg.get("fork_updates", 8)
    
    print(f"Running Canonical GRPO Fork ({updates} updates)...")
    os.system(f"python -m task3_grpo.continue_train --updates {updates} --loss-type grpo --output outputs/task3_grpo/canonical --run-name canonical")
    
    print(f"Running Dr. GRPO Fork ({updates} updates)...")
    os.system(f"python -m task3_grpo.continue_train --updates {updates} --loss-type dr_grpo --output outputs/task3_grpo/dr_grpo --run-name dr_grpo")

if __name__ == "__main__":
    main()
