from __future__ import annotations

import argparse
from common.data import load_yaml
from task2_ppo.continue_train import run_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    kl_values = cfg["kl_values"]
    fork_updates = cfg["fork_updates"]
    
    print(f"Starting KL Beta Ablation over conditions: {kl_values} with fork budget: {fork_updates} updates each.")
    
    for beta in kl_values:
        run_name = f"kl_beta_{beta}"
        output_path = f"outputs/task2_ppo/{run_name}"
        print(f"\n--- Running PPO fork for KL Beta = {beta} ---")
        run_ppo(
            config_path=args.config,
            output=output_path,
            updates=fork_updates,
            kl_beta=float(beta),
            run_name=run_name
        )
        
    print("All KL beta ablation runs completed successfully!")


if __name__ == "__main__":
    main()
