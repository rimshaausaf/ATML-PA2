from __future__ import annotations

import argparse
import json
import torch

from common.data import load_yaml, repo_path
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, normalize_advantages


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eps_values = cfg["clip_values"]
    beta_kl = float(cfg["kl_beta"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])

    results = {}
    print("=== PPO Clipping Analysis ===")
    
    for row in rows:
        old_logp = row["old_logprobs"]
        ref_logp = row["ref_logprobs"]
        if not isinstance(old_logp, torch.Tensor):
            old_logp = torch.tensor(old_logp)
        if not isinstance(ref_logp, torch.Tensor):
            ref_logp = torch.tensor(ref_logp)
            
        steps = len(old_logp)
        mask = torch.ones((1, steps), dtype=torch.float32)
        old_logp = old_logp.unsqueeze(0)
        ref_logp = ref_logp.unsqueeze(0)
        
        task_rew = torch.tensor([row.get("raw_terminal_reward", 1.0)], dtype=torch.float32)
        rew = shaped_rewards(task_rew, old_logp, ref_logp, mask, beta_kl)
        
        vals = row.get("values", torch.zeros_like(old_logp))
        if not isinstance(vals, torch.Tensor):
            vals = torch.tensor(vals).unsqueeze(0)
        else:
            if vals.dim() == 1:
                vals = vals.unsqueeze(0)

        advantages, returns = compute_gae(rew, vals, mask, gamma=gamma, lam=lam)
        norm_adv = normalize_advantages(advantages, mask)
        
        new_logp = old_logp + torch.randn_like(old_logp) * 0.05
        
        row_eps_stats = {}
        for eps in eps_values:
            loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, norm_adv, mask, eps=eps)
            row_eps_stats[str(eps)] = {
                "loss": float(loss.item()),
                "clip_fraction": float(clip_frac.item())
            }
        results[row.get("source_index", 0)] = row_eps_stats

    out_path = repo_path(cfg.get("results_dir", "results/task2_ppo")) / "clipping_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Clipping analysis saved successfully to {out_path}")


if __name__ == "__main__":
    main()
