from __future__ import annotations
import argparse, json, numpy as np
from collections import defaultdict
from common.data import load_yaml, read_jsonl, repo_path

def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad: raise ValueError(f"Expected K=8; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt

def regroup_equal_generation_budget(by_prompt, k: int):
    groups = []
    for pid, completions in by_prompt.items():
        for i in range(0, 8, k):
            groups.append(completions[i:i+k])
    return groups

def analyze_subset(groups):
    if not groups: return {"informative_group_rate": 0.0, "mean_within_group_std": 0.0, "relative_signal_variance": 0.0}
    stds = [np.std([c.get("effective_reward", c.get("reward", c.get("score", 0.0))) for c in g]) for g in groups]
    info_rate = sum(1 for s in stds if s > 1e-6) / len(groups)
    return {
        "informative_group_rate": float(info_rate),
        "mean_within_group_std": float(np.mean(stds)),
        "relative_signal_variance": float(np.var(stds))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    
    easy_prompts, hard_prompts = {}, {}
    for pid, completions in by_prompt.items():
        avg_rew = np.mean([c.get("effective_reward", c.get("reward", c.get("score", 0.0))) for c in completions])
        if avg_rew > 0: easy_prompts[pid] = completions
        else: hard_prompts[pid] = completions
        
    results = {}
    for k in cfg.get("group_sizes", [2, 4, 8]):
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        easy_groups = regroup_equal_generation_budget(easy_prompts, k)
        hard_groups = regroup_equal_generation_budget(hard_prompts, k)
        
        results[str(k)] = {
            "all": analyze_subset(all_groups),
            "easy": analyze_subset(easy_groups),
            "hard": analyze_subset(hard_groups)
        }
    
    out_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "group_size_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Binned {len(easy_prompts)} Easy prompts and {len(hard_prompts)} Hard prompts.")
    print("✅ Group size analysis with difficulty bins complete!")

if __name__ == "__main__":
    main()
