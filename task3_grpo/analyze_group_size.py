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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    
    results = {}
    for k in cfg.get("group_sizes", [2, 4, 8]):
        groups = regroup_equal_generation_budget(by_prompt, k)
        stds = [np.std([c["effective_reward"] for c in g]) for g in groups]
        info_rate = sum(1 for s in stds if s > 1e-6) / len(groups)
        
        results[str(k)] = {
            "informative_group_rate": float(info_rate),
            "mean_within_group_std": float(np.mean(stds)),
            "relative_signal_variance": float(np.var(stds))
        }
    
    out_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "group_size_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Group size analysis complete. Saved to results/task3_grpo/group_size_analysis.json")

if __name__ == "__main__":
    main()
