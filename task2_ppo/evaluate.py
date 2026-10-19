from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.metrics import masked_mean, sampled_kl, sample_entropy, mean_response_length, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, token_values


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward_model": load_reward_model(cfg)[0],
        "reward_tokenizer": load_reward_model(cfg)[1],
    }


def _get_response_logps(logits: torch.Tensor, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    resp_logits = logits[:, prompt_len - 1 : -1, :].float()
    resp_ids = full_ids[:, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(resp_logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    
    bundle = load_evaluation_bundle(args.config, args.adapter)
    cfg, rows, tokenizer, policy, reward_model, reward_tokenizer = (
        bundle["cfg"], bundle["rows"], bundle["tokenizer"], bundle["policy"], bundle["reward_model"], bundle["reward_tokenizer"]
    )
    
    device = next(policy.parameters()).device
    max_prompt_len = int(cfg["max_prompt_length"])
    eval_max_len = int(cfg.get("eval_max_response_length", 768))
    reward_max_len = int(cfg.get("reward_max_length", 1280))
    
    policy.eval()
    reward_model.eval()
    
    rewards, kls, ents, lengths, compliances = [], [], [], [], []
    examples = []
    
    print(f"=== Evaluating PPO Adapter: {args.name} ===")
    for row in tqdm(rows, desc=f"Evaluating {args.name}"):
        messages = prompt_messages(row)
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        p_ids = enc["input_ids"].to(device)
        p_mask = enc["attention_mask"].to(device)
        if p_ids.shape[1] > max_prompt_len:
            p_ids = p_ids[:, -max_prompt_len:]
            p_mask = p_mask[:, -max_prompt_len:]
            
        with torch.no_grad():
            gen_out = policy.generate(
                input_ids=p_ids,
                attention_mask=p_mask,
                max_new_tokens=eval_max_len,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
            prompt_len = p_ids.shape[1]
            seq_ids = gen_out[0]
            resp_ids = seq_ids[prompt_len:]
            resp_len = resp_ids.shape[0]  # Fixed: 1D tensor dimension
            
            terminated_with_eos = bool((resp_ids == tokenizer.eos_token_id).any().item())
            mask = torch.ones((1, resp_len), dtype=torch.float32, device=device)
            if terminated_with_eos:
                eos_idx = (resp_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                mask[0, eos_idx + 1:] = 0.0
                
            full_att_mask = torch.ones_like(gen_out, dtype=torch.long, device=device)
            
            logits = policy(input_ids=gen_out, attention_mask=full_att_mask).logits
            policy_logps = _get_response_logps(logits, gen_out, prompt_len) * mask
            
            with reference_mode(policy):
                ref_logits = policy(input_ids=gen_out, attention_mask=full_att_mask).logits
                ref_logps = _get_response_logps(ref_logits, gen_out, prompt_len) * mask
                
            valid_len = int(mask.sum().item())
            resp_text = tokenizer.decode(resp_ids[:valid_len], skip_special_tokens=True)
            
            full_conv = messages + [{"role": "assistant", "content": resp_text}]
            rm_text = reward_tokenizer.apply_chat_template(full_conv, tokenize=False)
            rm_enc = reward_tokenizer(rm_text, return_tensors="pt", truncation=True, max_length=reward_max_len).to(device)
            rm_out = reward_model(**rm_enc)
            rw = float(rm_out.logits[0].item() if hasattr(rm_out, "logits") else rm_out[0].item())
            
            kl = float(sampled_kl(policy_logps, ref_logps, mask).item())
            ent = float(sample_entropy(policy_logps, mask).item())
            length = float(mask.sum(-1).item())
            
            prompt_text = messages[-1]["content"] if messages else ""
            comp = word_limit_compliance(prompt_text, resp_text)
            
            rewards.append(rw)
            kls.append(kl)
            ents.append(ent)
            lengths.append(length)
            if comp is not None:
                compliances.append(comp)
                
            examples.append({
                "prompt": prompt_text,
                "response": resp_text,
                "reward": rw,
                "kl": kl,
            })

    metrics = {
        "mean_reward": float(torch.tensor(rewards).mean().item()),
        "mean_kl": float(torch.tensor(kls).mean().item()),
        "mean_entropy": float(torch.tensor(ents).mean().item()),
        "mean_length": float(torch.tensor(lengths).mean().item()),
        "word_limit_compliance": float(torch.tensor(compliances).mean().item()) if compliances else None,
    }
    
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    out_json = results_dir / f"{args.name}_eval.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- Metrics for {args.name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Evaluation results saved to {out_json}")


if __name__ == "__main__":
    main()
