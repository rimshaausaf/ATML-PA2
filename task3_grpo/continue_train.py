from __future__ import annotations
import argparse, json, torch, numpy as np, gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.optim import AdamW
from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters, reference_mode
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss

def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "reward_model": reward_model, "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts, "optimizer": optimizer}

def _get_logps(logits: torch.Tensor, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    resp_logits = logits[:, prompt_len - 1 : -1, :].float()
    resp_ids = ids[:, prompt_len:]
    return torch.gather(torch.nn.functional.log_softmax(resp_logits, dim=-1), dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)

def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg, tokenizer, policy, reward_model = bundle["cfg"], bundle["tokenizer"], bundle["policy"], bundle["reward_model"]
    reward_tokenizer, prompts, optimizer = bundle["reward_tokenizer"], bundle["prompt_rows"], bundle["optimizer"]
    
    num_updates = updates if updates is not None else int(cfg.get("updates", 20))
    out_path = repo_path(output or cfg["output"])
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = next(policy.parameters()).device
    beta = float(cfg.get("kl_beta", 0.05)) 
    eps = float(cfg.get("clip_epsilon", 0.2))
    K = int(cfg.get("group_size", 8))
    max_prompt_len = int(cfg["max_prompt_length"])
    max_resp_len = int(cfg.get("max_response_length", 768))
    mb_size = 2 # Micro-batch size to prevent CUDA OOM
    
    metrics_history = []
    
    for update_idx in range(num_updates):
        row = prompts[update_idx % len(prompts)]
        msg = prompt_messages(row)
        prompt_str = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer([prompt_str]*K, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len, add_special_tokens=False).to(device)
        p_ids, p_mask = enc["input_ids"], enc["attention_mask"]
        prompt_len = p_ids.shape[1]
        
        with torch.no_grad():
            gen_out = policy.generate(input_ids=p_ids, attention_mask=p_mask, max_new_tokens=max_resp_len, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=1.0)
        
        resp_ids = gen_out[:, prompt_len:]
        B, resp_len = resp_ids.shape
        
        mask = torch.ones((B, resp_len), dtype=torch.float32, device=device)
        full_att_mask = torch.cat([p_mask, torch.ones((B, resp_len), dtype=p_mask.dtype, device=device)], dim=1)
        term_flags = []
        
        for b in range(B):
            eos_pos = (resp_ids[b] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                eos_idx = int(eos_pos[0].item())
                mask[b, eos_idx + 1 :] = 0.0
                full_att_mask[b, prompt_len + eos_idx + 1 :] = 0
                term_flags.append(True)
            else:
                term_flags.append(False)
        
        rm_texts = []
        for b in range(B):
            valid_len = int(mask[b].sum().item())
            r_text = tokenizer.decode(resp_ids[b, :valid_len], skip_special_tokens=True)
            rm_texts.append(reward_tokenizer.apply_chat_template(msg + [{"role": "assistant", "content": r_text}], tokenize=False))
        
        eff_rewards = []
        with torch.no_grad():
            for i in range(0, B, mb_size):
                mb_texts = rm_texts[i:i+mb_size]
                mb_flags = term_flags[i:i+mb_size]
                rm_enc = reward_tokenizer(mb_texts, return_tensors="pt", padding=True, truncation=True, max_length=1280).to(device)
                rm_scores = reward_model(**rm_enc).logits.squeeze(-1).float()
                if rm_scores.dim() == 0: rm_scores = rm_scores.unsqueeze(0)
                eos_pen = float(cfg.get("missing_eos_penalty", 1.0))
                mb_eff = rm_scores - torch.tensor([0.0 if f else eos_pen for f in mb_flags], device=device)
                eff_rewards.append(mb_eff)
                del rm_enc; torch.cuda.empty_cache()
        eff_rewards = torch.cat(eff_rewards)
        
        group_ids = torch.zeros(B, dtype=torch.long, device=device) 
        adv = group_relative_advantages(eff_rewards, group_ids)
        
        optimizer.zero_grad()
        accumulated_stats = defaultdict(float)
        
        for i in range(0, B, mb_size):
            mb_gen_out = gen_out[i:i+mb_size]
            mb_full_att_mask = full_att_mask[i:i+mb_size]
            mb_mask = mask[i:i+mb_size]
            mb_adv = adv[i:i+mb_size]
            scale = mb_gen_out.shape[0] / B
            
            with torch.no_grad():
                with reference_mode(policy):
                    mb_ref_logits = policy(input_ids=mb_gen_out, attention_mask=mb_full_att_mask).logits
                    mb_ref_logps = _get_logps(mb_ref_logits, mb_gen_out, prompt_len) * mb_mask
                    del mb_ref_logits
            
            mb_logits = policy(input_ids=mb_gen_out, attention_mask=mb_full_att_mask).logits
            mb_new_logps = _get_logps(mb_logits, mb_gen_out, prompt_len) * mb_mask
            
            mb_old_logps = mb_new_logps.detach()
            
            mb_loss, mb_stats = grpo_policy_loss(mb_new_logps, mb_old_logps, mb_adv, mb_mask, mb_ref_logps, eps, beta, loss_type=loss_type, max_completion_length=max_resp_len)
            
            (mb_loss * scale).backward()
            
            for k, v in mb_stats.items():
                accumulated_stats[k] += v.item() * scale
                
            del mb_logits, mb_new_logps, mb_old_logps, mb_ref_logps, mb_loss
            torch.cuda.empty_cache()
            
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), 1.0)
        optimizer.step()
        
        uninfo_frac = 1.0 if eff_rewards.std(unbiased=False).item() < 1e-6 else 0.0
        
        metrics = {
            "update": update_idx + 1, "task_reward": eff_rewards.mean().item(),
            "mean_kl": accumulated_stats["sampled_kl"], "policy_loss": accumulated_stats["policy_term"],
            "entropy": accumulated_stats["sample_entropy"], "response_length": mask.sum(dim=-1).mean().item(),
            "clip_fraction": accumulated_stats["clip_fraction"], "uninformative_fraction": uninfo_frac
        }
        metrics_history.append(metrics)
        print(f"[Update {update_idx+1:02d}/{num_updates}] Reward: {metrics['task_reward']:.3f} | KL: {metrics['mean_kl']:.4f} | Loss: {metrics['policy_loss']:.4f} | Len: {metrics['response_length']:.0f}")

    out_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{run_name}_train_metrics.json", "w") as f: json.dump(metrics_history, f, indent=2)
    policy.save_pretrained(str(out_path))
    print(f"Saved {run_name} adapter to {out_path}")
    
    del policy, reward_model; gc.collect(); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)

if __name__ == "__main__":
    main()
