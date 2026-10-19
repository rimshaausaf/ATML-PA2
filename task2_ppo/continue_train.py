from __future__ import annotations
import argparse, json, time, torch
from pathlib import Path
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy, mean_response_length
from common.models import (
    load_policy, load_reward_model, load_tokenizer, load_value_model,
    trainable_parameters, value_parameter_groups, reference_mode, token_values
)
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, value_mse_loss, normalize_advantages


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode=cfg.get("value_train_mode", "head_only"))
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]), eps=1e-4)
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
        eps=1e-4,
    )

    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer
    }


def _get_response_logps(logits: torch.Tensor, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    resp_logits = logits[:, prompt_len - 1 : -1, :].float()
    resp_ids = full_ids[:, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(resp_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None: cfg["updates"] = int(updates)
    if clip_epsilon is not None: cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None: cfg["kl_beta"] = float(kl_beta)
    
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    policy, value_model, reward_model, tokenizer, reward_tokenizer = (
        bundle["policy"], bundle["value_model"], bundle["reward_model"], bundle["tokenizer"], bundle["reward_tokenizer"]
    )
    p_opt, v_opt, prompts = bundle["policy_optimizer"], bundle["value_optimizer"], bundle["prompt_rows"]
    
    device = next(policy.parameters()).device
    max_prompt_len = int(cfg["max_prompt_length"])
    max_resp_len = int(cfg["max_response_length"])
    reward_max_len = int(cfg.get("reward_max_length", 1280))
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    v_coef = float(cfg["value_coef"])
    max_norm = float(cfg["max_grad_norm"])
    ppo_epochs = int(cfg["ppo_epochs"])
    eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    metrics_history = []
    t_start = time.time()
    
    for update in range(int(cfg["updates"])):
        step_t0 = time.time()
        policy.eval()
        value_model.eval()
        
        row = prompts[update % len(prompts)]
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
                max_new_tokens=max_resp_len,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
        prompt_len = p_ids.shape[1]
        full_input_ids = gen_out
        resp_ids = full_input_ids[:, prompt_len:]
        resp_len = resp_ids.shape[1]
        
        terminated_with_eos = bool((resp_ids[0] == tokenizer.eos_token_id).any().item())
        resp_step_mask = torch.ones((1, resp_len), dtype=torch.float32, device=device)
        if terminated_with_eos:
            eos_idx = (resp_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            resp_step_mask[0, eos_idx + 1:] = 0.0
            
        full_att_mask = torch.ones_like(full_input_ids, dtype=torch.long, device=device)
        if terminated_with_eos:
            full_att_mask[0, prompt_len + eos_idx + 1:] = 0
            
        with torch.no_grad():
            logits = policy(input_ids=full_input_ids, attention_mask=full_att_mask).logits
            old_logps_token = _get_response_logps(logits, full_input_ids, prompt_len) * resp_step_mask
            
            with reference_mode(policy):
                ref_logits = policy(input_ids=full_input_ids, attention_mask=full_att_mask).logits
                ref_logps_token = _get_response_logps(ref_logits, full_input_ids, prompt_len) * resp_step_mask
                
            all_values = token_values(value_model, full_input_ids, full_att_mask).float()
            values_token = all_values[:, prompt_len - 1 : -1] * resp_step_mask
            
            valid_resp_len = int(resp_step_mask.sum().item())
            resp_text = tokenizer.decode(resp_ids[0, :valid_resp_len], skip_special_tokens=True)
            full_conv_messages = messages + [{"role": "assistant", "content": resp_text}]
            rm_text = reward_tokenizer.apply_chat_template(full_conv_messages, tokenize=False)
            rm_enc = reward_tokenizer(rm_text, return_tensors="pt", truncation=True, max_length=reward_max_len).to(device)
            rm_out = reward_model(**rm_enc)
            raw_reward = float(rm_out.logits[0].item() if hasattr(rm_out, "logits") else rm_out[0].item())
            
            eff_reward = raw_reward - (0.0 if terminated_with_eos else eos_penalty)
            task_reward = torch.tensor([eff_reward], dtype=torch.float32, device=device)
            
            rew_token = shaped_rewards(task_reward, old_logps_token, ref_logps_token, resp_step_mask, beta_kl)
            advantages, returns = compute_gae(rew_token, values_token, resp_step_mask, gamma=gamma, lam=lam)
            norm_advantages = normalize_advantages(advantages, resp_step_mask)
            
            kl_val = float(sampled_kl(old_logps_token, ref_logps_token, resp_step_mask).item())
            ent_val = float(sample_entropy(old_logps_token, resp_step_mask).item())
            
        policy.train()
        value_model.train()
        
        total_policy_loss, total_value_loss, total_clip_frac = 0.0, 0.0, 0.0
        p_grad_norm_val, v_grad_norm_val = 0.0, 0.0
        
        for _ in range(ppo_epochs):
            p_logits = policy(input_ids=full_input_ids, attention_mask=full_att_mask).logits
            new_logps_token = _get_response_logps(p_logits, full_input_ids, prompt_len) * resp_step_mask
            
            p_loss, _, clip_frac = ppo_policy_loss(
                new_logps_token, old_logps_token.detach(), norm_advantages.detach(), resp_step_mask, eps=eps
            )
            
            p_opt.zero_grad()
            p_loss.backward()
            p_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm)
            p_opt.step()
            
            pred_all_vals = token_values(value_model, full_input_ids, full_att_mask).float()
            pred_vals = pred_all_vals[:, prompt_len - 1 : -1] * resp_step_mask
            v_loss = v_coef * value_mse_loss(pred_vals, returns.detach(), resp_step_mask)
            
            v_opt.zero_grad()
            v_loss.backward()
            v_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in value_model.parameters() if p.requires_grad], max_norm
            )
            v_opt.step()
            
            total_policy_loss += float(p_loss.item())
            total_value_loss += float(v_loss.item())
            total_clip_frac += float(clip_frac.item())
            p_grad_norm_val = float(p_grad_norm.item() if isinstance(p_grad_norm, torch.Tensor) else p_grad_norm)
            v_grad_norm_val = float(v_grad_norm.item() if isinstance(v_grad_norm, torch.Tensor) else v_grad_norm)
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        
        step_metrics = {
            "update": update + 1,
            "raw_reward": raw_reward,
            "task_reward": eff_reward,
            "kl": kl_val,
            "entropy": ent_val,
            "policy_loss": total_policy_loss / ppo_epochs,
            "value_loss": total_value_loss / ppo_epochs,
            "clip_fraction": total_clip_frac / ppo_epochs,
            "grad_norm": p_grad_norm_val,
            "value_grad_norm": v_grad_norm_val,
            "response_length": float(mean_response_length(resp_step_mask)),
            "terminated_with_eos": terminated_with_eos,
            "step_time_sec": time.time() - step_t0,
            "wall_clock_sec": time.time() - t_start,
            "peak_vram_mb": peak_vram,
        }
        metrics_history.append(step_metrics)
        print(
            f"[Update {update+1:02d}/{cfg['updates']}] "
            f"Reward: {eff_reward:.3f} (Raw: {raw_reward:.3f}) | "
            f"KL: {kl_val:.4f} | Ent: {ent_val:.3f} | "
            f"PLoss: {step_metrics['policy_loss']:.4f} | VLoss: {step_metrics['value_loss']:.4f} | "
            f"ClipFrac: {step_metrics['clip_fraction']:.3f} | GradNorm: {p_grad_norm_val:.3f} | "
            f"Len: {step_metrics['response_length']:.0f}"
        )
        
    policy.save_pretrained(out)
    with open(out / f"metrics_{run_name}.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    with open(results_dir / f"{run_name}_train_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"PPO training complete in {time.time() - t_start:.1f}s. Adapter saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
