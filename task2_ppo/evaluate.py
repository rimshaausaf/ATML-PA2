from __future__ import annotations
import argparse, json, numpy as np, torch, gc
from tqdm import tqdm
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.metrics import sampled_kl, sample_entropy, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode

def _get_response_logps(logits: torch.Tensor, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    resp_logits = logits[:, prompt_len - 1 : -1, :].float()
    resp_ids = full_ids[:, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(resp_logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)

def evaluate_adapter(config_path: str, adapter: str, name: str, batch_size: int = 8):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    tokenizer = load_tokenizer(cfg["base_model"], padding_side="left")
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    device = next(policy.parameters()).device
    max_prompt_len = int(cfg["max_prompt_length"])
    eval_max_len = int(cfg.get("eval_max_response_length", 768))
    reward_max_len = int(cfg.get("reward_max_length", 1280))
    eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))

    policy.eval()
    reward_model.eval()
    raw_rewards, eff_rewards, kls, ents, lengths, eos_flags, compliances, examples = [], [], [], [], [], [], [], []

    for start_idx in tqdm(range(0, len(rows), batch_size), desc=f"Evaluating {name}"):
        batch_rows = rows[start_idx : start_idx + batch_size]
        batch_msgs = [prompt_messages(r) for r in batch_rows]
        prompt_strs = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]

        enc = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len, add_special_tokens=False).to(device)
        p_ids, p_mask = enc["input_ids"], enc["attention_mask"]
        prompt_len = p_ids.shape[1]

        with torch.no_grad():
            gen_out = policy.generate(input_ids=p_ids, attention_mask=p_mask, max_new_tokens=eval_max_len, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)
            resp_ids_batch = gen_out[:, prompt_len:]
            B, resp_len = resp_ids_batch.shape

            mask = torch.ones((B, resp_len), dtype=torch.float32, device=device)
            full_att_mask = torch.cat([p_mask, torch.ones((B, resp_len), dtype=p_mask.dtype, device=device)], dim=1)
            term_flags = []

            for b in range(B):
                eos_pos = (resp_ids_batch[b] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    eos_idx = int(eos_pos[0].item())
                    mask[b, eos_idx + 1 :] = 0.0
                    full_att_mask[b, prompt_len + eos_idx + 1 :] = 0
                    term_flags.append(True)
                else:
                    term_flags.append(False)

            logits = policy(input_ids=gen_out, attention_mask=full_att_mask).logits
            policy_logps = _get_response_logps(logits, gen_out, prompt_len) * mask
            del logits
            torch.cuda.empty_cache()

            with reference_mode(policy):
                ref_logits = policy(input_ids=gen_out, attention_mask=full_att_mask).logits
                ref_logps = _get_response_logps(ref_logits, gen_out, prompt_len) * mask
            del ref_logits
            torch.cuda.empty_cache()

            rm_texts, resp_texts = [], []
            for b in range(B):
                valid_len = int(mask[b].sum().item())
                r_text = tokenizer.decode(resp_ids_batch[b, :valid_len], skip_special_tokens=True)
                resp_texts.append(r_text)
                rm_texts.append(reward_tokenizer.apply_chat_template(batch_msgs[b] + [{"role": "assistant", "content": r_text}], tokenize=False))

            rm_enc = reward_tokenizer(rm_texts, return_tensors="pt", padding=True, truncation=True, max_length=reward_max_len).to(device)
            rm_scores = reward_model(**rm_enc).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(rm_scores, float): rm_scores = [rm_scores]

            for b in range(B):
                raw_rw = float(rm_scores[b])
                eff_rw = raw_rw - (0.0 if term_flags[b] else eos_penalty)
                b_mask = mask[b : b + 1]
                kl = float(sampled_kl(policy_logps[b : b + 1], ref_logps[b : b + 1], b_mask).item())
                ent = float(sample_entropy(policy_logps[b : b + 1], b_mask).item())
                length = float(b_mask.sum().item())
                prompt_text = batch_msgs[b][-1]["content"] if batch_msgs[b] else ""
                comp = word_limit_compliance(prompt_text, resp_texts[b])

                raw_rewards.append(raw_rw); eff_rewards.append(eff_rw); kls.append(kl); ents.append(ent); lengths.append(length); eos_flags.append(float(term_flags[b]))
                if comp is not None: compliances.append(comp)
                examples.append({"index": start_idx + b, "prompt": prompt_text, "response": resp_texts[b], "raw_reward": raw_rw, "effective_reward": eff_rw, "kl": kl, "entropy": ent, "response_length": length, "terminated_with_eos": term_flags[b]})

            del gen_out, p_ids, p_mask, enc, full_att_mask, policy_logps, ref_logps, rm_enc, mask
            torch.cuda.empty_cache()

    lengths_arr = np.asarray(lengths, dtype=float)
    metrics = {
        "name": name, "mean_reward": float(np.mean(raw_rewards)), "std_reward": float(np.std(raw_rewards)),
        "mean_effective_reward": float(np.mean(eff_rewards)), "mean_kl": float(np.mean(kls)), "mean_entropy": float(np.mean(ents)),
        "mean_length": float(np.mean(lengths_arr)), "std_length": float(np.std(lengths_arr)),
        "iqr_length": float(np.percentile(lengths_arr, 75) - np.percentile(lengths_arr, 25)),
        "eos_termination_rate": float(np.mean(eos_flags)), "word_limit_compliance": float(np.mean(compliances)) if compliances else None,
    }
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}_eval.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(results_dir / f"{name}_examples.jsonl", "w") as f:
        for ex in examples: f.write(json.dumps(ex) + "\n")
    print(f"\n--- {name} ---", json.dumps(metrics, indent=2))
    
    del policy, reward_model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.batch_size)

if __name__ == "__main__":
    main()
