from __future__ import annotations
import argparse, json, torch
from tqdm import tqdm
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer

def _get_batch_logps(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ids = input_ids[..., 1:].contiguous()
    shift_mask = response_mask[..., 1:].contiguous()
    per_token_logps = torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_ids.unsqueeze(2)).squeeze(2)
    return (per_token_logps * shift_mask).sum(dim=-1)

def evaluate(config_path: str, adapter_path: str, run_name: str):
    cfg = load_yaml(config_path)
    eval_path = cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(eval_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.padding_side = "left"
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    policy.eval()
    device = policy.device
    reward_model, _ = load_reward_model(cfg)
    reward_model.eval()
    max_len = int(cfg["max_sequence_length"])
    max_gen_len = 256
    
    results = {"preference_accuracy": 0.0, "mean_kl": 0.0, "mean_reward": 0.0, "mean_length": 0.0, "generations": []}
    total_margin_correct, total_kl, total_reward, total_length, valid_count = 0, 0.0, 0.0, 0, 0
    
    for row in tqdm(rows, desc=f"Evaluating {run_name}"):
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        try:
            c_ids, c_mask = encode_prompt_response(tokenizer, prompt, yc, max_len)
            r_ids, r_mask = encode_prompt_response(tokenizer, prompt, yr, max_len)
        except ValueError:
            continue
            
        batch = pad_batch(tokenizer, [(c_ids, c_mask), (r_ids, r_mask)])
        input_ids = batch["input_ids"].to(device)
        att_mask = batch["attention_mask"].to(device)
        resp_mask = batch["response_mask"].to(device)
        
        with torch.no_grad():
            policy_logits = policy(input_ids=input_ids, attention_mask=att_mask).logits
            policy_logps = _get_batch_logps(policy_logits, input_ids, resp_mask)
            with policy.disable_adapter():
                ref_logits = policy(input_ids=input_ids, attention_mask=att_mask).logits
                ref_logps = _get_batch_logps(ref_logits, input_ids, resp_mask)
                
        m_theta = (policy_logps[0] - ref_logps[0]) - (policy_logps[1] - ref_logps[1])
        if m_theta.item() > 0:
            total_margin_correct += 1
            
        prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        gen_inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
        if gen_inputs["input_ids"].shape[1] >= max_len:
            continue
            
        with torch.no_grad():
            gen_outputs = policy.generate(**gen_inputs, max_new_tokens=max_gen_len, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
        prompt_len = gen_inputs["input_ids"].shape[1]
        gen_only_ids = gen_outputs[0][prompt_len:]
        gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        gen_length = len(gen_only_ids)
        total_length += gen_length
        
        gen_att_mask = (gen_outputs != tokenizer.eos_token_id).long()
        gen_resp_mask = torch.zeros_like(gen_outputs, dtype=torch.float32)
        gen_resp_mask[0, prompt_len:] = 1.0 
        
        with torch.no_grad():
            pol_gen_logits = policy(input_ids=gen_outputs, attention_mask=gen_att_mask).logits
            pol_gen_logps = _get_batch_logps(pol_gen_logits, gen_outputs, gen_resp_mask)[0]
            with policy.disable_adapter():
                ref_gen_logits = policy(input_ids=gen_outputs, attention_mask=gen_att_mask).logits
                ref_gen_logps = _get_batch_logps(ref_gen_logits, gen_outputs, gen_resp_mask)[0]
                
        kl_div = (pol_gen_logps - ref_gen_logps).item()
        total_kl += kl_div
        
        with torch.no_grad():
            rm_out = reward_model(input_ids=gen_outputs, attention_mask=gen_att_mask)
            rm_score = rm_out.logits[0].item() if hasattr(rm_out, 'logits') else rm_out[0].item()
            
        total_reward += rm_score
        valid_count += 1
        results["generations"].append({"prompt": prompt_str, "generated": gen_text, "reward": rm_score, "kl": kl_div, "length": gen_length, "margin": m_theta.item()})
        
    if valid_count > 0:
        results["preference_accuracy"] = total_margin_correct / valid_count
        results["mean_kl"] = total_kl / valid_count
        results["mean_reward"] = total_reward / valid_count
        results["mean_length"] = total_length / valid_count
        
    print(f"\n--- Metrics for {run_name} ---")
    print(f"Preference Accuracy: {results['preference_accuracy']:.4f}")
    print(f"Mean KL Divergence:  {results['mean_kl']:.4f}")
    print(f"Mean RM Score:       {results['mean_reward']:.4f}")
    print(f"Mean Length:         {results['mean_length']:.1f}")
    
    out_dir = repo_path("results") / "task1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_name}_eval.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name)

if __name__ == "__main__":
    main()
