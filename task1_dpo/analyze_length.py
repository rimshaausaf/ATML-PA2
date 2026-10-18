from __future__ import annotations
import argparse, json, torch
from collections import defaultdict
from tqdm import tqdm
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.models import load_policy, load_tokenizer

def _get_batch_logps(logits, input_ids, response_mask):
    shift_logits, shift_ids, shift_mask = logits[..., :-1, :].contiguous(), input_ids[..., 1:].contiguous(), response_mask[..., 1:].contiguous()
    return (torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_ids.unsqueeze(2)).squeeze(2) * shift_mask).sum(dim=-1)

def eval_length(model, tokenizer, eval_path, prompts_path, max_len, device, name):
    strata_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in tqdm(read_jsonl(eval_path), desc=f"Evaluating Strata for {name}"):
        try:
            c_ids, c_mask = encode_prompt_response(tokenizer, prompt_messages_from_preference(row), preference_responses(row)[0], max_len)
            r_ids, r_mask = encode_prompt_response(tokenizer, prompt_messages_from_preference(row), preference_responses(row)[1], max_len)
        except ValueError: continue
        batch = pad_batch(tokenizer, [(c_ids, c_mask), (r_ids, r_mask)])
        input_ids, att_mask, resp_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["response_mask"].to(device)
        with torch.no_grad():
            pol_logps = _get_batch_logps(model(input_ids=input_ids, attention_mask=att_mask).logits, input_ids, resp_mask)
            with model.disable_adapter(): ref_logps = _get_batch_logps(model(input_ids=input_ids, attention_mask=att_mask).logits, input_ids, resp_mask)
        m_theta = (pol_logps[0] - ref_logps[0]) - (pol_logps[1] - ref_logps[1])
        strata_stats[row["length_stratum"]]["total"] += 1
        if m_theta.item() > 0: strata_stats[row["length_stratum"]]["correct"] += 1
            
    limit_compliance, total_limits, generated_lengths = 0, 0, []
    for row in tqdm(read_jsonl(prompts_path), desc=f"Evaluating Compliance for {name}"):
        gen_inputs = tokenizer(tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad(): gen_outputs = model.generate(**gen_inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        gen_only_ids = gen_outputs[0][gen_inputs["input_ids"].shape[1]:]
        generated_lengths.append(len(gen_only_ids))
        total_limits += 1
        if len(tokenizer.decode(gen_only_ids, skip_special_tokens=True).split()) <= 40: limit_compliance += 1

    return {
        "strata_accuracy": {k: (v["correct"] / v["total"] if v["total"] > 0 else 0) for k, v in strata_stats.items()},
        "word_limit_compliance_rate": limit_compliance / total_limits if total_limits > 0 else 0,
        "mean_generation_length": sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    cfg = load_yaml(ap.parse_args().config)
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.padding_side = "left"
    results = {}
    
    for name, adapter in [("Standard", cfg["standard_output"]), ("Length-Balanced", cfg["length_output"])]:
        print(f"\n--- Loading {name} DPO Model ---")
        model = load_policy(cfg, adapter_path=adapter, trainable=False).eval()
        results[name] = eval_length(model, tokenizer, cfg["paths"]["dpo_length_eval"], cfg["paths"]["word_limit_prompts"], int(cfg["max_sequence_length"]), model.device, name)
        del model; torch.cuda.empty_cache()
    
    for model_name, metrics in results.items():
        print(f"\n[{model_name} Model]\nPreference Accuracy by Stratum:")
        for stratum, acc in metrics["strata_accuracy"].items(): print(f"  - {stratum}: {acc:.4f}")
        print(f"Word-Limit Compliance (<= 40 words): {metrics['word_limit_compliance_rate']:.2%}\nMean Generated Tokens: {metrics['mean_generation_length']:.1f}")
        
    out_file = repo_path("results") / "task1" / "length_confounding.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__": main()
