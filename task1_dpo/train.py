from __future__ import annotations
import argparse, json, torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss

def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate

def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    all_rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    valid_rows = [r for r in all_rows if len(tokenizer.apply_chat_template(prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True)) < max_len]
    if max_examples is not None: valid_rows = valid_rows[: int(max_examples)]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(valid_rows, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=make_collate(tokenizer, max_len))
    optimizer = AdamW(trainable_parameters(model), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    return {"cfg": cfg, "rows": valid_rows, "tokenizer": tokenizer, "model": model, "loader": loader, "optimizer": optimizer, "beta": float(cfg["beta"] if beta is None else beta)}

def _get_batch_logps(logits, input_ids, response_mask):
    shift_logits, shift_ids, shift_mask = logits[..., :-1, :].contiguous(), input_ids[..., 1:].contiguous(), response_mask[..., 1:].contiguous()
    return (torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_ids.unsqueeze(2)).squeeze(2) * shift_mask).sum(dim=-1)

def run_training(config_path, run_name, dataset_path=None, output_path=None, beta=None, max_examples=None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, optimizer, loader, dpo_beta = bundle["cfg"], bundle["model"], bundle["optimizer"], bundle["loader"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    device, grad_accum_steps = model.device, int(cfg.get("gradient_accumulation_steps", 1))
    model.train()
    metrics_history = []
    
    for step, (c_batch, r_batch) in enumerate(tqdm(loader)):
        c_ids, c_mask, c_rmask = c_batch["input_ids"].to(device), c_batch["attention_mask"].to(device), c_batch["response_mask"].to(device)
        r_ids, r_mask, r_rmask = r_batch["input_ids"].to(device), r_batch["attention_mask"].to(device), r_batch["response_mask"].to(device)
        
        pol_c_logp = _get_batch_logps(model(input_ids=c_ids, attention_mask=c_mask).logits, c_ids, c_rmask)
        pol_r_logp = _get_batch_logps(model(input_ids=r_ids, attention_mask=r_mask).logits, r_ids, r_rmask)
        
        with torch.no_grad(), model.disable_adapter():
            ref_c_logp = _get_batch_logps(model(input_ids=c_ids, attention_mask=c_mask).logits, c_ids, c_rmask)
            ref_r_logp = _get_batch_logps(model(input_ids=r_ids, attention_mask=r_mask).logits, r_ids, r_rmask)
            
        loss, metrics = dpo_loss(pol_c_logp, pol_r_logp, ref_c_logp, ref_r_logp, dpo_beta)
        (loss / grad_accum_steps).backward()
        
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        metrics_history.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})
        
    model.save_pretrained(output)
    with open(output / f"metrics_{run_name}.json", "w") as f: json.dump(metrics_history, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)

if __name__ == "__main__": main()
