from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
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

    valid_rows = []
    for row in all_rows:
        prompt = prompt_messages_from_preference(row)
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) < max_len:
            valid_rows.append(row)

    if max_examples is not None:
        valid_rows = valid_rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        valid_rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": valid_rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _get_batch_logps(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ids = input_ids[..., 1:].contiguous()
    shift_mask = response_mask[..., 1:].contiguous()

    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1), dim=2, index=shift_ids.unsqueeze(2)
    ).squeeze(2)

    return (per_token_logps * shift_mask).sum(dim=-1)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    dpo_beta = bundle["beta"]
    device = model.device
    grad_accum_steps = int(cfg.get("gradient_accumulation_steps", 1))

    model.train()
    metrics_history = []
    
    print(f"Starting DPO Training: {run_name} (beta={dpo_beta}, examples={len(bundle['rows'])})")

    for step, (chosen_batch, rejected_batch) in enumerate(tqdm(loader)):
        
        chosen_input_ids = chosen_batch["input_ids"].to(device)
        chosen_mask = chosen_batch["attention_mask"].to(device)
        chosen_response_mask = chosen_batch["response_mask"].to(device)

        rejected_input_ids = rejected_batch["input_ids"].to(device)
        rejected_mask = rejected_batch["attention_mask"].to(device)
        rejected_response_mask = rejected_batch["response_mask"].to(device)

        policy_chosen_logits = model(input_ids=chosen_input_ids, attention_mask=chosen_mask).logits
        policy_rejected_logits = model(input_ids=rejected_input_ids, attention_mask=rejected_mask).logits

        policy_chosen_logp = _get_batch_logps(policy_chosen_logits, chosen_input_ids, chosen_response_mask)
        policy_rejected_logp = _get_batch_logps(policy_rejected_logits, rejected_input_ids, rejected_response_mask)

        with torch.no_grad(), model.disable_adapter():
            ref_chosen_logits = model(input_ids=chosen_input_ids, attention_mask=chosen_mask).logits
            ref_rejected_logits = model(input_ids=rejected_input_ids, attention_mask=rejected_mask).logits

            ref_chosen_logp = _get_batch_logps(ref_chosen_logits, chosen_input_ids, chosen_response_mask)
            ref_rejected_logp = _get_batch_logps(ref_rejected_logits, rejected_input_ids, rejected_response_mask)

        loss, metrics = dpo_loss(
            policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, dpo_beta
        )

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        metrics_history.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})

    model.save_pretrained(output)
    with open(output / f"metrics_{run_name}.json", "w") as f:
        json.dump(metrics_history, f)


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


if __name__ == "__main__":
    main()
