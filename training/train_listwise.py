"""
Step 4.2 — LIPO-λ listwise training.

Uses the custom ListwiseTrainer from listwise_trainer.py.
Input data is the listwise_pairs.jsonl produced by build_listwise.py.

Usage:
    python train_listwise.py --config configs/listwise_config.yaml
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../generation/code"))
from utils import load_jsonl
from listwise_trainer import ListwiseDataset, ListwiseDataCollator, ListwiseTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/listwise_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (trained)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], torch_dtype=torch.bfloat16
    )

    # Reference model (frozen SFT checkpoint — same weights at start)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], torch_dtype=torch.bfloat16
    )

    data = load_jsonl(cfg["dataset_path"])
    split = max(1, int(len(data) * cfg.get("eval_split", 0.05)))
    train_data, eval_data = data[split:], data[:split]

    max_length = cfg.get("max_length", 1024)
    train_dataset = ListwiseDataset(train_data, tokenizer, max_length)
    eval_dataset  = ListwiseDataset(eval_data,  tokenizer, max_length)
    collator      = ListwiseDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir                  = cfg["output_dir"],
        learning_rate               = cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size = cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs            = cfg.get("num_train_epochs", 2),
        logging_steps               = 10,
        save_strategy               = "epoch",
        bf16                        = torch.cuda.is_available(),
        remove_unused_columns       = False,
        dataloader_pin_memory       = False,
    )

    lambdas = tuple(cfg.get("lambdas", [1.0, 0.75, 0.5, 0.25]))

    trainer = ListwiseTrainer(
        ref_model         = ref_model,
        beta              = cfg.get("beta", 0.1),
        lambdas           = lambdas,
        model             = model,
        args              = training_args,
        train_dataset     = train_dataset,
        eval_dataset      = eval_dataset,
        data_collator     = collator,
        tokenizer         = tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Listwise model saved → {cfg['output_dir']}")


if __name__ == "__main__":
    main()
