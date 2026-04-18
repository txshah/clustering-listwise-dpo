"""
Step 4.1 — DPO training on binary (chosen, rejected) pairs.

Uses TRL's DPOTrainer directly. Input data is the message-list format
produced by build_pairs.py, which apply_chat_template knows how to handle.

Usage:
    python train_dpo.py --config configs/dpo_config.yaml
"""

import argparse
import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../generation/code"))
from utils import load_jsonl


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_hf_dataset(jsonl_path: str, test_split: float = 0.05) -> tuple:
    """
    Loads dpo_pairs.jsonl and returns (train_dataset, eval_dataset).
    Expected schema: {chosen: [messages], rejected: [messages]}
    """
    data = load_jsonl(jsonl_path)
    split = max(1, int(len(data) * test_split))
    train_data, eval_data = data[split:], data[:split]
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], torch_dtype=torch.bfloat16
    )

    train_dataset, eval_dataset = build_hf_dataset(cfg["dataset_path"])

    training_args = DPOConfig(
        output_dir                  = cfg["output_dir"],
        beta                        = cfg.get("beta", 0.1),
        learning_rate               = cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size = cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs            = cfg.get("num_train_epochs", 2),
        max_length                  = cfg.get("max_length", 1024),
        max_prompt_length           = cfg.get("max_prompt_length", 512),
        logging_steps               = 10,
        save_strategy               = "epoch",
        bf16                        = torch.cuda.is_available(),
        remove_unused_columns       = False,
    )

    trainer = DPOTrainer(
        model         = model,
        ref_model     = None,   # TRL creates a frozen copy automatically
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        tokenizer     = tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"DPO model saved → {cfg['output_dir']}")


if __name__ == "__main__":
    main()
