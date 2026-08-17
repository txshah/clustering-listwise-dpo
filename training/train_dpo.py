"""
Step 4.1 — DPO training on binary (chosen, rejected) pairs.

Uses LoRA (PEFT) so the base model is loaded once and shared as the reference.
Without LoRA, DPO needs two full model copies (~28GB for 7B) which OOMs on MPS.
With LoRA, only the adapter weights are trained (~few hundred MB overhead).

Usage:
    python train_dpo.py --config configs/dpo_config.yaml
    python train_dpo.py --config configs/dpo_config.yaml \
        --dataset_path generation/traces/dpo_pairs_small.jsonl \
        --output_dir   outputs/dpo_small
"""

import argparse
import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../generation/code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_jsonl
from wandb_setup import init_wandb


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_to_strings(data: list[dict]) -> list[dict]:
    """
    Convert message-list format → plain strings that TRL 0.9.6 expects.
    build_pairs.py outputs: {chosen: [messages], rejected: [messages]}
    TRL 0.9.6 expects:      {prompt: str, chosen: str, rejected: str}
    """
    flat = []
    for item in data:
        # chosen[:-1] = prompt turns, chosen[-1] = response
        prompt_turns   = item["chosen"][:-1]
        chosen_turn    = item["chosen"][-1]["content"]
        rejected_turn  = item["rejected"][-1]["content"]
        # Simple prompt string — base model has no chat template
        prompt_str = " ".join(t["content"] for t in prompt_turns)
        flat.append({"prompt": prompt_str, "chosen": chosen_turn, "rejected": rejected_turn})
    return flat


def build_hf_dataset(jsonl_path: str, test_split: float = 0.05) -> tuple:
    data = flatten_to_strings(load_jsonl(jsonl_path))
    n_eval = min(max(1, int(len(data) * test_split)), len(data) - 1)
    train_data, eval_data = data[n_eval:], data[:n_eval]
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/dpo_config.yaml")
    parser.add_argument("--dataset_path", default=None, help="Override config dataset_path")
    parser.add_argument("--output_dir",   default=None, help="Override config output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_path:
        cfg["dataset_path"] = args.dataset_path
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # Start tracking before the slow parts so aborted runs still show up
    report_to = init_wandb(
        run_name = os.path.basename(cfg["output_dir"].rstrip("/")),
        job_type = "train-dpo",
        config   = cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float16 for MPS, bfloat16 for CUDA
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], torch_dtype=dtype
    )

    # LoRA: train only small adapter weights, base model shared as frozen reference.
    # r=16, alpha=32 are standard starting values — tune up for the full run.
    peft_config = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = cfg.get("lora_r", 16),
        lora_alpha   = cfg.get("lora_alpha", 32),
        lora_dropout = cfg.get("lora_dropout", 0.05),
        target_modules = "all-linear",  # matches LPOI (lpoi_llava7b.py); SC-DPO full-finetunes
    )

    train_dataset, eval_dataset = build_hf_dataset(cfg["dataset_path"])

    training_args = DPOConfig(
        output_dir                  = cfg["output_dir"],
        beta                        = cfg.get("beta", 0.1),
        learning_rate               = cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size = cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs            = cfg.get("num_train_epochs", 2),
        max_length                  = cfg.get("max_length", 512),
        max_prompt_length           = cfg.get("max_prompt_length", 256),
        logging_steps               = 1,
        save_strategy               = "epoch",
        bf16                        = torch.cuda.is_available(),
        fp16                        = False,  # MPS + torch<2.5 doesn't support accelerate fp16
        remove_unused_columns       = False,
        report_to                   = report_to,  # "wandb" when a key is available, else "none"
    )

    trainer = DPOTrainer(
        model         = model,
        ref_model     = None,     # None + peft_config = base weights used as reference
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        tokenizer     = tokenizer,
        peft_config   = peft_config,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"DPO model saved → {cfg['output_dir']}")


if __name__ == "__main__":
    main()
