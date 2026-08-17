"""
Step 4.2 — LIPO-λ listwise training.

Uses LoRA so the base model is loaded once. The LoRA-wrapped model is the policy;
the frozen base weights (accessed via model.disable_adapter()) are the reference.
This avoids loading two full 7B copies.

Usage:
    python train_listwise.py --config configs/listwise_config.yaml
    python train_listwise.py --config configs/listwise_config.yaml \
        --dataset_path generation/traces/listwise_pairs_small.jsonl \
        --output_dir   outputs/listwise_small
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../generation/code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_jsonl
from listwise_trainer import ListwiseDataset, ListwiseDataCollator, ListwiseTrainer
from wandb_setup import init_wandb


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/listwise_config.yaml")
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
        job_type = "train-listwise",
        config   = cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], torch_dtype=dtype
    )

    # Wrap base model with LoRA for the policy. The ListwiseTrainer uses
    # model.disable_adapter() to get reference log-probs from the same base weights.
    peft_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = cfg.get("lora_r", 16),
        lora_alpha     = cfg.get("lora_alpha", 32),
        lora_dropout   = cfg.get("lora_dropout", 0.05),
        target_modules = "all-linear",  # matches LPOI (lpoi_llava7b.py); SC-DPO full-finetunes
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    data = load_jsonl(cfg["dataset_path"])
    # Ensure at least 1 training sample even on tiny smoke-test datasets
    n_eval = min(max(1, int(len(data) * cfg.get("eval_split", 0.05))), len(data) - 1)
    train_data, eval_data = data[n_eval:], data[:n_eval]

    max_length = cfg.get("max_length", 512)
    train_dataset = ListwiseDataset(train_data, tokenizer, max_length)
    eval_dataset  = ListwiseDataset(eval_data,  tokenizer, max_length)
    collator      = ListwiseDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir                  = cfg["output_dir"],
        learning_rate               = cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size = cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs            = cfg.get("num_train_epochs", 2),
        logging_steps               = 1,
        save_strategy               = "epoch",
        bf16                        = torch.cuda.is_available(),
        fp16                        = False,  # MPS + torch<2.5 doesn't support accelerate fp16
        remove_unused_columns       = False,
        dataloader_pin_memory       = False,
        report_to                   = report_to,  # "wandb" when a key is available, else "none"
    )

    lambdas = tuple(cfg.get("lambdas", [1.0, 0.75, 0.5, 0.25]))

    # ref_model=None signals ListwiseTrainer to use disable_adapter() for reference
    trainer = ListwiseTrainer(
        ref_model     = None,
        beta          = cfg.get("beta", 0.1),
        lambdas       = lambdas,
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = collator,
        tokenizer     = tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Listwise model saved → {cfg['output_dir']}")


if __name__ == "__main__":
    main()
