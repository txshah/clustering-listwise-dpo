"""
Optional Weights & Biases tracking, shared by the training and eval scripts.

Logging turns on automatically when a W&B API key is available (WANDB_API_KEY
env var or a previous `wandb login`). Without a key everything stays a no-op,
so runs never hang on a login prompt (Colab included). Force off with
WANDB_DISABLED=true.

Project defaults to "clustering-listwise-dpo"; override with WANDB_PROJECT.
"""

import netrc
import os

PROJECT = os.environ.get("WANDB_PROJECT", "clustering-listwise-dpo")


def _api_key_available() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def init_wandb(run_name: str, job_type: str, config: dict) -> str:
    """
    Start a W&B run and return "wandb" (usable as TrainingArguments.report_to).
    Returns "none" without side effects when logging is unavailable.
    HF's WandbCallback reuses an already-initialized run instead of starting
    its own, so calling this before Trainer keeps one run per process.
    """
    if not _api_key_available():
        print("wandb: no API key found - run not logged (set WANDB_API_KEY or `wandb login`)")
        return "none"
    try:
        import wandb
    except ImportError:
        print("wandb: API key found but package missing - `pip install wandb` to log runs")
        return "none"
    wandb.init(project=PROJECT, name=run_name, job_type=job_type, config=config)
    return "wandb"
