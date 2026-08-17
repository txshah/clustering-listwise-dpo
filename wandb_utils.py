"""
Shared W&B setup for training and evaluation scripts.

Auth resolution order (no code changes needed on a new machine):
  1. WANDB_API_KEY env var
  2. ~/.netrc entry for api.wandb.ai  (what `wandb login` writes)
  3. neither → falls back to offline mode so a run never blocks on a login prompt

Import from anywhere in the repo:
    import sys, os
    sys.path.insert(0, <repo root>)
    from wandb_utils import init_wandb, log_metrics, finish
"""

import os

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "clustering-listwise-dpo")


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
    try:
        with open(netrc_path) as f:
            return "api.wandb.ai" in f.read()
    except OSError:
        return False


def resolve_mode(requested: str | None = None) -> str:
    """online / offline / disabled — explicit request wins, then env, then creds."""
    if requested:
        return requested
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if _has_credentials() else "offline"


def init_wandb(
    project: str | None = None,
    run_name: str | None = None,
    config: dict | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
):
    """
    Returns the wandb run, or None if wandb is unavailable/disabled.
    Never raises — a broken W&B setup must not kill a multi-hour job.
    """
    mode = resolve_mode(mode)
    if mode == "disabled":
        print("[wandb] disabled")
        return None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed — skipping logging")
        return None

    try:
        run = wandb.init(
            project  = project or DEFAULT_PROJECT,
            name     = run_name,
            config   = config or {},
            group    = group,
            job_type = job_type,
            tags     = tags,
            mode     = mode,
        )
    except Exception as e:  # network hiccup, bad key, etc.
        print(f"[wandb] init failed ({e}) — continuing without logging")
        return None

    print(f"[wandb] mode={mode} project={run.project} run={run.name} url={run.url}")
    return run


def log_metrics(run, metrics: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        print(f"[wandb] log failed ({e})")


def log_summary(run, summary: dict) -> None:
    if run is None:
        return
    try:
        for k, v in summary.items():
            run.summary[k] = v
    except Exception as e:
        print(f"[wandb] summary update failed ({e})")


def finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def add_wandb_args(parser, default_job_type: str | None = None):
    """Attach the standard --wandb_* flags to an argparse parser."""
    parser.add_argument("--wandb_project",  default=None, help=f"W&B project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name (default: auto)")
    parser.add_argument("--wandb_group",    default=None, help="W&B group, e.g. 'gsm8k-paper-eval'")
    parser.add_argument("--wandb_job_type", default=default_job_type)
    parser.add_argument("--wandb_tags",     default=None, help="Comma-separated tags")
    parser.add_argument("--wandb_mode",     default=None,
                        choices=["online", "offline", "disabled"],
                        help="Override W&B mode (default: online if logged in, else offline)")
    parser.add_argument("--no_wandb", action="store_true", help="Shorthand for --wandb_mode disabled")
    return parser


def mode_from_args(args) -> str:
    return "disabled" if getattr(args, "no_wandb", False) else resolve_mode(getattr(args, "wandb_mode", None))


def tags_from_args(args) -> list[str] | None:
    raw = getattr(args, "wandb_tags", None)
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else None
