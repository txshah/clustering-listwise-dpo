"""
GSM8K evaluation — Mistral-paper protocol (8-shot, maj@8) plus pass@k.

Works for the base model, the DPO LoRA adapter, and the listwise LoRA adapter:
just point --model at an HF id or an adapter directory (adapter_config.json is
detected automatically and merged into its base model for fast inference).

One generation pass produces every metric:
    * avg@1   — mean accuracy of an individual sample
    * maj@k   — self-consistency majority vote over the first k samples
                (k=8 → the Mistral 7B paper's "GSM8K 8-shot maj@8")
    * pass@k  — unbiased estimator (Codex eq. 1) over all n samples

So --n_samples 10 gives maj@8, pass@10, pass@5 and pass@1 from a single run.

Usage:
    # Mistral paper number: 8-shot, maj@8, plus pass@10 / pass@5
    python evaluation/eval_gsm8k.py \
        --model outputs/dpo_mistral7b \
        --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 \
        --output results/dpo_paper_eval.json \
        --wandb_group gsm8k-paper-eval

    # cheap greedy sanity check (pass@1, deterministic)
    python evaluation/eval_gsm8k.py --model outputs/dpo_small --greedy --limit 100

Long runs write per-question records incrementally; re-run with --resume to
pick up where an interrupted job stopped.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

# vLLM starts its engine in a subprocess; with the default fork method that dies
# with "Cannot re-initialize CUDA in forked subprocess" if anything in this
# process has already touched CUDA. Must be set before vllm is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wandb_utils import (  # noqa: E402
    init_wandb, log_metrics, log_summary, finish,
    add_wandb_args, mode_from_args, tags_from_args,
)

# Few-shot completions run on until the model starts inventing the next question.
STOP_STRINGS = ["\nQuestion:", "\nQ:", "\n\n\n"]

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


# ── Answer extraction ─────────────────────────────────────────────────────────

def _clean_number(raw: str) -> str:
    return raw.replace(",", "").replace("$", "").rstrip(".").strip()


def extract_answer(text: str) -> str | None:
    """
    Pull the final numeric answer out of a completion.
    Priority: 'The answer is X'  →  '#### X'  →  last number in the text.
    """
    matches = re.findall(rf"[Tt]he answer is\s*({_NUM})", text)
    if matches:
        return _clean_number(matches[-1])

    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return _clean_number(m.group(1))

    numbers = re.findall(_NUM, text)
    return _clean_number(numbers[-1]) if numbers else None


def normalize(answer: str | None) -> str | None:
    """Canonical form so '42', '42.0' and '$42' vote together."""
    if answer is None:
        return None
    try:
        value = float(answer)
    except ValueError:
        return answer.strip()
    return str(int(value)) if value == int(value) else str(value)


try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except ImportError:               # pragma: no cover — math-verify is in uv.lock
    _mv_parse = None


def answers_match(pred: str | None, gold: str | None) -> bool:
    """
    math-verify (the sympy-based checker used by lighteval / Open R1) is the
    primary comparator; the numeric fallback covers anything it can't parse.
    Voting still happens on the normalized extracted strings — maj@k needs a
    canonical answer to count, not just a boolean.
    """
    if pred is None or gold is None:
        return False
    if _mv_parse is not None:
        try:
            return bool(_mv_verify(_mv_parse(gold), _mv_parse(pred)))
        except Exception:
            pass
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k)/C(n, k)  (Chen et al. 2021, eq. 1)."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    prob = 1.0
    for i in range(n - c + 1, n + 1):
        prob *= 1.0 - k / i
    return 1.0 - prob


def majority_answer(preds: list[str | None]) -> str | None:
    """Self-consistency vote. Ties broken by first occurrence (most-common order)."""
    votes = [p for p in preds if p is not None]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    for p in votes:                      # first answer that reached the top count
        if counts[p] == best:
            return p
    return None


def aggregate(records: list[dict], maj_ks: list[int], pass_ks: list[int]) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0}

    metrics = {"total": total, "n_samples": len(records[0]["predictions"])}

    flat = [c for r in records for c in r["sample_correct"]]
    metrics["avg@1"] = sum(flat) / len(flat)

    for k in maj_ks:
        hits = 0
        for r in records:
            if len(r["predictions"]) < k:
                continue
            hits += int(answers_match(majority_answer(r["predictions"][:k]), r["gold"]))
        metrics[f"maj@{k}"] = hits / total

    for k in pass_ks:
        n = min(len(r["sample_correct"]) for r in records)
        if k > n:
            print(f"[warn] skipping pass@{k}: only {n} samples per question")
            continue
        metrics[f"pass@{k}"] = sum(
            pass_at_k(len(r["sample_correct"]), sum(r["sample_correct"]), k) for r in records
        ) / total

    return metrics


# ── Prompting ─────────────────────────────────────────────────────────────────

def build_fewshot_prefix(n_shot: int) -> str:
    """
    Standard GSM8K few-shot prompt: the first n_shot train examples, calculator
    annotations (<<...>>) stripped and '#### N' rewritten as 'The answer is N.'
    Deterministic, so every model is scored against the identical prefix.
    """
    if n_shot <= 0:
        return ""
    train = load_dataset("openai/gsm8k", "main", split="train")
    blocks = []
    for item in train.select(range(n_shot)):
        reasoning, _, final = item["answer"].partition("####")
        reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()
        final = _clean_number(final)
        blocks.append(f"Question: {item['question']}\nAnswer: {reasoning}\nThe answer is {final}.")
    return "\n\n".join(blocks) + "\n\n"


def build_prompt(question: str, prefix: str) -> str:
    return f"{prefix}Question: {question}\nAnswer:"


def truncate_at_stop(text: str) -> str:
    cut = len(text)
    for stop in STOP_STRINGS:
        pos = text.find(stop)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut].strip()


# ── Backends ──────────────────────────────────────────────────────────────────
# Both expose the same contract:
#   .chunk_size        how many questions to hand over at a time
#   .tokenizer         used only to measure the few-shot prefix
#   .generate(prompts) → per prompt, a list of n completions
#
# vLLM is 5-10× faster (continuous batching, paged KV cache) and is the default
# when importable; HF is the always-available fallback.

def resolve_adapter(model_path: str, base_model: str | None, verbose: bool = True) -> tuple[str | None, str]:
    """(adapter_dir, base_or_model_id) — adapter_dir is None for plain checkpoints."""
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg) as f:
            base = base_model or json.load(f)["base_model_name_or_path"]
        if verbose:
            print(f"LoRA adapter detected → base model {base}")
        return model_path, base
    return None, model_path


class HFBackend:
    """transformers.generate with manual left-padded batching."""

    name = "hf"

    def __init__(self, args):
        from transformers import AutoModelForCausalLM

        # Probing CUDA here (not at import) keeps the vLLM path fork-safe.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        adapter_dir, base = resolve_adapter(args.model, args.base_model)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        if adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.merge_and_unload()      # fold LoRA in for full-speed generation
        tok_src = adapter_dir if adapter_dir and os.path.isfile(
            os.path.join(adapter_dir, "tokenizer_config.json")) else base

        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"      # required for batched decoder-only generation

        self.model = model.to(self.device).eval()
        self.args = args
        self.chunk_size = args.batch_size

    def generate(self, prompts: list[str]) -> list[list[str]]:
        args = self.args
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens = args.max_new_tokens,
            num_return_sequences = args.n_samples,
            pad_token_id = self.tokenizer.pad_token_id,
            stop_strings = STOP_STRINGS,
            tokenizer = self.tokenizer,
        )
        if args.greedy:
            gen_kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        completions = self.tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        n = gen_kwargs["num_return_sequences"]
        return [
            [truncate_at_stop(c) for c in completions[i * n : (i + 1) * n]]
            for i in range(len(prompts))
        ]


class VLLMBackend:
    """vLLM with continuous batching. LoRA adapters are served via LoRARequest."""

    name = "vllm"

    def __init__(self, args):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        adapter_dir, base = resolve_adapter(args.model, args.base_model)
        llm_kwargs = dict(
            model = base,
            dtype = "bfloat16",
            gpu_memory_utilization = args.gpu_memory_utilization,
            max_model_len = args.max_model_len,
            seed = args.seed,
        )
        if adapter_dir:
            with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
                rank = json.load(f).get("r", 16)
            llm_kwargs.update(enable_lora=True, max_lora_rank=max(rank, 16))

        self.llm = LLM(**llm_kwargs)
        self.lora_request = LoRARequest("adapter", 1, adapter_dir) if adapter_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(base)

        self.sampling = SamplingParams(
            n = 1 if args.greedy else args.n_samples,
            temperature = 0.0 if args.greedy else args.temperature,
            top_p = 1.0 if args.greedy else args.top_p,
            max_tokens = args.max_new_tokens,
            stop = STOP_STRINGS,
            seed = args.seed,
        )
        # vLLM schedules across the whole call, so chunks only exist to keep the
        # incremental record writing (and therefore --resume) working.
        self.chunk_size = args.vllm_chunk

    def generate(self, prompts: list[str]) -> list[list[str]]:
        outputs = self.llm.generate(
            prompts, self.sampling, lora_request=self.lora_request, use_tqdm=False
        )
        return [[truncate_at_stop(o.text) for o in out.outputs] for out in outputs]


def build_backend(args):
    requested = args.backend
    if requested == "auto":
        try:
            import vllm  # noqa: F401
            requested = "vllm"
        except ImportError:
            requested = "hf"
            print("[backend] vLLM not importable — falling back to transformers "
                  "(run `uv sync` for the fast path)")
    print(f"[backend] {requested}")
    return VLLMBackend(args) if requested == "vllm" else HFBackend(args)


def evaluate(backend, dataset, args, prefix, detail_path, done_idx, run):
    records: list[dict] = []

    # Re-load anything an earlier interrupted run already finished.
    if done_idx:
        with open(detail_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"Resuming: {len(records)} questions already evaluated")

    pending = [(i, item) for i, item in enumerate(dataset) if i not in done_idx]
    detail_file = open(detail_path, "a" if done_idx else "w")
    started = time.time()

    try:
        chunk = backend.chunk_size
        for start in tqdm(range(0, len(pending), chunk), desc="evaluating"):
            batch = pending[start : start + chunk]
            prompts = [build_prompt(item["question"], prefix) for _, item in batch]
            batch_completions = backend.generate(prompts)

            for (idx, item), completions in zip(batch, batch_completions):
                gold = normalize(extract_answer(item["answer"]))
                preds = [normalize(extract_answer(c)) for c in completions]
                sample_correct = [int(answers_match(p, gold)) for p in preds]

                record = {
                    "idx": idx,
                    "question": item["question"],
                    "gold": gold,
                    "predictions": preds,
                    "sample_correct": sample_correct,
                    "completions": completions if args.save_completions else [],
                }
                records.append(record)
                detail_file.write(json.dumps(record) + "\n")
            detail_file.flush()

            if run is not None and records:
                running = aggregate(records, args.maj_k, args.pass_k)
                log_metrics(run, {f"running/{k}": v for k, v in running.items() if k != "total"},
                            step=len(records))
    finally:
        detail_file.close()

    elapsed = time.time() - started
    print(f"Generation finished in {elapsed / 60:.1f} min ({len(pending)} questions this run)")
    return records, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_k_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id, checkpoint dir, or LoRA adapter dir")
    parser.add_argument("--base_model", default=None, help="Override the adapter's base model")
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")

    parser.add_argument("--n_shot", type=int, default=8, help="Few-shot exemplars (paper: 8)")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples per question (>= max k)")
    parser.add_argument("--maj_k", type=parse_k_list, default=[8], help="maj@k values, e.g. 8")
    parser.add_argument("--pass_k", type=parse_k_list, default=[1, 5, 10], help="pass@k values")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true",
                        help="Deterministic single sample — pass@1 only, no maj@k")
    parser.add_argument("--max_new_tokens", type=int, default=400)

    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"],
                        help="auto = vLLM when importable, else transformers")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="HF backend: questions per batch (× n_samples sequences)")
    parser.add_argument("--vllm_chunk", type=int, default=128,
                        help="vLLM backend: questions per generate() call (checkpoint granularity)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="vLLM backend: fraction of VRAM for weights + KV cache")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="vLLM backend: prompt + completion budget (8-shot prefix is ~1150 tokens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_completions", action="store_true",
                        help="Store raw generations in the detail file (large)")
    parser.add_argument("--resume", action="store_true", help="Skip questions already in the detail file")
    add_wandb_args(parser, default_job_type="eval")
    args = parser.parse_args()

    if args.greedy:
        args.n_samples, args.maj_k, args.pass_k = 1, [], [1]

    max_k = max(args.pass_k + args.maj_k + [1])
    if args.n_samples < max_k:
        parser.error(f"--n_samples {args.n_samples} is below the largest requested k ({max_k})")

    # NB: don't probe torch.cuda here — initialising CUDA in this process breaks
    # vLLM's engine subprocess. Each backend picks its own device/dtype.
    torch.manual_seed(args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    detail_path = args.output.replace(".json", "_detail.jsonl")

    done_idx: set[int] = set()
    if args.resume and os.path.isfile(detail_path):
        with open(detail_path) as f:
            done_idx = {json.loads(line)["idx"] for line in f if line.strip()}

    run_name = args.wandb_run_name or (
        f"eval-{os.path.basename(args.model.rstrip('/'))}-"
        f"{args.n_shot}shot-{'greedy' if args.greedy else f'n{args.n_samples}'}"
    )
    run = init_wandb(
        project  = args.wandb_project,
        run_name = run_name,
        config   = vars(args),
        group    = args.wandb_group,
        job_type = args.wandb_job_type,
        tags     = tags_from_args(args),
        mode     = mode_from_args(args),
    )

    print(f"Loading GSM8K ({args.split}) ...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Size the prompt before the engine starts — vLLM's max_model_len is fixed at
    # init, and a prefix longer than it would make every request fail.
    prefix = build_fewshot_prefix(args.n_shot)
    _, base_id = resolve_adapter(args.model, args.base_model, verbose=False)
    prefix_tokens = len(AutoTokenizer.from_pretrained(base_id)(prefix)["input_ids"])
    needed = prefix_tokens + 512 + args.max_new_tokens          # 512 ≈ longest GSM8K question
    if args.max_model_len < needed:
        print(f"[backend] raising --max_model_len {args.max_model_len} → {needed} "
              f"({args.n_shot}-shot prefix is {prefix_tokens} tokens)")
        args.max_model_len = needed
    print(f"{args.n_shot}-shot prefix: {prefix_tokens} tokens")

    print(f"Loading model {args.model} ...")
    backend = build_backend(args)

    records, elapsed = evaluate(backend, dataset, args, prefix, detail_path, done_idx, run)

    metrics = aggregate(records, args.maj_k, args.pass_k)
    metrics["eval_seconds"] = round(elapsed, 1)

    print("\n=== GSM8K results ===")
    print(f"model      : {args.model}")
    print(f"backend    : {backend.name}")
    print(f"setting    : {args.n_shot}-shot, "
          f"{'greedy' if args.greedy else f'{args.n_samples} samples @ T={args.temperature}'}")
    for key, value in metrics.items():
        if key in ("total", "n_samples", "eval_seconds"):
            continue
        print(f"{key:<10}: {value:.4f}")
    print(f"questions  : {metrics['total']}")

    summary = {
        "model": args.model,
        "backend": backend.name,
        "split": args.split,
        "n_shot": args.n_shot,
        "n_samples": args.n_samples,
        "temperature": None if args.greedy else args.temperature,
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nSummary → {args.output}")
    print(f"Per-question records → {detail_path}")

    log_summary(run, dict(metrics) | {"backend": backend.name})
    log_metrics(run, {f"final/{k}": v for k, v in metrics.items() if k != "total"})
    finish(run)


if __name__ == "__main__":
    main()
