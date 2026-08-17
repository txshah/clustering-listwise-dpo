"""
Step 1 — Raw trace generation (mirrors lce_solution_gen_gsm8k.py).

For each question in the input JSONL, sample the model N times at temperature=1
(the paper's D_naive setting) and save every raw response.

Backends (same pattern as evaluation/eval_gsm8k.py):
  vllm  — continuous batching, ~15× faster; the default when importable
  hf    — transformers.generate, one question at a time; always available

Traces are appended to the output file as each chunk finishes, so a multi-hour
run can be interrupted and rerun with --resume to pick up where it stopped.

Input JSONL schema  : {"idx": int, "question": str, "answer": str, ...}
Output JSONL schema : same fields + "trace": str  (n_samples rows per question)

Usage:
    uv run generation/code/generate_traces.py \
        --input  generation/traces/questions.jsonl \
        --output generation/traces/raw/raw_traces.jsonl \
        --model  mistralai/Mistral-7B-v0.1 \
        --n_samples 30 --max_new_tokens 512 --resume
"""

import argparse
import json
import os
import time

# vLLM starts its engine in a subprocess; must be set before vllm is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import load_jsonl, format_prompt


# ── Backends ──────────────────────────────────────────────────────────────────

class HFBackend:
    """transformers.generate, one question per call (n_samples sequences each)."""

    name = "hf"

    def __init__(self, args):
        from transformers import AutoModelForCausalLM

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        # MPS doesn't support bfloat16 before macOS 14 — use float16 instead
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        self.model.to(self.device).eval()
        self.args = args
        self.chunk_size = 1                      # one question per generate call

    def generate(self, prompts: list[str]) -> list[list[str]]:
        args = self.args
        out = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,              # ← temperature sampling (D_naive)
                    temperature=args.temperature,
                    num_return_sequences=args.n_samples,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            out.append([
                self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                for seq in output_ids
            ])
        return out


class VLLMBackend:
    """vLLM with continuous batching — the whole chunk is scheduled at once."""

    name = "vllm"

    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model = args.model,
            dtype = "bfloat16",
            gpu_memory_utilization = args.gpu_memory_utilization,
            max_model_len = args.max_model_len,
            seed = args.seed,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.sampling = SamplingParams(
            n = args.n_samples,
            temperature = args.temperature,      # paper: temp=1 for the candidate pool
            top_p = 1.0,                         # pure temperature sampling, like HF defaults
            max_tokens = args.max_new_tokens,
            seed = args.seed,
        )
        # Chunks only set how often traces hit disk (checkpoint/resume granularity).
        self.chunk_size = args.vllm_chunk

    def generate(self, prompts: list[str]) -> list[list[str]]:
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outputs]


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True,  help="Path to questions JSONL")
    parser.add_argument("--output",         required=True,  help="Path to save raw traces JSONL")
    parser.add_argument("--model",          required=True,  help="HF model name or local path")
    parser.add_argument("--n_samples",      type=int, default=8,   help="Traces per question")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions whose traces are already in the output file")

    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    parser.add_argument("--vllm_chunk", type=int, default=128,
                        help="vLLM: questions per generate() call (checkpoint granularity)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=1024,
                        help="vLLM: prompt + completion budget (zero-shot prompts are short)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    questions = load_jsonl(args.input)

    done_idx: set = set()
    if args.resume and os.path.isfile(args.output):
        with open(args.output) as f:
            done_idx = {json.loads(line)["idx"] for line in f if line.strip()}
        print(f"Resuming: {len(done_idx)} questions already have traces")
    pending = [q for q in questions if q["idx"] not in done_idx]

    # Keep the vLLM sequence budget ahead of the longest question + completion.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_prompt = max(
        (len(tokenizer(format_prompt(q["question"], tokenizer))["input_ids"]) for q in pending),
        default=0,
    )
    needed = max_prompt + args.max_new_tokens
    if args.max_model_len < needed:
        print(f"[backend] raising --max_model_len {args.max_model_len} → {needed} "
              f"(longest prompt is {max_prompt} tokens)")
        args.max_model_len = needed

    print(f"Loading model {args.model} ...")
    backend = build_backend(args)

    print(f"{len(pending)}/{len(questions)} questions to go, "
          f"{args.n_samples} traces each at T={args.temperature}")

    total = 0
    started = time.time()
    with open(args.output, "a" if done_idx else "w") as out_file:
        chunk = backend.chunk_size
        for start in tqdm(range(0, len(pending), chunk), desc="generating"):
            batch = pending[start : start + chunk]
            prompts = [format_prompt(q["question"], backend.tokenizer) for q in batch]
            for item, traces in zip(batch, backend.generate(prompts)):
                for trace in traces:
                    out_file.write(json.dumps({
                        "idx": item["idx"],
                        "question": item["question"],
                        "answer": item.get("answer", ""),
                        "trace": trace,
                    }, ensure_ascii=False) + "\n")
                    total += 1
            out_file.flush()

    elapsed = time.time() - started
    print(f"Saved {total} raw traces → {args.output}  "
          f"({elapsed / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
