"""
Step 1 — Raw trace generation (mirrors lce_solution_gen_gsm8k.py).

For each question in the input JSONL, call the model N times at temperature=1
(do_sample=True) and save every raw response to traces/raw/.

Input JSONL schema  : {"idx": int, "question": str, "answer": str, ...}
Output JSONL schema : same fields + "trace": str  (one file per question batch)

Usage:
    python generate_traces.py \
        --input  ../traces/questions.jsonl \
        --output ../traces/raw/raw_traces.jsonl \
        --model  mistralai/Mistral-7B-v0.1 \
        --n_samples 8 \
        --max_new_tokens 512
"""

import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# torch.isin on Long tensors is broken on MPS before macOS 14.
# Cast to float, run isin, cast result back to bool.
if torch.backends.mps.is_available():
    _orig_isin = torch.isin
    def _mps_isin(elements, test_elements, **kwargs):
        if elements.is_floating_point() or test_elements.is_floating_point():
            return _orig_isin(elements, test_elements, **kwargs)
        return _orig_isin(elements.float(), test_elements.float(), **kwargs).bool()
    torch.isin = _mps_isin

from utils import load_jsonl, save_jsonl


def build_prompt(question: str, tokenizer) -> str:
    """
    Format the question into a prompt.
    Uses chat template if available (instruct models); falls back to a plain
    Question/Answer format for base models like Mistral-7B-v0.1.
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Base model fallback — matches the paper's simple prompt style
    return f"Question: {question}\nAnswer:"


def generate_traces(
    questions: list[dict],
    model,
    tokenizer,
    n_samples: int,
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    """
    For each question dict, produce n_samples traces at temperature=1.
    Returns flat list of {idx, question, answer, trace} dicts.
    """
    results = []

    for item in tqdm(questions, desc="generating"):
        prompt = build_prompt(item["question"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,       # ← temperature sampling (D_naive)
                temperature=1.0,      # ← paper uses temp=1 for negative candidates
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in output_ids:
            trace = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            results.append({
                "idx": item["idx"],
                "question": item["question"],
                "answer": item.get("answer", ""),
                "trace": trace,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True,  help="Path to questions JSONL")
    parser.add_argument("--output",         required=True,  help="Path to save raw traces JSONL")
    parser.add_argument("--model",          required=True,  help="HF model name or local path")
    parser.add_argument("--n_samples",      type=int, default=8,   help="Traces per question")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading model {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # MPS (Apple Silicon) doesn't support bfloat16 before macOS 14 — use float16 instead
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype
    ).to(device)
    model.eval()

    questions = load_jsonl(args.input)
    print(f"Loaded {len(questions)} questions. Generating {args.n_samples} traces each...")

    raw_traces = generate_traces(questions, model, tokenizer, args.n_samples, args.max_new_tokens, device)

    save_jsonl(raw_traces, args.output)
    print(f"Saved {len(raw_traces)} raw traces → {args.output}")


if __name__ == "__main__":
    main()
