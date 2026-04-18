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

from utils import load_jsonl, save_jsonl


def build_prompt(question: str, tokenizer) -> str:
    """
    Format the question into a chat prompt.
    Swap this out if your model expects a different template.
    """
    messages = [{"role": "user", "content": question}]
    # apply_chat_template handles [INST] / <|user|> etc. automatically
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    questions = load_jsonl(args.input)
    print(f"Loaded {len(questions)} questions. Generating {args.n_samples} traces each...")

    raw_traces = generate_traces(questions, model, tokenizer, args.n_samples, args.max_new_tokens, device)

    save_jsonl(raw_traces, args.output)
    print(f"Saved {len(raw_traces)} raw traces → {args.output}")


if __name__ == "__main__":
    main()
