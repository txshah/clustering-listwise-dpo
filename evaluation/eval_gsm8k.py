"""
Evaluates a trained model on GSM8K test split.
Works for both the DPO-trained and listwise-trained models — just pass --model.

Metric: end accuracy (does the final number match ground truth?)

Usage:
    python eval_gsm8k.py --model outputs/dpo_mistral7b      --output results_dpo.json
    python eval_gsm8k.py --model outputs/listwise_mistral7b --output results_listwise.json
"""

import argparse
import json
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_answer(text: str) -> str | None:
    """
    Tries to extract the final numeric answer.
    Looks for #### <number> first (GSM8K format), then falls back to the
    last standalone number in the text.
    """
    # GSM8K style answer marker
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: last number in the text
    numbers = re.findall(r"\b\d[\d,\.]*\b", text)
    return numbers[-1].replace(",", "") if numbers else None


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = extract_answer(prediction)
    gt   = extract_answer(ground_truth) or ground_truth.strip()
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred == gt


def evaluate(model, tokenizer, dataset, max_new_tokens: int, device: str) -> dict:
    correct = 0
    total   = len(dataset)
    results = []

    for item in tqdm(dataset, desc="evaluating"):
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": item["question"]}]
            prompt   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,   # greedy — deterministic eval
                temperature    = 1.0,
                pad_token_id   = tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        response   = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        correct_ans = item["answer"]

        ok = is_correct(response, correct_ans)
        correct += int(ok)

        results.append({
            "question": item["question"],
            "ground_truth": correct_ans,
            "prediction": response,
            "correct": ok,
        })

    accuracy = correct / total
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True,  help="Path to trained model or HF id")
    parser.add_argument("--output",         default="eval_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--split",          default="test", choices=["train", "test"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    summary = evaluate(model, tokenizer, dataset, args.max_new_tokens, device)

    print(f"\nAccuracy: {summary['accuracy']:.4f}  ({summary['correct']}/{summary['total']})")

    with open(args.output, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "results"}, f, indent=2)
        f.write("\n")
    print(f"Summary saved → {args.output}")

    # also save per-question results
    detail_path = args.output.replace(".json", "_detail.jsonl")
    with open(detail_path, "w") as f:
        for r in summary["results"]:
            f.write(json.dumps(r) + "\n")
    print(f"Per-question results → {detail_path}")


if __name__ == "__main__":
    main()
