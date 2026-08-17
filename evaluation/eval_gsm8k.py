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
import os
import re
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wandb_setup import init_wandb

# math-verify (optional, pip install math-verify): sympy-based answer
# equivalence as used by lighteval / Open R1.  When installed it replaces the
# regex comparison in is_correct; otherwise the regex fallback below is used.
try:
    from math_verify import parse as mv_parse, verify as mv_verify
except ImportError:
    mv_parse = None


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
    gt = extract_answer(ground_truth) or ground_truth.strip()

    if mv_parse is not None:
        try:
            return bool(mv_verify(mv_parse(gt), mv_parse(prediction)))
        except Exception:
            pass  # fall back to the regex comparison below

    pred = extract_answer(prediction)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred == gt


def build_prompt(tokenizer, question: str) -> str:
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"Question: {question}\nAnswer:"


def evaluate(model, tokenizer, dataset, max_new_tokens: int, device: str,
             batch_size: int) -> dict:
    correct = 0
    total   = len(dataset)
    results = []

    # left padding so every sequence's prompt ends at the same position and
    # the generated continuation starts right after it
    tokenizer.padding_side = "left"

    for start in tqdm(range(0, total, batch_size), desc="evaluating"):
        batch   = dataset.select(range(start, min(start + batch_size, total)))
        prompts = [build_prompt(tokenizer, item["question"]) for item in batch]
        inputs  = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,   # greedy — deterministic eval
                temperature    = 1.0,
                pad_token_id   = tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for item, out in zip(batch, output_ids):
            response    = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
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
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Questions decoded per forward pass (lower if OOM)")
    parser.add_argument("--split",          default="test", choices=["train", "test"])
    parser.add_argument("--limit",          type=int, default=None,
                        help="Evaluate only the first N questions (default: all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = "math-verify" if mv_parse is not None else "regex fallback"
    print(f"Answer scorer: {scorer}")

    # Start tracking before the slow parts so aborted evals still show up
    tracking = init_wandb(
        run_name = f"eval-{os.path.basename(args.model.rstrip('/'))}",
        job_type = "eval",
        config   = {**vars(args), "scorer": scorer, "decoding": "greedy"},
    )

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
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    summary = evaluate(model, tokenizer, dataset, args.max_new_tokens, device,
                       args.batch_size)

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

    if tracking == "wandb":
        import wandb
        wandb.log({"accuracy": summary["accuracy"],
                   "correct":  summary["correct"],
                   "total":    summary["total"]})
        # keep the result files with the run so they survive Colab disconnects
        wandb.save(args.output, policy="now")
        wandb.save(detail_path, policy="now")
        wandb.finish()


if __name__ == "__main__":
    main()
