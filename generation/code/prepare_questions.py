"""
Downloads GSM8K training split and writes traces/questions.jsonl.

GSM8K answer format: "...chain of thought... #### 42"
We extract just the number after #### as the ground truth so is_correct()
can do a clean match against the model's final answer.

Usage:
    python prepare_questions.py --output ../traces/questions.jsonl
    python prepare_questions.py --output ../traces/questions.jsonl --split train
"""

import argparse
import re
import os
from datasets import load_dataset
from utils import save_jsonl


def extract_gsm8k_answer(raw_answer: str) -> str:
    """GSM8K stores answers as '...reasoning... #### <number>'"""
    match = re.search(r"####\s*([\d,\.]+)", raw_answer)
    if match:
        return match.group(1).replace(",", "").strip()
    return raw_answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../traces/questions.jsonl")
    parser.add_argument("--split",  default="train", choices=["train", "test"])
    parser.add_argument("--limit",  type=int, default=None,
                        help="Keep only the first N questions (default: all)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading GSM8K ({args.split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    questions = [
        {
            "idx":      idx,
            "question": item["question"],
            "answer":   extract_gsm8k_answer(item["answer"]),
        }
        for idx, item in enumerate(dataset)
    ]

    if args.limit is not None:
        questions = questions[:args.limit]

    save_jsonl(questions, args.output)
    print(f"Saved {len(questions)} questions → {args.output}")


if __name__ == "__main__":
    main()
