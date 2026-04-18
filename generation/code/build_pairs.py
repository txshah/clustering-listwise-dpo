"""
Step 3 — Build DPO pairs (mirrors get_prompt_chosen_rejected.py).

Reads the processed results JSONL and emits one pair per
(correct_solution, wrong_solution) combination, up to `max_pairs` per question.

Input JSONL  : traces/processed/results.jsonl  (from process_traces.py)
Output JSONL : traces/dpo_pairs.jsonl

Output schema — message-list format required by data.py's apply_chat_template:
  {
    "chosen":   [{"role": "user", "content": "<question>"},
                 {"role": "assistant", "content": "<correct trace>"}],
    "rejected": [{"role": "user", "content": "<question>"},
                 {"role": "assistant", "content": "<wrong trace>"}]
  }
data.py splits this as: prompt = chosen[:-1], chosen = chosen[-1], rejected = rejected[-1]

Usage:
    python build_pairs.py \
        --input     ../traces/processed/results.jsonl \
        --output    ../traces/dpo_pairs.jsonl \
        --max_pairs 3
"""

import argparse
import os
import random
from tqdm import tqdm

from utils import load_jsonl, save_jsonl


def build_pairs(
    processed: list[dict],
    max_pairs: int,
) -> list[dict]:
    pairs = []

    for item in tqdm(processed, desc="building pairs"):
        question = item["question"]
        corrects = item["correct_solutions"]
        wrongs   = item["wrong_solutions"]

        all_combos = [(c, w) for c in corrects for w in wrongs]
        random.shuffle(all_combos)

        for chosen, rejected in all_combos[:max_pairs]:
            pairs.append({
                "chosen": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": chosen},
                ],
                "rejected": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": rejected},
                ],
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Max (chosen, rejected) pairs per question")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    processed = load_jsonl(args.input)
    print(f"Loaded {len(processed)} processed questions.")

    pairs = build_pairs(processed, args.max_pairs)

    save_jsonl(pairs, args.output)
    print(f"Saved {len(pairs)} DPO pairs → {args.output}")


if __name__ == "__main__":
    main()
