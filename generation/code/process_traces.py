"""
Step 2 — Sort raw traces into correct / wrong buckets (mirrors process_gsm8k.py).

Applies correctness check and deduplication. Every question is saved regardless
of how many traces it collected — no minimum threshold.

Input JSONL  : traces/raw/raw_traces.jsonl  (from generate_traces.py)
               schema: {idx, question, answer, trace}

Output JSONL : traces/processed/results.jsonl
               schema: {idx, question, answer,
                        correct_solutions: [str, ...],
                        wrong_solutions:   [str, ...]}

Usage:
    python process_traces.py \
        --input  ../traces/raw/raw_traces.jsonl \
        --output ../traces/processed/results.jsonl
"""

import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from utils import load_jsonl, save_jsonl, is_correct, no_similar, exist_error


def process(raw_traces: list[dict], length_window: int = 500) -> list[dict]:
    pools: dict[int, dict] = defaultdict(lambda: {
        "question": "",
        "answer": "",
        "correct_solutions": [],
        "wrong_solutions": [],
    })

    for item in tqdm(raw_traces, desc="processing"):
        idx   = item["idx"]
        trace = item["trace"]
        gt    = item["answer"]

        pool = pools[idx]
        pool["question"] = item["question"]
        pool["answer"]   = gt

        if is_correct(trace, gt):
            if not exist_error(trace) and no_similar(trace, pool["correct_solutions"], length_window):
                pool["correct_solutions"].append(trace)
        else:
            if no_similar(trace, pool["wrong_solutions"], length_window):
                pool["wrong_solutions"].append(trace)

    return [{"idx": idx, **pool} for idx, pool in pools.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length_window", type=int, default=500,
                        help="Min char-length difference to treat two traces as distinct. "
                             "Use 0 to disable dedup (useful for small smoke-test runs).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    raw = load_jsonl(args.input)
    print(f"Loaded {len(raw)} raw traces.")

    results = process(raw, length_window=args.length_window)
    save_jsonl(results, args.output)

    for r in results:
        print(f"  idx={r['idx']}  correct={len(r['correct_solutions'])}  wrong={len(r['wrong_solutions'])}")
    print(f"Saved {len(results)} questions → {args.output}")


if __name__ == "__main__":
    main()
