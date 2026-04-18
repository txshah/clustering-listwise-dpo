"""
Step 2 — Sort raw traces into correct / wrong buckets (mirrors process_gsm8k.py).

Reads raw traces JSONL, applies the correctness oracle and deduplication,
then accumulates per-question result files until each question has enough
correct and wrong solutions.

Input JSONL  : traces/raw/raw_traces.jsonl  (from generate_traces.py)
               schema: {idx, question, answer, trace}

Output JSONL : traces/processed/results.jsonl
               schema: {idx, question, answer,
                        correct_solutions: [str, ...],
                        wrong_solutions:   [str, ...]}

Usage:
    python process_traces.py \
        --input  ../traces/raw/raw_traces.jsonl \
        --output ../traces/processed/results.jsonl \
        --correct_thresh 4 \
        --wrong_thresh   4
"""

import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from utils import load_jsonl, save_jsonl, is_correct, no_similar, exist_error


def process(
    raw_traces: list[dict],
    correct_thresh: int,
    wrong_thresh: int,
) -> tuple[list[dict], list[dict]]:
    """
    Groups traces by question idx, applies oracle + dedup, returns:
      - completed : questions that reached both thresholds
      - incomplete: questions still needing more traces (re-run generate_traces.py on these)
    """
    # Build per-question pools from raw traces
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

        correct = is_correct(trace, gt)

        if correct:
            if exist_error(trace):
                # paper keeps these separately; we just skip them
                continue
            if no_similar(trace, pool["correct_solutions"]):
                pool["correct_solutions"].append(trace)
        else:
            if no_similar(trace, pool["wrong_solutions"]):
                pool["wrong_solutions"].append(trace)

    completed  = []
    incomplete = []

    for idx, pool in pools.items():
        entry = {"idx": idx, **pool}
        if (len(pool["correct_solutions"]) >= correct_thresh and
                len(pool["wrong_solutions"]) >= wrong_thresh):
            completed.append(entry)
        else:
            incomplete.append(entry)

    return completed, incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--correct_thresh",  type=int, default=4)
    parser.add_argument("--wrong_thresh",    type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    raw = load_jsonl(args.input)
    print(f"Loaded {len(raw)} raw traces.")

    completed, incomplete = process(raw, args.correct_thresh, args.wrong_thresh)

    save_jsonl(completed, args.output)

    if incomplete:
        stem   = args.output.replace(".jsonl", "")
        rerun  = f"{stem}_needs_more.jsonl"
        # Save just the question + answer fields so you can re-feed into generate_traces.py
        rerun_items = [{"idx": e["idx"], "question": e["question"], "answer": e["answer"]}
                       for e in incomplete]
        save_jsonl(rerun_items, rerun)
        print(f"{len(incomplete)} questions need more samples → {rerun}")

    print(f"{len(completed)} questions completed → {args.output}")


if __name__ == "__main__":
    main()
