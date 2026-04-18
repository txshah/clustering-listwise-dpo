"""
Step 4.2 — Build listwise preference data for LIPO-λ training.

Takes the processed traces (5 correct + 5 incorrect per question, i.e. N/2 each)
and produces:  1 chosen  +  4 ranked rejected  per question.

Ranking proxy (acknowledged limitation — no reward model):
  Shorter incorrect trace = ranked better (fewer wasted/hallucinated steps).
  rejected1 = least bad (shortest wrong trace)
  rejected4 = worst     (longest wrong trace)

Output schema (flat strings — the custom ListwiseTrainer handles tokenisation):
  {
    "prompt":    "question text",
    "chosen":    "correct trace (shortest = most efficient correct solution)",
    "rejected1": "ranked negative 1 (least bad)",
    "rejected2": "ranked negative 2",
    "rejected3": "ranked negative 3",
    "rejected4": "ranked negative 4 (worst)"
  }

Usage:
    python build_listwise.py \
        --input  ../traces/processed/results.jsonl \
        --output ../traces/listwise_pairs.jsonl \
        --n_per_class 5
"""

import argparse
import os
from tqdm import tqdm
from utils import load_jsonl, save_jsonl


def rank_by_length(traces: list[str]) -> list[str]:
    """Sort ascending by length — shorter = better quality proxy."""
    return sorted(traces, key=len)


def build_listwise(processed: list[dict], n_per_class: int) -> list[dict]:
    pairs = []
    skipped = 0

    for item in tqdm(processed, desc="building listwise"):
        corrects = item["correct_solutions"][:n_per_class]
        wrongs   = item["wrong_solutions"][:n_per_class]

        if len(corrects) < 1 or len(wrongs) < 4:
            skipped += 1
            continue

        # chosen = shortest correct trace (most efficient solution)
        chosen = rank_by_length(corrects)[0]

        # rejected1-4: 4 of the n_per_class wrongs ranked by length (shortest=best)
        ranked_wrongs = rank_by_length(wrongs)[:4]

        pairs.append({
            "prompt":    item["question"],
            "chosen":    chosen,
            "rejected1": ranked_wrongs[0],
            "rejected2": ranked_wrongs[1],
            "rejected3": ranked_wrongs[2],
            "rejected4": ranked_wrongs[3],
        })

    if skipped:
        print(f"Skipped {skipped} questions (not enough traces — need ≥1 correct and ≥4 wrong)")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--n_per_class",  type=int, default=5,
                        help="How many traces per class to draw from (N/2, default=5)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    processed = load_jsonl(args.input)
    print(f"Loaded {len(processed)} processed questions.")

    pairs = build_listwise(processed, args.n_per_class)

    save_jsonl(pairs, args.output)
    print(f"Saved {len(pairs)} listwise pairs → {args.output}")


if __name__ == "__main__":
    main()
