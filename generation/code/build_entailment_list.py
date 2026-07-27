"""
Step 4.3 — Build preference lists labeled by entailment only (PORT-style data).

Unlike build_listwise.py (which splits good/bad by answer correctness and only
uses entailment to rank), this builder ignores correctness entirely.  The
good/bad label is the binarized entailment score:

    label = 1 (good) if entailment_score >= threshold else 0 (bad)

A trace whose final answer is wrong but whose reasoning entails the reference
counts as good; a correct-answer trace with low entailment counts as bad.

The good:bad ratio is an input (default 1:4, matching ListwiseTrainer's
expected 1 chosen + 4 rejected shape).  When n_good > 1, each good trace
becomes its own output record sharing the same ranked bad traces.

Filtering — a question is dropped when:
  - it has no correct solution (no reference trace, so scores are meaningless)
  - fewer than n_good traces binarize to good, or fewer than n_bad to bad

Input JSONL  : traces/processed/results_scored.jsonl  (from score_entailment.py)
Output JSONL : traces/entailment_pairs.jsonl

Output schema (1:4 default — compatible with ListwiseTrainer):
  {
    "prompt":    "question text",
    "chosen":    "good trace (highest entailment)",
    "rejected1": "bad trace 1 (highest entailment among bads)",
    ...
    "rejected4": "bad trace 4",
    "chosen_score":    0.91,            # metadata, ignored by the trainer
    "rejected_scores": [0.42, ...]      # metadata, ignored by the trainer
  }

Usage:
    python build_entailment_list.py \\
        --input     ../traces/processed/results_scored.jsonl \\
        --output    ../traces/entailment_pairs.jsonl \\
        --threshold 0.5 --n_good 1 --n_bad 4
"""

import argparse
import os
from tqdm import tqdm
from utils import load_jsonl, save_jsonl


def pool_traces(item: dict) -> list[tuple[str, float]]:
    """Merge correct and wrong traces into one (trace, score) pool."""
    scores = item.get("entailment_scores")
    if scores is None:
        raise ValueError(
            f"idx={item.get('idx')} has no entailment_scores. "
            "Run score_entailment.py first."
        )

    corrects, wrongs = item["correct_solutions"], item["wrong_solutions"]
    if len(scores["correct"]) != len(corrects) or len(scores["wrong"]) != len(wrongs):
        raise ValueError(
            f"idx={item.get('idx')}: entailment_scores not parallel to solutions."
        )

    return list(zip(corrects, scores["correct"])) + list(zip(wrongs, scores["wrong"]))


def build_entailment_list(
    processed: list[dict],
    threshold: float,
    n_good: int,
    n_bad: int,
) -> list[dict]:
    pairs = []
    skipped_no_reference = 0
    skipped_ratio = 0

    for item in tqdm(processed, desc="building entailment lists"):
        if not item["correct_solutions"]:
            skipped_no_reference += 1
            continue

        pooled = pool_traces(item)

        goods = sorted((p for p in pooled if p[1] >= threshold),
                       key=lambda x: x[1], reverse=True)
        bads  = sorted((p for p in pooled if p[1] < threshold),
                       key=lambda x: x[1], reverse=True)

        if len(goods) < n_good or len(bads) < n_bad:
            skipped_ratio += 1
            continue

        ranked_bads = bads[:n_bad]
        for chosen, chosen_score in goods[:n_good]:
            record = {
                "prompt": item["question"],
                "chosen": chosen,
            }
            for i, (bad, _) in enumerate(ranked_bads, start=1):
                record[f"rejected{i}"] = bad
            record["chosen_score"]    = chosen_score
            record["rejected_scores"] = [s for _, s in ranked_bads]
            pairs.append(record)

    if skipped_no_reference:
        print(f"Skipped {skipped_no_reference} questions (no correct solution → no reference)")
    if skipped_ratio:
        print(f"Skipped {skipped_ratio} questions (fewer than {n_good} good or {n_bad} bad "
              f"traces at threshold {threshold})")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,
                        help="Scored results JSONL (from score_entailment.py)")
    parser.add_argument("--output",    required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment score >= threshold → good (1), else bad (0)")
    parser.add_argument("--n_good",    type=int, default=1,
                        help="Good traces per question (each emits its own record)")
    parser.add_argument("--n_bad",     type=int, default=4,
                        help="Ranked bad traces per record")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed = load_jsonl(args.input)
    print(f"Loaded {len(processed)} processed questions.")

    pairs = build_entailment_list(processed, args.threshold, args.n_good, args.n_bad)

    save_jsonl(pairs, args.output)
    print(f"Saved {len(pairs)} preference lists ({args.n_good}:{args.n_bad} good:bad) "
          f"→ {args.output}")


if __name__ == "__main__":
    main()
