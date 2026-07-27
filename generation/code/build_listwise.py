"""
Step 4.2 — Build listwise preference data for LIPO-λ training.

Takes the processed results JSONL and produces 1 chosen + 4 ranked rejected
per question.  Supports two ranking methods:

  length     (default fallback)
    Shorter trace = better quality proxy.  Fast, no model needed.

  entailment (used automatically when entailment_scores field is present)
    Correctness stays the hard gate: chosen always comes from the correct
    traces, rejected only from the wrong ones.  Entailment only ranks within
    each group, over the FULL trace pool (no first-N truncation):
      chosen      = correct trace with the highest score
      rejected1-4 = wrong traces by descending score (least bad first)
    Higher score = reasoning more aligned with a known-correct trace.
    Note: correct-trace scores are measured against the shortest correct
    trace (the reference), which scores ~1 against itself — so chosen is
    usually the reference itself unless another correct trace out-entails it.

Ranking method selection (--ranking_method):
  auto        use entailment when scores are present, otherwise length [default]
  length      always rank by length
  entailment  always rank by entailment score (fails if scores are missing)

Output schema (flat strings — the custom ListwiseTrainer handles tokenisation):
  {
    "prompt":    "question text",
    "chosen":    "best correct trace",
    "rejected1": "ranked negative 1 (least bad)",
    "rejected2": "ranked negative 2",
    "rejected3": "ranked negative 3",
    "rejected4": "ranked negative 4 (worst)"
  }

Usage:
    # with entailment scores (run score_entailment.py first):
    python build_listwise.py \\
        --input  ../traces/processed/results_scored.jsonl \\
        --output ../traces/listwise_pairs.jsonl

    # length fallback (original behaviour):
    python build_listwise.py \\
        --input          ../traces/processed/results.jsonl \\
        --output         ../traces/listwise_pairs.jsonl \\
        --ranking_method length
"""

import argparse
import os
from tqdm import tqdm
from utils import load_jsonl, save_jsonl


# ── Ranking helpers ────────────────────────────────────────────────────────────

def rank_by_length(traces: list[str]) -> list[str]:
    """Ascending by length — shorter = better quality proxy."""
    return sorted(traces, key=len)


def rank_by_entailment(traces: list[str], scores: list[float]) -> list[str]:
    """Descending by entailment score — higher = less bad = ranked first."""
    return [t for t, _ in sorted(zip(traces, scores), key=lambda x: x[1], reverse=True)]


def best_correct_by_entailment(corrects: list[str], scores: list[float]) -> str:
    """Return the correct trace with the highest entailment score."""
    return max(zip(corrects, scores), key=lambda x: x[1])[0]


# ── Core builder ───────────────────────────────────────────────────────────────

def build_listwise(
    processed: list[dict],
    n_per_class: int,
    ranking_method: str,
) -> list[dict]:
    pairs   = []
    skipped = 0
    use_entailment_count = 0

    for item in tqdm(processed, desc="building listwise"):
        corrects = item["correct_solutions"]
        wrongs   = item["wrong_solutions"]

        if len(corrects) < 1 or len(wrongs) < 4:
            skipped += 1
            continue

        scores = item.get("entailment_scores", {})
        has_scores = bool(scores.get("correct")) and bool(scores.get("wrong"))

        use_entailment = (
            (ranking_method == "auto"       and has_scores) or
            (ranking_method == "entailment")
        )

        if use_entailment and not has_scores:
            raise ValueError(
                f"--ranking_method entailment requested but idx={item['idx']} "
                "has no entailment_scores. Run score_entailment.py first."
            )

        if use_entailment:
            # rank over the FULL pools so the best trace can't be cut by an
            # arbitrary first-N truncation, then keep the top of each ranking
            chosen        = best_correct_by_entailment(corrects, scores["correct"])
            ranked_wrongs = rank_by_entailment(wrongs, scores["wrong"])[:4]
            use_entailment_count += 1
        else:
            # length mode mirrors the original paper: draw from the first
            # n_per_class traces, then rank those by length
            chosen        = rank_by_length(corrects[:n_per_class])[0]
            ranked_wrongs = rank_by_length(wrongs[:n_per_class])[:4]

        pairs.append({
            "prompt":    item["question"],
            "chosen":    chosen,
            "rejected1": ranked_wrongs[0],
            "rejected2": ranked_wrongs[1],
            "rejected3": ranked_wrongs[2],
            "rejected4": ranked_wrongs[3],
        })

    if skipped:
        print(f"Skipped {skipped} questions (need ≥1 correct and ≥4 wrong traces)")
    if use_entailment_count:
        print(f"Used entailment ranking for {use_entailment_count}/{len(pairs)} questions.")
    else:
        print("Used length ranking.")

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--n_per_class",    type=int, default=5,
                        help="How many traces per class to draw from (length mode only; "
                             "entailment mode ranks the full pool)")
    parser.add_argument("--ranking_method", default="auto",
                        choices=["auto", "length", "entailment"],
                        help="How to rank traces: auto (entailment if available, else length), "
                             "length, or entailment")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed = load_jsonl(args.input)
    print(f"Loaded {len(processed)} processed questions.")

    pairs = build_listwise(processed, args.n_per_class, args.ranking_method)

    save_jsonl(pairs, args.output)
    print(f"Saved {len(pairs)} listwise pairs → {args.output}")


if __name__ == "__main__":
    main()
