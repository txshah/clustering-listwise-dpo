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
from utils import load_jsonl, save_jsonl, format_prompt


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

    # third element: does the trace's final answer match the ground truth?
    return ([(t, s, True)  for t, s in zip(corrects, scores["correct"])] +
            [(t, s, False) for t, s in zip(wrongs,   scores["wrong"])])


def build_entailment_list(
    processed: list[dict],
    threshold: float,
    n_good: int,
    n_bad: int,
) -> list[dict]:
    pairs = []
    skipped_no_reference = 0
    skipped_ratio = 0
    label_flips = {"good_wrong_answer": 0, "chosen_total": 0,
                   "bad_correct_answer": 0, "bad_total": 0}

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
        for chosen, chosen_score, chosen_is_correct in goods[:n_good]:
            record = {
                # Same scaffold as generation and eval (utils.format_prompt)
                "prompt": format_prompt(item["question"]),
                "chosen": chosen,
            }
            for i, (bad, _, _) in enumerate(ranked_bads, start=1):
                record[f"rejected{i}"] = bad
            record["chosen_score"]      = chosen_score
            record["chosen_is_correct"] = chosen_is_correct   # metadata: answer matches gold?
            record["rejected_scores"]   = [s for _, s, _ in ranked_bads]
            pairs.append(record)
            label_flips["good_wrong_answer"] += int(not chosen_is_correct)
            label_flips["chosen_total"]      += 1
        label_flips["bad_correct_answer"] += sum(c for _, _, c in ranked_bads)
        label_flips["bad_total"]          += len(ranked_bads)

    if skipped_no_reference:
        print(f"Skipped {skipped_no_reference} questions (no correct solution → no reference)")
    if skipped_ratio:
        print(f"Skipped {skipped_ratio} questions (fewer than {n_good} good or {n_bad} bad "
              f"traces at threshold {threshold})")

    # The mechanism this arm tests: how often does the entailment label disagree
    # with answer correctness? All zeros would mean the gate is just correctness
    # by another name and the ungated arm can't differ from the gated ones.
    if label_flips["chosen_total"]:
        gw, ct = label_flips["good_wrong_answer"], label_flips["chosen_total"]
        bc, bt = label_flips["bad_correct_answer"], label_flips["bad_total"]
        print(f"Label flips vs correctness @ threshold {threshold}: "
              f"chosen-with-wrong-answer {gw}/{ct} ({gw/ct:.1%}), "
              f"rejected-with-correct-answer {bc}/{bt} ({bc/bt:.1%})")

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
