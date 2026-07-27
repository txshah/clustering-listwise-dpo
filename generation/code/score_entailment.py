"""
Step 2.5 — Score traces with NLI entailment against the best correct solution.

Sits between process_traces.py and build_listwise.py.  Takes the processed
results JSONL and adds an `entailment_scores` field to each question.  The
scores are stored parallel to the existing solution lists so the file remains
backwards-compatible with any downstream step that ignores the new field.

Why entailment for ranking?
  The current length-based proxy assumes shorter wrong trace = less wrong.
  Entailment asks a more direct question: does this trace's reasoning lead
  to the same conclusion as a known-correct trace?  Higher score = the
  wrong trace is structurally closer to correct = less bad = should be r1.

Why bidirectional?
  One-directional entailment (reference => trace) favors vague hypotheses:
  "the answer is a number" is trivially entailed by any reference.  Scoring
  both directions and taking the min follows the mutual-entailment criterion
  from the semantic entropy literature (Kuhn et al. 2023): a trace only
  scores high when it follows from the reference AND the reference follows
  from it, i.e. the two say the same thing.

Reference trace per question:
  Shortest correct solution (same as the "chosen" in build_listwise).
  All traces (correct and wrong) are scored against it.

Output schema — adds to the existing processed JSONL:
  {
    "idx": 0,
    "question": "...",
    "answer":   "42",
    "correct_solutions": ["trace_a", "trace_b"],
    "wrong_solutions":   ["trace_c", "trace_d", "trace_e"],
    "entailment_scores": {
        "correct": [0.91, 0.84],   # parallel to correct_solutions
        "wrong":   [0.54, 0.31, 0.12]  # parallel to wrong_solutions
    }
  }

Usage:
    python score_entailment.py \\
        --input  ../traces/processed/results.jsonl \\
        --output ../traces/processed/results_scored.jsonl

    # smaller/faster model for smoke tests:
    python score_entailment.py \\
        --input  ../traces/processed/results_small.jsonl \\
        --output ../traces/processed/results_small_scored.jsonl \\
        --model  cross-encoder/nli-deberta-v3-small \\
        --batch_size 32
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import load_jsonl, save_jsonl

# Truncate traces to their last N chars before feeding to the NLI model.
# Math reasoning traces end with the answer, so the tail carries the most
# signal; this also keeps every input well within the 512-token limit.
_TAIL_CHARS = 800


# ── Model loading ──────────────────────────────────────────────────────────────

def load_nli_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def _entailment_index(model) -> int:
    """Return the logit index for the ENTAILMENT label (varies by model)."""
    for idx, label in model.config.id2label.items():
        if label.lower() == "entailment":
            return idx
    return 2  # safe fallback for most cross-encoder NLI models


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_pairs(
    premises: list[str],
    hypotheses: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> list[float]:
    """
    Return P(entailment) for each (premise, hypothesis) pair.
    premises and hypotheses must have the same length.
    """
    entail_idx = _entailment_index(model)
    scores = []

    for i in range(0, len(premises), batch_size):
        p_batch = premises[i : i + batch_size]
        h_batch = hypotheses[i : i + batch_size]

        enc = tokenizer(
            p_batch,
            h_batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits          # [B, num_labels]

        probs = F.softmax(logits, dim=-1)         # [B, num_labels]
        scores.extend(probs[:, entail_idx].cpu().tolist())

    return scores


def score_pairs_bidirectional(
    references: list[str],
    traces: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> list[float]:
    """
    Return min(P(ref entails trace), P(trace entails ref)) for each pair.
    The min enforces mutual entailment: high only when both texts carry
    the same conclusion, so vague traces can't score high for free.
    """
    forward  = score_pairs(references, traces, tokenizer, model, device, batch_size)
    backward = score_pairs(traces, references, tokenizer, model, device, batch_size)
    return [min(f, b) for f, b in zip(forward, backward)]


# ── Per-question scoring ───────────────────────────────────────────────────────

def score_question(
    item: dict,
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> dict:
    corrects = item["correct_solutions"]
    wrongs   = item["wrong_solutions"]

    # no correct traces → nothing to score against; return zeros
    if not corrects:
        return {
            **item,
            "entailment_scores": {
                "correct": [],
                "wrong":   [0.0] * len(wrongs),
            },
        }

    # reference = shortest correct trace (same selection as build_listwise "chosen")
    reference = min(corrects, key=len)
    ref_tail  = reference[-_TAIL_CHARS:]

    # score every correct trace against the reference
    # (single-element list → score is 1.0 by definition; skip the model call)
    if len(corrects) == 1:
        correct_scores = [1.0]
    else:
        correct_scores = score_pairs_bidirectional(
            references  = [ref_tail] * len(corrects),
            traces      = [c[-_TAIL_CHARS:] for c in corrects],
            tokenizer   = tokenizer,
            model       = model,
            device      = device,
            batch_size  = batch_size,
        )

    # score every wrong trace against the reference
    wrong_scores = []
    if wrongs:
        wrong_scores = score_pairs_bidirectional(
            references  = [ref_tail] * len(wrongs),
            traces      = [w[-_TAIL_CHARS:] for w in wrongs],
            tokenizer   = tokenizer,
            model       = model,
            device      = device,
            batch_size  = batch_size,
        )

    return {
        **item,
        "entailment_scores": {
            "correct": correct_scores,  # parallel to item["correct_solutions"]
            "wrong":   wrong_scores,    # parallel to item["wrong_solutions"]
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Processed results JSONL (from process_traces.py)")
    parser.add_argument("--output",     required=True,
                        help="Output path for scored JSONL")
    parser.add_argument("--model",      default="cross-encoder/nli-deberta-v3-small",
                        help="HuggingFace NLI model name")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="NLI inference batch size (lower if OOM)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading NLI model '{args.model}' on {device}...")
    tokenizer, model = load_nli_model(args.model, device)

    data = load_jsonl(args.input)
    print(f"Scoring {len(data)} questions...")

    scored = []
    for item in tqdm(data, desc="scoring"):
        scored.append(score_question(item, tokenizer, model, device, args.batch_size))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_jsonl(scored, args.output)

    # summary
    total_correct = sum(len(s["entailment_scores"]["correct"]) for s in scored)
    total_wrong   = sum(len(s["entailment_scores"]["wrong"])   for s in scored)
    print(f"Scored {total_correct} correct traces and {total_wrong} wrong traces.")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
