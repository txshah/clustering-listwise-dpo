"""
Shared utilities: IO, correctness check, deduplication.

Swap out `is_correct` for your own oracle (exact match, regex, judge model, etc.)
"""

import json
import re


# ── IO ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Correctness oracle ────────────────────────────────────────────────────────

def is_correct(trace: str, ground_truth: str) -> bool:
    """
    Checks whether the ground_truth appears as a STANDALONE number in the
    final portion of the trace (last 300 chars), i.e. the model's final answer.
    Works with GSM8K answers (already extracted to plain numbers by prepare_questions.py).

    The boundary guards replace the original raw substring check, which
    produced false positives whenever the gold digits appeared inside a
    larger number: gold "85" matched a phone number "91-85943-57126",
    gold "5" matched "$15" or "$65".  Commas are stripped from both sides
    so "1,200" in a trace matches gold "1200".
    """
    gold = ground_truth.strip().replace(",", "")
    tail = trace[-300:].replace(",", "")
    return re.search(rf"(?<![\d.]){re.escape(gold)}(?!\.?\d)", tail) is not None


# ── Deduplication ─────────────────────────────────────────────────────────────
# Mirrors the paper's heuristic: two solutions are "similar" if they have the
# same number of turns AND their total text lengths are within 500 chars.

def trace_text_length(trace: str) -> int:
    return len(trace)


def no_similar(candidate: str, existing: list[str], length_window: int = 500) -> bool:
    """Returns True if `candidate` is sufficiently different from all `existing` traces."""
    for existing_trace in existing:
        if abs(trace_text_length(candidate) - trace_text_length(existing_trace)) < length_window:
            return False
    return True


# ── Error phrase filter ───────────────────────────────────────────────────────
# The paper skips solutions that contain apology/error language even when the
# final answer happens to be correct.

_ERROR_PHRASES = ["error", "apolog", "i cannot", "i'm unable", "i am unable"]

def exist_error(trace: str) -> bool:
    lowered = trace.lower()
    return any(phrase in lowered for phrase in _ERROR_PHRASES)
