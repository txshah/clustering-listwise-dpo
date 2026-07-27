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
    Checks whether the ground_truth appears as a STANDALONE number where a
    final answer would actually live:

      1. after a "####" marker (GSM8K answer format), anywhere in the trace
      2. at the very start of the trace (answer-first format: "85 trees. ...")
      3. in the last 100 chars (the conclusion of the trace)

    Standalone means boundary-guarded: gold "85" must not match inside a
    larger number like "91-85943-57126" or "185", and gold "5" must not
    match inside "$15" or "8.5".  Commas are stripped from both sides so
    "1,200" in a trace matches gold "1200".

    The narrow windows replace earlier anywhere-in-tail checks, which
    produced false positives whenever the gold appeared as an INTERMEDIATE
    quantity: gold "10" matched "(12 + 10)/60" mid-reasoning in a trace
    whose actual conclusion was "it is 1 hour".
    """
    gold = re.escape(ground_truth.strip().replace(",", ""))
    standalone = rf"(?<![\d.]){gold}(?!\.?\d)"
    text = trace.replace(",", "")

    if re.search(rf"####\s*{gold}(?!\.?\d)", text):
        return True
    if re.match(rf"\s*\$?\s?{gold}(?!\.?\d)", text):
        return True
    return re.search(standalone, text[-100:]) is not None


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
