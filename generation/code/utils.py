"""
Shared utilities: IO, correctness check, deduplication.

Swap out `is_correct` for your own oracle (exact match, regex, judge model, etc.)
"""

import json

from math_verify import parse as _mv_parse, verify as _mv_verify


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
    Scores the trace with math-verify (the sympy-based checker used by
    lighteval / Open R1): does the trace's concluding expression equal the
    ground truth?

    Because the SFT-less base model often states the answer FIRST and then
    explains ("85 trees.\\nSolution: ..."), the opening line is scored as a
    second candidate; either location counts.  Everything else — extraction,
    normalisation, numeric equivalence — is the library's.
    """
    gold = _mv_parse(ground_truth.strip())
    for candidate in (trace, trace.strip().split("\n", 1)[0]):
        try:
            if _mv_verify(gold, _mv_parse(candidate)):
                return True
        except Exception:
            continue
    return False


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
