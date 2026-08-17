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


# ── Prompt format ─────────────────────────────────────────────────────────────
# One definition shared by trace generation, pair building and evaluation, so the
# model is always conditioned on the same scaffold it was trained on.

def format_prompt(question: str, tokenizer=None) -> str:
    """Chat template when the model has one, plain Question/Answer otherwise."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


# ── Correctness oracle ────────────────────────────────────────────────────────

def is_correct(trace: str, ground_truth: str) -> bool:
    """
    Mirrors the paper: checks whether the ground_truth appears in the final
    portion of the trace (last 300 chars), i.e. the model's final answer.
    Works with GSM8K answers (already extracted to plain numbers by prepare_questions.py).
    """
    return ground_truth.strip() in trace[-300:]


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
