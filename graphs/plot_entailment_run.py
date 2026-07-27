"""
Plot the results of the 100-question entailment-labeled preference-list run.

Reads the scored traces and the built 1:4 lists, renders a histogram of
entailment scores split at the labeling threshold plus a question-yield bar.

Usage (from repo root):
    python graphs/plot_entailment_run.py
"""

import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORED_PATH = os.path.join(REPO, "generation/traces/processed/results_100_scored.jsonl")
PAIRS_PATH  = os.path.join(REPO, "generation/traces/entailment_pairs_100.jsonl")
OUT_PATH    = os.path.join(REPO, "graphs/entailment_run_100.png")

THRESHOLD = 0.5

# palette (validated: CVD-safe, >=3:1 on surface)
SURFACE   = "#fcfcfb"
GOOD      = "#2a78d6"   # blue  — traces labeled good (chosen pool)
BAD       = "#eb6834"   # orange — traces labeled bad (rejected pool)
GRAY      = "#8a8985"
GRAY_LT   = "#c3c2bc"
INK       = "#0b0b0b"
INK_2     = "#52514e"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


scored = load_jsonl(SCORED_PATH)
pairs  = load_jsonl(PAIRS_PATH)

with_ref = [r for r in scored if r["correct_solutions"]]
scores = [s for r in with_ref
          for s in r["entailment_scores"]["correct"] + r["entailment_scores"]["wrong"]]
n_good = sum(s >= THRESHOLD for s in scores)
n_bad  = len(scores) - n_good

n_kept   = len(pairs)
n_no_ref = len(scored) - len(with_ref)
n_thin   = len(scored) - n_kept - n_no_ref

fig = plt.figure(figsize=(8.6, 5.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.0], hspace=0.55,
                      left=0.085, right=0.97, top=0.76, bottom=0.09)

# ── histogram of entailment scores, split at the labeling threshold ───────────
ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURFACE)

bins = [i / 20 for i in range(21)]
counts, edges, patches = ax.hist(scores, bins=bins, edgecolor=SURFACE, linewidth=1.4)
for left_edge, patch in zip(edges[:-1], patches):
    patch.set_facecolor(GOOD if left_edge >= THRESHOLD - 1e-9 else BAD)

ax.axvline(THRESHOLD, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
ax.annotate(f"threshold {THRESHOLD}", xy=(THRESHOLD, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 4), textcoords="offset points", ha="center",
            fontsize=8.5, color=INK)

ax.annotate(f"labeled bad → rejected pool\n{n_bad} traces", xy=(0.245, 235),
            ha="center", fontsize=9, color=BAD, fontweight="bold", linespacing=1.4)
ax.annotate(f"labeled good → chosen pool\n{n_good} traces", xy=(0.755, 235),
            ha="center", fontsize=9, color=GOOD, fontweight="bold", linespacing=1.4)

ax.set_xlim(0, 1)
ax.set_xlabel("NLI entailment score vs reference trace  ·  P(entailment)",
              fontsize=9, color=INK_2)
ax.set_ylabel("traces", fontsize=9, color=INK_2)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.tick_params(colors=INK_2, labelsize=8)
ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#d8d7d2")

# ── question yield ────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(SURFACE)

segments = [
    (n_kept,   GOOD,    f"{n_kept} kept → 1:4 lists"),
    (n_no_ref, GRAY,    f"{n_no_ref} no correct trace"),
    (n_thin,   GRAY_LT, f"{n_thin} <1 good or <4 bad"),
]
x = 0
for i, (width, color, label) in enumerate(segments):
    ax2.barh(0, width, left=x, height=0.85, color=color,
             edgecolor=SURFACE, linewidth=1.6)
    if i < len(segments) - 1:
        ax2.annotate(label, xy=(x + width / 2, -0.85), ha="center", va="top",
                     fontsize=8, color=INK_2)
    else:
        # narrow trailing segment: right-aligned on its own row to avoid collisions
        ax2.annotate(label, xy=(100, -2.1), ha="right", va="top",
                     fontsize=8, color=INK_2)
    x += width

ax2.set_xlim(0, 100)
ax2.set_ylim(-3.3, 0.8)
ax2.set_title("question yield (100 GSM8K questions)", fontsize=9, color=INK_2,
              loc="left", pad=4)
ax2.axis("off")

fig.text(0.085, 0.975, "Entailment scores split cleanly at the 0.5 labeling threshold",
         fontsize=13, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.085, 0.915,
         "Experiment: PORT-style 1:4 preference lists labeled by binarized entailment only "
         "(answer correctness ignored).\n"
         "100 GSM8K questions × 10 Mistral-7B traces; each trace NLI-scored against the "
         "shortest correct trace of its question.",
         fontsize=8.5, color=INK_2, linespacing=1.5, va="top")

fig.savefig(OUT_PATH, dpi=200)
print(f"saved → {OUT_PATH}")
