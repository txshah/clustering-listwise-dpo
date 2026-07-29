"""
Plot GSM8K accuracy of the four preference-training arms against the base model.

Reads results_*.json summaries (written by eval_gsm8k.py via run_arms.sh) and the
listwise training config, renders a bar chart with Wilson 95% CI whiskers and a
hyperparameter panel so the chart is self-describing.

Usage (from repo root, after run_arms.sh eval):
    python graphs/plot_arms.py
    python graphs/plot_arms.py --results_dir /path/to/jsons --tag 1000
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# palette (validated: CVD-safe, >=3:1 on surface)
SURFACE = "#fcfcfb"
BLUE    = "#2a78d6"   # trained arms
GRAY    = "#8a8985"   # base reference
GRAY_LT = "#c3c2bc"
INK     = "#0b0b0b"
INK_2   = "#52514e"
RED     = "#e34948"   # collapsed run marker

ARMS = [
    ("base",       "base\nMistral-7B",             GRAY),
    ("length",     "length-ranked\n(gated)",        BLUE),
    ("gated_sm",   "entailment gated\n(small NLI)", BLUE),
    ("gated_lg",   "entailment gated\n(large NLI)", BLUE),
    ("ungated_lg", "entailment ungated\n(large NLI)", BLUE),
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def result_path(results_dir: str, key: str, tag: str) -> str:
    name = f"results_base_{tag}.json" if key == "base" else f"results_arm_{key}_{tag}.json"
    return os.path.join(results_dir, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=REPO,
                        help="Directory containing results_*.json (default: repo root)")
    parser.add_argument("--tag", default="1000", help="Run tag in the filenames")
    parser.add_argument("--output", default=os.path.join(REPO, "graphs/arms_accuracy.png"))
    args = parser.parse_args()

    rows = []
    for key, label, color in ARMS:
        path = result_path(args.results_dir, key, args.tag)
        if not os.path.exists(path):
            print(f"skipping {key}: {path} not found")
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({"key": key, "label": label, "color": color,
                     "acc": r["accuracy"], "k": r["correct"], "n": r["total"]})
    if not rows:
        raise SystemExit("no results_*.json files found")

    with open(os.path.join(REPO, "training/configs/listwise_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    n_eval = rows[0]["n"]
    fig, ax = plt.subplots(figsize=(11.6, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.685, top=0.82, bottom=0.14)
    ax.set_facecolor(SURFACE)

    xs = range(len(rows))
    for i, row in enumerate(rows):
        lo, hi = wilson_ci(row["k"], row["n"])
        ax.bar(i, row["acc"], width=0.62, color=row["color"], zorder=3)
        ax.errorbar(i, row["acc"], yerr=[[row["acc"] - lo], [hi - row["acc"]]],
                    fmt="none", ecolor=INK_2, elinewidth=1.4, capsize=4, zorder=4)
        ax.annotate(f"{row['acc']*100:.1f}%", (i, hi), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    fontsize=10.5, fontweight="bold", color=INK)
        if row["acc"] < 0.02 and row["key"] != "base":
            ax.annotate("collapsed:\nempty generations", (i, 0.02), xytext=(0, 14),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=RED)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9.5, color=INK)
    ax.set_ylabel("GSM8K test accuracy (greedy pass@1)", fontsize=10, color=INK_2)
    ax.set_ylim(0, max(r["acc"] for r in rows) * 1.35 + 0.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(axis="y", color=GRAY_LT, linewidth=0.7, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRAY_LT)
    ax.tick_params(colors=INK_2)

    fig.suptitle("Listwise preference training on GSM8K - one seed per arm",
                 fontsize=13, fontweight="bold", color=INK, x=0.08, ha="left")
    fig.text(0.08, 0.855, f"Mistral-7B-v0.1 base | eval: first {n_eval} test questions, "
             "zero-shot, scored by math-verify | whiskers: Wilson 95% CI",
             fontsize=9, color=INK_2)

    eff_batch = (cfg.get("per_device_train_batch_size", 1)
                 * cfg.get("gradient_accumulation_steps", 16))
    lam = "/".join(str(x) for x in cfg.get("lambdas", []))
    hp = [
        ("data",  "1000 questions x 10 samples"),
        ("",      "temp 1.0, Mistral-7B base"),
        ("lists", "1 chosen + 4 ranked rejected"),
        ("",      "correctness gate: math-verify"),
        ("loss",  f"LIPO-lambda, beta {cfg.get('beta')}"),
        ("",      f"lambda {lam}"),
        ("LoRA",  f"all-linear, r={cfg.get('lora_r', 16)}, "
                  f"a={cfg.get('lora_alpha', 32)}"),
        ("optim", f"lr {cfg.get('learning_rate')}, "
                  f"{cfg.get('num_train_epochs')} epochs, batch {eff_batch}"),
        ("NLI",   "nli-deberta-v3 sm / lg"),
        ("",      "bidir min vs shortest correct"),
    ]
    panel = "hyperparameters\n" + "\n".join(f"{k:>6}  {v}" for k, v in hp)
    fig.text(0.705, 0.80, panel, fontsize=8.6, color=INK_2, family="monospace",
             va="top", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f3f3f0",
                       edgecolor=GRAY_LT, linewidth=0.8))

    fig.savefig(args.output, dpi=180, facecolor=SURFACE)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
