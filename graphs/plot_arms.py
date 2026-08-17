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
import glob
import json
import math
import statistics
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
    ("base",       "base\nMistral-7B",               GRAY),
    ("length",     "length-ranked\n(gated)",          BLUE),
    ("gated_lg",   "entailment gated\n(large NLI)",   BLUE),
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


def result_paths(results_dir: str, key: str, tag: str) -> list[str]:
    """
    run_arms.sh writes $RESULTS_DIR/base_{tag}.json for the base model and
    $RESULTS_DIR/arm_{key}_s{seed}_{tag}.json per training seed
    (arm_{key}_{tag}.json accepted for single-seed runs from older revisions).
    """
    if key == "base":
        candidates = [f"base_{tag}.json"]
    else:
        candidates = sorted(glob.glob(os.path.join(results_dir, f"arm_{key}_s*_{tag}.json")))
        candidates = [os.path.basename(c) for c in candidates] or [f"arm_{key}_{tag}.json"]
    paths = [os.path.join(results_dir, name) for name in candidates]
    return [p for p in paths if os.path.exists(p)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(REPO, "results"),
                        help="Directory containing the eval summaries (default: repo results/)")
    parser.add_argument("--tag", default="1000", help="Run tag in the filenames")
    parser.add_argument("--metric", default="maj@8",
                        help="Which metric to plot (maj@8, pass@1, pass@5, pass@10, avg@1)")
    parser.add_argument("--output", default=os.path.join(REPO, "graphs/arms_accuracy.png"))
    args = parser.parse_args()

    rows = []
    for key, label, color in ARMS:
        paths = result_paths(args.results_dir, key, args.tag)
        if not paths:
            print(f"skipping {key}: no results found")
            continue
        # eval_gsm8k.py summary: {"metrics": {"maj@8": ..., "pass@10": ..., "total": N}}
        accs, n = [], 0
        for path in paths:
            with open(path) as f:
                metrics = json.load(f)["metrics"]
            accs.append(metrics[args.metric])
            n = metrics["total"]
        acc = statistics.mean(accs)
        rows.append({"key": key, "label": label, "color": color,
                     "acc": acc, "k": round(acc * n), "n": n,
                     "std": statistics.stdev(accs) if len(accs) > 1 else None,
                     "n_seeds": len(accs)})
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
        # Multi-seed arms: whiskers = ±1 std across seeds (training noise).
        # Single runs: Wilson 95% CI on the eval questions (sampling noise).
        if row["std"] is not None:
            lo, hi = row["acc"] - row["std"], row["acc"] + row["std"]
        else:
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
    ax.set_ylabel(f"GSM8K test {args.metric} (8-shot, 10 samples)", fontsize=10, color=INK_2)
    ax.set_ylim(0, max(r["acc"] for r in rows) * 1.35 + 0.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(axis="y", color=GRAY_LT, linewidth=0.7, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRAY_LT)
    ax.tick_params(colors=INK_2)

    max_seeds = max(r["n_seeds"] for r in rows)
    seed_note = (f"{max_seeds} seeds per arm | whiskers: ±1 std over seeds"
                 if max_seeds > 1 else "one seed per arm | whiskers: Wilson 95% CI")
    fig.suptitle("Listwise preference training on GSM8K",
                 fontsize=13, fontweight="bold", color=INK, x=0.08, ha="left")
    fig.text(0.08, 0.855, f"Mistral-7B-v0.1 base | eval: {n_eval} test questions, "
             f"8-shot, 10 samples @ T=0.8, scored by math-verify | {seed_note}",
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
        ("NLI",   "nli-deberta-v3-large"),
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
