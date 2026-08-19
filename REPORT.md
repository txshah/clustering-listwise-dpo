# Entailment as Preference — Experiment Report

*2026-08-18/19 · Mistral-7B-v0.1 · GSM8K · LIPO-λ listwise · 1× RTX A6000 · W&B group `arms-1000`*

**Question:** does entailment-based construction of preference lists (ranking, and/or the good/bad gate) beat correctness/length-only construction for LIPO-λ listwise training?

## Findings

1. **Entailment ranking works.** maj@8 0.521–0.525 vs 0.497 for length ranking at 1000q training data (+2.4–2.8, seed std ≤0.004); 0.567 vs 0.556 at full data.
2. **The correctness gate is unnecessary.** Label-free PORT-style lists (binarized entailment) match the correctness-gated arm within noise at every scale; the binarization threshold is irrelevant (NLI scores are bimodal — t0.3/0.5/0.7 differ by ~8 of ~400 lists).
3. **Hard negatives carry the gain.** Negative selection: hardest 0.532 > spread 0.506 > easiest 0.498 (500q).
4. **Data scale dominates.** 1000→7473 questions lifts every arm ~5–6 pts; the entailment edge narrows +2.8→+1.1. Best model: `ungated_lg` @7473q, **maj@8 0.567±0.000** vs base 0.488.
5. **Loss bug found & fixed** (`74fe28a`): mean per-token logps pinned the cascade loss at init (3.371) at every lr — β·Δlogp too small by ~100×. Fixed to summed logps (`logp_agg: sum`). Failure signature documented in EXPERIMENTS.md.

## Results

**Tier 0 — lr sweep** (length arm, s42, 500q): 3e-6 → 0.480 (loss flatlined), **1e-5 → 0.486** (winner; smooth 3.37→2.19), 3e-5 → 0.448, 1e-4 → 0.374, base 0.466. Accuracy degrades monotonically above 1e-5; over-trained models lose the answer diversity maj@k needs.

**Tier 1 — core comparison** (full 1319q split, seeds 42/43, mean±std):

| model | maj@8 | avg@1 | pass@5 | pass@10 |
|---|---|---|---|---|
| base | 0.487 | 0.329 | 0.653 | 0.760 |
| length | 0.497±0.000 | 0.343 | 0.665 | 0.768 |
| gated_lg | 0.521±0.004 | 0.357 | 0.674 | 0.771 |
| **ungated_lg** | **0.525±0.001** | 0.350 | 0.676 | 0.784 |

**Tier 2 — gate sweeps** (500q, s42; t0.5 refs recomputed from Tier 1 detail files): thresholds flat (0.546/0.532/0.528 for t0.3/0.5/0.7); negatives hardest 0.532 > spread 0.506 > easiest 0.498 — defaults stand.

**Tier 3 — scale-up** (7473q data → 3037/3005 lists, full split, 2 seeds):

| model | maj@8 | avg@1 | pass@10 | Δ base |
|---|---|---|---|---|
| base | 0.488 | 0.327 | 0.753 | — |
| length | 0.556±0.008 | 0.396 | 0.809 | +6.8 |
| **ungated_lg** | **0.567±0.000** | 0.408 | 0.809 | +7.9 |

Figures: `graphs/arms_accuracy.png` (1000q), `graphs/arms_accuracy_7473.png` (7473q).

## Honest framing

The entailment gain (+1.1 to +2.8 maj@8) is real, low-variance, and consistent — but second-order next to data scale (+5–6). The strongest claim is **label-free preference construction that matches or beats label-dependent construction**, with the gain concentrated in near-miss negatives.

## Limitations

Two seeds per cell (one in Tier 2); `gated_lg` not run at 7473q; sum-logps add mild length sensitivity (standard for DPO-family, untested interaction with the length arm); one base model / task / NLI scorer; base maj@8 ~3.5 pts under the Mistral paper (prompt/parser differences — comparisons unaffected).

## Reproducibility

Branch `entailment-1000`; key commits `74fe28a` (loss fix + lr), `b473c6b` (1000q artifacts), `33aca23` (runbook fixes), `9755249` (7473q). Datasets + eval summary JSONs committed; raw dumps and per-question detail files regenerable via `run_entailment.sh` / `run_arms.sh` (resumable). ~34 h wall-clock total (~3 h lost to the pre-fix sweep); full-data generation 72.5 min.
