# Experiment Runbook

What to run to test the idea, in what order, and how to watch it on W&B.
All times are measured on this pod (1× RTX A6000 48GB, vLLM backends).

## The idea being tested

Does **entailment-based construction of preference lists** improve LIPO-λ listwise
training over correctness/length-only construction? Three arms, identical loss and
list shape (1 chosen + 4 ranked rejected), identical trace pool:

| Arm | "Good" label (gate) | Reject ranking | NLI model |
|---|---|---|---|
| `length` | answer correctness (math-verify) | shortest wrong first | none |
| `gated_lg` | answer correctness | entailment vs reference | deberta-v3-large |
| `ungated_lg` | entailment ≥ 0.5 only (PORT-style, correctness ignored) | entailment | deberta-v3-large |

Reference trace = shortest correct trace per question; scores are bidirectional
(min of both NLI directions). Baseline = untrained Mistral-7B-v0.1.

Every eval reports **maj@8** (the Mistral-paper headline), **pass@1/5/10**, and
avg@1 from one 10-sample pass (8-shot, T=0.8, scored by math-verify).

Key pairwise reads:
- `length` vs `gated_*` → does entailment **ranking** matter?
- `gated_lg` vs `ungated_lg` → does the entailment **gate** (labels) matter? ← the headline comparison

---

## Prerequisites (one-time, ~30 min)

```bash
uv sync                                            # env (Python 3.12, vLLM, TRL 1.x)
bash run_entailment.sh prepare generate process    # steps 1-3 → results_1000.jsonl
```

Generation is checkpointed (`--resume`), safe to interrupt. Verify afterwards:
`generation/traces/processed/results_1000.jsonl` exists and
`uv run python -c "from generation.code.utils import load_jsonl"`-style spot checks
aren't needed — `run_arms.sh score` fails loudly if the file is missing.

---

## Tier 0 — Learning-rate sweep (~5 h, run FIRST)

The current lr (3e-5) was adopted after observing 1e-6 leave the loss pinned at
init — one data point, not a sweep. Since every arm shares the lr, a bad choice
suppresses (or fakes) the arm differences, so nail it down before the core runs.

Sweep on the `length` baseline arm (tuning on the baseline, not the novel arms,
avoids biasing the comparison toward our method), one seed, 500-question eval:

```bash
bash run_arms.sh score build          # one-time: scoring + datasets
for lr in 1e-5 3e-5 1e-4; do
    ARMS=length SEEDS=42 LR=$lr bash run_arms.sh train
done
for lr in 1e-5 3e-5 1e-4; do
    ARMS=length SEEDS=42 LR=$lr EVAL_LIMIT=500 bash run_arms.sh eval
done
```

3 trainings (~1 h each) + 3 evals (~10 min each at limit 500) + the shared base
eval. Pick the lr by **final maj@8** (W&B group `arms-1000`, filter `lr:` tags),
sanity-checked against the loss curves: the winner should show a smooth decrease
from 3.37 without oscillation (1e-4 risk) and without flatlining (1e-5 risk).
If the winner isn't 3e-5, set `learning_rate` in `listwise_config.yaml` before
Tier 1. Borderline (two lrs within noise)? Prefer the smaller one.

> **Done (2026-08-18): winner 1e-5** (maj@8 0.486 vs 0.480 @3e-6, 0.448 @3e-5,
> 0.374 @1e-4; 500q) — config updated. The sweep also surfaced a loss bug: mean
> per-token logps pinned the cascade loss at init (3.37) at every lr; fixed to
> sum-over-response-tokens (`logp_agg` in the config), the DPO/LPOI convention.
> An extra 3e-6 point was run because the winner sat at the grid edge; it
> flatlined (final loss 3.07), so the prefer-smaller rule didn't apply.

## Tier 1 — Core comparison (~10 h total)

3 arms × **2 seeds** (42/43) + base, full 1319-question test split, at the
Tier-0 winning lr.

```bash
bash run_arms.sh                      # score → build → train ×6 → eval ×7 → summary
```

Or step by step (each step is independently re-runnable; training skips
finished output dirs, eval resumes per-question):

```bash
bash run_arms.sh score                # NLI scoring (deberta-v3-large) (~30-45 min)
bash run_arms.sh build                # 3 datasets                  (seconds)
bash run_arms.sh train                # 6 runs × ~1 h               (~6 h)
bash run_arms.sh eval                 # 7 evals × ~25 min           (~3 h)
bash run_arms.sh summary              # mean ± std table, no GPU
uv run graphs/plot_arms.py            # figure → graphs/arms_accuracy.png
```

Outputs: adapters in `outputs/arm_<arm>_1000_s<seed>/`, eval summaries in
`results/<name>_1000.json`, per-question records in `results/*_detail.jsonl`.

**Decision point:** look at `gated_lg` vs `length` and `ungated_lg` vs `gated_lg`
(mean ± std). Differences smaller than ~1 std across seeds are noise — don't
over-read them.

> **Done (2026-08-18)**, full 1319q split, maj@8 mean±std: base 0.487,
> length 0.497±0.000, gated_lg 0.521±0.004, ungated_lg 0.525±0.001.
> Entailment *ranking* beats length (+2.4–2.8, ≫ noise); the correctness
> *gate* does not matter (ungated ≈ gated).

## Tier 2 — Gate sweeps (~5 h, run after Tier 1)

One seed each, `EVAL_LIMIT=500` to keep it cheap. Promote any winner to 2 seeds
+ full eval before believing it.

> **Warning:** the original recipe here (`THRESHOLD=0.3 bash run_arms.sh build
> train`) silently no-ops: `build` overwrites the datasets in place while
> `train` skips output dirs that already hold an adapter. Use explicitly
> suffixed dataset/output names instead:

```bash
# Threshold sensitivity for the ungated arm (Tier 1 used 0.5)
SC=generation/traces/processed/results_1000_scored_lg.jsonl
for t in 0.3 0.7; do
    uv run generation/code/build_entailment_list.py \
        --input "$SC" --output "generation/traces/arm_ungated_lg_t${t}_1000.jsonl" \
        --threshold "$t"
    uv run training/train_listwise.py --config training/configs/listwise_config.yaml \
        --dataset_path "generation/traces/arm_ungated_lg_t${t}_1000.jsonl" \
        --output_dir "outputs/arm_ungated_lg_t${t}_1000_s42" --seed 42 \
        --wandb_group arms-1000 --wandb_run_name "train-ungated_lg_t${t}-1000-s42"
    uv run evaluation/eval_gsm8k.py --model "outputs/arm_ungated_lg_t${t}_1000_s42" \
        --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 --temperature 0.8 \
        --output "results/arm_ungated_lg_t${t}_s42_1000.json" \
        --wandb_group arms-1000 --wandb_run_name "eval-arm_ungated_lg_t${t}_s42-1000" \
        --resume --limit 500
done

# Negative-selection strategy for gated_lg (default: hardest)
uv run generation/code/build_listwise.py \
    --input generation/traces/processed/results_1000_scored_lg.jsonl \
    --output generation/traces/arm_gated_lg_easiest_1000.jsonl \
    --ranking_method entailment --negative_selection easiest
uv run training/train_listwise.py --config training/configs/listwise_config.yaml \
    --dataset_path generation/traces/arm_gated_lg_easiest_1000.jsonl \
    --output_dir outputs/arm_gated_lg_easiest_1000_s42 --seed 42 \
    --wandb_group arms-1000 --wandb_run_name train-gated_lg_easiest-1000-s42
# (repeat with --negative_selection spread)
```

Also report, per threshold, what fraction of "good" traces have a **wrong final
answer** — that's the mechanism the ungated arm is supposed to exploit
(`build_entailment_list.py` prints this at build time).

> **Done (2026-08-18)**, 500q s42 (same-subset t0.5 references computed from the
> Tier 1 detail files): threshold flat — t0.3 0.546, t0.5 0.532, t0.7 0.528
> (NLI scores are bimodal; the threshold moves ~8 of ~400 lists, label flips
> stay at 1.2–1.3% / 8.2–8.3%). Negative selection: hardest 0.532 > spread
> 0.506 > easiest 0.498 — hard negatives matter; the default wins, nothing to
> promote.

## Tier 3 — Scale & robustness (~20 h, optional until Tiers 1-2 conclude)

```bash
# Full-data run: best arm + length baseline at 7473 questions
N_QUESTIONS=7473 bash run_entailment.sh prepare generate process   # ~7 h, resumable
N_QUESTIONS=7473 SEEDS=42 bash run_arms.sh score build
# then train/eval just the two arms of interest (edit ARMS env or run manually)

# (lr already swept in Tier 0 — if Tier 3 changes data scale a quick re-check
#  of the winning lr on the full-data length arm is cheap insurance)
```

---

## Monitoring on W&B

Project: **`clustering-listwise-dpo`** → https://wandb.ai/ (logged in via ~/.netrc;
runs go online automatically, offline fallback if the key ever disappears).

Everything from one tier lands in **one group** (`arms-1000` by default), so the
group page is the experiment dashboard. Naming scheme:

| Run name | job_type | What it is |
|---|---|---|
| `train-<arm>-1000-s<seed>` | `train-listwise` | one training run |
| `eval-base-1000` | `eval` | base model eval |
| `eval-arm_<arm>_s<seed>-1000` | `eval` | one adapter eval |

Tags on training runs: `arm:<arm>`, `seed:<seed>`, `tag:1000` — filter by
`arm:` to overlay all seeds of one arm. lr-sweep runs (Tier 0) additionally get
an `lr:<value>` tag and an `_lr<value>` infix in run names and output dirs, so
they never mix with the main runs.

**During training, watch (job_type = `train-listwise`):**
- `train/loss` — must *decrease from ~3.37* (that's `-log(1/5)·Σλ`, the
  random-init value of the cascade loss). A healthy run at lr 1e-5 ends near
  ~2.2 after 3 epochs. A flat line pinned at 3.3-3.4 for 50+ steps = either
  lr far too low or `logp_agg: mean` (the bug fixed 2026-08-18: mean logps
  compress β-scaled scores ~100×, pinning the loss) → kill and check config.
- `train/learning_rate` — confirms the 1e-5 schedule is live.
- `eval/loss` (per epoch) — divergence from train loss = overfitting the small list set.
- Run config panel carries the full yaml (lr, LoRA, lambdas, seed) — verify the
  seed differs across the three runs of an arm.

**During eval, watch (job_type = `eval`):**
- `running/maj@8`, `running/pass@10`, `running/avg@1` — live-updating as
  questions complete (x-axis = questions evaluated). Curves stabilize after a
  few hundred questions; a run stuck near the base model's curve is an early
  signal the arm didn't help.
- `final/*` + the run summary — the numbers that go in the paper table.
- Red flag: `running/avg@1` near 0 → collapsed/empty generations (we mark these
  in the plot).

**Comparing arms:** on the group page, filter job_type=`eval`, group runs by the
`arm:` tag, and pin `final/maj@8`, `final/pass@10` as columns. The same numbers
are in `results/*.json` locally; `bash run_arms.sh summary` prints the
mean ± std table without touching W&B.

## Interruptions & resume

Everything is resumable — safe to kill the pod mid-tier:
- **generation**: `--resume` skips questions already in the raw file
- **training**: finished arms (adapter file exists) are skipped; a *partially*
  trained run restarts from scratch (runs are ~1 h, acceptable loss)
- **eval**: `--resume` skips questions already in `results/*_detail.jsonl`
- **summary/plot**: pure post-processing, run any time on whatever has finished

GPU is single-tenant: run one step at a time (training and vLLM eval both want
most of the 48GB).
