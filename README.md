# Clustering Listwise DPO

## What This Is

Listwise preference optimization (LIPO-λ) on Mistral-7B for math reasoning (GSM8K). The main experiment is a **3-arm comparison** of how the preference lists are labeled and ranked — all arms share the same trace pool, the same LIPO-λ loss, and the same 1 chosen + 4 ranked rejected shape, and are scored against the untrained base model:

| Arm | Gate (good vs bad) | Ranking within lists |
|---|---|---|
| `length` | answer correctness | trace length |
| `gated_lg` | answer correctness | bidirectional NLI entailment (deberta-v3-large) |
| `ungated_lg` | entailment score ≥ threshold only (PORT-style, correctness ignored) | entailment |

`run_arms.sh` drives the whole experiment. A binary D_naive DPO path (`train_dpo.py`, mirroring Step-Controlled DPO) is kept as auxiliary tooling but is not one of the arms.

---

## What We Did (Logical Flow)

### 1. Trace Generation (mirrors Step-Controlled-DPO)

We replicate the D_naive data generation from [Step-Controlled DPO](https://github.com/mathllm/Step-Controlled_DPO) (`src/positive_negative_lce_gen/`), stripped of the LCE (Language-Code-Execution) Jupyter kernel and the multi-GPU distributed setup. The core loop is identical:

1. Load GSM8K training questions
2. Call the SFT model at **temperature=1** (`do_sample=True`) — same as the paper
3. Check each trace for correctness by matching the final answer against the ground truth (mirrors `is_equal()` in the paper)
4. Deduplicate by trace length (mirrors `no_similar()` in the paper)

The paper uses 4+4 with 6 parallel GPU workers and a threshold gate. We removed the threshold — every question is saved with however many correct/wrong traces it collected.

### 2. Step 4.1 — D_naive DPO (mirrors Step-Controlled-DPO training)

From the trace pool, pair every correct trace with every incorrect trace (capped at `max_pairs` per question). Flattened to plain strings (`prompt`, `chosen`, `rejected`) — TRL's "standard" dataset format.

Train with **TRL's DPOTrainer** + LoRA so the base model is loaded once and shared as the frozen reference — avoids the ~28GB OOM of loading two full 7B copies.

### 3. Step 4.2 — Listwise LIPO-λ (mirrors LPOI)

From the same trace pool, build one listwise sample per question:
- **chosen**: shortest correct trace (efficiency proxy)
- **rejected1–4**: 4 incorrect traces ranked by length (shortest = least bad)

Questions with fewer than 4 wrong traces are skipped. The ranking is a weak proxy (acknowledged limitation — no reward model).

Train with a custom **ListwiseTrainer** that implements the LIPO-λ loss from [LPOI](https://github.com/fatemehpesaran310/lpoi) (`lpoi_dpo_trainer_5img.py:1537–1542`), adapted for text. Uses `model.disable_adapter()` to get reference log-probs from the frozen base weights — same LoRA memory trick as DPO.

### 3.5. Step 4.3 — Entailment-labeled listwise data (PORT-style)

A second listwise dataset where the good/bad label comes only from the binarized entailment score, ignoring answer correctness.
`score_entailment.py` scores every trace with an NLI cross-encoder against a reference (the shortest correct trace of its question), taking the min of both entailment directions (mutual entailment, as in the semantic entropy literature) so vague traces cannot score high one-directionally.
`build_entailment_list.py` then merges the correct and wrong pools, labels each trace good when `score >= threshold` (default 0.5) and bad otherwise, and emits lists at a configurable good:bad ratio (default 1:4, `ListwiseTrainer`-compatible).
Questions are filtered out when they have no correct trace (no reference) or cannot fill the requested ratio.
This isolates the labeling metric as the experimental variable versus the correctness-labeled data of Step 4.2.

### 4. Evaluation

Both models are evaluated on the GSM8K test split under the **Mistral 7B paper protocol: 8-shot prompting, maj@8** (self-consistency majority vote over 8 samples), alongside **pass@k** for k = 10, 5, 1.

A single generation pass produces every metric — sample `n = 10` completions per question at T=0.8, then:

| Metric | How it's computed |
|---|---|
| `avg@1` | mean accuracy of an individual sample (= `pass@1`) |
| `maj@8` | majority vote over the first 8 samples — the paper's headline number |
| `pass@k` | unbiased estimator `1 − C(n−c, k)/C(n, k)` (Codex eq. 1) for k = 1, 5, 10 |

So maj@8 + pass@10 + pass@5 cost exactly one run. `--greedy` gives a cheap deterministic pass@1 sanity check instead.

Answer extraction order: `The answer is X` → `#### X` → last number in the completion. Generation stops at `\nQuestion:` so few-shot completions don't run on into the next fabricated question.

---

## Key Changes from the Source Repos

### From Step-Controlled-DPO

| Original | Here | Why |
|---|---|---|
| Jupyter kernel for code execution (LCE) | Removed | Not needed for plain-text traces |
| 6 GPU workers, IP-based sharding | Single machine | Lightweight setup |
| `InferenceClient` (vLLM server) | vLLM offline `LLM` engine (HF fallback) | Same speed, no separate server process |
| Threshold gate (4+4 minimum per question) | Removed | Every question saved as-is |
| `utils.py` in sibling folder | Consolidated into `generation/code/utils.py` | Single import path |
| `get_initial_data.py` (GPU split utility) | Replaced by `prepare_questions.py` | Just load GSM8K directly |
| Full model fine-tune | LoRA (r=16, α=32, all-linear — matches LPOI) | Fits on one GPU, shared frozen reference |

### From LPOI

| Original | Here | Why |
|---|---|---|
| 5 image variants (pixel masking) | 5 text traces (correct + 4 ranked wrong) | Text domain, no images |
| Ranking from pixel degradation | Ranking by trace length | No reward model available |
| `lpoi.py` data loader (image paths) | `ListwiseDataset` (plain text) | Text-only |
| `lpoi_dpo_trainer_5img.py` full trainer | `listwise_trainer.py` (loss unchanged) | Stripped image batching |
| 3-component loss (text DPO + image DPO + anchor) | LIPO-λ only | Simplified; anchor = implicit via β |
| Separate frozen ref model | `model.disable_adapter()` | Single model load, same result |

---

## File Hierarchy

```
clustering-listwise-dpo/
│
├── generation/
│   ├── code/
│   │   ├── prepare_questions.py   # Download GSM8K → questions.jsonl
│   │   ├── generate_traces.py     # Sampling at temp=1 (vLLM/HF backends) → raw_traces.jsonl
│   │   ├── process_traces.py      # Correctness check + dedup → results.jsonl (no threshold)
│   │   ├── score_entailment.py    # Step 2.5: NLI entailment scores vs best correct trace
│   │   ├── build_pairs.py         # Step 4.1: (prompt, chosen, rejected) string pairs
│   │   ├── build_listwise.py      # Step 4.2: (chosen, rejected1-4) ranked pairs
│   │   ├── build_entailment_list.py # Step 4.3: entailment-only labels, ratio as input
│   │   └── utils.py               # IO, is_correct, no_similar, exist_error
│   └── traces/                    # All output files land here
│
├── training/
│   ├── configs/
│   │   ├── dpo_config.yaml        # β, lr, LoRA settings for DPO
│   │   └── listwise_config.yaml   # β, lambdas [1.0, 0.75, 0.5, 0.25], lr, LoRA settings
│   ├── listwise_trainer.py        # ListwiseDataset + LIPO-λ loss + ListwiseTrainer
│   ├── train_dpo.py               # TRL DPOTrainer + LoRA
│   └── train_listwise.py          # ListwiseTrainer + LoRA
│
├── evaluation/
│   └── eval_gsm8k.py              # 8-shot maj@8 + pass@k on GSM8K test (vLLM/HF backends)
│
├── graphs/
│   └── plot_arms.py               # Accuracy bar chart for the 3-arm run
│
├── wandb_utils.py                 # Shared W&B init (used by both trainers + eval)
├── run_arms.sh                    # 3-arm experiment: score → build → train → eval
├── run_entailment.sh              # Steps 1-3: prepare → generate → process
├── smoke_test.sh                  # Full pipeline on 10 questions
├── EXPERIMENTS.md                 # Tiered run plan + W&B monitoring guide
├── NAUTILUS.md                    # Nautilus (NRP) A6000 setup: pod, PVC env, ssh, herdr
└── pod-init.sh                    # Pod bootstrap, re-runnable after evictions (see NAUTILUS.md)
```

---

## Environment (uv)

One environment for everything — training, generation, NLI scoring, and vLLM eval. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; [uv](https://docs.astral.sh/uv/) manages it all:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
uv sync                                            # creates .venv (Python 3.12) from uv.lock
uv run training/train_listwise.py ...              # every command runs through uv
```

Locked stack: **Python 3.12 · torch 2.13 · transformers 5.15 · trl 1.10 · peft 0.20 · vllm 0.27.1**. To change a dependency, edit `pyproject.toml` and run `uv lock && uv sync` — don't `pip install` into the venv by hand.

`eval_gsm8k.py --backend auto` (the default) uses vLLM when importable and falls back to plain transformers otherwise. LoRA adapters need no merging on either backend — vLLM serves them through `LoRARequest`, HF merges them in memory.

> The repo previously ran on a pinned torch 2.4.1 / trl 0.9.6 stack; both trainers were ported to TRL 1.x (`processing_class=`, `max_length`-only truncation) and revalidated. The old `/pvcvolume/venv` is kept as a rollback but is no longer used.

### Pod environment (verified)

| | |
|---|---|
| GPU | 1 × RTX A6000, 48GB |
| HF cache | `HF_HOME=/pvcvolume/hf_cache` (Mistral-7B-v0.1 already pulled, ~14GB) |
| W&B | logged in via `~/.netrc` → runs go online automatically |
| uv cache | `UV_CACHE_DIR=/pvcvolume/uv_cache` |

---

## Experiment Tracking (W&B)

All three scripts (`train_dpo.py`, `train_listwise.py`, `eval_gsm8k.py`) share `wandb_utils.py`.

Auth resolves in this order — no code change needed per machine:
1. `WANDB_API_KEY` env var
2. `~/.netrc` entry for `api.wandb.ai` (what `wandb login` writes)
3. neither → **offline** mode, so a job never blocks on a login prompt (sync later with `wandb sync wandb/offline-run-*`)

Flags available on every script:

```
--wandb_project NAME     # default: clustering-listwise-dpo (or WANDB_PROJECT)
--wandb_run_name NAME    # default: auto (e.g. eval-dpo_mistral7b-8shot-n10)
--wandb_group NAME       # e.g. gsm8k-paper-eval — groups the 3 model runs together
--wandb_tags a,b
--wandb_mode online|offline|disabled
--no_wandb               # shorthand for --wandb_mode disabled
```

Training runs log loss/lr/eval-loss through HF's `WandbCallback` (attached to the run we open first, so the full YAML config lands in `wandb.config`). Eval runs log `running/*` metrics as questions complete and `final/*` + run summary at the end.

Project defaults also live in the training YAMLs (`wandb_project`, `wandb_group`); CLI flags win.

---

## End-to-End Commands

### Smoke test (10 questions — validates full pipeline)

```bash
# 1. Prepare questions
uv run generation/code/prepare_questions.py --output generation/traces/questions.jsonl

# Slice 10
uv run python -c "
import json
data = [json.loads(l) for l in open('generation/traces/questions.jsonl')]
with open('generation/traces/questions_small.jsonl','w') as f:
    [f.write(json.dumps(d)+'\n') for d in data[:10]]
"

# 2. Generate (10 questions × 10 samples)
uv run generation/code/generate_traces.py \
    --input generation/traces/questions_small.jsonl \
    --output generation/traces/raw/raw_traces_small.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --n_samples 10 --max_new_tokens 256

# 3. Sort correct / wrong (no threshold — saves all questions)
uv run generation/code/process_traces.py \
    --input  generation/traces/raw/raw_traces_small.jsonl \
    --output generation/traces/processed/results_small.jsonl

# 4.1 DPO
uv run generation/code/build_pairs.py \
    --input generation/traces/processed/results_small.jsonl \
    --output generation/traces/dpo_pairs_small.jsonl
uv run training/train_dpo.py --config training/configs/dpo_config.yaml \
    --dataset_path generation/traces/dpo_pairs_small.jsonl --output_dir outputs/dpo_small

# 4.2 Listwise (skips questions with <4 wrong traces)
uv run generation/code/build_listwise.py \
    --input generation/traces/processed/results_small.jsonl \
    --output generation/traces/listwise_pairs_small.jsonl --n_per_class 4
uv run training/train_listwise.py --config training/configs/listwise_config.yaml \
    --dataset_path generation/traces/listwise_pairs_small.jsonl --output_dir outputs/listwise_small

# 5. Eval (20 questions, 8-shot, maj@8 + pass@10/5/1)
uv run evaluation/eval_gsm8k.py --model outputs/dpo_small \
    --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 --limit 20 \
    --output results/dpo_small.json --wandb_group smoke-test
```

Or just `bash smoke_test.sh` (any single step: `prepare | generate | process | dpo | listwise | eval`).

---

### Full run (GPU recommended)

```bash
# 1. Prepare
uv run generation/code/prepare_questions.py --output generation/traces/questions.jsonl

# 2. Generate (7473 questions × 30 samples, vLLM continuous batching;
#    --resume lets an interrupted run pick up where it stopped)
uv run generation/code/generate_traces.py \
    --input generation/traces/questions.jsonl \
    --output generation/traces/raw/raw_traces.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --n_samples 30 --max_new_tokens 512 --resume

# 3. Sort
uv run generation/code/process_traces.py \
    --input  generation/traces/raw/raw_traces.jsonl \
    --output generation/traces/processed/results.jsonl

# 4.1 DPO
uv run generation/code/build_pairs.py \
    --input generation/traces/processed/results.jsonl \
    --output generation/traces/dpo_pairs.jsonl --max_pairs 5
uv run training/train_dpo.py --config training/configs/dpo_config.yaml

# 4.2 Listwise
uv run generation/code/build_listwise.py \
    --input generation/traces/processed/results.jsonl \
    --output generation/traces/listwise_pairs.jsonl --n_per_class 5
uv run training/train_listwise.py --config training/configs/listwise_config.yaml

# 4.3 Entailment-labeled listwise (PORT-style)
uv run generation/code/score_entailment.py \
    --input  generation/traces/processed/results.jsonl \
    --output generation/traces/processed/results_scored.jsonl
uv run generation/code/build_entailment_list.py \
    --input  generation/traces/processed/results_scored.jsonl \
    --output generation/traces/entailment_pairs.jsonl \
    --threshold 0.5 --n_good 1 --n_bad 4
uv run training/train_listwise.py --config training/configs/listwise_config.yaml \
    --dataset_path generation/traces/entailment_pairs.jsonl \
    --output_dir outputs/listwise_entailment

# 5. Paper eval — 8-shot maj@8 + pass@10/5/1 for every arm + base
bash run_arms.sh eval
```

---

### The 3-arm experiment (`run_arms.sh`)

The main experiment: three LIPO-λ listwise arms sharing one trace pool, all evaluated against the untrained base model.

| Arm | Gate | Ranking |
|---|---|---|
| `length` | correctness | trace length |
| `gated_lg` | correctness | bidirectional entailment (nli-deberta-v3-**large**) |
| `ungated_lg` | **entailment only** (score ≥ threshold, correctness ignored — PORT-style) | entailment |

```bash
bash run_entailment.sh prepare generate process   # steps 1-3 → results_${N}.jsonl
bash run_arms.sh                                   # score → build → train ×4 → eval ×5
EVAL_LIMIT=500 bash run_arms.sh eval               # quicker eval pass
```

Env knobs: `MODEL N_QUESTIONS THRESHOLD EVAL_LIMIT N_SHOT EVAL_SAMPLES MAJ_K PASS_K EVAL_TEMP RESULTS_DIR WANDB_GROUP`.

---

### Paper evaluation (maj@8 + pass@k)

`run_arms.sh eval` runs base + all three arms through the identical protocol and puts them in one W&B group so they overlay. Backend flags on `eval_gsm8k.py`:

```
--backend auto|vllm|hf         # auto: vLLM if importable, else transformers
--vllm_chunk 128               # questions per generate() call — only the checkpoint/resume granularity
--gpu_memory_utilization 0.90  # vLLM VRAM budget (weights + KV cache)
--max_model_len 2048           # auto-raised if the few-shot prefix needs more
--batch_size 4                 # HF backend only
```

**Cost (measured on this pod's A6000).** 1319 questions × 10 samples = 13,190 completions per model, with a 1150-token 8-shot prefix.

| Backend | Measured throughput | Full test split, per model |
|---|---|---|
| vLLM (default) | **0.87 s/question** (64 q in 56 s, continuous batching) | **~20 min** + ~2 min engine startup |
| HF transformers | 13 s/question at `BATCH_SIZE=4` (~28GB) | ~5 h |

Trace generation uses the same backends: 10 questions × 10 samples × 256 tokens ran in ~6 s on vLLM (vs minutes on HF), which extrapolates to roughly **7 h for the full 7473 × 30 × 512-token generation run** — checkpointed, so it survives interruptions via `--resume`.

So all four evals (base + 3 arms) finish in about 100 minutes on vLLM. Practical notes:

- Every run writes per-question records incrementally and `run_arms.sh` passes `--resume`, so an interrupted job restarts where it stopped (delete `results/*_detail.jsonl` to force a clean rerun).
- vLLM's engine schedules everything itself — `--vllm_chunk` (default 128) only sets how often records hit disk. `BATCH_SIZE` matters only on the HF fallback; there, 4 beat 8 (a batch waits on its slowest sequence).
- First vLLM start on a model compiles CUDA graphs (~1 min extra); later starts reuse the cache.

Direct invocation for one model:

```bash
uv run evaluation/eval_gsm8k.py \
    --model outputs/arm_gated_lg_1000 \
    --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 \
    --temperature 0.8 --batch_size 4 --max_new_tokens 400 \
    --output results/arm_gated_lg_1000.json \
    --wandb_group gsm8k-paper-eval --resume
```

`--model` accepts an HF id, a full checkpoint, or a **LoRA adapter directory** — adapters are detected via `adapter_config.json`, loaded onto their recorded base model and merged for full-speed inference (`--base_model` overrides the base).

---

## LIPO-λ Loss (verbatim from LPOI)

Scoring function per response:
```
score_i = β × (log π_policy(response_i | prompt) − log π_ref(response_i | prompt))
```

Cascading softmax (`lpoi_dpo_trainer_5img.py:1537–1542`):
```
losses1 = −log[ exp(s_chosen) / (s_chosen + s_r1 + s_r2 + s_r3 + s_r4) ]
losses2 = −log[ exp(s_r1)     / (s_r1 + s_r2 + s_r3 + s_r4) ]
losses3 = −log[ exp(s_r2)     / (s_r2 + s_r3 + s_r4) ]
losses4 = −log[ exp(s_r3)     / (s_r3 + s_r4) ]

total = λ1×losses1 + λ2×losses2 + λ3×losses3 + λ4×losses4
      = 1.0×L1    + 0.75×L2    + 0.5×L3     + 0.25×L4
```

---

## Known Limitations

- **Ranking proxy is heuristic**: `build_listwise.py` ranks by bidirectional NLI entailment when scores are present (length is the fallback), but NLI measures semantic consistency, not arithmetic correctness - a trace with one calculation slip still scores high.
- **Chosen is anchored to the reference**: correct-trace scores are measured against the shortest correct trace, which scores ~1 against itself, so `chosen` is almost always the reference (56/60 on the 100-question run).
- **Sampling nuance on vLLM**: generation and eval both default to the vLLM backend now. vLLM samples all `n` completions in one continuous-batching pass, so per-question wall-time collapses, but token-level RNG differs from HF's `generate` — trace pools produced by the two backends are statistically equivalent (same temperature), not bit-identical.
- **Train/eval prompt scaffold**: `utils.format_prompt` is now the single definition used by `generate_traces.py`, `build_pairs.py`, `build_listwise.py` and the eval. Preference data built before this change used the bare question as the prompt while generation used `Question: …\nAnswer:` — rebuild any `*_pairs*.jsonl` / `arm_*.jsonl` that predate it.
- **Eval prompt is few-shot, training prompt is zero-shot**: the paper protocol prepends 8 exemplars at eval time; the preference lists are single-turn. Pass `--n_shot 0` to score the models in their training format instead.
