# Clustering Listwise DPO

## What This Is

This repo runs two preference-optimization experiments on Mistral-7B for math reasoning (GSM8K), then compares their end accuracy:

| Experiment | Method | Loss |
|---|---|---|
| 4.1 | D_naive DPO | Binary sigmoid: chosen vs rejected |
| 4.2 | Listwise (LIPO-λ) | Cascading softmax over 1 chosen + 4 ranked rejected |

Both experiments share the same trace generation pipeline. They differ only in how the traces are formatted into training pairs and what loss is applied.

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

From the trace pool, pair every correct trace with every incorrect trace (capped at `max_pairs` per question). Flattened to plain strings (`prompt`, `chosen`, `rejected`) that TRL 0.9.6's `DPOTrainer.tokenize_row` expects.

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

Both models evaluated on GSM8K test split with greedy decoding (temp=0). Metric: **end accuracy** — does the model's final number match the ground truth?

---

## Key Changes from the Source Repos

### From Step-Controlled-DPO

| Original | Here | Why |
|---|---|---|
| Jupyter kernel for code execution (LCE) | Removed | Not needed for plain-text traces |
| 6 GPU workers, IP-based sharding | Single machine | Lightweight setup |
| `InferenceClient` (vLLM server) | HF `AutoModelForCausalLM` | No separate server needed |
| Threshold gate (4+4 minimum per question) | Removed | Every question saved as-is |
| `utils.py` in sibling folder | Consolidated into `generation/code/utils.py` | Single import path |
| `get_initial_data.py` (GPU split utility) | Replaced by `prepare_questions.py` | Just load GSM8K directly |
| Full model fine-tune | LoRA (r=16, α=32 on q/v projections) | Fits in ~14GB, shared reference |

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
│   │   ├── generate_traces.py     # LLM sampling at temp=1 → raw_traces.jsonl
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
│   └── eval_gsm8k.py              # Greedy decoding + end accuracy on GSM8K test
│
├── smoke_test_colab.ipynb         # Self-contained 10-question pipeline (DPO + listwise)
└── entailment_run_colab.ipynb     # Colab wrapper: 4-arm 1000-question run (Drive persistence + repo scripts)
```

---

## Dependencies

```
torch==2.4.1
transformers==4.45.1
trl==0.9.6
peft==0.13.2
accelerate==1.13.0
datasets==4.8.4
pyyaml
math-verify   # sympy-based answer checking (correctness oracle in utils.py + eval_gsm8k.py)
```

Install:
```bash
pip install "torch==2.4.1" "transformers==4.45.1" "trl==0.9.6" "peft==0.13.2" "accelerate==1.13.0" "datasets==4.8.4" pyyaml math-verify
```

---

## End-to-End Commands

### Smoke test (10 questions — validates full pipeline)

```bash
# 1. Prepare questions
python generation/code/prepare_questions.py --output generation/traces/questions.jsonl

# Slice 10
python -c "
import json
data = [json.loads(l) for l in open('generation/traces/questions.jsonl')]
with open('generation/traces/questions_small.jsonl','w') as f:
    [f.write(json.dumps(d)+'\n') for d in data[:10]]
"

# 2. Generate (10 questions × 10 samples)
python generation/code/generate_traces.py \
    --input generation/traces/questions_small.jsonl \
    --output generation/traces/raw/raw_traces_small.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --n_samples 10 --max_new_tokens 256

# 3. Sort correct / wrong (no threshold — saves all questions)
python generation/code/process_traces.py \
    --input  generation/traces/raw/raw_traces_small.jsonl \
    --output generation/traces/processed/results_small.jsonl

# 4.1 DPO
python generation/code/build_pairs.py \
    --input generation/traces/processed/results_small.jsonl \
    --output generation/traces/dpo_pairs_small.jsonl
python training/train_dpo.py --config training/configs/dpo_config.yaml \
    --dataset_path generation/traces/dpo_pairs_small.jsonl --output_dir outputs/dpo_small

# 4.2 Listwise (skips questions with <4 wrong traces)
python generation/code/build_listwise.py \
    --input generation/traces/processed/results_small.jsonl \
    --output generation/traces/listwise_pairs_small.jsonl --n_per_class 4
python training/train_listwise.py --config training/configs/listwise_config.yaml \
    --dataset_path generation/traces/listwise_pairs_small.jsonl --output_dir outputs/listwise_small

# 5. Eval
python evaluation/eval_gsm8k.py --model outputs/dpo_small --output results_dpo_small.json
python evaluation/eval_gsm8k.py --model outputs/listwise_small --output results_listwise_small.json
```

---

### Full run (GPU recommended)

```bash
# 1. Prepare
python generation/code/prepare_questions.py --output generation/traces/questions.jsonl

# 2. Generate (7473 questions × 30 samples — use A100/H100)
python generation/code/generate_traces.py \
    --input generation/traces/questions.jsonl \
    --output generation/traces/raw/raw_traces.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --n_samples 30 --max_new_tokens 512

# 3. Sort
python generation/code/process_traces.py \
    --input  generation/traces/raw/raw_traces.jsonl \
    --output generation/traces/processed/results.jsonl

# 4.1 DPO
python generation/code/build_pairs.py \
    --input generation/traces/processed/results.jsonl \
    --output generation/traces/dpo_pairs.jsonl --max_pairs 5
python training/train_dpo.py --config training/configs/dpo_config.yaml

# 4.2 Listwise
python generation/code/build_listwise.py \
    --input generation/traces/processed/results.jsonl \
    --output generation/traces/listwise_pairs.jsonl --n_per_class 5
python training/train_listwise.py --config training/configs/listwise_config.yaml

# 4.3 Entailment-labeled listwise (PORT-style)
python generation/code/score_entailment.py \
    --input  generation/traces/processed/results.jsonl \
    --output generation/traces/processed/results_scored.jsonl
python generation/code/build_entailment_list.py \
    --input  generation/traces/processed/results_scored.jsonl \
    --output generation/traces/entailment_pairs.jsonl \
    --threshold 0.5 --n_good 1 --n_bad 4
python training/train_listwise.py --config training/configs/listwise_config.yaml \
    --dataset_path generation/traces/entailment_pairs.jsonl \
    --output_dir outputs/listwise_entailment

# 5. Eval
python evaluation/eval_gsm8k.py --model outputs/dpo_mistral7b --output results_dpo.json
python evaluation/eval_gsm8k.py --model outputs/listwise_mistral7b --output results_listwise.json
```

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
- **Single-machine generation**: parallelise with vLLM for large-scale runs.
