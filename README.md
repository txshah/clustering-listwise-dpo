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
5. Iterate until each question has **N/2 = 5 correct + 5 incorrect** traces

The paper uses 4+4 with 6 parallel GPU workers. We use 5+5 on a single machine.

### 2. Step 4.1 — D_naive DPO (mirrors Step-Controlled-DPO training)

From the trace pool, pair every correct trace with every incorrect trace (capped at `max_pairs` per question). Format as `{chosen: [messages], rejected: [messages]}` — the message-list schema that `apply_chat_template` in the paper's `data.py` expects.

Train with **TRL's DPOTrainer**, same hyperparameters as `config_full.yaml` in the paper (β=0.1, lr=1e-6).

### 3. Step 4.2 — Listwise LIPO-λ (mirrors LPOI)

From the same trace pool, build one listwise sample per question:
- **chosen**: shortest correct trace (efficiency proxy)
- **rejected1–4**: 4 incorrect traces ranked by length (shortest = least bad)

The ranking is a weak proxy (acknowledged limitation — no reward model). The next step would be scoring with log-probabilities or a verifier.

Train with a custom **ListwiseTrainer** that implements the LIPO-λ loss from [LPOI](https://github.com/fatemehpesaran310/lpoi) (`lpoi_dpo_trainer_5img.py:1537–1542`), adapted for text (image-specific batching removed, loss function unchanged).

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
| 4 correct + 4 wrong threshold | 5 + 5 (N/2) | Match N=10 experiment design |
| `utils.py` in sibling folder | Consolidated into `generation/code/utils.py` | Single import path |
| `get_initial_data.py` (GPU split utility) | Replaced by `prepare_questions.py` | Just load GSM8K directly |

### From LPOI

| Original | Here | Why |
|---|---|---|
| 5 image variants (pixel masking) | 5 text traces (correct + 4 ranked wrong) | Text domain, no images |
| Ranking from pixel degradation | Ranking by trace length | No reward model available |
| `lpoi.py` data loader (image paths) | `ListwiseDataset` (plain text) | Text-only |
| `lpoi_dpo_trainer_5img.py` full trainer | `listwise_trainer.py` (loss unchanged) | Stripped image batching |
| 3-component loss (text DPO + image DPO + anchor) | LIPO-λ only | Simplified; anchor = implicit via β |

---

## File Hierarchy

```
clustering-listwise-dpo/
│
├── generation/
│   ├── code/
│   │   ├── prepare_questions.py   # Download GSM8K → questions.jsonl
│   │   ├── generate_traces.py     # LLM sampling at temp=1 → raw_traces.jsonl
│   │   ├── process_traces.py      # Sort correct/wrong, dedup → results.jsonl
│   │   ├── build_pairs.py         # Step 4.1: (chosen, rejected) message-list pairs
│   │   ├── build_listwise.py      # Step 4.2: (chosen, rejected1-4) ranked pairs
│   │   └── utils.py               # IO, is_correct, no_similar, exist_error
│   └── traces/                    # All output files land here
│
├── training/
│   ├── configs/
│   │   ├── dpo_config.yaml        # β, lr, batch size for DPO
│   │   └── listwise_config.yaml   # β, lambdas [1.0, 0.75, 0.5, 0.25], lr
│   ├── listwise_trainer.py        # ListwiseDataset + LIPO-λ loss + ListwiseTrainer
│   ├── train_dpo.py               # Wraps TRL DPOTrainer
│   └── train_listwise.py          # Runs ListwiseTrainer
│
└── evaluation/
    └── eval_gsm8k.py              # Greedy decoding + end accuracy on GSM8K test
```

---

## End-to-End Commands

### Smoke test first (10 questions — validates full pipeline end-to-end)

Runs the entire pipeline on 10 questions so you can confirm every step works
before committing to the full overnight job on a GPU machine.

Threshold=2 means: a question is "done" once we have 2 correct + 2 incorrect
traces for it. Lower = easier to satisfy with fewer samples, so the small test
completes quickly. The full run uses 5+5.

```bash
# Slice 10 questions from the full set
python -c "
import json
data = [json.loads(l) for l in open('generation/traces/questions.jsonl')]
with open('generation/traces/questions_small.jsonl','w') as f:
    [f.write(json.dumps(d)+'\n') for d in data[:10]]
"

# 2. Generate (10 questions × 10 samples = 100 traces)
python generation/code/generate_traces.py \
    --input          generation/traces/questions_small.jsonl \
    --output         generation/traces/raw/raw_traces_small.jsonl \
    --model          mistralai/Mistral-7B-v0.1 \
    --n_samples      10 \
    --max_new_tokens 256

# 3. Sort into correct / wrong (threshold=2 per class)
python generation/code/process_traces.py \
    --input          generation/traces/raw/raw_traces_small.jsonl \
    --output         generation/traces/processed/results_small.jsonl \
    --correct_thresh 2 \
    --wrong_thresh   2

# 4.1 Build DPO pairs and train
python generation/code/build_pairs.py \
    --input     generation/traces/processed/results_small.jsonl \
    --output    generation/traces/dpo_pairs_small.jsonl

python training/train_dpo.py \
    --config       training/configs/dpo_config.yaml \
    --dataset_path generation/traces/dpo_pairs_small.jsonl \
    --output_dir   outputs/dpo_small

# 4.2 Build listwise pairs and train
python generation/code/build_listwise.py \
    --input       generation/traces/processed/results_small.jsonl \
    --output      generation/traces/listwise_pairs_small.jsonl \
    --n_per_class 2

python training/train_listwise.py \
    --config       training/configs/listwise_config.yaml \
    --dataset_path generation/traces/listwise_pairs_small.jsonl \
    --output_dir   outputs/listwise_small

# 5. Evaluate both
python evaluation/eval_gsm8k.py --model outputs/dpo_small      --output results_dpo_small.json
python evaluation/eval_gsm8k.py --model outputs/listwise_small --output results_listwise_small.json
```

Accuracy on 10 training questions won't be meaningful, but every step
producing output without errors means the full run is safe to launch.

---

### Full run (GPU recommended — A100/H100 for the generation step)

```bash
# 1. Prepare questions (GSM8K train split)
python generation/code/prepare_questions.py \
    --output generation/traces/questions.jsonl

# 2. Generate raw traces (temp=1, 30 samples/question)
#    7473 questions × 30 samples — run on GPU, not MPS
python generation/code/generate_traces.py \
    --input          generation/traces/questions.jsonl \
    --output         generation/traces/raw/raw_traces.jsonl \
    --model          mistralai/Mistral-7B-v0.1 \
    --n_samples      30 \
    --max_new_tokens 512

# 3. Sort into correct / wrong pools (5 of each)
python generation/code/process_traces.py \
    --input          generation/traces/raw/raw_traces.jsonl \
    --output         generation/traces/processed/results.jsonl \
    --correct_thresh 5 \
    --wrong_thresh   5

# If results_needs_more.jsonl is non-empty, re-run steps 2-3 on it:
python generation/code/generate_traces.py \
    --input  generation/traces/processed/results_needs_more.jsonl \
    --output generation/traces/raw/raw_traces_round2.jsonl \
    --model  mistralai/Mistral-7B-v0.1 \
    --n_samples 30 --max_new_tokens 512
python generation/code/process_traces.py \
    --input  generation/traces/raw/raw_traces_round2.jsonl \
    --output generation/traces/processed/results.jsonl \
    --correct_thresh 5 --wrong_thresh 5

# 4.1 Build DPO pairs and train
python generation/code/build_pairs.py \
    --input     generation/traces/processed/results.jsonl \
    --output    generation/traces/dpo_pairs.jsonl \
    --max_pairs 5

python training/train_dpo.py --config training/configs/dpo_config.yaml

# 4.2 Build listwise pairs and train
python generation/code/build_listwise.py \
    --input        generation/traces/processed/results.jsonl \
    --output       generation/traces/listwise_pairs.jsonl \
    --n_per_class  5

python training/train_listwise.py --config training/configs/listwise_config.yaml

# 5. Evaluate both
python evaluation/eval_gsm8k.py --model outputs/dpo_mistral7b      --output results_dpo.json
python evaluation/eval_gsm8k.py --model outputs/listwise_mistral7b --output results_listwise.json
```

---

## LIPO-λ Loss (verbatim from LPOI)

Scoring function per response:
```
score_i = β × (log π_policy(response_i | prompt) − log π_ref(response_i | prompt))
```

Cascading softmax (from `lpoi_dpo_trainer_5img.py:1537–1542`):
```
losses1 = −log[ exp(s_chosen) / (s_chosen + s_r1 + s_r2 + s_r3 + s_r4) ]
losses2 = −log[ exp(s_r1)     / (s_r1 + s_r2 + s_r3 + s_r4) ]
losses3 = −log[ exp(s_r2)     / (s_r2 + s_r3 + s_r4) ]
losses4 = −log[ exp(s_r3)     / (s_r3 + s_r4) ]

total = λ1×losses1 + λ2×losses2 + λ3×losses3 + λ4×losses4
      = 1.0×L1    + 0.75×L2    + 0.5×L3     + 0.25×L4
```

Higher λ weights the top of the ranking more heavily — the model is penalised most for getting the chosen-vs-all comparison wrong.

---

## Known Limitations

- **Ranking proxy is weak**: incorrect traces are ranked by length (shorter = better), not by a reward model or verifier. This is the primary limitation of the listwise experiment. Replacing `rank_by_length` in `build_listwise.py` with a reward model score would make this a proper listwise DPO setup.
- **No preference within correct traces**: all correct traces are treated equally; only one is used as `chosen`. A future improvement would rank correct traces too (e.g., by solution length or log-probability).
- **Single-machine generation**: the original paper uses 6 parallel GPU workers. For large-scale runs, parallelise `generate_traces.py` with vLLM batch inference.
