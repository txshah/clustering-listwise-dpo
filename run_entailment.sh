#!/usr/bin/env bash
# Entailment run — full pipeline on N_QUESTIONS (default 1000), then LIPO training + eval.
# Sized for a single A6000; generation and eval use the vLLM backend automatically.
#
# Usage:
#   bash run_entailment.sh                # run all steps (1000 questions, threshold 0.5)
#   bash run_entailment.sh prepare        # step 1 only
#   bash run_entailment.sh generate       # step 2 only (the slow one)
#   bash run_entailment.sh process        # step 3 only
#   bash run_entailment.sh score          # step 4 only (NLI entailment scoring)
#   bash run_entailment.sh build          # step 5 only (binarize + 1:4 lists)
#   bash run_entailment.sh train          # step 6 only (LIPO-lambda listwise training)
#   bash run_entailment.sh eval           # step 7 only (GSM8K test accuracy)
#
# Parameters (env vars):
#   N_QUESTIONS=1000 THRESHOLD=0.5 bash run_entailment.sh
#   THRESHOLD=0.7 bash run_entailment.sh build train eval   # rebuild lists + retrain at 0.7

set -e
cd "$(dirname "$0")"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
N_QUESTIONS="${N_QUESTIONS:-1000}"
N_SAMPLES="${N_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
THRESHOLD="${THRESHOLD:-0.5}"       # entailment score >= THRESHOLD -> good, else bad
N_GOOD="${N_GOOD:-1}"
N_BAD="${N_BAD:-4}"
LENGTH_WINDOW="${LENGTH_WINDOW:-0}" # 0 = dedup disabled, same as the notebook

TAG="${N_QUESTIONS}"
QUESTIONS="generation/traces/questions_${TAG}.jsonl"
RAW="generation/traces/raw/raw_traces_${TAG}.jsonl"
PROCESSED="generation/traces/processed/results_${TAG}.jsonl"
SCORED="generation/traces/processed/results_${TAG}_scored.jsonl"
PAIRS="generation/traces/entailment_pairs_${TAG}_t${THRESHOLD}.jsonl"
OUTPUT_DIR="outputs/listwise_entailment_${TAG}_t${THRESHOLD}"
EVAL_OUT="results_listwise_entailment_${TAG}_t${THRESHOLD}.json"

step_prepare() {
    echo "=== Step 1: prepare ${N_QUESTIONS} questions ==="
    uv run generation/code/prepare_questions.py \
        --output "$QUESTIONS" \
        --limit "$N_QUESTIONS"
}

step_generate() {
    echo "=== Step 2: generate traces (${N_QUESTIONS} questions x ${N_SAMPLES} samples) ==="
    uv run generation/code/generate_traces.py \
        --input "$QUESTIONS" \
        --output "$RAW" \
        --model "$MODEL" \
        --n_samples "$N_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --resume
}

step_process() {
    echo "=== Step 3: sort correct / wrong (length_window=${LENGTH_WINDOW}) ==="
    uv run generation/code/process_traces.py \
        --input "$RAW" \
        --output "$PROCESSED" \
        --length_window "$LENGTH_WINDOW"
}

step_score() {
    echo "=== Step 4: NLI entailment scoring ==="
    uv run generation/code/score_entailment.py \
        --input "$PROCESSED" \
        --output "$SCORED" \
        --model "$NLI_MODEL" \
        --batch_size 32
}

step_build() {
    echo "=== Step 5: build ${N_GOOD}:${N_BAD} entailment lists (threshold=${THRESHOLD}) ==="
    uv run generation/code/build_entailment_list.py \
        --input "$SCORED" \
        --output "$PAIRS" \
        --threshold "$THRESHOLD" \
        --n_good "$N_GOOD" \
        --n_bad "$N_BAD"
}

step_train() {
    echo "=== Step 6: LIPO-lambda listwise training ==="
    uv run training/train_listwise.py \
        --config training/configs/listwise_config.yaml \
        --dataset_path "$PAIRS" \
        --output_dir "$OUTPUT_DIR"
}

step_eval() {
    echo "=== Step 7: evaluate on GSM8K test (8-shot maj@8 + pass@10/5/1) ==="
    uv run evaluation/eval_gsm8k.py \
        --model "$OUTPUT_DIR" \
        --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 \
        --output "$EVAL_OUT" \
        --wandb_group "entailment-${TAG}" --resume
    cat "$EVAL_OUT"
}

if [ $# -eq 0 ]; then
    step_prepare
    step_generate
    step_process
    step_score
    step_build
    step_train
    step_eval
else
    for step in "$@"; do
        "step_${step}"
    done
fi
