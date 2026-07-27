#!/usr/bin/env bash
# Four-arm comparison: length baseline, gated small, gated large, ungated large.
#
# Arms (all trained with the same LIPO-lambda loss and 1 chosen + 4 rejected shape):
#   length      gated: chosen = shortest correct, rejected ranked by length
#   gated_sm    gated: ranked by bidirectional entailment (nli-deberta-v3-small)
#   gated_lg    gated: ranked by bidirectional entailment (nli-deberta-v3-large)
#   ungated_lg  no gate: good/bad from binarized large-NLI score (PORT-style)
#
# Assumes steps 1-3 already ran (see run_entailment.sh prepare/generate/process),
# i.e. generation/traces/processed/results_${N_QUESTIONS}.jsonl exists.
#
# Usage:
#   bash run_arms.sh                 # score -> build -> train -> eval
#   bash run_arms.sh score           # any subset of steps, in order
#   bash run_arms.sh build train
#   EVAL_LIMIT=500 bash run_arms.sh eval   # quicker eval pass
#
# Parameters (env vars):
#   MODEL=mistralai/Mistral-7B-v0.1 N_QUESTIONS=1000 THRESHOLD=0.5
#   EVAL_BATCH=8 EVAL_LIMIT=<empty = full test split>

set -e
cd "$(dirname "$0")"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
N_QUESTIONS="${N_QUESTIONS:-1000}"
THRESHOLD="${THRESHOLD:-0.5}"
EVAL_BATCH="${EVAL_BATCH:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

TAG="${N_QUESTIONS}"
PROCESSED="generation/traces/processed/results_${TAG}.jsonl"
SCORED_SM="generation/traces/processed/results_${TAG}_scored_sm.jsonl"
SCORED_LG="generation/traces/processed/results_${TAG}_scored_lg.jsonl"

ARMS="length gated_sm gated_lg ungated_lg"

step_score() {
    echo "=== Score: bidirectional entailment, small then large NLI ==="
    python generation/code/score_entailment.py \
        --input "$PROCESSED" --output "$SCORED_SM" \
        --model cross-encoder/nli-deberta-v3-small --batch_size 32
    python generation/code/score_entailment.py \
        --input "$PROCESSED" --output "$SCORED_LG" \
        --model cross-encoder/nli-deberta-v3-large --batch_size 16
}

step_build() {
    echo "=== Build: 4 datasets ==="
    python generation/code/build_listwise.py \
        --input "$SCORED_SM" --output "generation/traces/arm_length_${TAG}.jsonl" \
        --ranking_method length
    python generation/code/build_listwise.py \
        --input "$SCORED_SM" --output "generation/traces/arm_gated_sm_${TAG}.jsonl" \
        --ranking_method entailment
    python generation/code/build_listwise.py \
        --input "$SCORED_LG" --output "generation/traces/arm_gated_lg_${TAG}.jsonl" \
        --ranking_method entailment
    python generation/code/build_entailment_list.py \
        --input "$SCORED_LG" --output "generation/traces/arm_ungated_lg_${TAG}.jsonl" \
        --threshold "$THRESHOLD"
}

step_train() {
    for arm in $ARMS; do
        echo "=== Train: $arm ==="
        python training/train_listwise.py \
            --config training/configs/listwise_config.yaml \
            --dataset_path "generation/traces/arm_${arm}_${TAG}.jsonl" \
            --output_dir "outputs/arm_${arm}_${TAG}"
    done
}

step_eval() {
    LIMIT_FLAG=""
    [ -n "$EVAL_LIMIT" ] && LIMIT_FLAG="--limit $EVAL_LIMIT"
    echo "=== Eval: base model ==="
    python evaluation/eval_gsm8k.py \
        --model "$MODEL" --output "results_base_${TAG}.json" \
        --batch_size "$EVAL_BATCH" $LIMIT_FLAG
    for arm in $ARMS; do
        echo "=== Eval: $arm ==="
        python evaluation/eval_gsm8k.py \
            --model "outputs/arm_${arm}_${TAG}" --output "results_arm_${arm}_${TAG}.json" \
            --batch_size "$EVAL_BATCH" $LIMIT_FLAG
    done
    echo "=== Summary ==="
    for f in "results_base_${TAG}.json" $(for a in $ARMS; do echo "results_arm_${a}_${TAG}.json"; done); do
        [ -f "$f" ] && echo "$f: $(python -c "import json;print(json.load(open('$f'))['accuracy'])")"
    done
}

if [ $# -eq 0 ]; then
    step_score
    step_build
    step_train
    step_eval
else
    for step in "$@"; do
        "step_${step}"
    done
fi
