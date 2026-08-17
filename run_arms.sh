#!/usr/bin/env bash
# Three-arm comparison: length baseline, gated large, ungated large.
#
# Arms (all trained with the same LIPO-lambda loss and 1 chosen + 4 rejected shape):
#   length      gated: chosen = shortest correct, rejected ranked by length
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
#   bash run_arms.sh summary         # aggregate finished evals (mean±std over seeds)
#
# Training runs once per arm per seed (SEEDS, default "42 43") and skips
# output dirs that already contain an adapter, so the whole script is re-runnable.
#
# lr sweep: LR=1e-4 SEEDS=42 bash run_arms.sh train eval
#   LR overrides the config learning_rate and suffixes every output/result
#   (arm_gated_lg_1000_lr1e-4_s42), so sweep runs never collide with the
#   main runs at the config lr.
#
# Parameters (env vars):
#   MODEL=mistralai/Mistral-7B-v0.1 N_QUESTIONS=1000 THRESHOLD=0.5
#   SEEDS="42 43"  LR=<empty = config lr>  EVAL_LIMIT=<empty = full test split>
#   N_SHOT=8 EVAL_SAMPLES=10 MAJ_K=8 PASS_K=1,5,10 EVAL_TEMP=0.8
#   RESULTS_DIR=results WANDB_GROUP=arms-${N_QUESTIONS}

set -e
cd "$(dirname "$0")"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
N_QUESTIONS="${N_QUESTIONS:-1000}"
THRESHOLD="${THRESHOLD:-0.5}"
SEEDS="${SEEDS:-42 43}"        # one training run per arm per seed
LR="${LR:-}"                   # override learning_rate; runs get an _lr<LR> suffix
EVAL_LIMIT="${EVAL_LIMIT:-}"

# Paper eval protocol (Mistral 7B paper: 8-shot maj@8; one pass gives pass@k too)
N_SHOT="${N_SHOT:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10}"
MAJ_K="${MAJ_K:-8}"
PASS_K="${PASS_K:-1,5,10}"
EVAL_TEMP="${EVAL_TEMP:-0.8}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WANDB_GROUP="${WANDB_GROUP:-arms-${N_QUESTIONS}}"
mkdir -p "$RESULTS_DIR"

TAG="${N_QUESTIONS}"
PROCESSED="generation/traces/processed/results_${TAG}.jsonl"
SCORED_LG="generation/traces/processed/results_${TAG}_scored_lg.jsonl"

ARMS="${ARMS:-length gated_lg ungated_lg}"   # override to sweep a subset, e.g. ARMS=length

step_score() {
    echo "=== Score: bidirectional entailment (nli-deberta-v3-large) ==="
    uv run generation/code/score_entailment.py \
        --input "$PROCESSED" --output "$SCORED_LG" \
        --model cross-encoder/nli-deberta-v3-large --batch_size 16
}

step_build() {
    echo "=== Build: 3 datasets ==="
    # length ranking ignores the scores; sharing the scored file keeps one input
    uv run generation/code/build_listwise.py \
        --input "$SCORED_LG" --output "generation/traces/arm_length_${TAG}.jsonl" \
        --ranking_method length
    uv run generation/code/build_listwise.py \
        --input "$SCORED_LG" --output "generation/traces/arm_gated_lg_${TAG}.jsonl" \
        --ranking_method entailment
    uv run generation/code/build_entailment_list.py \
        --input "$SCORED_LG" --output "generation/traces/arm_ungated_lg_${TAG}.jsonl" \
        --threshold "$THRESHOLD"
}

step_train() {
    LR_SUFFIX=""
    LR_FLAG=""
    if [ -n "$LR" ]; then
        LR_SUFFIX="_lr${LR}"
        LR_FLAG="--learning_rate $LR"
    fi
    for arm in $ARMS; do
        for seed in $SEEDS; do
            out="outputs/arm_${arm}_${TAG}${LR_SUFFIX}_s${seed}"
            if [ -f "$out/adapter_model.safetensors" ]; then
                echo "=== Train: $arm seed=$seed${LR:+ lr=$LR} — already done, skipping ==="
                continue
            fi
            echo "=== Train: $arm seed=$seed${LR:+ lr=$LR} ==="
            uv run training/train_listwise.py \
                --config training/configs/listwise_config.yaml \
                --dataset_path "generation/traces/arm_${arm}_${TAG}.jsonl" \
                --output_dir "$out" \
                --seed "$seed" $LR_FLAG \
                --wandb_group "$WANDB_GROUP" \
                --wandb_run_name "train-${arm}-${TAG}${LR_SUFFIX}-s${seed}" \
                --wandb_tags "arm:${arm},seed:${seed},tag:${TAG}${LR:+,lr:${LR}}"
        done
    done
}

run_one_eval() {
    local name="$1" model="$2"
    local LIMIT_FLAG=""
    [ -n "$EVAL_LIMIT" ] && LIMIT_FLAG="--limit $EVAL_LIMIT"
    echo "=== Eval: $name (${N_SHOT}-shot, ${EVAL_SAMPLES} samples @ T=${EVAL_TEMP}) ==="
    uv run evaluation/eval_gsm8k.py \
        --model "$model" \
        --n_shot "$N_SHOT" --n_samples "$EVAL_SAMPLES" \
        --maj_k "$MAJ_K" --pass_k "$PASS_K" --temperature "$EVAL_TEMP" \
        --output "$RESULTS_DIR/${name}_${TAG}.json" \
        --wandb_group "$WANDB_GROUP" --wandb_run_name "eval-${name}-${TAG}" \
        --resume $LIMIT_FLAG
}

step_eval() {
    LR_SUFFIX=""
    [ -n "$LR" ] && LR_SUFFIX="_lr${LR}"
    run_one_eval base "$MODEL"
    for arm in $ARMS; do
        for seed in $SEEDS; do
            run_one_eval "arm_${arm}${LR_SUFFIX}_s${seed}" \
                         "outputs/arm_${arm}_${TAG}${LR_SUFFIX}_s${seed}"
        done
    done
    step_summary
}

step_summary() {
    echo "=== Summary: mean ± std across seeds (maj@${MAJ_K} + pass@k)${LR:+ [lr=$LR]} ==="
    RESULTS_DIR="$RESULTS_DIR" TAG="$TAG" ARMS="$ARMS" SEEDS="$SEEDS" MAJ_K="$MAJ_K" LR="$LR" \
    uv run python - <<'EOF'
import json, os, statistics

results_dir, tag = os.environ["RESULTS_DIR"], os.environ["TAG"]
arms, seeds = os.environ["ARMS"].split(), os.environ["SEEDS"].split()
lr_suffix = f"_lr{os.environ['LR']}" if os.environ.get("LR") else ""
metrics_keys = [f"maj@{os.environ['MAJ_K']}", "pass@1", "pass@5", "pass@10"]

def show(name, runs):
    if not runs:
        print(f"{name:<14} (no results yet)")
        return
    parts = []
    for k in metrics_keys:
        vals = [r[k] for r in runs if k in r]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        parts.append(f"{k} {mean:.3f}±{std:.3f}")
    print(f"{name:<14} n_seeds={len(runs)}  " + "  ".join(parts))

def load(path):
    try:
        with open(path) as f:
            return json.load(f)["metrics"]
    except FileNotFoundError:
        return None

base = load(os.path.join(results_dir, f"base_{tag}.json"))
show("base", [base] if base else [])
for arm in arms:
    runs = [m for s in seeds
            if (m := load(os.path.join(results_dir, f"arm_{arm}{lr_suffix}_s{s}_{tag}.json")))]
    show(arm, runs)
EOF
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
