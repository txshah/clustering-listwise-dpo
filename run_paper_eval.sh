#!/usr/bin/env bash
# Mistral-paper GSM8K evaluation: 8-shot, maj@8, plus pass@10 / pass@5 / pass@1.
#
# One generation pass per model (N samples per question) yields every metric, so
# maj@8 and pass@k cost the same as maj@8 alone.
#
# Usage:
#   bash run_paper_eval.sh                 # base + dpo + listwise, full test split
#   bash run_paper_eval.sh base            # one model only
#   LIMIT=200 bash run_paper_eval.sh       # quick subset (good for timing first)
#   N_SAMPLES=8 bash run_paper_eval.sh     # maj@8 only, skip pass@10
#
# Every run logs to W&B under group $WANDB_GROUP so the three models overlay.

set -e
cd "$(dirname "$0")"

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
DPO_MODEL="${DPO_MODEL:-outputs/dpo_mistral7b}"
LISTWISE_MODEL="${LISTWISE_MODEL:-outputs/listwise_mistral7b}"

N_SHOT="${N_SHOT:-8}"          # Mistral paper: 8-shot
N_SAMPLES="${N_SAMPLES:-10}"   # 10 → maj@8 + pass@10 + pass@5 + pass@1
MAJ_K="${MAJ_K:-8}"
PASS_K="${PASS_K:-1,5,10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
BACKEND="${BACKEND:-auto}"     # auto → vLLM when importable, else transformers
BATCH_SIZE="${BATCH_SIZE:-4}"  # HF backend only: questions per batch
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WANDB_GROUP="${WANDB_GROUP:-gsm8k-paper-eval}"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

mkdir -p "$RESULTS_DIR"

run_eval() {
    local tag="$1" model="$2"
    if [ ! -e "$model" ] && [[ "$model" == outputs/* ]]; then
        echo "!! skipping $tag — $model does not exist yet"
        return
    fi
    echo "=== $tag : $model (${N_SHOT}-shot, ${N_SAMPLES} samples @ T=${TEMPERATURE}) ==="
    uv run evaluation/eval_gsm8k.py \
        --model "$model" \
        --backend "$BACKEND" \
        --n_shot "$N_SHOT" \
        --n_samples "$N_SAMPLES" \
        --maj_k "$MAJ_K" \
        --pass_k "$PASS_K" \
        --temperature "$TEMPERATURE" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output "$RESULTS_DIR/${tag}_paper_eval.json" \
        --wandb_group "$WANDB_GROUP" \
        --wandb_run_name "${tag}-${N_SHOT}shot-maj${MAJ_K}" \
        --resume \
        $LIMIT_ARG
}

case "${1:-all}" in
    base)     run_eval base     "$BASE_MODEL" ;;
    dpo)      run_eval dpo      "$DPO_MODEL" ;;
    listwise) run_eval listwise "$LISTWISE_MODEL" ;;
    all)
        run_eval base     "$BASE_MODEL"
        run_eval dpo      "$DPO_MODEL"
        run_eval listwise "$LISTWISE_MODEL"
        ;;
    *)
        echo "Unknown target: $1  (valid: base dpo listwise all)"
        exit 1
        ;;
esac

echo "=== Summary ==="
for f in "$RESULTS_DIR"/*_paper_eval.json; do
    [ -e "$f" ] || continue
    echo "--- $f"
    uv run python -c "import json,sys; d=json.load(open('$f')); print(json.dumps(d['metrics'], indent=2))"
done
