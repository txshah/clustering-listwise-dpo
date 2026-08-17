#!/usr/bin/env bash
# Smoke test — runs the full pipeline on 10 questions.
# Usage:
#   bash smoke_test.sh            # run all steps
#   bash smoke_test.sh prepare    # step 1 only
#   bash smoke_test.sh generate   # step 2 only
#   bash smoke_test.sh process    # step 3 only
#   bash smoke_test.sh dpo        # step 4.1 (build pairs + train DPO)
#   bash smoke_test.sh listwise   # step 4.2 (build listwise + train listwise)
#   bash smoke_test.sh eval       # step 5 (evaluate both)

set -e
cd "$(dirname "$0")"

MODEL="mistralai/Mistral-7B-v0.1"

step_prepare() {
    echo "=== Step 1: prepare questions ==="
    uv run generation/code/prepare_questions.py --output generation/traces/questions.jsonl
    uv run python -c "
import json
data = [json.loads(l) for l in open('generation/traces/questions.jsonl')]
with open('generation/traces/questions_small.jsonl','w') as f:
    [f.write(json.dumps(d)+'\n') for d in data[:10]]
print('Sliced 10 questions -> generation/traces/questions_small.jsonl')
"
}

step_generate() {
    echo "=== Step 2: generate traces (10 questions x 10 samples) ==="
    uv run generation/code/generate_traces.py \
        --input generation/traces/questions_small.jsonl \
        --output generation/traces/raw/raw_traces_small.jsonl \
        --model "$MODEL" \
        --n_samples 10 \
        --max_new_tokens 256
}

step_process() {
    echo "=== Step 3: sort correct / wrong (dedup disabled for smoke test) ==="
    uv run generation/code/process_traces.py \
        --input generation/traces/raw/raw_traces_small.jsonl \
        --output generation/traces/processed/results_small.jsonl \
        --length_window 0
}

step_dpo() {
    echo "=== Step 4.1: build DPO pairs ==="
    uv run generation/code/build_pairs.py \
        --input generation/traces/processed/results_small.jsonl \
        --output generation/traces/dpo_pairs_small.jsonl
    echo "=== Step 4.1: train DPO ==="
    uv run training/train_dpo.py \
        --config training/configs/dpo_config.yaml \
        --dataset_path generation/traces/dpo_pairs_small.jsonl \
        --output_dir outputs/dpo_small \
        --wandb_group smoke-test
}

step_listwise() {
    echo "=== Step 4.2: build listwise pairs ==="
    uv run generation/code/build_listwise.py \
        --input generation/traces/processed/results_small.jsonl \
        --output generation/traces/listwise_pairs_small.jsonl \
        --n_per_class 4
    echo "=== Step 4.2: train listwise ==="
    uv run training/train_listwise.py \
        --config training/configs/listwise_config.yaml \
        --dataset_path generation/traces/listwise_pairs_small.jsonl \
        --output_dir outputs/listwise_small \
        --wandb_group smoke-test
}

step_eval() {
    echo "=== Step 5: evaluate (20 questions, 8-shot, maj@8 + pass@10/5/1) ==="
    for tag in dpo listwise; do
        uv run evaluation/eval_gsm8k.py \
            --model "outputs/${tag}_small" \
            --n_shot 8 --n_samples 10 --maj_k 8 --pass_k 1,5,10 \
            --limit 20 --batch_size 4 --max_new_tokens 400 \
            --output "results/${tag}_small.json" \
            --wandb_group smoke-test --wandb_run_name "smoke-${tag}"
    done
    echo "--- DPO results ---"
    head -40 results/dpo_small.json
    echo "--- Listwise results ---"
    head -40 results/listwise_small.json
}

case "${1:-all}" in
    prepare)  step_prepare ;;
    generate) step_generate ;;
    process)  step_process ;;
    dpo)      step_dpo ;;
    listwise) step_listwise ;;
    eval)     step_eval ;;
    all)
        step_prepare
        step_generate
        step_process
        step_dpo
        step_listwise
        step_eval
        ;;
    *)
        echo "Unknown step: $1"
        echo "Valid steps: prepare generate process dpo listwise eval all"
        exit 1
        ;;
esac

echo "=== Done ==="
