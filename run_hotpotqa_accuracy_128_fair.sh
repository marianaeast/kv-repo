#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zrd/bypasskv_repo
DATA=/mnt/disk0/zrd_data/datasets/hotpotqa/test-00000-of-00001.parquet
LIMIT=${1:-200}
RUN_NAME=${RUN_NAME:-hotpotqa_k128_fair_v2_$(date +%Y%m%d_%H%M%S)}
OUT=${OUTPUT_DIR:-$ROOT/results/$RUN_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"

CLUSTER_ROOT=$ROOT/kv-repos/ClusterKV
CLUSTER_PY=/home/zrd/miniconda3/envs/clusterkv-quest/bin/python
COMMON=(--model llama3.1-8b-chat-32k --task hotpotqa --token_budget 128
        --dtype bfloat16 --data_path "$DATA" --limit "$LIMIT"
        --output_root "$OUT/cluster_driver" --resume)

cd "$CLUSTER_ROOT/accuracy/LongBench"

# FullKV is the reference. Naive, Quest, and ClusterKV all use one shared
# selection per KV head. Quest and ClusterKV also record selection recall.
PYTHONPATH="$CLUSTER_ROOT" "$CLUSTER_PY" -u pred.py "${COMMON[@]}"
PYTHONPATH="$CLUSTER_ROOT" "$CLUSTER_PY" -u pred.py "${COMMON[@]}" \
    --quest --chunk_size 1 --gqa_policy qavg --strict_total_budget
PYTHONPATH="$CLUSTER_ROOT" "$CLUSTER_PY" -u pred.py "${COMMON[@]}" \
    --quest --chunk_size 16 --gqa_policy qavg --strict_total_budget --recall_stat
PYTHONPATH="$CLUSTER_ROOT" "$CLUSTER_PY" -u pred.py "${COMMON[@]}" \
    --cluster --sink 16 --nlist 400 --fit_iter 20 --gqa_policy qavg --strict_total_budget --recall_stat

SHADOW_ROOT=$ROOT/kv-repos/ShadowKV
SHADOW_PY=/home/zrd/miniconda3/envs/ShadowKV/bin/python
cd "$SHADOW_ROOT"
PYTHONPATH="$SHADOW_ROOT" "$SHADOW_PY" -u test/eval_hotpotqa_equal_budget.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_path "$DATA" --output "$OUT/shadowkv.jsonl" \
    --total_budget 128 --chunk_size 8 --rank 160 --limit "$LIMIT" \
    --device cuda:0 --resume --recall_stat --strict_total_budget

MODEL_DIR=$OUT/cluster_driver/llama3.1-8b-chat-32k
PYTHON=/home/zrd/miniconda3/envs/vllm-cu129/bin/python

"$PYTHON" "$ROOT/summarize_hotpotqa_accuracy.py" \
    --fullkv "$MODEL_DIR/hotpotqa.jsonl" \
    --naive "$MODEL_DIR/hotpotqa-128c1-gqaqavg-totalv2.jsonl" \
    --quest "$MODEL_DIR/hotpotqa-128-gqaqavg-totalv2-recall.jsonl" \
    --clusterkv "$MODEL_DIR/hotpotqa-htruc400fi20sink16_128-gqaqavg-totalv2-recall.jsonl" \
    --shadowkv "$OUT/shadowkv.jsonl" --output "$OUT/accuracy_summary.csv"

"$PYTHON" "$ROOT/summarize_hotpotqa_recall.py" \
    --quest "$MODEL_DIR/hotpotqa-128-gqaqavg-totalv2-recall.jsonl" \
    --clusterkv "$MODEL_DIR/hotpotqa-htruc400fi20sink16_128-gqaqavg-totalv2-recall.jsonl" \
    --shadowkv "$OUT/shadowkv.jsonl" --output "$OUT/recall_summary.csv"

echo "Results: $OUT"
