#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zrd/bypasskv_repo
DATA=/mnt/disk0/zrd_data/datasets/hotpotqa/test-00000-of-00001.parquet
LIMIT=${1:-200}
ACCURACY_MODEL=${ACCURACY_MODEL:-llama3.1-8b}
case "$ACCURACY_MODEL" in
  llama3.1-8b)
    CLUSTER_MODEL=llama3.1-8b-chat-32k
    SHADOW_MODEL=meta-llama/Llama-3.1-8B-Instruct
    CLUSTER_PY=/home/zrd/miniconda3/envs/clusterkv-quest/bin/python
    SHADOW_PY=/home/zrd/miniconda3/envs/ShadowKV/bin/python
    ;;
  qwen3-8b)
    CLUSTER_MODEL=qwen3-8b-chat-32k
    SHADOW_MODEL=/mnt/disk0/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
    CLUSTER_PY=/home/zrd/miniconda3/envs/clusterkv-qwen3/bin/python
    SHADOW_PY=/home/zrd/miniconda3/envs/ShadowKV-qwen3/bin/python
    ;;
  *)
    echo "Unsupported ACCURACY_MODEL=$ACCURACY_MODEL (use llama3.1-8b or qwen3-8b)" >&2
    exit 2
    ;;
esac
RUN_NAME=${RUN_NAME:-hotpotqa_${ACCURACY_MODEL}_k128_fair_v2_$(date +%Y%m%d_%H%M%S)}
OUT=${OUTPUT_DIR:-$ROOT/results/$RUN_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"

CLUSTER_ROOT=$ROOT/kv-repos/ClusterKV
COMMON=(--model "$CLUSTER_MODEL" --task hotpotqa --token_budget 128
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
cd "$SHADOW_ROOT"
PYTHONPATH="$SHADOW_ROOT" "$SHADOW_PY" -u test/eval_hotpotqa_equal_budget.py \
    --model_name "$SHADOW_MODEL" \
    --data_path "$DATA" --output "$OUT/shadowkv.jsonl" \
    --total_budget 128 --chunk_size 8 --rank 160 --limit "$LIMIT" \
    --device cuda:0 --resume --recall_stat --strict_total_budget

MODEL_DIR=$OUT/cluster_driver/$CLUSTER_MODEL
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
