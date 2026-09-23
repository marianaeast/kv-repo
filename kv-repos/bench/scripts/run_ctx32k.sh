#!/bin/bash
set -u
source /home/zrd/miniconda3/etc/profile.d/conda.sh
conda activate clusterkv-quest
cd /home/zrd/bypasskv_repo/kv-repos/bench
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=1

: > /tmp/gpu_ctx32k_samples.log
( while true; do
    printf "%s %s\n" "$(date +%H:%M:%S)" "$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' '|')"
    sleep 5
  done ) >> /tmp/gpu_ctx32k_samples.log &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

COMMON="--context_len 32768 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512"

run() {
  local tag="$1"; shift
  echo ""
  echo "================ $tag  $(date +%H:%M:%S) ================"
  python scripts/bench_decode.py "$@" --tag "$tag" --out_json "results/${tag}.json" 2>&1 \
    | grep -vE "FutureWarning|weights_only|prepare_inputs_for_generation|GenerationMixin|trust_remote_code|auto class|Loading checkpoint shards|^$"
  echo "---- $tag exit=$? ----"
}

run ctx32k_full               --method full      $COMMON
run ctx32k_quest              --method quest     $COMMON
run ctx32k_clusterkv          --method clusterkv $COMMON
run ctx32k_clusterkv_offload  --method clusterkv $COMMON --offload
kill $SAMPLER 2>/dev/null
echo ""
echo "ALL DONE $(date +%H:%M:%S)"
