#!/bin/bash
# yq9 (2x NVIDIA H800 NVL, sm_90, 95.8GB) 上跑 32K 四配置 decode 延迟实验
# 与 yq7 版 run_ctx32k.sh 的差异：模型路径 + CUDA_HOME + 输出 tag
set -euo pipefail

MC=/home/zrd/miniconda3
ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH   # $ENV/bin 必须在前，否则 python 会落到 base
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/zrd/bypasskv_repo/kv-repos/bench

COMMON="--context_len 32768 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512"

echo "[yq9] model = $KV_BENCH_MODEL"
echo "[yq9] GPU   = $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "########## 1/4 full (dense) ##########"
python scripts/bench_decode.py --method full $COMMON --tag yq9_full \
  --out_json results/yq9_ctx32k_full.json

echo "########## 2/4 clusterkv ##########"
python scripts/bench_decode.py --method clusterkv $COMMON --tag yq9_clusterkv \
  --out_json results/yq9_ctx32k_clusterkv.json

echo "########## 3/4 quest ##########"
python scripts/bench_decode.py --method quest $COMMON --tag yq9_quest \
  --out_json results/yq9_ctx32k_quest.json

echo "########## 4/4 clusterkv + offload ##########"
python scripts/bench_decode.py --method clusterkv $COMMON --offload --tag yq9_clusterkv_offload \
  --out_json results/yq9_ctx32k_clusterkv_offload.json

echo "########## ALL DONE ##########"
