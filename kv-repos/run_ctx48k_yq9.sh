#!/bin/bash
# yq9 (H800 NVL 96GB) 上跑 48K 四配置 decode 延迟实验
# 目的：检验「ClusterKV 收益被带宽压制」假说——KV 量增 1.5x 后收益是否重现
set -uo pipefail

MC=/home/zrd/miniconda3
ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/zrd/bypasskv_repo/kv-repos/bench

COMMON="--context_len 49152 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512"
IN="--input_pt data/ctx48k_pg19.pt"

echo "[yq9/48K] GPU=$CUDA_VISIBLE_DEVICES  $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "########## 1/4 full ##########"
python scripts/bench_decode.py --method full      $COMMON $IN --tag yq9_ctx48k_full      --out_json results/yq9_ctx48k_full.json;      echo "[exit=$?]"

echo "########## 2/4 clusterkv ##########"
python scripts/bench_decode.py --method clusterkv $COMMON $IN --tag yq9_ctx48k_clusterkv --out_json results/yq9_ctx48k_clusterkv.json; echo "[exit=$?]"

echo "########## 3/4 quest ##########"
python scripts/bench_decode.py --method quest     $COMMON $IN --tag yq9_ctx48k_quest     --out_json results/yq9_ctx48k_quest.json;     echo "[exit=$?]"

echo "########## 4/4 clusterkv + offload ##########"
python scripts/bench_decode.py --method clusterkv $COMMON $IN --offload --tag yq9_ctx48k_clusterkv_offload --out_json results/yq9_ctx48k_clusterkv_offload.json; echo "[exit=$?]"

echo "########## ALL DONE $(date -Is) ##########"
