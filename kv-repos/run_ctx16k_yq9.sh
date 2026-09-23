#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd /home/zrd/bypasskv_repo/kv-repos/bench
C16="--context_len 16384 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx16k_pg19.pt"
C32="--context_len 32768 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512"
echo "########## 16K full ##########"
python scripts/bench_decode.py --method full      $C16 --tag yq9_ctx16k_full      --out_json results/yq9_ctx16k_full.json; echo "[exit=$?]"
echo "########## 16K clusterkv ##########"
python scripts/bench_decode.py --method clusterkv $C16 --tag yq9_ctx16k_clusterkv --out_json results/yq9_ctx16k_clusterkv.json; echo "[exit=$?]"
echo "########## 32K full (rerun) ##########"
python scripts/bench_decode.py --method full      $C32 --tag yq9_full_rerun      --out_json results/yq9_ctx32k_full_rerun.json; echo "[exit=$?]"
echo "########## 32K clusterkv (rerun) ##########"
python scripts/bench_decode.py --method clusterkv $C32 --tag yq9_clusterkv_rerun --out_json results/yq9_ctx32k_clusterkv_rerun.json; echo "[exit=$?]"
echo "########## DONE ##########"
