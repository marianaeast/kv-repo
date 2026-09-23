#!/bin/bash
# 48K 交叉 A/B 复现：full / clusterkv 交替跑 2 轮，抵消跨 run 漂移
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 49152 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx48k_pg19.pt"
for r in a b; do
  echo "##### round $r full #####"
  python scripts/bench_decode.py --method full      $C --tag ab48k_${r}_full      --out_json results/ab48k_${r}_full.json; echo "[exit=$?]"
  echo "##### round $r clusterkv #####"
  python scripts/bench_decode.py --method clusterkv $C --tag ab48k_${r}_clusterkv --out_json results/ab48k_${r}_clusterkv.json; echo "[exit=$?]"
done
echo "##### DONE #####"
