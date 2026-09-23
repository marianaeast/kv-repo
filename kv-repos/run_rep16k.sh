#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 16384 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx16k_pg19.pt"
for i in 1 2 3; do
  echo "##### ckv rep$i #####"
  python scripts/bench_decode.py --method clusterkv $C --tag rep${i}_ckv --out_json results/rep${i}_ckv.json; echo "[exit=$?]"
  echo "##### full rep$i #####"
  python scripts/bench_decode.py --method full      $C --tag rep${i}_full --out_json results/rep${i}_full.json; echo "[exit=$?]"
done
echo "##### DONE #####"
