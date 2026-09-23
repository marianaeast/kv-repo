#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 32768 --decode_len 64 --iteration 2 --warmup_iter 1 --token_budget 512 --input_pt data/ctx32k_pg19.pt --prof_steps 20"
python scripts/bench_breakdown.py --method full      $C --tag bd_full_32k      --out_json results/bd_full_32k.json
python scripts/bench_breakdown.py --method clusterkv $C --tag bd_clusterkv_32k --out_json results/bd_clusterkv_32k.json
python scripts/bench_breakdown.py --method quest     $C --tag bd_quest_32k     --out_json results/bd_quest_32k.json
echo "########## BD1 DONE ##########"
