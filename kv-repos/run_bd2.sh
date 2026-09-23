#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 32768 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx32k_pg19.pt --prof_steps 24"
echo "########## quest+offload 32K ##########"
python scripts/bench_breakdown.py --method quest --offload $C --tag bd_quest_off_32k --out_json results/bd_quest_off_32k.json
echo "########## quest+offload 48K ##########"
C48="--context_len 49152 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx48k_pg19.pt --prof_steps 24"
python scripts/bench_breakdown.py --method quest --offload $C48 --tag bd_quest_off_48k --out_json results/bd_quest_off_48k.json
echo "########## clusterkv+offload 32K ##########"
python scripts/bench_breakdown.py --method clusterkv --offload $C --tag bd_ckv_off_32k --out_json results/bd_ckv_off_32k.json
echo "########## BD2 DONE ##########"
