#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
C48="--context_len 49152 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx48k_pg19.pt --prof_steps 24"
echo "##### 1/4 full 48K #####"
python scripts/bench_breakdown.py --method full $C48 --tag full_48k --out_json results/bd2_full_48k.json
echo "##### 2/4 quest 48K (GPU) #####"
python scripts/bench_breakdown.py --method quest $C48 --tag quest_48k --out_json results/bd2_quest_48k.json
echo "##### 3/4 clusterkv 48K (GPU) #####"
python scripts/bench_breakdown.py --method clusterkv $C48 --tag ckv_48k --out_json results/bd2_ckv_48k.json
echo "##### 4/4 clusterkv+offload 48K #####"
python scripts/bench_breakdown.py --method clusterkv --offload $C48 --tag ckv_off_48k --out_json results/bd2_ckv_off_48k.json
echo "##### BD5 DONE #####"
