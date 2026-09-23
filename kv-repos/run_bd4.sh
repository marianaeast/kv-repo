#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 32768 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx32k_pg19.pt --prof_steps 24"
C48="--context_len 49152 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx48k_pg19.pt --prof_steps 24"
echo "##### 1/6 full 32K #####"
python scripts/bench_breakdown.py --method full $C --tag full_32k --out_json results/bd2_full_32k.json
echo "##### 2/6 quest 32K (GPU) #####"
python scripts/bench_breakdown.py --method quest $C --tag quest_32k --out_json results/bd2_quest_32k.json
echo "##### 3/6 clusterkv 32K (GPU) #####"
python scripts/bench_breakdown.py --method clusterkv $C --tag ckv_32k --out_json results/bd2_ckv_32k.json
echo "##### 4/6 quest+offload 32K #####"
python scripts/bench_breakdown.py --method quest --offload $C --tag quest_off_32k --out_json results/bd2_quest_off_32k.json
echo "##### 5/6 quest+offload 48K #####"
python scripts/bench_breakdown.py --method quest --offload $C48 --tag quest_off_48k --out_json results/bd2_quest_off_48k.json
echo "##### 6/6 clusterkv+offload 32K #####"
python scripts/bench_breakdown.py --method clusterkv --offload $C --tag ckv_off_32k --out_json results/bd2_ckv_off_32k.json
echo "##### BD4 DONE #####"
