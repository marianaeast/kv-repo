#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=1
cd /home/zrd/bypasskv_repo/kv-repos/bench
C="--context_len 32768 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx32k_pg19.pt --prof_steps 24"
C48="--context_len 49152 --decode_len 96 --iteration 3 --warmup_iter 1 --token_budget 512 --input_pt data/ctx48k_pg19.pt --prof_steps 24"
echo "##### quest+offload 32K #####"
python scripts/bench_breakdown.py --method quest --offload $C --tag quest_off_32k --out_json results/bd_quest_off_32k.json
echo "##### quest+offload 48K #####"
python scripts/bench_breakdown.py --method quest --offload $C48 --tag quest_off_48k --out_json results/bd_quest_off_48k.json
echo "##### clusterkv 32K (GPU) #####"
python scripts/bench_breakdown.py --method clusterkv $C --tag ckv_32k --out_json results/bd_ckv_32k.json
echo "##### clusterkv+offload 32K #####"
python scripts/bench_breakdown.py --method clusterkv --offload $C --tag ckv_off_32k --out_json results/bd_ckv_off_32k.json
echo "##### BD3 DONE #####"
