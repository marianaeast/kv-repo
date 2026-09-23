#!/bin/bash
# 移动/重编之后的端到端自检：小 context 跑通 3 条关键路径
# 用法：bash scripts/smoke_yq9.sh 2>&1 | tee logs/smoke_yq9.log
set -uo pipefail
MC=/home/zrd/miniconda3
ENV=$MC/envs/clusterkv-quest
SKENV=$MC/envs/ShadowKV
export CONDA_PREFIX=$ENV
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST=9.0
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench || exit 1

C="--input_pt data/ctx32k_pg19.pt --context_len 4096 --token_budget 512"
C="$C --decode_len 16 --iteration 1 --warmup_iter 0 --prof_steps 8"

echo "##### smoke 1/3 quest + offload (offload 全链 + gather_pages UVA kernel) #####"
python -u scripts/bench_breakdown.py --method quest --offload $C --tag smoke_quest_off

echo "##### smoke 2/3 clusterkv + offload (ClusterKV kernel + recall 换页) #####"
python -u scripts/bench_breakdown.py --method clusterkv --offload $C --tag smoke_ckv_off

echo "##### smoke 3/3 shadowkv (重编的 sm_90 .so) #####"
# 注意：ShadowKV 的 layer_compute 用 `if q_len > 4*1024: prefill else: decode` 分支，
# sparse_end 只在 prefill 分支赋值，所以 ctx 必须 > 4096，否则报
# AttributeError: 'ShadowKVCache_CPU' object has no attribute 'sparse_end'。
$SKENV/bin/python -u scripts/bench_shadowkv.py \
    --input_pt data/ctx32k_pg19.pt --context_len 16384 \
    --sparse_budget 1024 --dtype bfloat16 --decode_len 8 --iteration 1 \
    --warmup_iter 0 --prof_steps 4 --tag smoke_shadowkv --out_json /tmp/smoke_shadowkv.json

echo "##### SMOKE DONE #####"
