#!/bin/bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
ENVP=/home/zrd/miniconda3/envs/ShadowKV
cd /home/zrd/bypasskv_repo/kv-repos/bench
P=$ENVP/bin/python
C="--input_pt data/ctx32k_pg19.pt --context_len 32768"
echo "##### 1/3 bf16 budget=1024 #####"
$P -u scripts/bench_shadowkv.py $C --sparse_budget 1024 --dtype bfloat16 --tag shadowkv_32k_b1024_bf16 --out_json results/bd_shadowkv_32k_bf16.json
echo "##### 2/3 fp16 budget=1024 #####"
$P -u scripts/bench_shadowkv.py $C --sparse_budget 1024 --dtype float16 --tag shadowkv_32k_b1024_fp16 --out_json results/bd_shadowkv_32k_fp16.json
echo "##### 3/3 bf16 budget=512 #####"
$P -u scripts/bench_shadowkv.py $C --sparse_budget 512 --dtype bfloat16 --tag shadowkv_32k_b512_bf16 --out_json results/bd_shadowkv_32k_b512.json
echo "##### ALL DONE #####"
