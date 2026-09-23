#!/bin/bash
set -uo pipefail
MC=/home/zrd/miniconda3; ENV=$MC/envs/clusterkv-quest
export CONDA_PREFIX=$ENV; export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH
export KV_BENCH_MODEL=/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
export CUDA_VISIBLE_DEVICES=0
cd /home/zrd/bypasskv_repo/kv-repos/bench
for cfg in "32768:ctx32k_pg19.pt" "49152:ctx48k_pg19.pt"; do
  ctx="${cfg%%:*}"; pt="${cfg##*:}"
  echo "########## prof quest ctx=$ctx ##########"
  python scripts/prof_decode.py --method quest --context_len $ctx \
    --input_pt data/$pt --steps 24 --topk 30 --tag "quest_ctx${ctx}"
done
echo "########## quest PROF DONE ##########"
