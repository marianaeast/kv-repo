#!/usr/bin/env bash
# Add naive measurements to an existing pair of 32K result directories.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ID=${1:?Usage: bash run_naive_32k.sh EXISTING_RUN_ID}
ENV_PATH=/home/zrd/miniconda3/envs/clusterkv-quest
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CONDA_PREFIX="$ENV_PATH"
export PATH="$ENV_PATH/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/kv-repos/ClusterKV${PYTHONPATH:+:$PYTHONPATH}"
BENCH="$ROOT/kv-repos/bench"
for budget in 512 1024; do
    out="$BENCH/results/bd32k_b${budget}_${RUN_ID}"
    [[ -f "$out/summary_model.csv" ]] || { echo "Missing completed base run: $out" >&2; exit 2; }
    nvidia-smi > "$out/naive_hardware.txt"
    sha256sum "$BENCH/scripts/naive_attention.py" "$BENCH/scripts/bench_breakdown.py" > "$out/naive_source.sha256"
    python -u "$BENCH/scripts/bench_breakdown.py" --method naive \
        --input_pt "$BENCH/data/ctx32k_pg19.pt" --context_len 32768 --token_budget "$budget" \
        --decode_len 96 --iteration 3 --warmup_iter 1 --prof_steps 24 \
        --tag "naive_32k_b${budget}" --out_json "$out/naive.json" 2>&1 | tee "$out/naive_supplement.log"
    python "$BENCH/scripts/summarize_breakdown_v2.py" "$out" --budget "$budget" --require-naive
    echo "Completed naive budget=$budget: $out"
done
