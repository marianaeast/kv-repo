#!/usr/bin/env bash
# 默认跑32K/512和32K/1024；每次新建目录，失败即停止。
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH="$ROOT/kv-repos/bench"
MC=/home/zrd/miniconda3
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-2}
export KV_BENCH_MODEL=${KV_BENCH_MODEL:-/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$BENCH"
mkdir -p results logs
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $# == 0 ]]; then set -- 512 1024; fi
for budget in "$@"; do
    [[ $budget == 512 || $budget == 1024 ]] || { echo 'budgets must be 512 or 1024' >&2; exit 2; }
done
DECODE_LEN=${DECODE_LEN:-96}
ITERATION=${ITERATION:-3}
WARMUP_ITER=${WARMUP_ITER:-1}
PROF_STEPS=${PROF_STEPS:-24}
INF_WARMUP=${INF_WARMUP:-10}
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
run() {
    local env_name=$1 log=$2
    shift 2
    local env_path="$MC/envs/$env_name"
    if [[ $DRY_RUN == 1 ]]; then
        printf '%q ' "$env_path/bin/python" "$@"; printf '\n'
        return
    fi
    env CONDA_PREFIX="$env_path" PATH="$env_path/bin:$CUDA_HOME/bin:/usr/bin:/bin" \
        PYTHONPATH="$ROOT/kv-repos/ClusterKV:$ROOT/kv-repos/ShadowKV${PYTHONPATH:+:$PYTHONPATH}" \
        "$env_path/bin/python" -u "$@" 2>&1 | tee "$log"
}
for budget in "$@"; do
    out="results/bd32k_b${budget}_${RUN_ID}"
    log="logs/bd32k_b${budget}_${RUN_ID}"
    mkdir -p "$out" "$log"
    common=(--input_pt "$BENCH/data/ctx32k_pg19.pt" --context_len 32768 \
        --decode_len "$DECODE_LEN" --iteration "$ITERATION" --warmup_iter "$WARMUP_ITER" --prof_steps "$PROF_STEPS")
    if [[ $DRY_RUN == 0 ]]; then
        nvidia-smi > "$out/hardware.txt"
        cp "$ROOT/run_bd_32k.sh" "$out/run_script.sh"
        sha256sum scripts/{bench_breakdown,bench_shadowkv,microbench_infinigen,breakdown_timeline,llama31_rope_patch,shadow_gather_patch,naive_attention}.py > "$out/script_sha256.txt"
    fi
    for method in full clusterkv quest clusterkv_offload quest_offload naive; do
        flags=(--method "${method%_offload}")
        if [[ $method == *_offload ]]; then flags+=(--offload); fi
        run clusterkv-quest "$log/$method.log" scripts/bench_breakdown.py \
            "${flags[@]}" "${common[@]}" --token_budget "$budget" \
            --tag "${method}_32k_b${budget}" --out_json "$BENCH/$out/$method.json"
    done
    run ShadowKV "$log/shadowkv.log" scripts/bench_shadowkv.py "${common[@]}" \
        --model_path "$KV_BENCH_MODEL" --sparse_budget "$budget" --dtype bfloat16 \
        --tag "shadowkv_32k_b${budget}" --out_json "$BENCH/$out/shadowkv.json"
    run ShadowKV "$log/infinigen_microbench.log" scripts/microbench_infinigen.py \
        --context_len 32768 --budget "$budget" --partial_ratio 0.3 --warmup "$INF_WARMUP" \
        --iters "$DECODE_LEN" --prof_steps "$PROF_STEPS" --tag "infini_micro_32k_b${budget}" \
        --out_json "$BENCH/$out/infinigen_microbench.json"
    run clusterkv-quest "$log/summary.log" scripts/summarize_breakdown_v2.py "$BENCH/$out" --budget "$budget" --require-naive
    echo "Completed budget=$budget: $BENCH/$out"
done
