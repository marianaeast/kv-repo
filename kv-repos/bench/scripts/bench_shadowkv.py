#!/usr/bin/env python
"""ShadowKV (attn_mode=\"shadowkv_cpu\") decode 延迟 stage-level breakdown.

口径与 bench_breakdown.py 对齐：bsz=1、单序列 decode。
ShadowKV 的 decode 每层长这样（models/base.py: layer_compute）：
  1. rope(q,k) + update_kv_cache                 -> COMPUTE(misc)
  2. get_retrieval_position_ids                  -> SELECT
       batch_gemm_softmax(landmark 打分) + topk + reorder_keys_and_compute_offsets
  3. [copy_stream] get_value_cache               -> LOAD  (CPU pinned -> GPU)
       gather_copy_with_offsets
  4. get_key_cache                               -> COMPUTE (GPU D2D + 低秩重建)
       gather_copy_d2d_with_offsets + batch_gather_gemm_rotary_pos_emb_cuda
  5. flash_attn_with_kvcache                     -> COMPUTE(attn)
  6. 权重 GEMM / norm / silu                      -> COMPUTE(gemm/misc)
"""
import argparse, json, os, sys, time, math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SK = "/home/zrd/bypasskv_repo/kv-repos/ShadowKV"
sys.path.insert(0, SK)
sys.path.insert(0, os.path.join(SK, "kernels"))

import torch
from breakdown_timeline import summarize_profile

DEFAULT_MODEL = ("/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
                 "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")

# ---------------------------------------------------------------- kernel 归类
# 顺序敏感：先匹配到的赢。
SELECT_PATTERNS = [
    "BatchGemmWithEpilogueVisitor", "ApplySoftmax", "ApplySoftmaxFinalReduction",
    "batch_gemm_softmax",                   # landmark 打分（fused gemm+softmax）
    "reorder_keys_and",                     # selected_chunks -> gather offsets
    "topk",                                 # torch.topk（选 chunk）
    "gatherTopK", "sortKeyValue", "radixSortKVInPlace",
    "DeviceRadixSort", "DeviceSelect", "DeviceScan", "DeviceCompact",
    "adjacent_difference",
]
LOAD_PATTERNS = [
    # V: CPU pinned -> GPU staging；K的GPU内gather归入compute_misc
    "gather_copy_var_midpoint",
    "Memcpy HtoD", "Memcpy DtoH", "Memcpy DeviceToHost",
]
COMPUTE_ATTN_PATTERNS = [
    "flash_fwd", "fmha", "paged_attention", "BatchDecodeWithPagedKVCache",
]
COMPUTE_GEMM_PATTERNS = [
    "batch_gather_gemm",                    # U @ SV 低秩重建 K + rope
    "xmma_gemm", "gemv", "cutlass", "hopper_",
]
COMPUTE_MISC_PATTERNS = [
    "gather_copy_d2d", "Memcpy DtoD",
    "rmsnorm", "rms_norm", "silu_and_mul", "act_and_mul",
    "vectorized_elementwise", "unrolled_elementwise", "elementwise_kernel",
    "reduce_kernel", "rotary", "rope", "Memset", "__nv_",
]


def classify(name):
    for p in SELECT_PATTERNS:
        if p in name:
            return "select"
    for p in LOAD_PATTERNS:
        if p in name:
            return "load"
    for p in COMPUTE_ATTN_PATTERNS:
        if p in name:
            return "compute_attn"
    for p in COMPUTE_GEMM_PATTERNS:
        if p in name:
            return "compute_gemm"
    for p in COMPUTE_MISC_PATTERNS:
        if p in name:
            return "compute_misc"
    return "other"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--input_pt", required=True)
    p.add_argument("--context_len", type=int, required=True)
    p.add_argument("--sparse_budget", type=int, default=1024)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--decode_len", type=int, default=96)
    p.add_argument("--iteration", type=int, default=3)
    p.add_argument("--warmup_iter", type=int, default=1)
    p.add_argument("--prof_steps", type=int, default=24)
    p.add_argument("--topk_kernels", type=int, default=40)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    if not (a.iteration > a.warmup_iter >= 0 and 0 < a.decode_len <= 128 and 0 < a.prof_steps <= 116):
        p.error('iteration must exceed warmup; decode_len<=128, prof_steps<=116')
    if a.dtype != 'bfloat16' or a.chunk_size != 8:
        p.error('ShadowKV CUDA gather requires bf16 and chunk_size=8')
    return a


def main():
    a = parse()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    dt = torch.bfloat16 if a.dtype == "bfloat16" else torch.float16

    a.input_pt = os.path.abspath(a.input_pt)
    if a.out_json:
        a.out_json = os.path.abspath(a.out_json)
    d = torch.load(a.input_pt)
    ids = d["input_ids"][: a.context_len].unsqueeze(0).to(dev)
    os.chdir(SK)   # models/ 里的相对路径假设
    assert ids.shape[1] == a.context_len

    from kernels import shadowkv
    from shadow_gather_patch import install
    install(shadowkv)
    from models import choose_model_class
    LLM = choose_model_class(a.model_path)
    t_build = time.time()
    llm = LLM(model_name=a.model_path, device=a.device, batch_size=1,
              max_length=a.context_len + a.decode_len + 512,
              attn_mode="shadowkv_cpu", sparse_budget=a.sparse_budget,
              chunk_size=a.chunk_size, rank=a.rank, dtype=dt)
    print("[build] %.1f s | %s" % (time.time() - t_build, llm.kv_cache))

    def prefill():
        # 必须重建同一prompt的cache；仅重置offset会保留上轮内容。
        llm.kv_cache.position_ids.fill_(-1)
        llm.kv_cache.signals.zero_()
        logits = llm.batch_prefill(ids)
        llm.kv_cache.H2D()
        torch.cuda.synchronize()
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with torch.inference_mode():
        llm.warmup()
        # ---------------- 1) 墙钟（逐 step CUDA Event）
        CTX = a.context_len
        wall_steps = []
        for it in range(a.iteration):
            nxt = prefill()
            torch.cuda.synchronize()
            for _ in range(a.decode_len):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
                e.record()
                torch.cuda.synchronize()
                wall_steps.append(s.elapsed_time(e))
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                del out
        drop = a.decode_len * a.warmup_iter
        wall = wall_steps[drop:]
        wall_mean = sum(wall) / len(wall)
        ws = sorted(wall)
        wall_median = ws[len(ws) // 2]
        wall_p99 = ws[int(len(ws) * 0.99) - 1] if len(ws) > 10 else ws[-1]
        wall_std = (sum((x - wall_mean) ** 2 for x in wall) / len(wall)) ** 0.5

        # ---------------- 2) kernel 级
        nxt = prefill()
        for _ in range(12):
            out = llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
            nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
            del out
        torch.cuda.synchronize()

        from torch.profiler import profile, ProfilerActivity, record_function
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof_steps):
                with record_function('bench_decode_step'):
                    out = llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
                    torch.cuda.synchronize()
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                del out
            torch.cuda.synchronize()

    def dev_us(e):
        # torch 版本间属性名不同
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(e, attr):
                return float(getattr(e, attr))
        return 0.0

    evs = [e for e in prof.key_averages()
           if e.device_type == torch.autograd.DeviceType.CUDA and e.key != "bench_decode_step"]

    stages = {"select": 0.0, "load": 0.0, "compute_attn": 0.0,
              "compute_gemm": 0.0, "compute_misc": 0.0, "other": 0.0}
    detail = []
    total = 0.0
    for e in evs:
        ms = dev_us(e) / a.prof_steps / 1000.0
        if ms <= 0:
            continue
        st = classify(e.key)
        stages[st] += ms
        total += ms
        detail.append({"name": e.key, "stage": st, "ms_per_step": ms,
                       "calls_per_step": e.count / a.prof_steps})
    detail.sort(key=lambda x: -x["ms_per_step"])

    rec = {
        "tag": a.tag or "shadowkv_%d" % a.context_len,
        "method": "shadowkv", "offload": True,
        "context_len": a.context_len, "sparse_budget": a.sparse_budget,
        "chunk_size": a.chunk_size, "rank": a.rank, "dtype": a.dtype,
        "decode_len": a.decode_len, "iteration": a.iteration,
        "prof_steps": a.prof_steps,
        "num_layers": llm.kv_cache.num_layers,
        "wall_ms_mean": wall_mean, "wall_ms_median": wall_median,
        "wall_ms_p99": wall_p99, "wall_ms_std": wall_std,
        "wall_ms_min": min(wall), "wall_ms_max": max(wall),
        "schema_version": 2, "measure": "model_decode_forward",
        "wall_steps_ms": wall, "wall_definition": "CUDA event span around synchronized forward; sampling excluded",
        "budget_notes": "dynamic sparse_budget excludes local/outlier/generated tokens; per KV head selection",
        "outlier_chunks": llm.kv_cache.outlier_chunk,
        "local_chunks": llm.kv_cache.local_chunk,
        "sparse_start": llm.kv_cache.sparse_start,
        "gather_patch": "64 chunks and current stream support",
        "kernels": detail[: a.topk_kernels],
    }
    rec.update(summarize_profile(prof, classify, a.prof_steps))
    if a.out_json:
        prof.export_chrome_trace(a.out_json.replace('.json', '.trace.json'))
        with open(a.out_json, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: rec[k] for k in ('tag', 'wall_ms_median', 'profile_wall_ms', 'stages_ms', 'gap_ms', 'raw_stages_ms')}, indent=2))
    if a.out_json:
        print('saved ->', a.out_json)


if __name__ == '__main__':
    main()
