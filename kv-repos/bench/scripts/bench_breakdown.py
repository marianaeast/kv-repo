#!/usr/bin/env python
"""真实模型decode计时。LOAD只计实际搬运；重叠在同一profiler时间线上去重。
独立wall统计不与profile各项相加；LOAD不等同于依赖图关键路径等待。
"""
import argparse, json, os, sys
import torch
from breakdown_timeline import summarize_profile

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEFAULT_MODEL = os.environ.get(
    "KV_BENCH_MODEL",
    "/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)

# ---------------------------------------------------------------- kernel 归类
# 顺序敏感：先匹配到的赢。写成子串匹配。
SELECT_PATTERNS = [
    "naive_identify_load_keys_kernel", "naive_scan_keys", "gatherTopK", "topk", "radixSortKVInPlace",
    "get_sel_indices",          # clusterkv: 按 cluster 取 token 索引
    "get_neigh_c",              # clusterkv: query -> 最近 cluster 质心
    "MaxPossibleSample",        # quest: page min/max 上界打分 (estimate)
    "radix_topk",               # quest: topk_filtering
    "topk_filtering",
    "estimate_attn",
    "indexSelect",              # torch 侧的 gather（offload recall）
    "index_select",
    "swap_in_out",              # clusterkv offload: 页表换入换出决策
    # offload 页号去重 torch.unique 拉起的 cub kernel（radix sort/select/scan）
    "DeviceRadixSort",
    "DeviceSelect",
    "DeviceScan",
    "DeviceCompact",
    "adjacent_difference",
]
LOAD_PATTERNS = [
    "naive_load_values_kernel",
    "Memcpy HtoD",
    "Memcpy DtoH",
    "Memcpy DeviceToHost",
    "recall",                   # clusterkv offload: CPU->GPU 页换入
    "copy_kernel<__half>",      # clusterkv offload: recall 的实际实现 kernel
    "gather_pages",             # quest offload: UVA zero-copy 页搬运
]
COMPUTE_ATTN_PATTERNS = [
    "BatchDecodeWithPagedKVCacheKernel",
    "single_prefill_with_kv_cache",
    "paged_attention",
    "flash_fwd",
    "fmha",
]
COMPUTE_GEMM_PATTERNS = [
    "xmma_gemm",                # cublas tensor-core GEMM（权重投影）
    "gemv",                    # cublas GEMV
    "cutlass",
    "ampere_", "hopper_",
]
COMPUTE_MISC_PATTERNS = [
    "rmsnorm", "rms_norm",
    "AppendPagedKVCache",
    "VariableLengthMergeStates",
    "QKApplyRotary",
    "apply_rope",
    "vectorized_elementwise",
    "unrolled_elementwise",
    "elementwise_kernel",
    "reduce_kernel",
    "Memcpy DtoD", "Memset",
    "_clusterkv_knl",
    "_quest_knl",
]


def classify(name: str) -> str:
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
    p.add_argument("--method", required=True, choices=["full", "quest", "clusterkv", "naive"])
    p.add_argument("--offload", action="store_true")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--input_pt", required=True)
    p.add_argument("--context_len", type=int, required=True)
    p.add_argument("--token_budget", type=int, default=512)
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--nlist", type=int, default=200)
    p.add_argument("--niter", type=int, default=20)
    p.add_argument("--sink", type=int, default=16)
    p.add_argument("--window", type=int, default=320)
    p.add_argument("--window_nlist", type=int, default=8)
    p.add_argument("--decode_len", type=int, default=128)
    p.add_argument("--iteration", type=int, default=3)
    p.add_argument("--warmup_iter", type=int, default=1)
    p.add_argument("--prof_steps", type=int, default=24)
    p.add_argument("--topk_kernels", type=int, default=40)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    if a.offload:
        assert a.method in ("quest", "clusterkv"), "offload 只对 quest / clusterkv 实现"
    if not (a.iteration > a.warmup_iter >= 0 and a.decode_len > 0 and a.prof_steps > 0):
        p.error("iteration must exceed warmup_iter; positive step counts required")
    return a


def build(a, dev):
    if a.method == "quest":
        from clusterkv.quest_models.llama import LlamaForCausalLM
    else:
        from clusterkv.clusterkv_models.llama import LlamaForCausalLM
    m = LlamaForCausalLM.from_pretrained(a.model_path, device_map=dev,
                                         torch_dtype=torch.float16)
    m.eval()
    return m


def init_controller(a, model, dev, max_seq_len):
    if a.method == "naive":
        from naive_attention import install
        install(model, max_seq_len, a.token_budget, dev)
        return
    if a.method == "quest":
        if a.offload:
            model.quest_init(page_size=a.page_size, max_seq_len=max_seq_len,
                             token_budget=a.token_budget, dtype=torch.float16,
                             device=dev, offload=True)
        else:
            model.quest_init(page_size=a.page_size, max_seq_len=max_seq_len,
                             token_budget=a.token_budget, dtype=torch.float16,
                             device=dev)
    else:
        init_tb = max(102400, max_seq_len) if a.method == "full" else a.token_budget
        model.clusterkv_init(nlist=a.nlist, niter=a.niter, max_seq_len=max_seq_len,
                             token_budget=init_tb, dtype=torch.float16, device=dev,
                             full=(a.method == "full"), sink=a.sink, window=a.window,
                             window_nlist=a.window_nlist, offload=a.offload)


def prime_offload(model, a):
    """初始化全量CPU副本属于prefill/setup；失败不能继续输出性能结果。"""
    if a.method == 'quest' and a.offload:
        store = model.model.iController.kv_offload
        if store is None or not store.use_gpu_gather:
            raise RuntimeError('Quest offload requires the UVA gather extension')
        # 每次新prefill即使页数相同，也必须重新同步。
        store._synced_pages = 0
        store.sync_from_gpu(model.model.iController)
        torch.cuda.synchronize()


def main():
    a = parse()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.set_default_dtype(torch.float16)

    d = torch.load(a.input_pt)
    ids = d["input_ids"][: a.context_len].unsqueeze(0).to(dev)
    assert ids.shape[1] == a.context_len

    model = build(a, dev)
    max_seq_len = a.context_len + a.decode_len + 512
    from llama31_rope_patch import install
    rope_impl = install(model.config, dev, max_seq_len)
    init_controller(a, model, dev, max_seq_len)
    clear = model.quest_clear if a.method == "quest" else model.clusterkv_clear

    n_layers = model.config.num_hidden_layers

    # ---------------- 1) 墙钟（逐 step CUDA Event），不受 profiler 影响
    wall_steps = []
    with torch.inference_mode():
        for it in range(a.iteration):
            clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            out = model(input_ids=ids)
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del out
            prime_offload(model, a)
            for _ in range(a.decode_len):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = model(input_ids=pred)
                e.record()
                torch.cuda.synchronize()
                wall_steps.append(s.elapsed_time(e))
                pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del out
        # 丢掉第一轮（含 JIT / 首次 metadata 构建）
        drop = a.decode_len * a.warmup_iter
        wall = wall_steps[drop:]
        wall_mean = sum(wall) / len(wall)
        wall_sorted = sorted(wall)
        wall_median = wall_sorted[len(wall_sorted) // 2]
        wall_p99 = wall_sorted[int(len(wall_sorted) * 0.99) - 1] if len(wall_sorted) > 10 else wall_sorted[-1]
        wall_std = (sum((x - wall_mean) ** 2 for x in wall) / len(wall)) ** 0.5

    # ---------------- 2) kernel 级
    with torch.inference_mode():
        clear()
        out = model(input_ids=ids)
        pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out
        prime_offload(model, a)
        for _ in range(12):  # 预热
            out = model(input_ids=pred)
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del out
        torch.cuda.synchronize()

        from torch.profiler import profile, ProfilerActivity, record_function
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof_steps):
                with record_function('bench_decode_step'):
                    out = model(input_ids=pred)
                    torch.cuda.synchronize()
                pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del out
            torch.cuda.synchronize()

    evs = [e for e in prof.key_averages()
           if e.device_type == torch.autograd.DeviceType.CUDA and e.key != "bench_decode_step"]

    stages = {"select": 0.0, "load": 0.0, "compute_attn": 0.0,
              "compute_gemm": 0.0, "compute_misc": 0.0, "other": 0.0}
    detail = []
    total = 0.0
    for e in evs:
        ms = e.self_device_time_total / a.prof_steps / 1000.0
        if ms <= 0:
            continue
        st = classify(e.key)
        stages[st] += ms
        total += ms
        detail.append({"name": e.key, "stage": st, "ms_per_step": ms,
                       "calls_per_step": e.count / a.prof_steps})

    detail.sort(key=lambda x: -x["ms_per_step"])

    # offload 的实际 I/O 规模（唯一页数决定 gather 搬多少）
    off_stat = None
    try:
        st = getattr(getattr(model.model, "iController", None), "kv_offload", None)
        if st is not None and st.unique_hist:
            h = st.unique_hist
            off_stat = dict(st.stats())
            off_stat.update({
                "n_unique_mean": sum(h) / len(h),
                "n_unique_max": max(h), "n_unique_min": min(h),
                "n_unique_samples": len(h),
                "page_bytes": st.page_size * 2 * st.num_kv_heads * st.head_dim * 2,
            })
    except Exception as e:
        print("[warn] offload stats unavailable:", e)

    rec = {
        "tag": a.tag or f"{a.method}_{a.context_len}",
        "method": a.method, "offload": bool(a.offload or a.method == "naive"),
        "context_len": a.context_len, "token_budget": a.token_budget,
        "page_size": a.page_size, "decode_len": a.decode_len,
        "iteration": a.iteration, "prof_steps": a.prof_steps,
        "num_layers": n_layers,
        "wall_ms_mean": wall_mean, "wall_ms_median": wall_median,
        "wall_ms_p99": wall_p99, "wall_ms_std": wall_std,
        "wall_ms_min": min(wall), "wall_ms_max": max(wall),
        "schema_version": 2, "measure": "model_decode_forward",
        "dtype": "float16", "rope_impl": rope_impl,
        "wall_steps_ms": wall, "wall_definition": "CUDA event span around synchronized forward; sampling excluded",
        "budget_notes": "First 2 layers dense. ClusterKV adds sink and growing recent window; Quest page rounding/current page.",
        "sink": a.sink, "window": a.window,
        "offload_notes": "Quest custom UVA union-page gather retains original GPU KV allocation; not a capacity-saving implementation" if a.method == "quest" and a.offload else None,
        "offload_stats": off_stat,
        "kernels": detail[: a.topk_kernels],
    }
    if a.method == 'naive':
        rec['budget_notes'] = 'Every layer scans all K; per-query-head Top-K with no extra sink/window or dense layers'
        rec['offload_notes'] = 'Full post-RoPE K/V on pinned CPU; fetch ALL K before QK/Top-K, then selected V; one reusable layer-size GPU K staging buffer; no prefetch or cross-step reuse'
        rec['identification_definition'] = 'Full CPU-to-GPU K load + QK scores + Top-K; K load is not also counted in retrieval'
        rec['logical_key_load_bytes_at_context'] = n_layers*model.config.num_key_value_heads*a.context_len*(model.config.hidden_size//model.config.num_attention_heads)*2
        rec['identification_key_load_ms'] = sum(k['ms_per_step'] for k in detail if 'naive_identify_load_keys_kernel' in k['name'])
        rec['scan_implementation'] = 'Triton exhaustive FP32 dot/reduction over FP16 keys, torch Top-K'
        rec['logical_v_load_bytes_per_step'] = n_layers*model.config.num_attention_heads*a.token_budget*(model.config.hidden_size//model.config.num_attention_heads)*2
        rec['sink'] = rec['window'] = 0
    rec.update(summarize_profile(prof, classify, a.prof_steps))
    stages = rec['stages_ms']
    if a.out_json:
        prof.export_chrome_trace(a.out_json.replace('.json', '.trace.json'))
        with open(a.out_json, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: rec[k] for k in ('tag', 'wall_ms_median', 'profile_wall_ms', 'stages_ms', 'gap_ms', 'raw_stages_ms')}, indent=2))
    if a.out_json:
        print('saved ->', a.out_json)


if __name__ == '__main__':
    main()
