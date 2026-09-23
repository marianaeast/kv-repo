#!/usr/bin/env python
"""逐 decode step 计时 harness.

改造自 ClusterKV 官方 efficiency/bench_textgen.py:
  - 固定外部 input_ids (所有方法读同一份序列)
  - 逐 step 用 CUDA Event 计时 (elapsed_time -> ms)
  - prefill 走 dense (各方法默认路径)
  - 统计 mean / median / P90 / P99 / std / min / max
"""
import argparse, json, os, platform, time
import numpy as np
import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 模型路径可用环境变量 KV_BENCH_MODEL 覆盖（不同机器 HF 缓存位置不同）：
#   yq7: /data/lrc/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f2...
#   yq9: /home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f2...
DEFAULT_MODEL = os.environ.get(
    "KV_BENCH_MODEL",
    "/data/lrc/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["full", "quest", "clusterkv"])
    p.add_argument("--offload", action="store_true", help="KV offload to CPU pinned mem (clusterkv only)")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--input_pt", default="/home/zrd/bypasskv_repo/kv-repos/bench/data/ctx32k_pg19.pt")
    p.add_argument("--token_budget", type=int, default=512)
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--context_len", type=int, default=32768)
    p.add_argument("--decode_len", type=int, default=128)
    p.add_argument("--iteration", type=int, default=3)
    p.add_argument("--warmup_iter", type=int, default=1)
    p.add_argument("--nlist", type=int, default=200)
    p.add_argument("--niter", type=int, default=20)
    p.add_argument("--sink", type=int, default=16)
    p.add_argument("--window", type=int, default=320)
    p.add_argument("--window_nlist", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    assert a.warmup_iter < a.iteration
    if a.offload:
        assert a.method == "clusterkv", "offload is only implemented for clusterkv"
    return a


def build_model(a, dev):
    if a.method == "quest":
        from clusterkv.quest_models.llama import LlamaForCausalLM
    else:
        from clusterkv.clusterkv_models.llama import LlamaForCausalLM
    m = LlamaForCausalLM.from_pretrained(a.model_path, device_map=dev, torch_dtype=torch.float16)
    m.eval()
    return m


def main():
    a = parse()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.set_default_dtype(torch.float16)

    d = torch.load(a.input_pt)
    ids = d["input_ids"][: a.context_len].unsqueeze(0).to(dev)
    assert ids.shape[1] == a.context_len, f"input len {ids.shape[1]} != {a.context_len}"
    print(f"[info] input  : {a.context_len} tokens from {os.path.basename(a.input_pt)}", flush=True)

    model = build_model(a, dev)
    max_seq_len = a.context_len + a.decode_len + 512

    if a.method == "quest":
        model.quest_init(page_size=a.page_size, max_seq_len=max_seq_len,
                         token_budget=a.token_budget, dtype=torch.float16, device=dev)
    else:
        # dense baseline：用超大 token_budget 把 infer_token_budget 顶到 kv_seqlen，
        # 从而 need_estimate()==False，走 full attention（官方 bench_textgen 的做法）。
        init_tb = max(102400, max_seq_len) if a.method == "full" else a.token_budget
        print(f"[info] clusterkv_init token_budget={init_tb} (sparse budget={a.token_budget})", flush=True)
        model.clusterkv_init(nlist=a.nlist, niter=a.niter, max_seq_len=max_seq_len,
                             token_budget=init_tb, dtype=torch.float16, device=dev,
                             full=(a.method == "full"), sink=a.sink, window=a.window,
                             window_nlist=a.window_nlist, offload=a.offload)

    def clear():
        (model.quest_clear if a.method == "quest" else model.clusterkv_clear)()

    prefill_ms, step_ms = [], []
    out_text = None
    tok = None
    gen = None

    with torch.inference_mode():
        for it in range(a.iteration):
            clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            s.record()
            out = model(input_ids=ids)
            e.record()
            torch.cuda.synchronize()
            prefill_ms.append(s.elapsed_time(e))
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = [int(pred[0, 0])]
            del out

            steps = []
            for _ in range(a.decode_len):
                s.record()
                out = model(input_ids=pred)
                e.record()
                torch.cuda.synchronize()
                steps.append(s.elapsed_time(e))
                pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen.append(int(pred[0, 0]))
                del out
            step_ms.extend(steps)
            if it == a.iteration - 1:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(a.model_path)
                out_text = tok.decode(gen, skip_special_tokens=True)[:200].replace("\n", " ")
            print(f"[info] iter {it}: prefill {prefill_ms[-1]:.1f} ms | "
                  f"decode mean {np.mean(steps):.3f} ms", flush=True)

    warm = a.warmup_iter * a.decode_len
    arr = np.array(step_ms[warm:], dtype=np.float64)
    pf = np.array(prefill_ms[a.warmup_iter:], dtype=np.float64)

    res = {
        "tag": a.tag or f"{a.method}{'_offload' if a.offload else ''}",
        "method": a.method, "offload": bool(a.offload),
        "model_path": a.model_path,
        "context_len": a.context_len, "decode_len": a.decode_len,
        "token_budget": a.token_budget, "page_size": a.page_size,
        "init_token_budget": (max(102400, a.context_len + a.decode_len + 512) if a.method == "full" else a.token_budget),
        "nlist": a.nlist, "sink": a.sink, "window": a.window,
        "iteration": a.iteration, "warmup_iter": a.warmup_iter,
        "n_steps_measured": int(arr.size),
        "prefill_ms_mean": float(pf.mean()),
        "decode_ms_mean": float(arr.mean()),
        "decode_ms_median": float(np.median(arr)),
        "decode_ms_p90": float(np.percentile(arr, 90)),
        "decode_ms_p99": float(np.percentile(arr, 99)),
        "decode_ms_std": float(arr.std()),
        "decode_ms_min": float(arr.min()),
        "decode_ms_max": float(arr.max()),
        "decode_tokens_per_s": float(1000.0 / arr.mean()),
        "sample_output": out_text,
        "gpu": torch.cuda.get_device_name(0),
        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("\n===== RESULT =====")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("saved ->", a.out_json)


if __name__ == "__main__":
    main()
