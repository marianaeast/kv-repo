#!/usr/bin/env python
"""kernel 级拆解：clusterkv 的 decode 每步时间花在哪？

对同一次 decode 循环用 torch.profiler (CUDA activities) 采样，
按 kernel 聚合总耗时，对比 full / clusterkv 在 16K/32K/48K 下的差异，
用于验证「indexing 段开销随上下文增长」这一假说。

用法:
  python prof_decode.py --method clusterkv --context_len 49152 \
      --input_pt data/ctx48k_pg19.pt --steps 24 --topk 14
"""
import argparse, json, os
import torch

torch.backends.cudnn.benchmark = False
DEFAULT_MODEL = os.environ.get(
    "KV_BENCH_MODEL",
    "/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="clusterkv", choices=["full", "quest", "clusterkv"])
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
    p.add_argument("--steps", type=int, default=24, help="profile 多少个 decode step")
    p.add_argument("--topk", type=int, default=14)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    a = parse()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.set_default_dtype(torch.float16)

    d = torch.load(a.input_pt)
    ids = d["input_ids"][: a.context_len].unsqueeze(0).to(dev)
    assert ids.shape[1] == a.context_len

    if a.method == "quest":
        from clusterkv.quest_models.llama import LlamaForCausalLM
    else:
        from clusterkv.clusterkv_models.llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(a.model_path, device_map=dev, torch_dtype=torch.float16)
    model.eval()

    max_seq_len = a.context_len + 128 + 512
    if a.method == "quest":
        model.quest_init(page_size=a.page_size, max_seq_len=max_seq_len,
                         token_budget=a.token_budget, dtype=torch.float16, device=dev)
    else:
        init_tb = max(102400, max_seq_len) if a.method == "full" else a.token_budget
        model.clusterkv_init(nlist=a.nlist, niter=a.niter, max_seq_len=max_seq_len,
                             token_budget=init_tb, dtype=torch.float16, device=dev,
                             full=(a.method == "full"), sink=a.sink, window=a.window,
                             window_nlist=a.window_nlist, offload=False)

    clear = model.quest_clear if a.method == "quest" else model.clusterkv_clear

    with torch.inference_mode():
        clear()
        out = model(input_ids=ids)
        pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out

        # 预热若干步（含 JIT / 首次 metadata 构建）
        for _ in range(12):
            out = model(input_ids=pred)
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del out
        torch.cuda.synchronize()

        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.steps):
                out = model(input_ids=pred)
                pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del out
            torch.cuda.synchronize()

    evs = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.self_device_time_total, reverse=True)
    total = sum(e.self_device_time_total for e in evs)
    per_step = total / a.steps / 1000.0  # ms

    print(f"\n===== {a.method} ctx={a.context_len} budget={a.token_budget} "
          f"steps={a.steps} =====")
    print(f"total CUDA busy = {total/1000.0:.3f} ms / {a.steps} steps = {per_step:.3f} ms per step")
    print(f"{'kernel':<58s} {'ms/step':>9s} {'%':>6s} {'calls/step':>10s}")
    print("-" * 88)
    for e in evs[: a.topk]:
        ms = e.self_device_time_total / a.steps / 1000.0
        if ms < 0.005:
            break
        print(f"{e.key[:57]:<58s} {ms:9.3f} {100*e.self_device_time_total/total:6.1f} "
              f"{e.count/a.steps:10.1f}")

    if a.tag:
        rec = {"tag": a.tag, "method": a.method, "context_len": a.context_len,
               "token_budget": a.token_budget, "steps": a.steps,
               "cuda_busy_ms_per_step": per_step,
               "kernels": [{"name": e.key,
                            "ms_per_step": e.self_device_time_total / a.steps / 1000.0,
                            "calls_per_step": e.count / a.steps,
                            "pct": 100 * e.self_device_time_total / total}
                           for e in evs[: a.topk]]}
        with open(f"results/prof_{a.tag}.json", "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print("saved ->", f"results/prof_{a.tag}.json")


if __name__ == "__main__":
    main()
