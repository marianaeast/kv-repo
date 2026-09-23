#!/usr/bin/env python
"""诊断 quest+offload 的 launch gap：把每步墙钟拆成 CPU 提交 / GPU 执行 / 同步。"""
import os, sys, time, statistics
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch

MODEL = os.environ.get("KV_BENCH_MODEL")
CTX, BUDGET, PAGE = 32768, 512, 16


def main():
    d = torch.load("/home/zrd/bypasskv_repo/kv-repos/bench/data/ctx32k_pg19.pt")
    ids = d["input_ids"][:CTX].unsqueeze(0).cuda()

    from clusterkv.quest_models.llama import LlamaForCausalLM
    m = LlamaForCausalLM.from_pretrained(MODEL, device_map="cuda:0", torch_dtype=torch.float16)
    m.eval()
    m.quest_init(page_size=PAGE, max_seq_len=CTX + 512, token_budget=BUDGET,
                 dtype=torch.float16, device="cuda:0", offload=True)
    m.quest_clear()

    import clusterkv.utils.offload_cache as oc
    ST = {"unique_ms": 0.0, "unique_n": 0, "sfg_ms": 0.0, "sfg_n": 0,
          "recall_ms": 0.0, "recall_n": 0}

    _u = torch.unique
    def tu(*a, **kw):
        t0 = time.perf_counter(); r = _u(*a, **kw); t1 = time.perf_counter()
        ST["unique_ms"] += (t1 - t0) * 1000; ST["unique_n"] += 1
        return r
    torch.unique = tu

    _sfg = oc.OffloadKvStore.sync_from_gpu
    def sfg(self, ctrl):
        t0 = time.perf_counter(); r = _sfg(self, ctrl); t1 = time.perf_counter()
        ST["sfg_ms"] += (t1 - t0) * 1000; ST["sfg_n"] += 1
        return r
    oc.OffloadKvStore.sync_from_gpu = sfg

    _rc = oc.OffloadKvStore.recall
    def rc(self, layer_idx, page_ids, ctrl):
        t0 = time.perf_counter(); r = _rc(self, layer_idx, page_ids, ctrl); t1 = time.perf_counter()
        ST["recall_ms"] += (t1 - t0) * 1000; ST["recall_n"] += 1
        return r
    oc.OffloadKvStore.recall = rc

    with torch.inference_mode():
        out = m(input_ids=ids)
        pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True); del out
        for _ in range(6):
            out = m(input_ids=pred); pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True); del out
        torch.cuda.synchronize()

        N = 24
        wall, cpu_sub, total = [], [], []
        for _ in range(N):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            s.record()
            out = m(input_ids=pred)
            e.record()
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            wall.append(s.elapsed_time(e)); cpu_sub.append((t1 - t0) * 1000); total.append((t2 - t0) * 1000)
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True); del out

        # 纯 CPU 提交时间（每步先 sync 清空队列）
        cpu_only = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = m(input_ids=pred)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            cpu_only.append((t1 - t0) * 1000)
            pred = out.logits[:, -1, :].argmax(dim=-1, keepdim=True); del out

    avg = lambda x: sum(x) / len(x)
    print("== quest+offload 32K 诊断（GPU=%s）==" % torch.cuda.get_device_name(0))
    print("event wall          : %8.3f ms" % avg(wall))
    print("perf_counter total  : %8.3f ms   (= CPU 提交 + GPU 完成, 不打断流水)" % avg(total))
    print("CPU 提交时间(不打断): %8.3f ms" % avg(cpu_sub))
    print("CPU-only 提交(每步先sync): %8.3f ms  <- 纯 python/launch 成本" % avg(cpu_only))
    print("-" * 60)
    for k, lab in [("unique", "torch.unique"), ("sfg", "sync_from_gpu"), ("recall", "OffloadKvStore.recall")]:
        n = ST[k + "_n"]
        if n:
            print("%-22s: %8.3f ms  total, %6.1f calls/step, %7.3f ms/call" %
                  (lab, ST[k + "_ms"] / N, n / N, ST[k + "_ms"] / n))
    print("-" * 60)
    print("注：每步 %d 层；recall 内部含 unique + gather launch + D2D（与 unique 有重叠统计）" % 32)


if __name__ == "__main__":
    main()
