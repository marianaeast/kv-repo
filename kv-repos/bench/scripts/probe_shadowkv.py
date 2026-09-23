import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SK = "/home/zrd/bypasskv_repo/kv-repos/ShadowKV"
sys.path.insert(0, SK); sys.path.insert(0, os.path.join(SK, "kernels"))
os.chdir(SK)
import torch
from models import choose_model_class

MODEL = "/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
CTX = 32768
BUDGET = 1024

d = torch.load("/home/zrd/bypasskv_repo/kv-repos/bench/data/ctx32k_pg19.pt")
ids = d["input_ids"][:CTX].unsqueeze(0).to("cuda:0")
print("input:", tuple(ids.shape), ids.dtype)

LLM = choose_model_class(MODEL)
llm = LLM(model_name=MODEL, device="cuda:0", batch_size=1,
          max_length=CTX + 2048, attn_mode="shadowkv_cpu",
          sparse_budget=BUDGET, dtype=torch.bfloat16)
print("model built | sparse_budget", BUDGET)

t0 = time.time()
with torch.inference_mode():
    logits = llm.batch_prefill(ids)
    torch.cuda.synchronize()
print("prefill: %.1f s | logits %s" % (time.time() - t0, tuple(logits.shape)))

llm.kv_cache.H2D()
print("kv_cache.H2D done")
llm.warmup()

with torch.inference_mode():
    nxt = logits[:, -1, :].argmax(-1, keepdim=True)
    t0 = time.time(); N = 8
    for i in range(N):
        out = llm.inference(input_ids=nxt, position_ids=llm.get_ctx(nxt))
        nxt = out[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
    print("decode: %.2f ms/step" % ((time.time() - t0) / N * 1000))
print("PROBE OK")
