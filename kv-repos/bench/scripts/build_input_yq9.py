import os, torch
from transformers import AutoTokenizer

MODEL = os.environ.get("KV_BENCH_MODEL",
    "/home/zrd/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
SRC = "/home/zrd/bypasskv_repo/kv-repos/bench/data/pg19_test10146.txt"
OUT = "/home/zrd/bypasskv_repo/kv-repos/bench/data"

tok = AutoTokenizer.from_pretrained(MODEL)
text = open(SRC, encoding="utf-8").read()
ids_full = tok(text, truncation=False, return_tensors="pt").input_ids[0]
print("book full tokens :", len(ids_full))

for target, name in [(32768, "ctx32k_pg19.pt"), (49152, "ctx48k_pg19.pt")]:
    if len(ids_full) < target:
        print(f"SKIP {name}: need {target} > {len(ids_full)}"); continue
    ids = ids_full[:target]
    torch.save({"input_ids": ids, "model_path": MODEL, "source": SRC,
                "target_len": target, "full_len": int(len(ids_full))},
               os.path.join(OUT, name))
    print(f"saved {name}: shape={tuple(ids.shape)}")
