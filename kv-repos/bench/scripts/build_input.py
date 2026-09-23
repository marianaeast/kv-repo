import torch, json, os
from transformers import AutoTokenizer

MODEL = "/data/lrc/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
SRC   = "/home/zrd/bypasskv_repo/kv-repos/bench/data/pg19_test10146.txt"
OUT   = "/home/zrd/bypasskv_repo/kv-repos/bench/data"
TARGET = 32768

tok = AutoTokenizer.from_pretrained(MODEL)
text = open(SRC, encoding="utf-8").read()
ids_full = tok(text, truncation=False, return_tensors="pt").input_ids[0]
print("全书 token 数 :", len(ids_full))

assert len(ids_full) >= TARGET, f"书太短 ({len(ids_full)} < {TARGET})"
ids = ids_full[:TARGET]

torch.save({"input_ids": ids, "model_path": MODEL, "source": SRC,
            "target_len": TARGET, "full_len": int(len(ids_full))},
           os.path.join(OUT, "ctx32k_pg19.pt"))
print("已保存 32K 序列, shape =", tuple(ids.shape))
print("前 20 token id :", ids[:20].tolist())
print("解码前 160 字符:", tok.decode(ids[:40])[:160].replace(chr(10)," "))
print("解码末 160 字符:", tok.decode(ids[-40:])[:160].replace(chr(10)," "))
