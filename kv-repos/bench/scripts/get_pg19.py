import os, sys
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from datasets import load_dataset

out = "/home/zrd/bypasskv_repo/kv-repos/bench/data/pg19_book0.txt"
try:
    ds = load_dataset("deepmind/pg19", split="test", streaming=True)
    row = next(iter(ds))
    text = row["text"]
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print("OK  chars =", len(text))
    print("head:", text[:180].replace("\n", " "))
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    sys.exit(1)
