import os, time, torch

# 多尺寸 HBM 带宽实测：copy 大 buffer，读+写各计一次
# 用法: python bw_test2.py            -> 512MB / 2GB / 4GB 三档
SIZES_MB = [int(x) for x in os.environ.get("BW_SIZES_MB", "512,2048,4096").split(",")]
ITERS = int(os.environ.get("BW_ITERS", "50"))
WARMUP = 5


def one(n_mb):
    n_el = n_mb * 1024 * 1024 // 2  # fp16 = 2 bytes
    a = torch.empty(n_el, dtype=torch.float16, device="cuda")
    b = torch.empty_like(a)
    for _ in range(WARMUP):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    moved = n_el * a.element_size() * 2  # read + write
    del a, b
    torch.cuda.empty_cache()
    return dt, moved / dt / 1e12


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"total mem    : {props.total_memory/1024**3:.1f} GiB")
    print(f"L2 cache     : {props.L2_cache_size/1024**2:.1f} MiB")
    print("-" * 46)
    for s in SIZES_MB:
        dt, gbps = one(s)
        print(f"{s:5d} MB copy : {dt*1e3:8.3f} ms  -> {gbps:5.2f} TB/s (r+w)")


if __name__ == "__main__":
    main()
