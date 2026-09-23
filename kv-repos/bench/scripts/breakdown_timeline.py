"""同一条 profiler 时间线上的互斥分解，避免跨 stream 重复计时。"""
from collections import Counter

STAGES = ("select", "load", "compute_attn", "compute_gemm", "compute_misc", "other")


def partition_intervals(intervals, start, end):
    """intervals为(start_us, end_us, stage)。重叠时计算/选择优先于LOAD。

    LOAD结果是无其他GPU工作覆盖的搬运时间，不声称是依赖图关键路径等待。
    host发射空洞计入gap；单条时间线各项恰好加总到end-start。
    """
    changes = [(start, None, 0), (end, None, 0)]
    for left, right, stage in intervals:
        left, right = max(left, start), min(right, end)
        if right > left:
            changes.extend([(left, stage, 1), (right, stage, -1)])
    changes.sort(key=lambda item: item[0])
    active = Counter()
    totals = dict.fromkeys(STAGES + ("gap",), 0.0)
    # SELECT本身也是计算，完整保留；并发的模型计算归入剩余时间。
    # LOAD只保留未被SELECT或其他计算覆盖的区间，不能直接减两个阶段总时长。
    priority = ("select", "compute_attn", "compute_gemm", "compute_misc", "other", "load")
    previous = start
    for instant, stage, change in changes:
        owner = next((key for key in priority if active[key] > 0), "gap")
        totals[owner] += instant - previous
        if stage is not None:
            active[stage] += change
        previous = instant
    return totals


def summarize_profile(prof, classify, expected_steps):
    events = prof.events()
    windows = [e for e in events if e.name == "bench_decode_step"
               and str(e.device_type).split(".")[-1] == "CPU"]
    if len(windows) != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} step ranges, found {len(windows)}")
    intervals = []
    raw = dict.fromkeys(STAGES, 0.0)
    for event in events:
        if str(event.device_type).split('.')[-1] != 'CUDA':
            continue
        if event.name == "bench_decode_step":
            continue  # GPU user annotation不是实际kernel
        left, right = event.time_range.start, event.time_range.end
        intervals.append((left, right, classify(event.name)))
    if not intervals:
        raise RuntimeError("Profiler returned no CUDA events; cannot emit a breakdown")
    return summarize_windows(intervals, [(w.time_range.start, w.time_range.end) for w in windows])


def summarize_windows(intervals, windows):
    """从实际kernel区间和step窗口统计；也供已保存的Chrome trace重新归类。"""
    totals = dict.fromkeys(STAGES + ("gap",), 0.0)
    raw = dict.fromkeys(STAGES, 0.0)
    wall = 0.0
    for start, end in windows:
        wall += end - start
        for left, right, stage in intervals:
            raw[stage] += max(0, min(right, end) - max(left, start))
        parts = partition_intervals(intervals, start, end)
        for stage, value in parts.items():
            totals[stage] += value
    scale = len(windows) * 1000.0
    return {
        "stages_ms": {key: totals[key] / scale for key in STAGES},
        "gap_ms": totals["gap"] / scale,
        "profile_wall_ms": wall / scale,
        "cuda_busy_ms": (wall - totals["gap"]) / scale,
        "raw_stages_ms": {key: value / scale for key, value in raw.items()},
        "raw_kernel_sum_ms": sum(raw.values()) / scale,
        "overlap_ms": (sum(raw.values()) - wall + totals["gap"]) / scale,
        "load_definition": "unoverlapped GPU load occupancy; not dependency-proven critical-path wait",
        "stack_definition": "select-first exclusive profiler timeline; sum(stages_ms)+gap_ms=profile_wall_ms; do not stack onto unprofiled median",
        "attribution_priority": ["select", "compute_attn", "compute_gemm", "compute_misc", "other", "load"],
    }
