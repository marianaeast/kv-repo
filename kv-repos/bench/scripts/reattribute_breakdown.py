"""用保存的真实kernel trace重新分配堆叠时间，不重跑模型、不改变wall测量。"""
import argparse
import json
import math
import shutil
from pathlib import Path

from bench_breakdown import classify as classify_model
from bench_shadowkv import classify as classify_shadow
from breakdown_timeline import summarize_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    pending = []
    for method in ('full', 'quest', 'clusterkv', 'naive', 'quest_offload',
                   'clusterkv_offload', 'shadowkv'):
        path = args.directory / (method + '.json')
        rec = json.loads(path.read_text())
        trace = path.with_suffix('.trace.json')
        events = json.loads(trace.read_text())['traceEvents']
        classify = classify_shadow if method == 'shadowkv' else classify_model
        windows = [(e['ts'], e['ts'] + e['dur']) for e in events
                   if e.get('cat') == 'user_annotation' and e.get('name') == 'bench_decode_step']
        intervals = [(e['ts'], e['ts'] + e['dur'], classify(e['name'])) for e in events
                     if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
        if len(windows) != rec['prof_steps']:
            raise ValueError(f'{method}: wrong number of step windows')
        new = summarize_windows(intervals, windows)
        # Chrome trace rounds timestamps; only allow that small serialization difference.
        for key in ('profile_wall_ms', 'gap_ms'):
            if not math.isclose(new[key], rec[key], abs_tol=0.001):
                raise ValueError(f'{method}: {key} changed unexpectedly')
        for stage, value in rec['raw_stages_ms'].items():
            if not math.isclose(new['raw_stages_ms'][stage], value, abs_tol=0.001):
                raise ValueError(f'{method}: raw {stage} changed unexpectedly')
        rec.update(new)
        rec['attribution_source'] = str(trace)
        pending.append((path, rec))
    backup = args.directory / 'before_select_priority'
    backup.mkdir(exist_ok=True)
    for path, rec in pending:
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f'{path.stem}: select={rec["stages_ms"]["select"]:.4f} ms; raw/wall preserved')


if __name__ == '__main__':
    main()
