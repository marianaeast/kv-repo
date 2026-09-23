"""只汇总本次同预算结果，校验互斥分解；InfiniGen微基准单独输出。"""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--budget', type=int, required=True)
p.add_argument('--require-naive', action='store_true')
a = p.parse_args()
methods = ['full', 'clusterkv', 'quest', 'clusterkv_offload', 'quest_offload', 'shadowkv', 'infinigen_microbench']
if a.require_naive or (a.directory/'naive.json').exists():
    methods.insert(-1, 'naive')
rows = []
for method in methods:
    r = json.loads((a.directory / (method + '.json')).read_text())
    budget = r.get('token_budget', r.get('sparse_budget', r.get('budget')))
    if r.get('schema_version') != 2 or r['context_len'] != 32768 or budget != a.budget:
        raise ValueError(f'{method}: wrong schema/context/budget')
    parts = r['stages_ms']
    numbers = list(parts.values()) + [r['gap_ms'], r['profile_wall_ms'], r['wall_ms_median']]
    if any(not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError(f'{method}: invalid/negative timings')
    if not math.isclose(sum(parts.values()) + r['gap_ms'], r['profile_wall_ms'], abs_tol=1e-5):
        raise ValueError(f'{method}: inconsistent stacked denominator')
    rows.append(dict(method=method, scope=r['measure'], context=32768, budget=budget, dtype=r['dtype'],
        wall_median_ms=r['wall_ms_median'], profile_wall_ms=r['profile_wall_ms'],
        select_ms=parts['select'], load_unoverlapped_ms=parts['load'],
        compute_ms=sum(parts[k] for k in ('compute_attn','compute_gemm','compute_misc')),
        other_ms=parts['other'], gap_ms=r['gap_ms'], raw_load_ms=r['raw_stages_ms']['load']))
for name, selected in [('summary_model.csv',rows[:-1]), ('summary_microbench.csv',rows[-1:])]:
    with (a.directory/name).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=selected[0].keys()); writer.writeheader(); writer.writerows(selected)
    print('saved ->',a.directory/name)
print(f'PASS: all {len(methods)} results have matching configuration and nonnegative additive timelines')
