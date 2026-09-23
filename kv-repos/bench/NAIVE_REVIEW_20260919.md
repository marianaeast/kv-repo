# Naive baseline validation

- Full K and V reside in pinned CPU memory. Each layer/step loads ALL K into a shared one-layer GPU staging buffer, computes QK/Top-K, then fetches selected V. K transfer is counted in Identification, not counted again in Load. Per-query-head Top-K, all 32 layers, no V reuse/prefetch/extra sink or window.
- Kernel checks: full CPU K load/reload, padded capacity and untouched tail, full-key scores, Top-K sets, V values, weighted output, noncontiguous head capacity, boundary/repeated indices: PASS.
- Standard HF dense-limit comparison: 256-token prompt, budget 512, prefill + 3 sequential decode steps: PASS. Maximum logit error 0.03125. Clear/replay: PASS.
- Full 32K/512 and 32K/1024: 96 steps x 3 rounds, discard first round (192 measured steps), separate 24-step profile: exit 0, eight-result summary checks PASS.
- Inputs/results: results/bd32k_b{512,1024}_20260919_022052_3074385/naive.json.
- Logs: logs/naive_cpu_keys_check.log, logs/naive_cpu_keys_full32k.log; hardware and source hashes stored beside each naive.json.
- Figures updated in /home/zrd/BypassKV/Eurosys27/images/motivation_breakdown_32k_*.pdf.
- Existing ClusterKV dense branch did not agree with naive in a short-context cross-check; naive agrees with standard HF. Existing Full measurements were not rerun or modified. See root README for the boundary of this finding.

- Corrected CPU-K results (512 / 1024): normal median 58.8377 / 60.6982 ms; raw Identification 45.8743 / 45.8733 ms, including full K transfer 41.8980 / 41.8955 ms.
- Previous GPU-resident-K results are superseded and archived under each result directory in superseded_gpu_key_naive/. Other seven methods were not rerun.
