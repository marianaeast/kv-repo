# KV baseline experiments

本仓库保存本地修改后的 KV-cache 对比方法、精度评测和性能实验代码。

- 最新共享预算精度版：[ACCURACY_FAIR_V2.md](ACCURACY_FAIR_V2.md)，入口 `run_hotpotqa_accuracy_128_fair.sh`。
- 共享预算精度入口支持 `ACCURACY_MODEL=llama3.1-8b`（默认）和
  `ACCURACY_MODEL=qwen3-8b`；Qwen3 使用独立 Conda 环境，不改动 Llama
  复现实验的依赖。
- 性能实验入口：`run_bd_32k.sh`。性能路径尚未接入精度 v2 的全部预算规则，不能用精度脚本耗时替代性能实验。
- `kv-repos/` 中各项目和已下载的依赖均按源码快照纳入版本管理，包含本地修改；上游版本记录在 `SOURCE_REPOSITORIES.json`，许可证保留在各项目内。
- 本机原有嵌套 Git 元数据保留，但不上传，也不作为新仓库的 submodule。克隆本仓库即可获得已纳入的源码；内层 `.gitmodules` 是上游配置留档。
- 数据集、模型权重、实验输出、编译产物和旧备份不上传。脚本目前使用本机 `/home/zrd` 路径和 Conda 环境，换机器后需按实际位置配置并重新构建 CUDA 扩展。

## 32K decode breakdown：运行与口径（2026-09-19 修订）

本目录用于检查论文图 2b 的计时来源。论文里的图目前仍标为 synthetic；此脚本不会修改论文或替换图。

## 运行

一次运行 32K/512 和 32K/1024，两组各 8 个配置：

```bash
bash /home/zrd/bypasskv_repo/run_bd_32k.sh 512 1024
```

只跑某个预算：`bash /home/zrd/bypasskv_repo/run_bd_32k.sh 512`。指定另一张卡：`CUDA_VISIBLE_DEVICES=1 bash /home/zrd/bypasskv_repo/run_bd_32k.sh 512 1024`。

脚本自动设置环境。full、ClusterKV、Quest 使用 `/home/zrd/miniconda3/envs/clusterkv-quest`，ShadowKV 和 InfiniGen 微基准使用 `/home/zrd/miniconda3/envs/ShadowKV`。CUDA 默认 `/usr/local/cuda-12.8`，架构默认 sm_90。切换环境时同时设置 PATH 和 CONDA_PREFIX；首次执行需 JIT 编译小型扩展。

默认模型是本地 Llama-3.1-8B-Instruct；输入为 `kv-repos/bench/data/ctx32k_pg19.pt`，PG19 book id 10146 的前 32768 个 token，batch size 1。可用 `KV_BENCH_MODEL` 环境变量指定模型路径，但当前参数与补丁仅检查了 Llama-3.1-8B。

真实模型配置：每轮 96 个 decode step、3 轮，丢弃第 1 轮；每轮重新 prefill 同一输入并重新生成。另从同一 prompt 重新 prefill、预热 12 步、采集 24 步 profiler。预填充/初始化不计入 decode。sampling/argmax 不计入 forward 延迟。

InfiniGen 微基准：10 步预热、96 步实际流水线计时、24 步带事件的拆解。它不生成文本，不等同于上述模型执行。

短跑检查（只能验证执行和计时字段，不能作为性能结论）：

```bash
DECODE_LEN=2 ITERATION=1 WARMUP_ITER=0 PROF_STEPS=2 INF_WARMUP=1 bash /home/zrd/bypasskv_repo/run_bd_32k.sh 512 1024
```

仅查看命令：`bash /home/zrd/bypasskv_repo/run_bd_32k.sh --dry-run 512 1024`。

## 输出

每次运行生成独立目录，不覆盖旧数据：

- `kv-repos/bench/results/bd32k_b512_<时间>_<pid>/`
- `kv-repos/bench/results/bd32k_b1024_<时间>_<pid>/`
- 对应日志在 `kv-repos/bench/logs/bd32k_b.../`，逐配置保存。

目录包含每配置 JSON、真实模型的 Chrome trace、硬件信息、脚本快照/校验值。
`summarize_breakdown_v2.py` 校验全部配置的 schema、32K context、预算与时间加和，生成 `summary_model.csv` 和单独的 `summary_microbench.csv`。任一运行失败就停止，不把旧文件混入新结果。

历史 `RESULTS_32K.md` 和原有 JSON 保留为旧实验记录；其中理论 LOAD、负 gap、混用预算的数值不可与新结果混用。原文件备份在 `review_backup_20260919/`。

## LOAD 怎么测、怎样画堆叠图

真实模型同时提供两套统计：

1. `wall_ms_*`：不启用 profiler，CUDA event 包住逐步 forward，并同步等待完成。它是实际运行时间，不由各阶段相加推算。
2. `profile_wall_ms` 和 `stages_ms`：在 profiler 的同一步时间窗口内，将 CUDA kernel/memcpy 时间区间求并集；重叠时依次归到选择、attention、GEMM、其他计算、other、LOAD。SELECT完整保留，和它同时执行的其他计算不再重复堆叠。没有 GPU 活动的区间计入 `gap_ms`。

因此 `sum(stages_ms) + gap_ms == profile_wall_ms`，不会因为 copy stream 与计算重叠而出现负 gap。`raw_stages_ms` 保留未去重的各阶段时长，用于审计；不要把它们直接堆叠。profiler 本身会带来开销，不能把 profile 各段直接堆到另一轮的 wall median 上。

**`stages_ms.load` 是未被其他 GPU 活动覆盖的加载占用，不是严格证明的关键路径等待。** 无重叠不等于依赖阻塞：LOAD 可能与 host 工作重叠，已重叠的 LOAD 也可能拖慢计算。图 2b 如果写 “exposed retrieval / remaining data wait”，还需要按运行时依赖记录等待事件，不能直接把本字段改名就使用。`gap_ms` 也不能全部说成 Python/launch overhead，它包括同步、调度和其他无 GPU kernel 的时间。

GPU 内部 D2D gather 计入 compute_misc，不当成 CPU→GPU I/O。普通 attention 中的 HBM 读访问不拆成理论 LOAD。删除了旧脚本的 HBM 带宽推算字段。

InfiniGen 微基准用 CUDA events 记录实际 SELECT/LOAD/COMPUTE 区间并去重，wall 单独同步实测；`measured_prefetch_wait_ms` 是主流等待预取 ready 的事件区间，包括尚未完成的选择/加载以及事件和 host 提交间隙，**不能全算成纯 I/O**。事件包围的是操作阶段，和真实模型的逐 kernel profiler 归因仍有差别。

## 各配置实际是什么

| 配置 | 当前执行路径 | 预算和限制 |
|---|---|---|
| full | ClusterKV 的 dense 分支，fp16 | 全 KV；512/1024 只是实验分组标签 |
| clusterkv | ClusterKV 原生 GPU cache 路径，fp16 | 前 2 层 dense；其余层还包含 sink=16 和增长的 recent window，window 上限 320 |
| clusterkv_offload | ClusterKV 原生 CPU cache/增量换入路径，fp16 | 前 2 层 dense；per-query-head 选择，动态集合缓存并仅换入 miss |
| quest | ClusterKV 仓库内 Quest 实现，fp16 | page_size=16、前 2 层 dense；预算经页取整并处理当前页 |
| quest_offload | 上述 Quest 加本地 UVA offload 补丁，fp16 | 每层跨 query heads 合并页号，加载这些页的所有 KV heads，可能明显放大搬运量；仍保留原 GPU cache 分配，不能声称已实现省显存的 offload |
| shadowkv | ShadowKV shadowkv_cpu，bf16 | rank=160、chunk=8；预算为动态检索 token 数，不含 local/outlier/新生成 token；选择按 KV head 合并；V 远程搬运和 K 重建重叠 |
| naive | Llama真实模型：完整post-RoPE K/V主副本均在CPU pinned，每步先加载完整K | 所有层先CPU→GPU加载完整K，再逐query head扫描、Top-K，最后加载所选V；不预取、不缓存上一步V、无额外sink/window；budget不等于总GPU占用 |
| infinigen_fixed_budget_microbench | 合成张量上的 InfiniGen 风格下一层预取 | **不是原版 InfiniGen 的模型端到端结果**，单独汇总 |

“512/1024”是各运行时的动态预算参数，不代表每层 attention 的总 token 数、总 GPU KV 占用或传输字节完全一致。ShadowKV 是 bf16、其他真实模型是 fp16；上述差别都需要在论文实验设置中公开。

### InfiniGen 改动

旧版 LOAD 是 `字节数 / 41 GB/s`，wall 是阶段相加，还存在 SELECT 使用 32 个 query heads、LOAD/attention 却按 8 个 KV heads 的不一致。

新版实际执行：

1. 部分通道 query 投影、与部分 K 打分并 Top-K；默认部分通道比例 0.3，即 head_dim=128 时 38 个通道，不再把 rank=16 的低秩预测叫原版 InfiniGen。
2. 真实 pinned CPU K/V，经 UVA kernel 随机 gather 到 GPU；32 个 query heads 独立选 K 个 token，映射到 8 个 KV heads，各 head 使用自己的所选集合。不跨 head 去重，传输字节按实际 kernel 计。
3. 独立预取 stream 用本层输入预测/加载下一层；计算流只有在当前层 attention 需要数据时等待当前层 ready。首层预取/流水线尾部均计入时间。
4. 每层独立权重，执行 QKV、attention、输出投影和 FFN；有 RMS 归一化避免随机激活发散。wall 来自整步同步实测；LOAD 来自真实 gather 事件。

仍然是随机权重/部分 K 和固定 Top-K，没有真实模型校准、alpha 自适应阈值、RoPE、embedding、lm_head 或文本生成；UVA 搬运后端也不是上游 CPU embedding。它能回答这个几何和调度下实际传输多少时间，**不能据此给出 InfiniGen 原版排名或宣称相同精度**。

### ShadowKV 512 修复

旧文档把失败归因于 outlier=0 并声称最小预算为 1024，不准确。直接问题是 gather/reorder 的 C++ 分派没有 64 chunks 分支（512/8），且 copy.cuh 的 `MAP_SIZE/128` 循环在 64 chunks 时执行零次。

本次补齐 64-chunk 分派和不足 128 的 offset 读取，并将原 default-stream 调用改成 current-stream。通过小扩展单独编译 gather 文件，不重编整个 CUTLASS 包。outlier 保留上游规则：512 时为 0、1024 时为 24；0 不会使 gather 空转。landmark 维度必须满足 kernel 的 8 对齐，不能随意设成 12 个 outlier。补齐了 512 的 gather 支持，但不能称作未经修改的上游配置。每轮重新 prefill，修复只重置 offset、却复用旧 cache 和 token 的错误。CPU V 搬运的 head stride 改用实际分配 stride（原来错误使用 prompt 长度，max_length 较大时会读错其他 head）；GPU buffer 为 local chunk 对齐额外预留空间。

### Llama3.1 RoPE

旧 ClusterKV/Quest 兼容补丁遇到 llama3 时仅设 scale=1，遗漏分段频率。bench 现在用已安装 Transformers 的 Llama3 频率函数建表，再用 FlashInfer 的 fused RoPE；安装时对 0/32767 等位置与直接公式核对。该修正同时应用于 full/ClusterKV/Quest。

Quest offload 首次 prefill 后显式构建 CPU 副本，每次新 prefill 都重新同步，只同步已完整的历史页并按真实物理页号定位，避免把尚在增长的当前页错误标成已同步；扩展失败时不再接受悄悄退回 CPU gather 后继续出结果。


## 新增 Naive：扫描全部 K，仅加载选中 V

`bench/scripts/naive_attention.py` 接在与其他真实模型相同的 Llama 外层上，32 个 query heads 各选 B 个 token，映射到 8 个 KV heads。扫描使用 FP32 dot/reduction 对完整 FP16 K 评分，选择后对所选分数做 softmax，再与真实 CPU→GPU 搬运的 V 相乘。当前 token 的 K/V 追加也计入 decode，均包含 D2H。prefill 使用完整 causal attention，初始化 CPU K/V 不计入 decode。每步每层重新从CPU读取完整K到共享的一层GPU暂存区，没有跨层或跨步保留完整K。

512/1024 预算每步逻辑 V 读量分别是 128/256 MiB（32层×32个query heads×budget×128×2字节），不跨 query head 去重。32K时还要每步加载约2 GiB完整K（32层×8 KV heads×32768×128×2字节）。GPU只复用一层K暂存区的分配，不复用其内容。Identification包含这部分K传输、QK和Top-K；后续Retrieval/LOAD是所选V，K传输不重复计算。

主脚本以后自动跑8项。已有两组七项结果可补测 naive：

```bash
bash /home/zrd/bypasskv_repo/run_naive_32k.sh 20260919_022052_3074385
```

补测结果为原结果目录中的 `naive.json`，另存硬件、源码摘要和日志。汇总器兼容旧7项；存在naive或指定`--require-naive`时汇总8项。

验证：随机非零数据检查所有K评分、Top-K集合、独立head的V加载、加权输出和边界/重复索引。`check_naive_model.py` 还在256-token前缀、budget=512覆盖全部历史的条件下，与标准HF SDPA模型核对prefill和连续3步decode，再检查清空后重放prompt；FP16最大logit误差依次为0.02344、0.01660、0.02344、0.03125，均通过设定容差。该检查验证实现，不等于32K稀疏生成质量评测。

交叉检查还发现：仓库已有ClusterKV dense分支在同一个短上下文测试的首个decode与naive最大logit差约0.86，而naive与标准HF参考通过。因此已有Full分支的生成数值正确性需单独核查；本次新增没有以旧Full输出作为正确性标准，也没有修改或重测已有Full结果。


### CPU K纠正与原始阶段成本

此前“完整K在GPU”的naive是对用户要求的误解，旧结果移至各结果目录的`superseded_gpu_key_naive/`，当前naive.json改用CPU完整K/V版本重测。CPU K复制/重载、Top-K及V读回测试和HF dense-limit检查均通过，日志为`logs/naive_cpu_keys_check.log`。

旧版去重规则会将与计算重叠的SELECT也归给计算，现已改为SELECT优先。旧图InfiniGen约0.42 ms不是选择操作的全部时长：512/1024的原始phase SELECT分别约3.94/3.99 ms；原始LOAD约5.87/11.08 ms，图中未重叠LOAD仅约0.17/0.16 ms。微基准为合成权重和CUDA事件阶段计时，不能据此证明原版InfiniGen几乎隐藏了全部I/O。

ShadowKV原始V加载路径为1.47/2.66 ms，去重后约0.99/1.77 ms。它按8个KV heads加载V的cache miss，K在GPU重建。即使没有任何V cache hit，512/1024的每步V读量上限也只有32/64 MiB；InfiniGen当前微基准按32个query heads分别读取K和V，不去重，对应256/512 MiB。这是两种实现在当前参数下的逻辑读量差异，不能把小时间直接判为假数据，也不能在未记录hit统计时宣称具体命中率。

新增`images/motivation_breakdown_32k_raw_costs.pdf`分别展示原始Identification和Retrieval成本。因并发重叠，两者与compute不能直接相加当wall。主堆叠图继续保留去重口径，并显式标注Naive identification包括CPU K加载。


### SELECT完整保留（2026-09-19修正）

SELECT也是GPU计算，可以与另一stream的模型计算并行；这不应让图中选择成本消失。现在所有堆叠图优先保留SELECT区间的并集，其次计其他计算，最后计尚未被覆盖的LOAD。Others相应去重，总和仍等于同一测量窗口。多个SELECT若自身同时执行，不能将其重复堆成wall。

同一批KV必须先完成选择才能开始加载；不能用LOAD总时长减SELECT总时长。当前InfiniGen微基准的SELECT和LOAD都在同一预取stream，彼此顺序执行，它们分别与主stream的模型计算重叠。只有实际时间区间的交集才扣除。

七个真实模型配置从已保存trace重新归类，校验raw各段、gap和profile wall未改变；脚本是bench/scripts/reattribute_breakdown.py。InfiniGen旧结果没有保存区间，按相同96步和24步事件设置补跑两个预算，并将逐步区间保存为infinigen_microbench.phases.json。旧JSON保存在各结果目录before_select_priority/。
