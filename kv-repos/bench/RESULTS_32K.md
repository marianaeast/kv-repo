# 32K 上下文下 KV 稀疏方法的 decode 逐 step 延迟对比

**日期**：2026-09-18
**机器**：yq7（2× NVIDIA A100 80GB PCIe，driver 580.95.05），实验跑在 GPU 1
**模型**：Meta-Llama-3.1-8B-Instruct（本地权重 `/data/lrc/hub/...`）
**状态**：A 层实验完成（4 个配置）；ShadowKV / InfiniGen 未纳入本轮

---

## 1. 实验口径

| 项 | 设定 |
|---|---|
| 上下文长度 | **32768 token**（prefill，dense） |
| 生成长度 | 128 token（decode 测 256 步：3 iteration，warmup 1） |
| 稀疏预算 | **512 token** |
| page size | 16（仅 Quest 使用） |
| 聚类数 nlist | 200（仅 ClusterKV 使用） |
| sink / window | 16 / 320（ClusterKV 算法内置） |
| batch size | 1 |
| dtype | float16 |
| 计时方式 | 逐 step `torch.cuda.Event` + `elapsed_time`，每步 `synchronize` |
| 统计口径 | 丢弃第 1 个 iteration（warmup），对剩余 256 步取统计量 |

### 输入构造（可复现）

- 语料：**PG19 test split 第一本**《Reminiscences of Pioneer Days in St. Paul》(book id 10146)
- 获取方式：**不执行 HF 数据集的 `trust_remote_code`**，直接从
  `https://storage.googleapis.com/deepmind-gutenberg/test/10146.txt` 取原始文本
  （加载脚本本身只是下载器，数据实体在 GCS）
- 全书 253872 字符 / 57158 token；用 Llama-3.1 tokenizer 编码后**取前 32768 个 token**
- 固化产物：`/home/zrd/bypasskv_repo/kv-repos/bench/data/ctx32k_pg19.pt`
- **四个配置读同一份 `input_ids`**，避免 tokenizer 差异引入偏差

---

## 2. 主结果

| 方法 | prefill (ms) | **decode 均值 (ms)** | median | p90 | p99 | std | tok/s | vs dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full`（dense 基线） | 5540.0 | **20.28** | 20.25 | 20.35 | 21.95 | 0.37 | 49.3 | 1.000× |
| `clusterkv` (512) | 6678.3 | **19.36** | 19.28 | 19.41 | 22.47 | 0.55 | 51.7 | **1.048×** |
| `quest` (512) | 5604.1 | **26.20** | 25.29 | 31.57 | 32.22 | 2.23 | 38.2 | 0.774× |
| `clusterkv --offload` (512) | 6233.8 | **33.37** | 33.18 | 33.93 | 37.49 | 0.94 | 30.0 | 0.608× |

峰值显存（PyTorch 统计）：full 46.0G / quest 46.2G / clusterkv 46.0G / offload 42.4G。

### 一句话结论

**在 32K + 512 预算下，只有 ClusterKV 比 dense 略快（4.8%）；Quest 反而慢 22.6%，
KV 卸载到 CPU 慢 39.2%。** 稀疏方法在这个设定下的收益远小于原论文报告的数字。

---

## 3. 为什么会这样：拆解 dense 的 20.28 ms

这个结果不神秘，把 dense 的单步时间拆开就很清楚：

| 成分 | 估算 | 稀疏能否优化 |
|---|---|---|
| 权重加载 16 GB ÷ 2039 GB/s（A100 HBM 带宽） | **≈ 7.9 ms** | ❌ 固定成本 |
| KV cache 加载 4.3 GB ÷ 2039 GB/s | **≈ 2.1 ms** | ✅ **理论上限** |
| kernel launch / Python 层 32 层循环 / FlashInfer wrapper / QKV GEMM / RMSNorm 等 | ≈ 10.3 ms | ⚠️ 部分 |

- **稀疏方法最多只能省掉那 2.1 ms 的 KV 加载，即 10% 的上限。**
- ClusterKV 实测省下 0.9 ms（20.28 → 19.36），已经接近这个上限的一半。

### 3.1 Quest 为什么比 dense 还慢

**索引粒度差异，索引开销差 10 倍：**

| | 选择粒度 | 每步要评估的候选数 |
|---|---|---|
| Quest | page（16 token） | 32768 ÷ 16 = **2048 个 page** |
| ClusterKV | cluster | **200 个 cluster** |

Quest 每步要对 2048 个 page 计算 criticality（用 query 与 page 的 min/max key 估计）
再做 top-k，这个开销远超它省下的 2 ms KV 加载。ClusterKV 只评估 200 个候选，
索引开销小一个数量级——**这正是 ClusterKV 论文的核心论点，在这里被独立复现了。**

### 3.2 offload 为什么慢 39%

按实现推算的 PCIe 传输量：

```
每层每步 = num_heads(32) × token_budget(512) × head_dim(128) × 2(K,V) × 2 B
         = 8.4 MB
× 30 层（前 2 层不走 offload）  ≈ 252 MB / step
PCIe 4.0 x16 实际带宽 ~20–25 GB/s  →  10–12 ms / step
```

实测增量 13.1 ms（33.37 − 20.28），与估算吻合。

> **一个可优化点**：`clusterkv_controller.py` 里
> `num_kv_heads_ = num_heads if offload else num_kv_heads`，
> 且 `g2c` 形状为 `(num_heads, token_budget-1)`——即 **逐 query head 各自选 top-k**。
> 对 GQA 模型（8 个 KV head）这会把传输量放大 **4 倍**。若改为只传 8 个 kv_head、
> 在 GPU 上做 `repeat_kv`，PCIe 开销可望降到 ~3 ms。

### 3.3 根本原因：GQA 压缩了稀疏化的收益空间

Llama-3.1-8B 是 **GQA（8 个 KV head）**，KV cache 只有 MHA 模型的 1/4
（32K 全量仅 4.3 GB）。而 A100 的 HBM 带宽高达 2039 GB/s，加载它只要 2.1 ms。

稀疏方法的加速比本质上取决于 `KV 加载时间 / 单步总时间`。分母里占大头的
**权重加载 7.9 ms 完全无法通过 KV 稀疏化优化**。文献里那些 2–7× 的加速比，
通常来自 MHA 模型、更长上下文、或更低带宽的硬件组合。

---

## 4. 修改过的上游代码（否则 Llama-3.1 跑不起来）

两个文件各打了同一组补丁，**原文件已备份为 `.orig`**：

- `clusterkv/clusterkv_models/ClusterKVAttention.py`
- `clusterkv/quest_models/QuestAttention.py`

| # | 问题 | 后果 | 处理 |
|---|---|---|---|
| 1 | `_init_rope` 读 `config.rope_scaling["type"]` | transformers ≥4.44 / Llama-3.1 用的是 **`rope_type`** 键 → **KeyError 直接崩** | 改为 `rs.get("type", rs.get("rope_type"))`；对 `llama3/dynamic/yarn/longrope` 回退 `rope_scale=1.0` |
| 2 | `apply_rope_in_place(rope_scale=self.config.rope_scaling)` | **把 dict 传给 float 形参**；且 kernel 默认 `rope_theta=1e4`，而 Llama-3.1 需要 **5e5** | 改为 `rope_scale=self.rope_scale` + `rope_theta=getattr(config,"rope_theta",None) or 1e4` |

第 2 条是潜伏 bug：对 Llama-2（`rope_scaling=None`）恰好蒙对默认值 1.0，所以原作者没暴露。

另有一处**不改代码的用法约束**：dense 基线必须用超大 `token_budget`（官方用 102400）来
关掉稀疏路径——因为 `need_estimate() = kv_seqlen > infer_token_budget`，
传 512 会让 dense 模式误入稀疏分支，而此时 `centroids` 为 None → `get_neigh_c` TypeError。

---

## 5. 数据质量与局限（重要）

1. **共享 GPU 环境**。zjm 的长任务同时占用 GPU 0/1（利用率 87–89%，显存 25.7 GB）。
   - 先做过对照：**同一配置在 GPU 空闲（17.41 ms）与被占用（17.05 ms）时差异 <2%**，
     说明单 batch decode 占不满 SM，共存影响小。
   - 但 **Quest 的 std 达 2.23（median 25.29 / p99 32.22）**，存在间歇性干扰。
     建议 Quest 在空闲环境重跑，或增加 iteration 数。
   - full / clusterkv 的 std 仅 0.37 / 0.55，数据稳定。

2. **显存峰值触及 97%**（79461 / 81920 MiB），未 OOM 属于运气。
   主因是 32K prefill 的 logits（32768 × 128256 × 2 B ≈ 8.4 GB）与激活。
   **加长上下文或增加 iteration 前需注意**。

3. **RoPE 是近似，输出文本不可用于精度评估**。
   Llama-3.1 的 `rope_type=llama3` 需要分段频率调整，而 kernel 只接受单一 `rope_scale`。
   已用 `scale=1.0 + theta=5e5`（theta 正确，缺少分段修正）。
   **延迟测量不受影响**，但四个方法在 128-token greedy 生成下都出现了重复退化：

   | 方法 | 输出样本（前 ~100 字符） |
   |---|---|
   | full | `the third floor of Ingersoll block, and the people of the city were in the street singing the "Honor and the` |
   | quest | `the In the evening of the 4th of July, 1862, the Inhabitants of the town were filled with a sense of patrioti` |
   | clusterkv | `the 12th of the same year, and the 12th of the same year, and the 12th of the same year, and the 12th of the` |
   | offload | `the office of the book of the of the of the of the of the of the of the of the` |

   若要输出质量可信，需先实现完整的 llama3 分段 RoPE。

4. **本轮未覆盖**：
   - **ShadowKV**：走 vLLM runtime（paged attention / CUDA graph / 连续批处理），
     与上面三者的 HF 朴素实现 **不可直接比绝对延迟**，应单独测并只比内部加速比。
   - **InfiniGen**：官方 speedup 实现是 **OPT-only**（`--embed_dim 5120 --n_head 40` = OPT-13B，
     prompt_len ≤ 1920），而 OPT 用绝对位置编码，**无法外推到 32K**。故排除。

5. **单 batch、单请求**。没有 batch 效应；稀疏方法在较大 batch 下的表现可能不同。

---

## 6. 复现命令

```bash
ssh yq7
source /home/zrd/miniconda3/etc/profile.d/conda.sh && conda activate clusterkv-quest
cd /home/zrd/bypasskv_repo/kv-repos/bench
export CUDA_HOME=/usr/local/cuda-12.4 && export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=1

COMMON="--context_len 32768 --decode_len 128 --iteration 3 --warmup_iter 1 --token_budget 512"

python scripts/bench_decode.py --method full      $COMMON --tag full
python scripts/bench_decode.py --method quest     $COMMON --tag quest
python scripts/bench_decode.py --method clusterkv $COMMON --tag clusterkv
python scripts/bench_decode.py --method clusterkv $COMMON --offload --tag clusterkv_offload
```

一键跑全部：`bash scripts/run_ctx32k.sh`
结果 JSON：`results/ctx32k_*.json`

---

## 7. 建议的下一步

1. **保存逐 step 时间序列**（当前只存了统计量），可观察时间随 KV 增长的漂移趋势。
2. **Quest 加 iteration 到 5+**，或在空闲 GPU 上重跑，压掉那 2.23 ms 的 std。
3. **做 MHA 对照实验**（Llama-2-7B-chat，32 个 KV head）验证「GQA 压缩收益」这一解释——
   若 MHA 下稀疏收益显著变大，则该解释成立。
4. **给 Quest 加 CPU offload**，才能与 `clusterkv+offload` 公平对比；
   同时这也是验证「ClusterKV 的连续 cluster 布局 vs Quest 的分散 page」这一论点的关键实验。
5. **优化 offload 的 GQA 展开**（只传 8 个 kv_head），预期把 PCIe 开销从 13 ms 降到 ~3 ms。
6. **补齐 ShadowKV**（vLLM runtime，单独测）与 **InfiniGen**（需自己写 32K 适配）。
