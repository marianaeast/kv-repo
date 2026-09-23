# KV Cache 压缩/稀疏推理 — 四仓库环境配置报告

**机器**：yq7（`10.214.241.232:2022`，`zrd@ubuntu`）
**配置时间**：2026-09-18
**落盘路径**：仓库 `/home/zrd/bypasskv_repo/kv-repos/`；环境 `/home/zrd/miniconda3/envs/`

---

## 一、结论先说

**一个环境跑不了全部四个工作，需要拆成 3 个。**

拆分的硬约束是 **torch 版本区间完全不交叠**，且各自锁死：

| 工作 | Python | torch | transformers | 冲突点 |
|---|---|---|---|---|
| ClusterKV | ≥3.10 | **2.5.0** | 4.45.2 | — |
| Quest | 3.10 | **2.5.0** | 4.45.2 | 与 ClusterKV 完全兼容 |
| InfiniGen | 3.9 | **2.0.1** | 4.38.2（实测） | torch 2.0.1 与 2.5.0 互斥 |
| ShadowKV | 3.10 | **2.3.1** | 4.43.1 | torch 2.3.1 与 2.5.0 互斥；numpy 1.23.2 与 2.1.2 互斥 |

**ClusterKV 与 Quest 可以合并**（torch 2.5.0 / transformers 4.45.2 / accelerate 1.0.1 / datasets 3.0.1 / flash-attn 2.6.3 全部一致，且共用同一份 libraft），所以最终是 **3 个环境**而不是 4 个。

---

## 二、环境清单

| 环境名 | 覆盖工作 | Python | torch | 体积 |
|---|---|---|---|---|
| `clusterkv-quest` | ClusterKV + Quest | 3.10.21 | 2.5.0+cu124 | 8.3 G |
| `infinigen` | InfiniGen | 3.9.25 | 2.0.1+cu117 | 5.2 G |
| `ShadowKV` | ShadowKV | 3.10.21 | 2.3.1+cu121 | 9.6 G |

### 环境 1：`clusterkv-quest`

```
conda activate clusterkv-quest
```

- torch 2.5.0+cu124 / transformers 4.45.2 / datasets 3.0.1 / accelerate 1.0.1
- flash-attn **2.6.3**（源码编译，180 MB wheel）
- flashinfer-python 0.2.5 / pylibraft-cu11 24.10.0 / rmm 24.10.0 / einops 0.8.0
- cmake ≥3.26.4 + ninja（conda 装，系统自带 3.22.1 版本不够）
- 已编译 CUDA 扩展：
  - `ClusterKV/clusterkv/_clusterkv_knl.*.so`（22 MB）
  - `ClusterKV/clusterkv/_quest_knl.*.so`（20 MB）
  - `quest/quest/_kernels.*.so`
- `libraft` 只编译了一次（`ClusterKV/3rdparty/raft`，两个仓库 pin 同一 commit `1e4961e`），安装到 `$CONDA_PREFIX`
- ClusterKV / Quest 均已 `pip install -e . --no-deps`

验证：`clusterkv.clusterkv_models` / `clusterkv.quest_models` / `clusterkv.utils` / `clusterkv._clusterkv_knl` / `clusterkv._quest_knl` / `quest._kernels` 全部导入通过。

### 环境 2：`infinigen`

```
conda activate infinigen
```

- torch 2.0.1+cu117 / transformers **4.38.2** / lm-eval 0.3.0 / accelerate / sentencepiece / ftfy
- 纯 Python 依赖，无需编译

**⚠️ 主动偏离了上游**：`requirements.txt` 未锁 transformers，pip 会装到 4.57.6，但新版强制要求 torch ≥ 2.1，直接把 PyTorch 后端禁用了。InfiniGen 代码又依赖 `LlamaRotaryEmbedding` / `LlamaAttention` / `apply_rotary_pos_emb` / `GPTNeoXRotaryEmbedding` 这些已被重构掉的内部 API，故锁定 **4.38.2**。五个关键导入全部验证通过。

### 环境 3：`ShadowKV`

```
conda activate ShadowKV
```

- torch 2.3.1+cu121 / transformers 4.43.1 / numpy 1.23.2 / vllm 0.5.3.post1
- flashinfer **0.1.6+cu121torch2.3**（官方 cu121/torch2.3 索引）
- flash-attn 2.6.3（预编译 wheel `cu123torch2.3cxx11abiFALSE`，32 秒装完）
- minference 0.1.6.0（源码编译）
- youtokentome / Cython / cutlass（`3rdparty/cutlass`，depth=1）
- 已编译：`ShadowKV/kernels/shadowkv.*.so`（3.2 MB）

验证：`kernels.shadowkv` / `models.tensor_op` / `models.base` / `models.llama` 全部导入通过。

---

## 三、踩到的坑与处理方式（重要）

### 1. minference 对 transformers ≥4.44 不兼容 → **已打补丁**

`minference` 装是装上了，但 import 就炸：

- `minference/modules/kvcompression.py:9` → `from transformers.models.glm.modeling_glm import apply_rotary_pos_emb`
- `minference/utils.py:12` → `from transformers.models.glm.modeling_glm import GlmMLP, GlmRotaryEmbedding`

`transformers.models.glm` 在新版已被移除，而 ShadowKV 锁死 4.43.1，撞车。这两个符号只在 **ChatGLM 专用分支**里用到，ShadowKV 不走。

处理：把两处 import 改成 `try/except ImportError` 并回落到 `None`。原文件已备份为 `*.bak`。

> `minference` 是 `models/tensor_op.py:28` 的**模块级 import**，不装的话 ShadowKV 模型根本 import 不进来，所以这一步不能省。

### 2. 构建隔离导致编译失败 → `--no-build-isolation`

- `minference` 编译期需要 torch → 必须 `--no-build-isolation`
- `youtokentome` 编译期需要 Cython → 先装 Cython 再 `--no-build-isolation`
- `flash-attn` 同理

### 3. 清华 PyPI 镜像在本机不可达

`pypi.tuna.tsinghua.edu.cn` 直连和走代理都是 `000`，**用不了**。pypi.org（2s）和 download.pytorch.org 都正常，全程走官方源。

### 4. 代理不稳定

本机 shell 默认挂 `7890` 代理。clone 时出现过 `gnutls_handshake() failed: The TLS connection was non-properly terminated`。子模块用 `--shallow-submodules` 后稳定通过。

### 5. CUDA 工具链

机器有 `12.4 / 12.9 / 13.0`（默认 `cuda` → 13.0），**没有 12.1**。

- 所有编译统一用 `CUDA_HOME=/usr/local/cuda-12.4`
- 已实测 torch 的 CUDA 校验只比 major 版本，12.4 对 12.1 可通过
- **以后要重编译扩展，必须先 `export CUDA_HOME=/usr/local/cuda-12.4`**

### 6. ClusterKV 的 `kernel/setup.sh` 是 uv 写法的

脚本里用 `$VIRTUAL_ENV`，conda 环境下为空。手动编译时需：

```bash
export VIRTUAL_ENV=$CONDA_PREFIX
```

---

## 四、复现命令

```bash
# ClusterKV — LongBench 精度评测（zjm 正在跑的就是这个）
conda activate clusterkv-quest
cd /home/zrd/bypasskv_repo/kv-repos/ClusterKV/accuracy/LongBench
CUDA_VISIBLE_DEVICES=0 python pred.py --model llama3.1-8b-chat-8k --task hotpotqa --cluster
# --quest 切换 Quest 基线；不带参数即 full KV

# ClusterKV — 效率测试
cd /home/zrd/bypasskv_repo/kv-repos/ClusterKV/efficiency
CUDA_VISIBLE_DEVICES=0 python bench_textgen.py --model llama3-8b --iteration 5 --warmup 3 --method clusterkv

# Quest
conda activate clusterkv-quest
cd /home/zrd/bypasskv_repo/kv-repos/quest
bash scripts/longbench.sh          # 需先按脚本内容改模型路径

# InfiniGen
conda activate infinigen
cd /home/zrd/bypasskv_repo/kv-repos/infinigen
# 精度评测在 accuracy/，加速评测在 speedup/

# ShadowKV
conda activate ShadowKV
cd /home/zrd/bypasskv_repo/kv-repos/ShadowKV
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  test/eval_acc.py --datalen 131072 --method shadowkv --model_name ...
```

---

## 五、资源与风险

### 磁盘（需要关注）

```
/dev/nvme0n1p2   1.5T  1.3T  122G  92%   /
```

配置前剩 159 G，现在剩 **122 G**。三个新环境占 23 G，pip 缓存另占数 G。**建议清理 `~/.cache/pip`**（可回收约 6 G+），并留意 `/data1`（99%）和 `/data`（94%）。

### GPU

配置完成时（11:47）状态：

- **GPU 0**：空闲（0%）
- **GPU 1**：被用户 **`zjm`** 占用，25898 MiB / 81920 MiB，利用率 62%
  - `python pred.py --model llama3.1-8b-chat-8k --task gov_report --cluster --token_budget 1024`

`zjm` 从上午 10:00 起持续在 yq7 上跑 **ClusterKV 的 LongBench 评测**（先后跑了 multifieldqa_en、triviaqa、gov_report），已运行 1 小时 18 分。**不是自己的进程，不要动。**

---

## 六、遗留事项

1. `minference` 的补丁写在 site-packages 里，**若重装 minference 会丢失**，需重新打。补丁内容见第三节第 1 条。
2. `infinigen` 环境的 transformers 是实测选定（4.38.2），上游未锁版本，属于主动决策。
3. ShadowKV 的 `nemo_toolkit[all]==1.23` **尚未安装**——它只用于构建 RULER 数据集，跑评测本身不需要。需要时再装（体积较大，且可能引入依赖冲突）。
4. 磁盘 92%，继续装大依赖前建议先清理。
