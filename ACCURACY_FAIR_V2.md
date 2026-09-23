# 本地共享预算精度对比 v2

入口：`run_hotpotqa_accuracy_128_fair.sh`。默认新建
`results/hotpotqa_<模型>_k128_fair_v2_<时间>/`，不会覆盖旧结果。新增的
`--strict_total_budget` 开关只用于精度对比；不加开关可复现旧版。

## 预算与选择范围

同一次实验中的所有方法使用相同模型、BF16、HotpotQA 输入顺序、
31500 token 输入上限和最多 32 个输出 token。当前支持
Llama-3.1-8B-Instruct 和 Qwen3-8B。prefill 保持完整注意力；稀疏 decode
覆盖所有层，包括前两层。每个 KV head 的所有 query heads 共用一份选择
结果；各 query head 仍独立计算最终 attention。Qwen3 显式关闭 thinking
模式，避免思考文本占用短答案预算。

| 方法 | 共享选择规则 | 新生成 token | 每 KV head 的 decode 上限 |
|---|---|---|---|
| Naive | 同组 query 的精确 QK 分数取平均，再做 token Top-K | 与全部历史一起选择 | 128 |
| Quest | 同组 query 的 page 估计分数取平均，再选择 page | 原本已参与全历史选择，继续保留 | 128，16 token/page |
| ClusterKV | 同组 query 取平均，按 centroid 分数选择簇并截断 | 按 cosine 最近的原有 centroid 插入索引，不再额外拼接 | 16 sink + 112 selected = 128 |
| ShadowKV | 保留原方法的同组 softmax 分数取最大值，再选择 chunk | 每步更新 chunk landmark，尾部未满 chunk 也竞争预算 | 至多 128，8 token/chunk |

GQA 聚合公式保留方法差异；统一的是选择集合的粒度和总预算，不是把所有 selector 改成同一种算法。ClusterKV 保留初始 centroid，不逐步重新聚类，新 token 在簇内不享有插队优先权。sink 计入总预算。ShadowKV 禁用免费 local/outlier/alignment/generated 池，prompt K 保留原来的低秩重建，生成 K 保留原来的精确存储；两者均须选中才能参与 attention。

Quest/ShadowKV 选到未满的尾块时，有效 token 数可能少于 128。ShadowKV 使用布尔掩码排除尾块 padding，绝不填入免费的真实 token。短历史不足 128 时可使用全部历史；历史超过预算后必须走稀疏选择。

## 指标和限制

- 汇总准确率为生成答案的 HotpotQA F1，不由召回率推断。
- Quest、ClusterKV、ShadowKV 统计召回率；参考集合仍为**各 query head 独立的精确 Top-128**，范围统一为 prompt + 已生成历史。共享集合相对于这个参考的召回率不必达到 100%。
- FullKV 和 Naive 在正式脚本中不统计召回率。
- v2 记录 `budget_policy=shared_kv_all_history_v2`。Cluster 驱动输出名含 `totalv2`；ShadowKV 拒绝接续不同 policy 的旧文件。
- 这是修改预算规则后的受控精度版本，不能作为原版实现的性能测量。ShadowKV 用带掩码的 SDPA 处理不同 KV head 的尾块有效长度；ClusterKV 每步维护索引也有额外成本。

## 验证

以下 GPU 检查已通过，但未启动完整 HotpotQA 精度实验：

- `test_total_budget_cluster.py`：新 token 可选中/可淘汰、共享选择集合、精确 Naive，以及输出对照显式 attention。
- `test_total_budget_forward.py`：小型随机权重 Llama 的实际 BF16/SDPA forward；120 和 151 token prompt、连续 16 次 decode，覆盖跨越 128 预算边界、前两层、动态簇索引。
- `kv-repos/ShadowKV/test/test_total_budget.py`：连续 20 步更新、未满 chunk、padding 掩码、缓存重置，以及真实 Llama head 形状的 BF16/CUDA RoPE 与低秩重建。
- Qwen3 Quest 兼容检查：随机两层 Qwen3 完成 prefill 和连续 decode；当预算
  覆盖全部历史时，原生模型与补丁模型的 prefill/decode logits 最大误差均为 0。
- Qwen3 ShadowKV 投影检查：Qwen3 的无 bias Q/K/V 投影及逐 head Q/K
  RMSNorm 与原生实现逐项对比，最大误差均为 0。

## 运行

确认所选 GPU 空闲后运行（示例使用 GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 bash /home/zrd/bypasskv_repo/run_hotpotqa_accuracy_128_fair.sh 200
```

Qwen3-8B 使用独立环境，不会改变原有 Llama 环境：

```bash
ACCURACY_MODEL=qwen3-8b CUDA_VISIBLE_DEVICES=1 bash /home/zrd/bypasskv_repo/run_hotpotqa_accuracy_128_fair.sh 200
```

Qwen3 的 ClusterKV/Quest 和 ShadowKV 分别使用
`/home/zrd/miniconda3/envs/clusterkv-qwen3` 与
`/home/zrd/miniconda3/envs/ShadowKV-qwen3`；两者均固定 Transformers 4.53.3。

只做数据集冒烟检查时将 `200` 改为 `1`。脚本依次运行 FullKV、Naive、Quest、ClusterKV、ShadowKV，并输出 `accuracy_summary.csv` 和 `recall_summary.csv`。
