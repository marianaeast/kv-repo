# Quest 的 KV offload 支持
#
# 设计（三段数据放置）：
#   CPU (pinned)  完整 KV 主副本          [L, cap, 2, page, kvH, D]
#   GPU staging   本步 attention 实际读的页（紧凑布局，gather 目标）
#
# decode 每步：
#   1) estimate / topk 照旧（metadata 常驻 GPU，这是 Quest 的关键假设）
#   2) 把 top-k 选中的页从 CPU 主副本 gather 到 staging —— 这是 LOAD 开销
#   3) attention 在 staging 上做，per-head 索引变成 staging 内的槽位
#
# 为什么可以这样：
#   Quest 的 sparse decode 接受 paged_kv_indices = [num_qo_heads, K]，
#   即每个 query head 有自己的 K 个页。只要把这些页物化成一段紧凑
#   连续内存，再给 kernel 一份"槽位映射"，语义就完全等价。
#
# 性能要点（踩过的坑）：
#   * index_select 作用在 pinned tensor 上返回值 **不是** pinned，
#     会导致 H2D 退化成 pageable copy（实测慢 3~5 倍）。
#   * 更严重的是：在 CPU 上做页 gather，实测有效带宽只有 ~7 GB/s
#     （100 页 x 32 层 = 29.7 ms/step），而且每层都要同步 D2H 拿页号，
#     CPU 会在关键路径上等 GPU。正解是 UVA zero-copy —— 让 GPU kernel
#     直接按索引读 pinned 主副本（见 quest_offload_ext.py），实测
#     45.6 GB/s，接近 PCIe Gen5 x16 上限。
#   * 页号去重 torch.unique 会拉起一串 cub kernel（radix sort /
#     select / scan），且因输出长度数据相关而引入一次同步。
import torch

# 前若干层不做稀疏（full attention，扫全部页），因此也不 offload。
# 必须与 InferenceController.SKIP_LAYERS 保持一致。
SKIP_LAYERS = 2


class OffloadKvStore:
    def __init__(self, num_layers, num_kv_heads, head_dim, max_seq_len,
                 page_size, dtype, device, max_sel_pages, ext=None):
        cap = (max_seq_len + page_size - 1) // page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.cap = cap
        self.device = device
        self.dtype = dtype
        self.max_sel_pages = max_sel_pages
        self.ext = ext

        page_shape = (2, page_size, num_kv_heads, head_dim)

        # CPU 主副本：pinned，UVA 下 GPU kernel 可直接寻址
        self.cpu = torch.zeros((num_layers, cap) + page_shape,
                               dtype=dtype, pin_memory=True)

        # GPU staging：最多 max_sel_pages 个被选中的页 + 1 个当前页
        self.stg = torch.zeros((num_layers, max_sel_pages + 1) + page_shape,
                               dtype=dtype, device=device)

        self._synced_pages = 0
        self.load_bytes = 0        # 本步 H2D 字节数（供 breakdown 用）
        self.n_unique = 0
        self.cur_slot = 0
        self.unique_hist = []      # 每层每步的唯一页数采样
        self.use_gpu_gather = ext is not None

    # ---------------------------------------------------------------- 主副本同步
    def sync_from_gpu(self, ctrl):
        """把 GPU kv_cache 里"刚写满的新页"同步到 CPU 主副本。

        只搬新页；正在写的那一页每步都在变，由 gather 直接从 GPU 取，
        不进 D2H 路径（否则每步都要为 1 个 token 搬 64 KB）。

        注意页表是全局的（kv_cache._indicies 所有层共享），所以用单个
        计数器是自洽的；但 prefill 重来（clear 后页表从头增长）时必须把
        计数器归零，否则会漏同步，且首步会撞出一次巨额全量 D2H。
        """
        idxs = ctrl.kv_cache.indicies
        # 当前页通过GPU D2D单独提供。它尚在增长，不能提前标成CPU已同步。
        # 只同步之前的完整页；页池clear后物理页号不一定从0递增。
        n = max(0, len(idxs) - 1)
        if n == 0:
            self._synced_pages = 0
            return
        if n < self._synced_pages:      # 页表被 clear 过（新一轮 prefill）
            self._synced_pages = 0
        if self._synced_pages == 0:
            todo = list(idxs[:n])
        else:
            todo = list(idxs[self._synced_pages:n])
        if todo:
            src = torch.tensor(todo, dtype=torch.long, device=self.device)
            dst = torch.tensor(todo, dtype=torch.long)
            # 只同步被 offload 的层（前 SKIP 层常驻 GPU，不进 CPU 主副本）
            for l in range(SKIP_LAYERS, self.num_layers):
                buf = ctrl.kv_cache.buf_layer(l)
                tmp = buf.index_select(0, src).to("cpu", non_blocking=False)
                self.cpu[l].index_copy_(0, dst, tmp)
            self._synced_pages = n

    # ---------------------------------------------------------------- gather
    def recall(self, layer_idx, page_ids, ctrl):
        """把 page_ids 指定的页从 CPU 主副本搬进 GPU staging。

        page_ids: [num_qo_heads, K] int32，绝对页号（不含当前页）
        返回:
          slot     [num_qo_heads, K] int32 —— 每页在 staging 里的槽位
          n_unique 唯一页数
        """
        H, K = page_ids.shape
        flat = page_ids.reshape(-1).long()
        uniq, inv = torch.unique(flat, return_inverse=True)
        n = int(uniq.numel())
        if n + 1 > self.stg.size(1):
            raise RuntimeError(
                f"[offload] staging overflow: n_unique={n} > stg={self.stg.size(1)}; "
                f"page_ids.shape={tuple(page_ids.shape)} page_ids.max={int(page_ids.max())}; "
                f"ctrl.inference_page_budget={getattr(ctrl, 'inference_page_budget', None)} "
                f"ctrl._page_budget={getattr(ctrl, '_page_budget', None)} "
                f"need_estimate={ctrl.need_estimate()} max_sel_pages={self.max_sel_pages}")

        stg = self.stg[layer_idx]
        if self.use_gpu_gather:
            # <-- LOAD：GPU kernel 直接读 pinned 主副本（UVA zero-copy），
            #     没有 CPU 拷贝、没有逐层同步
            self.ext.gather_pages(stg, self.cpu[layer_idx], uniq)
        else:
            idx = uniq.to("cpu", non_blocking=False)
            buf = torch.index_select(self.cpu[layer_idx], 0, idx)
            stg[:n].copy_(buf, non_blocking=True)

        # 当前页（top-k 已排除，但 attention 必须看得到它）：GPU 内 D2D
        last_abs = ctrl.kv_cache.indicies[-1]
        stg[n].copy_(ctrl.kv_cache.buf_layer(layer_idx)[last_abs], non_blocking=True)

        self.load_bytes = n * self.page_size * 2 * self.num_kv_heads * self.head_dim * 2
        self.n_unique = n
        self.cur_slot = n
        self.unique_hist.append(n)
        return inv.reshape(H, K).to(torch.int32), n

    # ---------------------------------------------------------------- 内存视图
    def staging_view(self, layer_idx, n_unique):
        """staging 前 n_unique+1 页的视图，交给 attention kernel。"""
        return self.stg[layer_idx][: n_unique + 1]

    def stats(self):
        mb = lambda t: t.numel() * t.element_size() / 1e6
        return {"cpu_master_MB": mb(self.cpu),
                "gpu_staging_MB": mb(self.stg),
                "num_layers": self.num_layers,
                "cap_pages": self.cap,
                "gather_impl": "gpu_uva_zero_copy" if self.use_gpu_gather else "cpu_index_select"}
