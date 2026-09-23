#!/bin/bash
# 在 yq9 (NVIDIA H800 NVL / sm_90) 上编译 ClusterKV + Quest 的 CUDA 扩展
#
# 与 yq7 (A100 / sm_80) 成功配置的关键差异：
#   1) 架构号 90（上游 kernel/CMakeLists.txt 硬编码 89，已改为 90）
#   2) CUDA_HOME 指向 12.8（yq9 没有 12.4）
#   3) rmm / raft / fmt / spdlog 复用从 yq7 $CONDA_PREFIX 同步来的本地包，
#      避免 CPM 从 GitHub 下载（代理不稳）以及重新编译 raft（耗时 30min+）
set -euo pipefail

MC=/home/zrd/miniconda3
ENV=$MC/envs/clusterkv-quest
REPO=/home/zrd/bypasskv_repo/kv-repos/ClusterKV

export CONDA_PREFIX=$ENV
export VIRTUAL_ENV=$ENV
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENV/bin:$MC/bin:$CUDA_HOME/bin:$PATH   # $ENV/bin 必须在前，否则 python 会落到 base
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=${MAX_JOBS:-24}

echo "===== 环境自检 ====="
echo "cmake : $(cmake --version | head -1)"
echo "ninja : $(which ninja)"
echo "nvcc  : $(nvcc --version | tail -2 | head -1)"
echo "python: $(which python) - $(python --version 2>&1)"
echo "fmt   : $(grep -m1 'define FMT_VERSION' $ENV/include/fmt/core.h 2>/dev/null || echo 'NOT FOUND')"
python - <<'PY'
import torch
print("torch :", torch.__version__, "| built for cuda:", torch.version.cuda)
print("torch arch list:", torch.cuda.get_arch_list())
PY

echo
echo "===== 清理旧构建（保留 _deps 中已下载的依赖源码）====="
if [ -d "$REPO/kernel/build/_deps" ]; then
    find "$REPO/kernel/build" -maxdepth 1 -mindepth 1 ! -name '_deps' -exec rm -rf {} +
    find "$REPO/kernel/build/_deps" -maxdepth 1 -name '*-build' -exec rm -rf {} +
else
    rm -rf "$REPO/kernel/build"
fi
mkdir -p "$REPO/kernel/build"
rm -rf "$REPO/3rdparty/raft/cpp/build"

echo
echo "===== cmake 配置 ====="
cd "$REPO/kernel/build"
cmake -GNinja \
  -DCMAKE_PREFIX_PATH="$VIRTUAL_ENV;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCPM_raft_SOURCE="$REPO/3rdparty/raft" \
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  ..
#  注 1：CPM_raft_SOURCE 指向本地 3rdparty/raft 源码，避免从 GitHub 下载（代理不稳）。
#        同时 kernel/cmake/get_raft.cmake 的 COMPILE_LIBRARY 已由 ON 改为 OFF
#        （原文件备份为 get_raft.cmake.orig）：raft 24.02 的
#        src/neighbors/ivfpq_search_uint8_t_int64_t.cu 在 sm_90 下报 7 个编译错误，
#        而 ClusterKV 只链接 header-only 的 raft::raft，并不需要 raft 的编译库。
#  注 2：FETCHCONTENT_UPDATES_DISCONNECTED=ON 禁止对已下载的 _deps 源码做网络更新。
#  注 3：不要加 -DCMAKE_DISABLE_FIND_PACKAGE_fmt=TRUE。必须让 $CONDA_PREFIX 的
#        fmt 10.1.1 压过 conda base 的 fmt 11.2.0 —— fmt 11 删除了
#        fmt::basic_format_string，而 spdlog 1.12 依赖它（否则 bsk_ops.cu 报 12 errors）。

echo
echo "===== 编译 (MAX_JOBS=$MAX_JOBS) ====="
ninja

echo
echo "===== 链接 .so 到 clusterkv/ ====="
cd "$REPO/kernel"
for f in $(find ./build -maxdepth 1 -name "*.so"); do
    abs=$(realpath "$f")
    base=$(basename "$f")
    rm -f "$REPO/clusterkv/$base"
    ln -s "$abs" "$REPO/clusterkv/"
    echo "linked: $base"
done

echo
echo "===== 验证导入 ====="
cd /home/zrd
python - <<'PY'
import clusterkv._clusterkv_knl as a
import clusterkv._quest_knl as b
print("import OK:", a.__file__)
print("import OK:", b.__file__)
PY
echo "===== BUILD DONE ====="
