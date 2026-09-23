#!/bin/bash
# 在 yq9 (H800, sm_90) 上重编 ShadowKV CUDA 扩展
# 原因：从 yq7 rsync 来的 .so 依赖 GLIBC_2.32（yq9 只有 2.31），且只有 sm_80 cubin
set -uo pipefail

ENVP=/home/zrd/miniconda3/envs/ShadowKV
REPO=/home/zrd/bypasskv_repo/kv-repos/ShadowKV

unset CONDA_PREFIX VIRTUAL_ENV
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$ENVP/bin:/home/zrd/miniconda3/bin:$CUDA_HOME/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=${MAX_JOBS:-16}
export NVCC_PREPEND_FLAGS=""

echo "===== ENV ====="
echo "python : $(which python)"
python -V
nvcc --version | tail -2 | head -1
gcc --version | head -1
python - <<PY
import torch; print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

cd $REPO
echo "===== clean ====="
rm -rf build
echo "===== build ====="
python setup.py build_ext --inplace 2>&1 | tail -60
RC=\${PIPESTATUS[0]}
echo "===== build rc=\$RC ====="

if [ -f $REPO/kernels/shadowkv.cpython-310-x86_64-linux-gnu.so ]; then
  echo "===== arch check ====="
  cuobjdump --list-elf $REPO/kernels/shadowkv.cpython-310-x86_64-linux-gnu.so | head
fi
echo "===== DONE rc=\$RC ====="
