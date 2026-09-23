"""Quest offload 用的 GPU 侧页 gather。

为什么要自己写这一个 kernel：
  CPU 侧 torch.index_select 从 pinned 主副本里取页，实测有效带宽只有
  ~7 GB/s（100 页 x 32 层 = 29.7 ms/step），而且每层都要一次同步 D2H
  拿页号，CPU 会在关键路径上等 GPU。ClusterKV 的 offload 之所以没这个
  问题，是因为它的 recall 是 CUDA kernel 直接读 pinned memory。

这里用同样的办法：UVA 下 pinned memory 的 host pointer 就是 device
pointer，所以 GPU kernel 可以按索引直接读 CPU 主副本，把选中的页搬到
GPU staging —— 没有 CPU 拷贝、没有逐层同步。

编译与使用：
    from quest_offload_ext import load_ext
    ext = load_ext()
    ext.gather_pages(dst_gpu, src_pinned_cpu, idx_gpu_int64)
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 每个 block 搬一页；src 是 pinned host memory（UVA 下 GPU 可直接寻址）
template <typename T>
__global__ void gather_pages_kernel(T* __restrict__ dst,
                                    const T* __restrict__ src,
                                    const int64_t* __restrict__ idx,
                                    int64_t page_elems) {
    const int64_t i = blockIdx.x;
    const int64_t src_page = idx[i];
    const T* __restrict__ s = src + src_page * page_elems;
    T* __restrict__ d = dst + i * page_elems;
    for (int64_t j = threadIdx.x; j < page_elems; j += blockDim.x) {
        d[j] = s[j];
    }
}

void gather_pages(torch::Tensor dst, torch::Tensor src, torch::Tensor idx) {
    TORCH_CHECK(src.device().is_cpu(), "src must be a CPU (pinned) tensor");
    TORCH_CHECK(src.is_pinned(), "src must be pinned memory for UVA access");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(idx.is_cuda(), "idx must be a CUDA tensor");
    TORCH_CHECK(idx.scalar_type() == at::kLong, "idx must be int64");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && idx.is_contiguous(),
                "all tensors must be contiguous");

    const int64_t n = idx.numel();
    if (n == 0) return;
    TORCH_CHECK(dst.size(0) >= n, "dst has fewer pages than requested");
    const int64_t page_elems = src.numel() / src.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int TPB = 256;
    if (src.scalar_type() == at::kHalf) {
        gather_pages_kernel<__half><<<(unsigned)n, TPB, 0, stream>>>(
            reinterpret_cast<__half*>(dst.data_ptr()),
            reinterpret_cast<const __half*>(src.data_ptr()),
            idx.data_ptr<int64_t>(), page_elems);
    } else if (src.scalar_type() == at::kBFloat16) {
        gather_pages_kernel<__nv_bfloat16><<<(unsigned)n, TPB, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
            idx.data_ptr<int64_t>(), page_elems);
    } else {
        TORCH_CHECK(false, "only fp16/bf16 supported");
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gather_pages failed: ", cudaGetErrorString(err));
}
"""

_CPP_SRC = "void gather_pages(torch::Tensor dst, torch::Tensor src, torch::Tensor idx);"

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    _ext = load_inline(
        name="quest_offload_ext",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["gather_pages"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )
    return _ext


if __name__ == "__main__":
    # 自测 + 与 CPU gather 对照
    import time
    torch.cuda.init()
    ext = load_ext(verbose=True)

    L, cap = 32, 2048
    page = (2, 16, 8, 128)
    src = torch.zeros((L, cap) + page, dtype=torch.float16, pin_memory=True)
    dst = torch.zeros((L, 1024) + page, dtype=torch.float16, device="cuda")
    idx = torch.arange(100, dtype=torch.long, device="cuda")

    # 正确性
    ref = torch.index_select(src[0], 0, idx.cpu())
    ext.gather_pages(dst[0][:100], src[0], idx)
    torch.cuda.synchronize()
    ok = torch.equal(ref, dst[0][:100].cpu())
    print("correctness:", "OK" if ok else "MISMATCH")

    # 吞吐
    for n in (32, 100, 300):
        i = torch.arange(n, dtype=torch.long, device="cuda")
        for _ in range(3):
            for l in range(L):
                ext.gather_pages(dst[l][:n], src[l], i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            for l in range(L):
                ext.gather_pages(dst[l][:n], src[l], i)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        mb = n * 16 * 8 * 128 * 2 * 2 * L / 1e6
        print(f"n={n:4d} pages x {L} layers : {dt*1000:7.3f} ms/step  "
              f"({mb:7.1f} MB, {mb/dt/1e3:6.2f} GB/s)")
