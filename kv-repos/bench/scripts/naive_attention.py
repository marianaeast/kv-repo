"""Naive: full post-RoPE K and V in pinned CPU memory.

Every step first loads all keys onto GPU, then each query head scans them,
selects Top-K tokens, fetches their V values,
then computes softmax(scores_selected) @ V_selected. No prefetch or V reuse.
The surrounding Llama layers are shared with the other real-model benchmarks.
"""
import math
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


@triton.jit
def naive_scan_keys(Q, K, Scores, N: tl.constexpr, CAP: tl.constexpr,
                    GROUPS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    tokens = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    q = tl.load(Q + head*D + dims).to(tl.float32)
    k = tl.load(K + (head//GROUPS)*CAP*D + tokens[:, None]*D + dims[None, :],
                mask=tokens[:, None] < N, other=0).to(tl.float32)
    scores = tl.sum(k*q[None, :], axis=1) * (D ** -0.5)
    tl.store(Scores + head*N + tokens, scores, mask=tokens < N)


CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
// Full K transfer is part of Identification, before QK and Top-K.
// Copy 16 bytes per thread; head strides include the unused cache capacity.
__global__ void naive_identify_load_keys_kernel(uint4* dst, const uint4* src,
    int64_t stride, int64_t count) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    int head = blockIdx.y;
    if (i < count) dst[head*stride+i] = src[head*stride+i];
}
void load_keys(torch::Tensor dst, torch::Tensor src, int64_t length) {
    TORCH_CHECK(src.device().is_cpu() && src.is_pinned() && dst.is_cuda(), "CPU pinned K and GPU staging required");
    TORCH_CHECK(src.dim()==3 && src.sizes()==dst.sizes(), "matching [heads,capacity,dim] required");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "contiguous keys required");
    TORCH_CHECK(src.scalar_type()==dst.scalar_type() && src.element_size()==2, "matching 16-bit keys required");
    TORCH_CHECK(src.size(2)%8==0 && length>0 && length<=src.size(1), "invalid length/dim");
    int64_t stride=src.size(1)*src.size(2)/8, count=length*src.size(2)/8;
    naive_identify_load_keys_kernel<<<dim3((count+255)/256,src.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(
        (uint4*)dst.data_ptr(),(uint4*)src.data_ptr(),stride,count);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"full K load failed");
}
__global__ void naive_load_values_kernel(uint16_t* dst, const uint16_t* src,
    const int64_t* idx, int qh, int kh, int cap, int budget, int dim) {
    int token = blockIdx.x, h = blockIdx.y;
    int64_t row = idx[h*budget+token];
    for (int d=threadIdx.x; d<dim; d+=blockDim.x)
        dst[(int64_t(h)*budget+token)*dim+d] = src[(int64_t(h/(qh/kh))*cap+row)*dim+d];
}
void load_values(torch::Tensor dst, torch::Tensor src, torch::Tensor idx) {
    TORCH_CHECK(src.device().is_cpu() && src.is_pinned(), "pinned CPU V required");
    TORCH_CHECK(dst.is_cuda() && idx.is_cuda(), "CUDA destination/indices required");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && idx.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(src.dim()==3 && dst.dim()==3 && idx.dim()==2, "invalid dimensions");
    TORCH_CHECK(src.scalar_type()==dst.scalar_type() && src.element_size()==2, "matching 16-bit types required");
    TORCH_CHECK(idx.scalar_type()==at::kLong, "int64 indices required");
    int qh=dst.size(0), kh=src.size(0), budget=dst.size(1), dim=dst.size(2);
    TORCH_CHECK(qh%kh==0 && src.size(2)==dim && idx.size(0)==qh && idx.size(1)==budget, "invalid shapes");
    naive_load_values_kernel<<<dim3(budget,qh),128,0,at::cuda::getCurrentCUDAStream()>>>(
        (uint16_t*)dst.data_ptr(), (uint16_t*)src.data_ptr(), idx.data_ptr<int64_t>(),
        qh,kh,src.size(1),budget,dim);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess, "V gather launch failed");
}
'''


def load_extension():
    return load_inline(name='naive_cpu_kv_v2',
        cpp_sources='void load_values(torch::Tensor, torch::Tensor, torch::Tensor); void load_keys(torch::Tensor, torch::Tensor, int64_t);',
        cuda_sources=CUDA, functions=['load_values', 'load_keys'], extra_cuda_cflags=['-O3'])


def select(q, keys, length, budget):
    heads, dim = q.shape
    scores = torch.empty(heads, length, device=q.device, dtype=torch.float32)
    naive_scan_keys[(triton.cdiv(length, 64), heads)](
        q, keys, scores, length, keys.shape[1], heads//keys.shape[0], dim, 64)
    return scores.topk(min(budget, length), dim=-1, sorted=False)


@torch.inference_mode()
def self_test(ext, device):
    # Unequal head counts, padded capacity, independent head selections and boundary tokens.
    torch.manual_seed(20260919)
    heads, kvheads, n, capacity, dim = 8, 2, 257, 300, 128
    q = torch.randn(heads, dim, device=device, dtype=torch.float16)
    keys = torch.randn(kvheads, capacity, dim, device=device, dtype=torch.float16)
    host_keys = keys.cpu().pin_memory()
    loaded_keys = torch.full_like(keys, -7)
    ext.load_keys(loaded_keys, host_keys, n)
    torch.cuda.synchronize()
    torch.testing.assert_close(loaded_keys[:,:n], keys[:,:n], rtol=0, atol=0)
    assert bool((loaded_keys[:,n:]==-7).all()), 'K copy overwrote unused capacity'
    # Changing CPU K must be reflected in the next load, with no persistent GPU reuse.
    host_keys[:,0].add_(1)
    ext.load_keys(loaded_keys, host_keys, n)
    torch.cuda.synchronize()
    torch.testing.assert_close(loaded_keys[:,:n].cpu(), host_keys[:,:n], rtol=0, atol=0)
    keys = loaded_keys
    values = torch.randn(kvheads, capacity, dim, dtype=torch.float16).pin_memory()
    ref_scores = torch.stack([(keys[h//4,:n].float()*q[h].float()).sum(-1)/math.sqrt(dim) for h in range(heads)])
    for budget in (17, n):
        scores, ids = select(q, keys, n, budget)
        torch.testing.assert_close(scores, ref_scores.gather(1, ids), rtol=1e-4, atol=1e-5)
        expected_ids = ref_scores.topk(budget, dim=-1).indices.sort(dim=-1).values
        torch.testing.assert_close(ids.sort(dim=-1).values, expected_ids, rtol=0, atol=0)
        output = torch.empty(heads, budget, dim, device=device, dtype=torch.float16)
        ext.load_values(output, values, ids)
        torch.cuda.synchronize()
        expected = torch.stack([values[h//4,ids[h].cpu()] for h in range(heads)])
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        actual = torch.bmm(scores.softmax(-1).half().unsqueeze(1), output).squeeze(1)
        reference = (scores.softmax(-1).cpu().unsqueeze(-1)*expected.float()).sum(1)
        torch.testing.assert_close(actual.cpu().float(), reference, rtol=0.005, atol=0.001)
    ids = torch.tensor([[0,n-1,0]]*heads, device=device)
    output = torch.empty(heads, 3, dim, device=device, dtype=torch.float16)
    ext.load_values(output, values, ids)
    torch.cuda.synchronize()
    expected = torch.stack([values[h//4,ids[h].cpu()] for h in range(heads)])
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print('Naive CPU K load/reload, full-key scores, Top-K sets, V gather, weighted output: PASS', flush=True)


class NaiveController:
    """Only track sequence length; do not allocate the old full GPU KV cache."""
    def __init__(self, max_length):
        self.kv_seqlen = 0
        self.max_length = max_length

    def prepare_metadata(self, length):
        self.kv_seqlen += length
        if self.kv_seqlen > self.max_length:
            raise ValueError('Naive cache capacity exceeded')

    def clean_states(self):
        self.kv_seqlen = 0

    def set_token_budget(self, budget):
        pass  # Llama's existing model loop calls these controller hooks.

    def begin_forward(self, length, updateTensor=True):
        pass

    def end_forward(self):
        pass


class NaiveAttention(torch.nn.Module):
    def __init__(self, original, capacity, budget, ext):
        super().__init__()
        self.q_proj, self.k_proj = original.q_proj, original.k_proj
        self.v_proj, self.o_proj = original.v_proj, original.o_proj
        self.heads, self.kvheads = original.num_heads, original.num_key_value_heads
        self.dim, self.hidden = original.head_dim, original.hidden_size
        self.budget, self.ext = budget, ext
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        self.keys = torch.empty(self.kvheads, capacity, self.dim, dtype=dtype, pin_memory=True)
        self.values = torch.empty(self.kvheads, capacity, self.dim, dtype=dtype, pin_memory=True)
        self.staging = torch.empty(self.heads, budget, self.dim, device=device, dtype=dtype)

    def forward(self, hidden_states, controller, **kwargs):
        import clusterkv.utils
        from flashinfer import single_prefill_with_kv_cache
        batch, length, _ = hidden_states.shape
        if batch != 1:
            raise ValueError('Naive benchmark supports batch=1')
        end = controller.kv_seqlen
        start = end-length
        q = self.q_proj(hidden_states).view(length, self.heads, self.dim)
        k = self.k_proj(hidden_states).view(length, self.kvheads, self.dim)
        v = self.v_proj(hidden_states).view(length, self.kvheads, self.dim)
        clusterkv.utils.apply_rope_in_place(q, k, start)
        self.keys[:,start:end].copy_(k.transpose(0,1), non_blocking=False)
        # Synchronous D2H makes current K/V visible to subsequent CPU-to-GPU loads.
        # Prefill copy is outside decode timing; decode append cost is included.
        self.values[:,start:end].copy_(v.transpose(0,1), non_blocking=False)
        if length > 1:
            if start != 0:
                raise ValueError('Chunked prefill is not supported by this benchmark')
            out = single_prefill_with_kv_cache(q, k, v, causal=True)
        else:
            # Reuse only the allocation: contents are replaced from CPU every layer/step.
            key_staging = controller.key_staging
            self.ext.load_keys(key_staging, self.keys, end)
            scores, ids = select(q[0], key_staging, end, self.budget)
            count = ids.shape[1]
            if count == self.budget:
                selected_v = self.staging
            else:
                selected_v = torch.empty(self.heads, count, self.dim, device=q.device, dtype=q.dtype)
            self.ext.load_values(selected_v, self.values, ids)
            weights = scores.softmax(-1).to(q.dtype)
            out = torch.bmm(weights.unsqueeze(1), selected_v).reshape(1,self.heads,self.dim)
        return self.o_proj(out.reshape(1,length,self.hidden)), None, None


def install(model, max_length, budget, device):
    ext = load_extension()
    self_test(ext, device)
    model.model.controller = NaiveController(max_length)
    config = model.config
    model.model.controller.key_staging = torch.empty(config.num_key_value_heads, max_length,
        config.hidden_size//config.num_attention_heads, device=device, dtype=model.dtype)
    model.model._skip_layer = 0
    model.model._token_budget = budget
    for layer in model.model.layers:
        layer.self_attn = NaiveAttention(layer.self_attn, max_length, budget, ext)
    print('Naive: CPU K/V -> load ALL K -> QK/Top-K -> load selected V; no prefetch/cache reuse', flush=True)
