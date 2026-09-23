#!/usr/bin/env python
"""固定预算InfiniGen风格微基准，不是原版端到端结果。

真实partial query/partial K选择、per-query-head随机加载、下一层预取。
使用随机权重而非校准模型；固定Top-K替代原文alpha阈值；UVA加载替代CPU embedding。
LOAD来自实际传输，wall来自实际流水线，不再用字节数/带宽或各段相加。
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from breakdown_timeline import partition_intervals, STAGES

CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
// src: [2,kv_heads,context,dim] pinned；dst: [2,q_heads,budget,dim] GPU。
__global__ void gather_tokens_kernel(uint16_t* dst, const uint16_t* src,
    const int64_t* indices, int qh, int kh, int n, int k, int d) {
    int i=blockIdx.x, h=blockIdx.y, kv=blockIdx.z;
    int token=indices[h*k+i], src_head=h/(qh/kh);
    for(int j=threadIdx.x; j<d; j+=blockDim.x) {
        dst[((int64_t(kv)*qh+h)*k+i)*d+j] = src[((int64_t(kv)*kh+src_head)*n+token)*d+j];
    }
}
void gather_tokens(torch::Tensor dst, torch::Tensor src, torch::Tensor idx) {
    TORCH_CHECK(src.device().is_cpu() && src.is_pinned(), "pinned CPU src required");
    TORCH_CHECK(dst.is_cuda() && idx.is_cuda(), "CUDA dst/idx required");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && idx.is_contiguous(), "contiguous required");
    TORCH_CHECK(src.dim()==4 && dst.dim()==4 && idx.dim()==2, "invalid dimensions");
    TORCH_CHECK(src.size(0)==2 && dst.size(0)==2 && src.size(3)==dst.size(3), "invalid KV shape");
    TORCH_CHECK(src.scalar_type()==dst.scalar_type() && src.element_size()==2, "same 16-bit dtype required");
    TORCH_CHECK(idx.scalar_type()==at::kLong, "int64 indices required");
    int h=dst.size(1), k=dst.size(2), d=dst.size(3), kh=src.size(1), n=src.size(2);
    TORCH_CHECK(h%kh==0 && idx.size(0)==h && idx.size(1)==k, "invalid heads/indices");
    gather_tokens_kernel<<<dim3(k,h,2),128,0,at::cuda::getCurrentCUDAStream()>>>(
      (uint16_t*)dst.data_ptr(),(uint16_t*)src.data_ptr(),idx.data_ptr<int64_t>(),h,kh,n,k,d);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"gather launch failed");
}
'''

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    for name, default in [('context_len',32768),('budget',512),('layers',32),('h_q',32),('h_kv',8),
                          ('head_dim',128),('hidden',4096),('inter',14336),('warmup',10),('iters',96),('prof_steps',24)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--partial_ratio',type=float,default=0.3)
    p.add_argument('--rank',type=int,default=None,help='部分通道数；默认floor(0.3*head_dim)，不是低秩SVD的秩')
    p.add_argument('--dtype',choices=['float16','bfloat16'],default='float16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--tag',default='')
    p.add_argument('--out_json',required=True)
    p.add_argument('--self_test',action='store_true')
    a=p.parse_args()
    a.rank=a.rank or int(a.partial_ratio*a.head_dim)
    if not (0<a.budget<=a.context_len and 0<a.rank<=a.head_dim and a.h_q%a.h_kv==0):
        p.error('invalid budget/channels/GQA geometry')
    if min(a.layers,a.iters,a.prof_steps)<=0 or a.hidden!=a.h_q*a.head_dim:
        p.error('positive counts and hidden=h_q*head_dim required')
    return a

@torch.inference_mode()
def main():
    a=parse()
    torch.cuda.set_device(a.device)
    torch.manual_seed(20260919)
    dt=getattr(torch,a.dtype)
    ext=load_inline(name='infini_token_gather_v2',
        cpp_sources='void gather_tokens(torch::Tensor, torch::Tensor, torch::Tensor);',
        cuda_sources=CUDA,functions=['gather_tokens'],extra_cuda_cflags=['-O3'])
    # 非零数据，逐head不同索引，包含边界位置和重复位置。
    src=torch.randn(2,2,257,128,dtype=dt).pin_memory()
    idx=torch.randint(257,(8,64),device=a.device)
    idx[:,0],idx[:,1]=0,256
    dst=torch.empty(2,8,64,128,device=a.device,dtype=dt)
    ext.gather_tokens(dst,src,idx)
    torch.cuda.synchronize()
    ref=torch.stack([src[:,h//4,idx[h].cpu()] for h in range(8)],dim=1)
    torch.testing.assert_close(dst.cpu(),ref,rtol=0,atol=0)
    if a.self_test:
        print('Per-query-head UVA gather: PASS')
        return
    from flash_attn import flash_attn_with_kvcache
    print('FIXED-BUDGET SYNTHETIC MICROBENCH, not original InfiniGen E2E',vars(a),flush=True)
    def rand(*shape):
        return torch.empty(*shape,device=a.device,dtype=dt).normal_(std=0.02)
    # 每层不同权重，不能循环使用一层权重伪造高cache命中。
    weights=[(rand((a.h_q+2*a.h_kv)*a.head_dim,a.hidden),rand(a.hidden,a.hidden),
              rand(2*a.inter,a.hidden),rand(a.hidden,a.inter)) for _ in range(a.layers)]
    partial_w=rand(a.layers,a.h_q*a.rank,a.hidden)
    partial_k=rand(a.layers,a.h_kv,a.rank,a.context_len)
    master=torch.empty(a.layers,2,a.h_kv,a.context_len,a.head_dim,dtype=dt,pin_memory=True).normal_(std=0.02)
    selected=torch.empty(a.layers,2,a.h_q,a.budget,a.head_dim,device=a.device,dtype=dt)
    x0=rand(1,a.hidden)
    stream=torch.cuda.Stream()
    ready=[torch.cuda.Event() for _ in range(a.layers)]

    def rms(x):
        # 合成层也要归一化，避免连续32层随机权重把激活放大到NaN。
        return (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-5)).to(dt)

    def step(measure=False):
        current=torch.cuda.current_stream()
        anchor=torch.cuda.Event(enable_timing=True)
        anchor.record()
        segments=[]
        def begin():
            if measure:
                e=torch.cuda.Event(enable_timing=True); e.record(); return e
        def end(stage,start):
            if measure:
                e=torch.cuda.Event(enable_timing=True); e.record(); segments.append((stage,start,e))
        def prefetch(layer,x):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                x.record_stream(stream)
                s=begin()
                pq=F.linear(x,partial_w[layer]).view(a.h_kv,a.h_q//a.h_kv,a.rank)
                scores=torch.bmm(pq,partial_k[layer]).view(a.h_q,a.context_len)
                indices=scores.topk(a.budget,dim=-1,sorted=False).indices.contiguous()
                end('select',s)
                s=begin()
                ext.gather_tokens(selected[layer],master[layer],indices)
                end('load',s)
                ready[layer].record()
        x=x0
        for layer,(wqkv,wo,wgu,wd) in enumerate(weights):
            s=begin()
            normalized=rms(x)
            end('compute_misc',s)
            if layer==0:
                prefetch(0,normalized)
            if layer+1<a.layers:
                prefetch(layer+1,normalized)
            s=begin()
            qkv=F.linear(normalized,wqkv)
            q=qkv[:,:a.hidden].reshape(a.h_q,1,1,a.head_dim)
            end('compute_gemm',s)
            s=begin()
            current.wait_event(ready[layer])
            end('load_wait',s)
            s=begin()
            # 每个query head各自k个token；head作为batch维，不能用共享8-head的缓存冒充。
            out=flash_attn_with_kvcache(q,selected[layer,0].unsqueeze(2),selected[layer,1].unsqueeze(2))
            end('compute_attn',s)
            s=begin()
            x=x+F.linear(out.reshape(1,a.hidden),wo)
            gate,up=F.linear(rms(x),wgu).chunk(2,dim=-1)
            x=x+F.linear(F.silu(gate)*up,wd)
            end('compute_gemm',s)
        stop=torch.cuda.Event(enable_timing=True); stop.record()
        torch.cuda.synchronize()
        if not measure:
            return x
        end_us=anchor.elapsed_time(stop)*1000
        intervals=[(anchor.elapsed_time(s)*1000,anchor.elapsed_time(e)*1000,name)
                   for name,s,e in segments if name!='load_wait']
        parts=partition_intervals(intervals,0,end_us)
        raw={key:sum(s.elapsed_time(e) for name,s,e in segments if name==key)
             for key in STAGES+('load_wait',)}
        return parts,raw,end_us/1000,intervals

    probe=step()
    if not bool(torch.isfinite(probe).all()):
        raise RuntimeError("nonfinite synthetic pipeline output")
    del probe
    for _ in range(a.warmup): step()
    wall=[]
    for _ in range(a.iters):
        start=time.perf_counter(); step(); wall.append((time.perf_counter()-start)*1000)
    samples=[step(True) for _ in range(a.prof_steps)]
    stages={key:statistics.mean(v[0][key]/1000 for v in samples) for key in STAGES}
    raw={key:statistics.mean(v[1][key] for v in samples) for key in STAGES+('load_wait',)}
    rec={'schema_version':2,'method':'infinigen_fixed_budget_microbench','measure':'synthetic_tensor_pipeline',
         'offload':True,'context_len':a.context_len,'budget':a.budget,'rank':a.rank,'layers':a.layers,
         'h_q':a.h_q,'h_kv':a.h_kv,'dtype':a.dtype,'tag':a.tag,
         'wall_ms_mean':statistics.mean(wall),'wall_ms_median':statistics.median(wall),'wall_steps_ms':wall,
         'stages_ms':stages,'raw_stages_ms':raw,'measured_prefetch_wait_ms':raw['load_wait'],
         'gap_ms':statistics.mean(v[0]['gap']/1000 for v in samples),
         'profile_wall_ms':statistics.mean(v[2] for v in samples),
         'io_detail':{'backend':'measured UVA random gather','logical_bytes_per_step':a.layers*2*a.h_q*a.budget*a.head_dim*2,
                      'raw_load_ms':raw['load'],'raw_wait_ms':raw['load_wait']},
         'load_definition':'exclusive event-measured load phase; raw_wait includes pending selection/load and marker/host overhead',
         'stack_definition':'select-first CUDA phase-event timeline; stages+gap=profile_wall_ms; separate from uninstrumented wall',
         'attribution_priority':['select','compute_attn','compute_gemm','compute_misc','other','load'],
         'limitations':['synthetic weights/keys, no calibrated selection or text generation',
                        'fixed Top-K replaces original alpha threshold; default partial channels=floor(0.3*head_dim)',
                        '32 independent query-head selections mapped to 8 KV heads, without deduplication',
                        'UVA backend replaces original CPU embedding; no RoPE/embedding/lm_head; simple RMS normalization',
                        'phase events include host submission gaps; not directly identical to kernel profiler attribution']}
    rec['phase_occupied_ms']=sum(stages.values())
    rec['overlap_ms']=sum(raw[k] for k in STAGES)-rec['phase_occupied_ms']
    target=Path(a.out_json); target.parent.mkdir(parents=True,exist_ok=True)
    timeline=target.with_suffix('.phases.json')
    timeline.write_text(json.dumps([{'end_us':v[2]*1000,'intervals_us':v[3]} for v in samples]))
    rec['phase_timeline_file']=str(timeline)
    target.write_text(json.dumps(rec,indent=2,ensure_ascii=False))
    print(f"Measured pipeline median={rec['wall_ms_median']:.3f}ms; raw load={raw['load']:.3f}ms; prefetch wait={raw['load_wait']:.3f}ms")
    print('saved ->',target)

if __name__=='__main__': main()
