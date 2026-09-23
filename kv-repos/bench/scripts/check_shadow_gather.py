"""验证512/1024预算下的真实V搬运、cache hit和stream顺序。"""
import sys
from pathlib import Path
import torch
from shadow_gather_patch import install, KERNELS

sys.path.insert(0,str(KERNELS))
import shadowkv
ext=install(shadowkv)
torch.manual_seed(11)
for count in (64,128):
    heads,chunk,dim,nchunks,prefix=2,8,128,512,128
    capacity=nchunks+76  # CPU head stride包含未使用预留区，不等于prompt长度
    budget=count*chunk
    src=torch.randn(1,heads,capacity*chunk,dim,dtype=torch.bfloat16).pin_memory()
    buf=torch.full((1,heads,prefix+budget+128,dim),-7,device='cuda',dtype=torch.bfloat16)
    temp=torch.empty(1,heads,budget,dim,device='cuda',dtype=torch.bfloat16)
    keys=buf.clone()
    cached=torch.full((1,heads,count),-1,device='cuda',dtype=torch.long)
    offsets=torch.zeros(heads*count,device='cuda',dtype=torch.int32)
    cnts=torch.zeros(heads,device='cuda',dtype=torch.int32)
    signals=torch.zeros_like(cnts)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for shift in (0,count//2,count//2):
        with torch.cuda.stream(stream):
            selection=torch.stack([torch.arange(shift,shift+count,device='cuda')+h*count for h in range(heads)]).unsqueeze(0)
            ext.reorder_keys_and_compute_offsets(cached,selection,offsets,cnts,1,heads,count)
            ext.gather_copy_d2d_with_offsets(keys,offsets,cnts,1,heads,
                budget*dim,prefix*dim,buf.size(-2)*dim,count)
            ext.gather_copy_with_offsets(src,buf,temp,offsets,cnts,signals,1,heads,
                src[0].stride(0),budget*dim,prefix*dim,buf.size(-2)*dim,count)
        stream.synchronize()
        # reorder可以改变槽位顺序，必须按返回的cached位置核对。
        for h in range(heads):
            expected=src[0,h].view(capacity,chunk,dim)[cached[0,h].cpu()].reshape(budget,dim)
            torch.testing.assert_close(buf[0,h,prefix:prefix+budget].cpu(),expected,rtol=0,atol=0)
            hit_tokens=int(cnts[h])*chunk
            torch.testing.assert_close(keys[0,h,prefix:prefix+hit_tokens].cpu(),expected[:hit_tokens],rtol=0,atol=0)
            # 用正确数据代替后续K重建，供下一轮hit重排检查。
            keys[0,h,prefix:prefix+budget].copy_(expected)
        torch.cuda.synchronize()
        assert bool((buf[:,:,:prefix]==-7).all()), 'overwrote static cache'
        print('budget',budget,'shift',shift,'hits',cnts.tolist(),'PASS',flush=True)
print('ShadowKV gather checks passed')
