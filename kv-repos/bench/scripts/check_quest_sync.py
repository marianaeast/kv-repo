"""检查CPU主副本只同步完整页，并按物理页号定位。"""
from types import SimpleNamespace
import torch
from clusterkv.utils.offload_cache import OffloadKvStore

store=OffloadKvStore(3,2,4,32,4,torch.float16,'cuda',4)
buf=torch.arange(3*8*2*4*2*4,device='cuda',dtype=torch.float32).reshape(3,8,2,4,2,4).half()
cache=SimpleNamespace(indicies=[5,1,7],buf_layer=lambda layer: buf[layer])
ctrl=SimpleNamespace(kv_cache=cache)
store.sync_from_gpu(ctrl)
for page in (5,1):
    torch.testing.assert_close(store.cpu[2,page],buf[2,page].cpu(),rtol=0,atol=0)
assert store._synced_pages==2
assert bool((store.cpu[2,7]==0).all())
buf[2,7].fill_(37)
cache.indicies.append(3)
store.sync_from_gpu(ctrl)
torch.testing.assert_close(store.cpu[2,7],buf[2,7].cpu(),rtol=0,atol=0)
assert store._synced_pages==3
assert bool((store.cpu[2,3]==0).all())
store._synced_pages=0
cache.indicies=[3,2]
buf[2,3].fill_(19)
store.sync_from_gpu(ctrl)
torch.testing.assert_close(store.cpu[2,3],buf[2,3].cpu(),rtol=0,atol=0)
print('Quest completed-page sync, nonsequential physical IDs, new prefill: PASS')
