"""仅重编ShadowKV的gather文件，支持64 chunks；无需重编全部CUTLASS kernel。"""
from pathlib import Path
from torch.utils.cpp_extension import load

KERNELS = Path(__file__).resolve().parents[2] / 'ShadowKV' / 'kernels'

def install(shadowkv):
    extension = load(name='shadowkv_gather64_v2',
                     sources=[str(KERNELS/'gather_copy.cu'), str(Path(__file__).with_name('shadow_gather_bind.cpp'))],
                     extra_include_paths=[str(KERNELS)], extra_cuda_cflags=['-O3'], verbose=False)
    for name in ('reorder_keys_and_compute_offsets', 'gather_copy_d2d_with_offsets', 'gather_copy_with_offsets'):
        setattr(shadowkv, name, getattr(extension, name))
    return extension
