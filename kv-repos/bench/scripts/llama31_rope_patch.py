"""为旧ClusterKV/Quest运行时接入模型配置中的Llama3.1分段RoPE。"""
import torch


def install(config, device, max_length):
    scaling = config.rope_scaling or {}
    if scaling.get('rope_type', scaling.get('type')) != 'llama3':
        return 'original'
    from transformers.modeling_rope_utils import _compute_llama3_parameters
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    import clusterkv.utils

    inv_freq, _ = _compute_llama3_parameters(config, device)
    positions = torch.arange(max_length, device=device, dtype=torch.long)
    angles = positions.float()[:, None] * inv_freq.float()[None, :]
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()

    def apply(q, k, past_kv_len, rope_scale=None, rope_theta=None):
        end = past_kv_len + q.shape[0]
        if end > max_length:
            raise ValueError('RoPE position exceeds precomputed cache')
        apply_rope_with_cos_sin_cache_inplace(
            positions[past_kv_len:end], q, k, q.shape[-1], cache, is_neox=True)

    # 与HF公式核对两个不同位置，包括长上下文位置。
    for pos in (0, min(32767, max_length-1)):
        q = torch.randn(1, 32, inv_freq.numel()*2, device=device, dtype=torch.float16)
        k = q[:, :8].clone()
        original = q.clone()
        cos, sin = cache[pos].chunk(2)
        cos, sin = cos.repeat(2), sin.repeat(2)
        left, right = original.float().chunk(2, dim=-1)
        expected = original.float()*cos + torch.cat((-right, left), dim=-1)*sin
        apply(q, k, pos)
        torch.testing.assert_close(q.float(), expected, rtol=0.002, atol=0.002)
    clusterkv.utils.apply_rope_in_place = apply
    return 'HF llama3 frequencies + FlashInfer fused RoPE (self-test passed)'
