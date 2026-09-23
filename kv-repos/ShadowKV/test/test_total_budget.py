"""Small CUDA checks for dynamic chunks and masked strict-budget attention."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.budgeted_kv_cache import BudgetedShadowKVCache
from models.tensor_op import apply_rotary_pos_emb_cuda


def check_native_rope():
    # Use the real CUDA RoPE kernel and BF16 shapes from Llama-3.1-8B.
    cfg = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8,
                          num_hidden_layers=1, hidden_size=4096)
    cache = BudgetedShadowKVCache(cfg, max_length=256, device='cuda',
                                 dtype=torch.bfloat16, sparse_budget=128, chunk_size=8, rank=16)
    keys = torch.randn(1, 8, 131, 128, device='cuda', dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    angles = torch.randn(256, 64, device='cuda')
    cos_sin = torch.cat([angles.cos(), angles.sin()], -1).to(torch.bfloat16)
    def rope(k, ids):
        return apply_rotary_pos_emb_cuda(k, cos_sin, ids)
    positions = torch.arange(131, device='cuda').view(1, 1, -1).expand(1, 8, -1)
    cache.get_svd(keys, 0)
    cache.prefill_kv_cache(values, 0, rope(keys, positions))
    new_k = torch.randn(1, 8, 1, 128, device='cuda', dtype=torch.bfloat16)
    cache.update_kv_cache(new_k, torch.randn_like(new_k), 0)
    query = torch.randn(1, 32, 1, 128, device='cuda', dtype=torch.bfloat16)
    ids = cache.get_retrieval_position_ids(0, query)
    reconstructed = cache.get_key_cache(0, ids, rope, cos_sin)
    output = cache.attention(query, reconstructed, cache.get_value_cache(0, ids), 0)
    assert output.shape == (1, 1, 32, 128) and output.isfinite().all()
    # CUDA RoPE must match an independent rotate-half expression on prompt keys.
    prompt_ids = ids.clamp(0, 130)
    u = cache.U[0].unsqueeze(1).expand(-1, 8, -1, -1)
    selected = u.gather(2, prompt_ids.unsqueeze(-1).expand(-1, -1, -1, 16))
    raw = torch.einsum('bhnr,bhrd->bhnd', selected, cache.SV[0])
    cosine, sine = cos_sin[prompt_ids].chunk(2, dim=-1)
    cosine, sine = cosine.repeat(1, 1, 1, 2), sine.repeat(1, 1, 1, 2)
    rotated_half = torch.cat([-raw[..., 64:], raw[..., :64]], -1)
    expected = (raw.float() * cosine.float() + rotated_half.float() * sine.float()).to(raw.dtype)
    prompt = (ids >= 0) & (ids < 131)
    torch.testing.assert_close(reconstructed[prompt], expected[prompt], atol=0.02, rtol=0.02)
    generated = ids == 131
    if generated.any():
        torch.testing.assert_close(reconstructed[generated], new_k.expand(-1, -1, 128, -1)[generated])
    print('PASS: BF16 Llama head shapes, native CUDA RoPE, low-rank reconstruction and masked attention')


def main():
    torch.manual_seed(6)
    cfg = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8,
                          num_hidden_layers=1, hidden_size=256)
    cache = BudgetedShadowKVCache(cfg, max_length=64, device='cuda',
                                 dtype=torch.float32, sparse_budget=8, chunk_size=4, rank=4)
    cache.enable_recall_stats(8)
    keys = torch.zeros(1, 8, 9, 8, device='cuda')
    keys[:, :, :4, 0] = 10
    keys[:, :, 4:8, 0] = 20
    keys[:, :, 8, 0] = -1000
    values = torch.randn_like(keys)
    cache.get_svd(keys, 0)
    cache.prefill_kv_cache(values, 0, keys)
    queries = torch.zeros(1, 32, 1, 8, device='cuda')
    queries[..., 0] = 1
    all_keys, all_values = keys, values
    for step in range(20):
        new_k = torch.zeros(1, 8, 1, 8, device='cuda')
        new_k[..., 0] = -1000 if step == 0 else 10000
        new_v = torch.randn_like(new_k)
        all_keys = torch.cat([all_keys, new_k], dim=2)
        all_values = torch.cat([all_values, new_v], dim=2)
        cache.update_kv_cache(new_k, new_v, 0)
        ids = cache.get_retrieval_position_ids(0, queries)
        valid = ids >= 0
        assert (valid.sum(-1) <= 8).all()
        assert cache.history_lengths[0] == 10 + step
        for row in ids.reshape(-1, ids.shape[-1]):
            assert row[row >= 0].unique().numel() == (row >= 0).sum()
        if step == 0:
            assert not (ids == 9).any(), 'newest low-score token must be excludable'
        if step == 1:
            assert (ids == 10).all(dim=0).any(), 'newest high-score token must enter a selected chunk'
            assert (~valid).any(), 'exercise the partial-chunk padding path'
        for chunk in range((10 + step + 3) // 4):
            torch.testing.assert_close(cache.landmark_sums[0, :, :, chunk],
                                       all_keys[:, :, chunk * 4:(chunk + 1) * 4].sum(2))
        ks = cache.get_key_cache(0, ids, lambda k, pos: k, None)
        vs = cache.get_value_cache(0, ids)
        output = cache.attention(queries, ks, vs, 0)
        assert ks.shape[-2] <= 8 and vs.shape[-2] <= 8
        # The fixture's prompt keys have rank one, so reconstruction is exact.
        for head in range(32):
            kv_head = head // 4
            selected = ids[0, kv_head][valid[0, kv_head]]
            k = all_keys[0, kv_head, selected]
            v = all_values[0, kv_head, selected]
            weights = (queries[0, head] @ k.T / (8 ** 0.5)).softmax(-1)
            torch.testing.assert_close(output[0, :, head], weights @ v, atol=2e-4, rtol=2e-4)
    assert cache.history_lengths[0] > 3 * cache.sparse_budget
    assert cache.max_attention_tokens <= 8
    stats = cache.get_recall_stats()
    assert stats['mean_selected_tokens'] <= 8
    cache.clear()
    assert cache.history_lengths == [0] and cache.generated_keys == [None]
    assert cache.landmark_sums.count_nonzero() == 0
    print('PASS: new tokens can be selected or rejected; dynamic partial/full chunks; 20-step strict shared budget; masked attention equals explicit reference; reset')


if __name__ == '__main__':
    main()
    check_native_rope()
