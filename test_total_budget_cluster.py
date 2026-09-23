"""CUDA checks for shared selection, online cluster membership, and outputs."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent / 'kv-repos/ClusterKV'))
import accuracy.cluster_attention as cluster
import accuracy.quest_attention as quest
from accuracy.online_cluster import append_decode_keys


def reference_attention(q, k, v, ids):
    selected_k = k.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1]))
    selected_v = v.gather(2, ids.unsqueeze(-1).expand_as(selected_k))
    weights = (q @ selected_k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    return (weights.softmax(-1) @ selected_v).transpose(1, 2).reshape(1, 1, -1)


def main():
    torch.manual_seed(5)
    device = 'cuda'
    heads, groups, dim = 8, 4, 8
    keys = torch.zeros(1, heads, 5, dim, device=device)
    keys[:, :, :3, 0] = 1
    keys[:, :, 3, 0] = -1
    keys[:, :, 4, 0] = 4  # Newly generated token.
    values = torch.randn_like(keys)
    centers = torch.zeros(1, heads, 2, dim, device=device)
    centers[:, :, 0, 0] = 1
    centers[:, :, 1, 0] = -1
    module = SimpleNamespace(
        key_centroids=centers,
        cluster_key_indices=torch.arange(4, device=device).repeat(heads, 1),
        cluster_key_ptr=torch.tensor([0, 0, 0, 1], device=device, dtype=torch.int16).repeat(heads, 1),
        cluster_key_size=torch.tensor([3, 1], device=device, dtype=torch.int32).repeat(heads, 1),
    )
    append_decode_keys(module, keys, sink=0)
    assert module.cluster_key_indices.shape == (heads, 5)
    assert torch.equal(module.cluster_key_indices[0], torch.tensor([0, 1, 2, 4, 3], device=device))
    # Calling again must not duplicate the generated token.
    append_decode_keys(module, keys, sink=0)
    assert module.cluster_key_indices.shape[-1] == 5
    q = torch.zeros(1, heads * groups, 1, dim, device=device)
    q[:, :, :, 0] = 1
    q[:, groups:2 * groups, :, 0] = -1
    k = keys.repeat_interleave(groups, 1)
    v = values.repeat_interleave(groups, 1)
    captured = {}
    def capture(stats, ids, scores, topk):
        captured['ids'] = ids.clone()
        assert scores.shape[-1] == 5, 'recall must include generated history'
    cluster.update_from_indices = capture
    out = cluster.cluster_attn_out(q, k, v, None, 5, centers,
        module.cluster_key_indices, module.cluster_key_size, module.cluster_key_size_ps,
        groups, 0, 4, 0, 'truc', None, gqa_policy='qavg')
    ids = captured['ids']
    assert ids.shape == (1, heads * groups, 4)
    assert (ids[:, :groups] == 4).any(), 'important newest token must be selectable'
    assert not (ids[:, groups:2 * groups] == 4).any(), 'newest token must not be free'
    assert (ids.reshape(1, heads, groups, 4) == ids.reshape(1, heads, groups, 4)[:, :, :1]).all()
    torch.testing.assert_close(out, reference_attention(q, k, v, ids))

    # Quest chunk_size=1 is Naive: its shared set must equal group-mean Top-K.
    q = torch.randn(1, heads * groups, 1, dim, device=device)
    keys = torch.randn(1, heads, 9, dim, device=device)
    k = keys.repeat_interleave(groups, 1)
    v = torch.randn_like(keys).repeat_interleave(groups, 1)
    def capture_mask(stats, mask, scores, topk):
        captured['mask'] = mask.clone()
        assert scores.shape[-1] == 9
    quest.update_from_mask = capture_mask
    out = quest.quest_attn_out(q, k, v, None, False, 4, 1, 4, 0, None,
                              num_key_value_groups=groups, gqa_policy='qavg')
    scores = (q @ k.transpose(-1, -2)).reshape(1, heads, groups, 1, 9).mean(2)
    expected_ids = scores.topk(4, -1).indices.squeeze(2).repeat_interleave(groups, 1)
    expected = torch.zeros_like(captured['mask']).scatter_(-1, expected_ids.unsqueeze(2), True)
    assert torch.equal(captured['mask'], expected)
    torch.testing.assert_close(out, reference_attention(q, k, v, expected_ids))
    print('PASS: Cluster new-token inclusion/exclusion, strict K, shared GQA, actual attention output; Naive exact shared Top-K')


if __name__ == '__main__':
    main()
