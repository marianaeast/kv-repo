"""Insert decode keys into the existing clusters for strict-budget accuracy."""
import torch
import torch.nn.functional as F


def append_decode_keys(module, all_keys, sink):
    # Cluster indices are relative to the sequence after the reserved sink.
    indexed = module.cluster_key_indices.shape[-1]
    new_keys = all_keys[0, :, sink + indexed:, :]
    if new_keys.shape[-2] == 0:
        return
    centroids = module.key_centroids[0]
    scores = torch.bmm(
        F.normalize(new_keys.float(), dim=-1),
        F.normalize(centroids.float(), dim=-1).transpose(1, 2),
    )
    new_labels = scores.argmax(dim=-1)
    new_indices = torch.arange(
        indexed, indexed + new_keys.shape[-2], device=all_keys.device
    ).expand(all_keys.shape[1], -1)
    labels = torch.cat([module.cluster_key_ptr.long(), new_labels], dim=-1)
    indices = torch.cat([module.cluster_key_indices, new_indices], dim=-1)
    # Stable order preserves the original within-cluster order. New tokens
    # receive ordinary membership, with no extra attention slots or priority.
    order = labels.argsort(dim=-1, stable=True)
    module.cluster_key_ptr = labels.gather(-1, order).to(torch.int16)
    module.cluster_key_indices = indices.gather(-1, order)
    sizes = module.cluster_key_size.clone()
    sizes.scatter_add_(-1, new_labels, torch.ones_like(new_labels, dtype=sizes.dtype))
    module.cluster_key_size = sizes
    module.cluster_key_size_ps = sizes.cumsum(dim=-1)
