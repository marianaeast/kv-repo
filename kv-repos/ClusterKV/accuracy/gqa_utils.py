"""Small helpers for sharing one sparse selection across a GQA KV head."""

import torch


def average_query_heads(tensor: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Average consecutive query heads that share the same KV head."""
    if num_key_value_groups == 1:
        return tensor
    if tensor.shape[1] % num_key_value_groups != 0:
        raise ValueError(
            f"{tensor.shape[1]} query heads cannot be split into "
            f"groups of {num_key_value_groups}"
        )
    num_key_value_heads = tensor.shape[1] // num_key_value_groups
    grouped_shape = (
        tensor.shape[0],
        num_key_value_heads,
        num_key_value_groups,
        *tensor.shape[2:],
    )
    return tensor.reshape(grouped_shape).mean(dim=2)


def repeat_kv_head_selection(
    tensor: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Give every query head the selection made by its shared KV head."""
    if num_key_value_groups == 1:
        return tensor
    expanded = tensor.unsqueeze(2).expand(
        tensor.shape[0],
        tensor.shape[1],
        num_key_value_groups,
        *tensor.shape[2:],
    )
    return expanded.reshape(
        tensor.shape[0],
        tensor.shape[1] * num_key_value_groups,
        *tensor.shape[2:],
    )
