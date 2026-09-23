"""Accuracy-only ShadowKV variant with a shared budget over ALL history.

Keep ShadowKV's chunk landmarks, group-max selection and low-rank prompt keys.
Generated keys update the chunk landmarks and compete for the same slots.
No local, outlier, alignment or generated token is appended unconditionally.
This is a controlled accuracy variant, not a timing/offload implementation.
"""
import math

import torch
import torch.nn.functional as F

from .kv_cache import ShadowKVCache


class BudgetedShadowKVCache(ShadowKVCache):
    strict_total_budget = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sparse_budget < self.chunk_size or self.sparse_budget % self.chunk_size:
            raise ValueError("budget must be a positive multiple of chunk_size")
        self.local_chunk = 0
        self.outlier_chunk = 0
        max_chunks = (self.max_length + self.chunk_size - 1) // self.chunk_size
        self.landmark_sums = torch.zeros(
            self.num_layers, self.batch_size, self.num_key_value_heads,
            max_chunks, self.head_dim, dtype=torch.float32, device=self.device,
        )
        self.history_lengths = [0] * self.num_layers
        self.generated_keys = [None] * self.num_layers
        self.selected_valid = [None] * self.num_layers
        self.selected_ids = [None] * self.num_layers
        self.max_attention_tokens = 0
        self.last_max_attention_tokens = None

    def prefill_kv_cache(self, new_v_cache, layer_idx, key_states_roped, query=None):
        length = new_v_cache.shape[-2]
        self.prefill = length
        self.prefill_local = 0
        self.sparse_start = 0
        self.sparse_end = min(self.sparse_budget, length)
        self.v_cache_cpu[layer_idx, :, :, :length].copy_(new_v_cache)
        if self.recall_enabled:
            self.recall_key_cache[layer_idx, :, :, :length].copy_(key_states_roped)
        padding = (-length) % self.chunk_size
        keys = F.pad(key_states_roped.float(), (0, 0, 0, padding))
        sums = keys.reshape(
            self.batch_size, self.num_key_value_heads, -1,
            self.chunk_size, self.head_dim,
        ).sum(dim=-2)
        self.landmark_sums[layer_idx, :, :, :sums.shape[-2]].copy_(sums)
        self.history_lengths[layer_idx] = length
        if layer_idx == self.num_layers - 1:
            self.kv_offset = length

    def update_kv_cache(self, new_k_cache, new_v_cache, layer_idx):
        start = self.history_lengths[layer_idx]
        incoming = new_k_cache.shape[-2]
        end = start + incoming
        if end > self.max_length:
            raise ValueError("decode history exceeds max_length")
        self.v_cache_cpu[layer_idx, :, :, start:end].copy_(new_v_cache)
        if self.recall_enabled:
            self.recall_key_cache[layer_idx, :, :, start:end].copy_(new_k_cache)
        old = self.generated_keys[layer_idx]
        self.generated_keys[layer_idx] = (
            new_k_cache.clone() if old is None else torch.cat([old, new_k_cache], dim=-2)
        )
        # Decode normally appends one token. The partial last chunk is updated
        # immediately; it never gets free admission while waiting to fill up.
        for i in range(incoming):
            chunk = (start + i) // self.chunk_size
            self.landmark_sums[layer_idx, :, :, chunk].add_(new_k_cache[:, :, i].float())
        self.history_lengths[layer_idx] = end
        if layer_idx == self.num_layers - 1:
            self.kv_offset = end
            self.gen_offset = end - self.prefill

    def get_retrieval_position_ids(self, layer_idx, query_states):
        if query_states.shape[-2] != 1:
            raise ValueError("strict ShadowKV supports one decode token at a time")
        length = self.history_lengths[layer_idx]
        num_chunks = (length + self.chunk_size - 1) // self.chunk_size
        counts = (length - torch.arange(num_chunks, device=self.device) * self.chunk_size)
        counts = counts.clamp(max=self.chunk_size)
        landmarks = (self.landmark_sums[layer_idx, :, :, :num_chunks]
                     / counts[None, None, :, None]).to(self.dtype)
        queries = query_states.reshape(
            self.batch_size, self.num_key_value_heads, self.num_key_value_groups,
            1, self.head_dim,
        )
        scores = torch.einsum('bhgqd,bhcd->bhgqc', queries, landmarks) / math.sqrt(self.head_dim)
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_scores = probabilities.sum(dim=-2).amax(dim=2)
        n_select = min(self.sparse_budget // self.chunk_size, num_chunks)
        chunks = chunk_scores.topk(n_select, dim=-1).indices
        ids = (chunks.unsqueeze(-1) * self.chunk_size
               + torch.arange(self.chunk_size, device=self.device)).flatten(-2)
        valid = ids < length
        ids = ids.masked_fill(~valid, -1)
        self.selected_valid[layer_idx] = valid
        self.selected_ids[layer_idx] = ids
        maximum = int(valid.sum(dim=-1).max().item())
        if maximum > self.sparse_budget:
            raise RuntimeError("attention selection exceeds the shared KV-head budget")
        self.max_attention_tokens = max(self.max_attention_tokens, maximum)
        if self.recall_enabled:
            self._update_recall(layer_idx, query_states, ids)
        return ids

    def get_value_cache(self, layer_idx, position_ids):
        ids = position_ids.clamp(min=0)
        return self.v_cache_cpu[layer_idx].gather(
            2, ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        # Retain native ShadowKV semantics: prompt keys are reconstructed from
        # low-rank factors, generated keys are exact but still must be selected.
        prompt_ids = position_ids.clamp(min=0, max=self.prefill - 1)
        u = self.U[layer_idx].unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)
        selected_u = u.gather(2, prompt_ids.unsqueeze(-1).expand(-1, -1, -1, u.shape[-1]))
        keys = rope_func(torch.einsum('bhnr,bhrd->bhnd', selected_u, self.SV[layer_idx]), prompt_ids)
        generated = self.generated_keys[layer_idx]
        if generated is not None:
            offsets = (position_ids - self.prefill).clamp(min=0, max=generated.shape[-2] - 1)
            exact = generated.gather(2, offsets.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            keys = torch.where((position_ids >= self.prefill).unsqueeze(-1), exact, keys)
        return keys

    def attention(self, query, keys, values, layer_idx):
        # A partial selected chunk can contain fewer than chunk_size real
        # tokens. Mask its padding per KV head so it contributes no softmax mass.
        groups = self.num_key_value_groups
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        mask = self.selected_valid[layer_idx].repeat_interleave(groups, dim=1).unsqueeze(-2)
        output = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        return output.transpose(1, 2)

    def _update_recall(self, layer_idx, query_states, position_ids):
        length = self.history_lengths[layer_idx]
        queries = query_states.reshape(
            self.batch_size, self.num_key_value_heads, self.num_key_value_groups,
            1, self.head_dim,
        )
        scores = torch.einsum('bhgqd,bhnd->bhgqn', queries,
                              self.recall_key_cache[layer_idx, :, :, :length])
        k = min(self.recall_topk, length)
        truth = scores.topk(k, dim=-1).indices
        # Use a separate sentinel column for invalid padding, so it cannot
        # overwrite the membership bit of a legitimately selected token zero.
        mask = torch.zeros(self.batch_size, self.num_key_value_heads, length + 1,
                           dtype=torch.bool, device=self.device)
        safe_ids = position_ids.masked_fill(position_ids < 0, length)
        mask.scatter_(-1, safe_ids, True)
        mask = mask[..., :length]
        expanded = mask[:, :, None, None, :].expand_as(scores)
        self.recall_hits.add_(expanded.gather(-1, truth).sum(dtype=torch.int64))
        self.recall_selected.add_(mask.sum(dtype=torch.int64) * self.num_key_value_groups)
        self.recall_total += truth.numel()
        self.recall_events += truth.numel() // k

    def snapshot_recall_stats(self):
        super().snapshot_recall_stats()
        self.last_max_attention_tokens = self.max_attention_tokens

    def clear(self):
        super().clear()
        self.landmark_sums.zero_()
        self.history_lengths = [0] * self.num_layers
        self.generated_keys = [None] * self.num_layers
        self.selected_valid = [None] * self.num_layers
        self.selected_ids = [None] * self.num_layers
        self.max_attention_tokens = 0
        self.last_max_attention_tokens = None
