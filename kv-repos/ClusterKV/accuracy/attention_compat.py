"""Small compatibility helpers shared by the Llama and Qwen3 patches."""

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def is_qwen3_attention(module):
    return getattr(module, "_bypasskv_model_type", "") == "qwen3"


def extract_cache(past_key_value, kwargs):
    """Accept both the old singular and the new plural cache argument."""
    return kwargs.pop("past_key_values", past_key_value)


def project_qkv(module, hidden_states):
    """Project Q/K/V and apply Qwen3's per-head Q/K RMSNorm when present."""
    batch, query_length, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(
        batch, query_length, module.num_heads, module.head_dim
    )
    key = module.k_proj(hidden_states).view(
        batch, query_length, module.num_key_value_heads, module.head_dim
    )
    value = module.v_proj(hidden_states).view(
        batch, query_length, module.num_key_value_heads, module.head_dim
    )
    if hasattr(module, "q_norm"):
        query = module.q_norm(query)
    if hasattr(module, "k_norm"):
        key = module.k_norm(key)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def apply_rope(module, query, key, value, position_ids, position_embeddings):
    if position_embeddings is None:
        cos, sin = module.rotary_emb(value, position_ids)
    else:
        cos, sin = position_embeddings
    return apply_rotary_pos_emb(query, key, cos, sin, position_ids)


def call_dense_attention(
    module,
    hidden_states,
    attention_mask,
    position_ids,
    cache,
    output_attentions,
    use_cache,
    position_embeddings,
    kwargs,
):
    if is_qwen3_attention(module):
        call_kwargs = dict(kwargs)
        call_kwargs.update(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        call_kwargs[module._bypasskv_cache_argument] = cache
        return module.flash_forward(**call_kwargs)
    legacy_kwargs = dict(kwargs)
    if position_embeddings is not None:
        legacy_kwargs["position_embeddings"] = position_embeddings
    return module.flash_forward(
        hidden_states,
        attention_mask,
        position_ids,
        cache,
        output_attentions,
        use_cache,
        **legacy_kwargs,
    )


def format_sparse_output(module, attention_output, cache):
    attention_weights = None
    if is_qwen3_attention(module):
        return attention_output, attention_weights
    return attention_output, attention_weights, cache


def layer_kv(cache, layer_id):
    """Return one cached layer across DynamicCache API versions."""
    try:
        return cache[layer_id]
    except (TypeError, AttributeError, KeyError):
        layer = cache.layers[layer_id]
        return layer.keys, layer.values
