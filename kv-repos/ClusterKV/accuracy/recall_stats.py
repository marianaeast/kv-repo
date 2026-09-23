"""Small helpers for measuring token-selection recall during decoding."""

import torch


def new_recall_stats():
    return {
        "hits": None,
        "total": 0,
        "selected": None,
        "events": 0,
    }


def reset_recall_stats(stats):
    stats.clear()
    stats.update(new_recall_stats())


def _add(stats, hits, total, selected, events):
    if stats["hits"] is None:
        stats["hits"] = torch.zeros((), dtype=torch.int64, device=hits.device)
        stats["selected"] = torch.zeros((), dtype=torch.int64, device=hits.device)
    stats["hits"].add_(hits)
    stats["selected"].add_(selected)
    stats["total"] += int(total)
    stats["events"] += int(events)


def update_from_mask(stats, selected_mask, exact_scores, topk):
    """Compare a boolean candidate mask with exact token-level Top-K."""
    if stats is None:
        return
    k = min(int(topk), exact_scores.shape[-1])
    truth = exact_scores.topk(k, dim=-1).indices
    hits = selected_mask.gather(-1, truth).sum(dtype=torch.int64)
    selected = selected_mask.sum(dtype=torch.int64)
    events = truth.numel() // k
    _add(stats, hits, truth.numel(), selected, events)


def update_from_indices(stats, selected_indices, exact_scores, topk):
    """Compare fixed-size candidate indices with exact token-level Top-K."""
    if stats is None:
        return
    prompt_len = exact_scores.shape[-1]
    valid = (selected_indices >= 0) & (selected_indices < prompt_len)
    safe_indices = selected_indices.clamp(0, max(prompt_len - 1, 0))
    candidate_mask = torch.zeros(
        selected_indices.shape[0],
        selected_indices.shape[1],
        prompt_len,
        dtype=torch.bool,
        device=selected_indices.device,
    )
    candidate_mask.scatter_(-1, safe_indices, valid)

    k = min(int(topk), prompt_len)
    truth = exact_scores.topk(k, dim=-1).indices
    expanded_mask = candidate_mask.unsqueeze(-2).expand(
        -1, -1, truth.shape[-2], -1
    )
    hits = expanded_mask.gather(-1, truth).sum(dtype=torch.int64)
    selected = candidate_mask.sum(dtype=torch.int64) * truth.shape[-2]
    events = truth.numel() // k
    _add(stats, hits, truth.numel(), selected, events)


def reset_model_recall_stats(model):
    for module in model.modules():
        stats = getattr(module, "recall_stats", None)
        if stats is not None:
            reset_recall_stats(stats)


def collect_model_recall_stats(model):
    hits = 0
    total = 0
    selected = 0
    events = 0
    for module in model.modules():
        stats = getattr(module, "recall_stats", None)
        if stats is None:
            continue
        if stats["hits"] is not None:
            hits += int(stats["hits"].item())
            selected += int(stats["selected"].item())
        total += stats["total"]
        events += stats["events"]
    return {
        "recall_hits": hits,
        "recall_total": total,
        "selection_recall": hits / total if total else None,
        "selected_tokens_total": selected,
        "recall_events": events,
        "mean_selected_tokens": selected / events if events else None,
    }
