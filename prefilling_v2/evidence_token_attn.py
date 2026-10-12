"""Attention CE onto needle tokens inside already-selected remote chunks."""

from __future__ import annotations

import math

import torch


def evidence_token_attn_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    query_mask: torch.Tensor,
    evidence_tokens: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    *,
    skip_inert_slots: bool = False,
    local_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-positive CE over selected remote tokens; gold is evidence digits.

    Softmax support is the selected top-k chunks only (not the local window).
    Queries with no gold token in that support are skipped.
    """

    if q.ndim != 4 or k.ndim != 4 or indices.ndim != 4:
        raise ValueError("q, k, and indices must be 4D")
    batch, seq_len, h_q, dim = q.shape
    h_kv = k.shape[2]
    if k.shape[:2] != (batch, seq_len) or k.shape[-1] != dim:
        raise ValueError("incompatible q/k shapes")
    if h_q % h_kv != 0:
        raise ValueError("query heads must be a multiple of kv heads")
    groups = h_q // h_kv
    topk = indices.shape[-1]
    if indices.shape[:3] != (batch, seq_len, h_kv):
        raise ValueError("invalid indices shape")
    if query_mask.shape != (batch, seq_len):
        raise ValueError("query mask must match sequence layout")
    if evidence_tokens.shape != (batch, seq_len):
        raise ValueError("evidence tokens must match sequence layout")
    if key_valid.shape != (batch, seq_len):
        raise ValueError("key_valid must be [B, L]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    zero = q.float().sum() * 0.0
    empty = {
        "remote_needle_qk_mass": zero.detach(),
        "local_weight": zero.detach(),
        "n_supervised": zero.detach(),
    }
    rows, query_pos = torch.where(query_mask.bool())
    if rows.numel() == 0:
        return zero, empty

    idx = indices[rows, query_pos]
    valid_idx = idx >= 0
    if not bool(valid_idx.any()):
        return zero, empty
    safe_idx = idx.clamp_min(0)
    slots = torch.arange(chunk_size, device=q.device)
    token_ids = safe_idx.unsqueeze(-1) * int(chunk_size) + slots
    token_ids = token_ids.clamp(0, seq_len - 1)
    token_valid = valid_idx.unsqueeze(-1) & (
        token_ids < seq_len
    )
    if skip_inert_slots:
        token_valid = token_valid & (slots != (chunk_size - 1))
    flat_ids = token_ids.reshape(rows.shape[0], h_kv, topk * chunk_size)
    flat_valid = token_valid.reshape(rows.shape[0], h_kv, topk * chunk_size)
    key_ok = torch.gather(
        key_valid[rows].unsqueeze(1).expand(-1, h_kv, seq_len),
        2,
        flat_ids,
    )
    gold = torch.gather(
        evidence_tokens.bool()[rows].unsqueeze(1).expand(-1, h_kv, seq_len),
        2,
        flat_ids,
    )
    support = flat_valid & key_ok
    gold = gold & support
    hit = gold.any(dim=-1)
    if not bool(hit.any()):
        stats = dict(empty)
        if local_weight is not None:
            stats["local_weight"] = local_weight[rows, query_pos].float().mean().detach()
        return zero, stats

    # Autocast can recast einsum to bf16 even after q/k.float(); keep the
    # partition in fp32 so -inf masks do not overflow.
    with torch.autocast(device_type=q.device.type, enabled=False):
        k_rows = k.float()[rows].permute(0, 2, 1, 3)
        k_sel = torch.gather(
            k_rows,
            2,
            flat_ids.unsqueeze(-1).expand(-1, -1, -1, dim),
        )
        q_rows = q.float()[rows, query_pos].reshape(
            rows.shape[0], h_kv, groups, dim
        )
        logits = torch.einsum("rhgd,rhtd->rhgt", q_rows, k_sel)
        logits = logits.float() * (1.0 / math.sqrt(dim))
        logits = logits.masked_fill(
            ~support.unsqueeze(2), torch.finfo(logits.dtype).min
        )
        all_masked = torch.isneginf(logits).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    gold_f = gold.unsqueeze(2).to(dtype=probs.dtype)
    mass = (probs * gold_f).sum(-1)
    denom = gold_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    losses = -(log_probs * (gold_f / denom)).sum(-1)
    head_hit = hit.unsqueeze(2).expand_as(losses)
    supervised = head_hit & (~all_masked.squeeze(-1))
    if not bool(supervised.any()):
        stats = dict(empty)
        if local_weight is not None:
            stats["local_weight"] = local_weight[rows, query_pos].float().mean().detach()
        return zero, stats
    loss = losses[supervised].mean()
    stats = {
        "remote_needle_qk_mass": mass[supervised].mean().detach(),
        "n_supervised": supervised.to(dtype=q.dtype).sum().detach(),
    }
    if local_weight is None:
        stats["local_weight"] = zero.detach()
    elif local_weight.shape[:2] != (batch, seq_len):
        raise ValueError("local_weight must match [B, L, ...]")
    else:
        stats["local_weight"] = local_weight[rows, query_pos].float().mean().detach()
    return loss, stats


def prepare_evidence_token_attn(
    model: torch.nn.Module,
    query_mask: torch.Tensor,
    evidence_tokens: torch.Tensor,
) -> list[torch.nn.Module]:
    from prefilling_v2.attention import KernelDreamFullRouteAttention

    layers = [
        module
        for module in model.modules()
        if isinstance(module, KernelDreamFullRouteAttention)
    ]
    if not layers:
        raise RuntimeError("evidence token attention requires route layers")
    for layer in layers:
        layer.evidence_token_query_mask = query_mask
        layer.evidence_token_positions = evidence_tokens
        layer.evidence_token_attn_loss = None
        layer._last_remote_needle_qk_mass = None
        layer._last_evidence_local_weight = None
    return layers


def collect_evidence_token_attn_loss(
    layers: list[torch.nn.Module],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = [layer.evidence_token_attn_loss for layer in layers]
    if any(loss is None for loss in losses):
        raise RuntimeError("missing evidence token attention loss from a route layer")
    masses = []
    locals_ = []
    for layer in layers:
        mass = getattr(layer, "_last_remote_needle_qk_mass", None)
        local = getattr(layer, "_last_evidence_local_weight", None)
        if mass is not None:
            masses.append(float(mass.detach()))
        if local is not None:
            locals_.append(float(local.detach()))
    stats = {
        "remote_needle_qk_mass": sum(masses) / max(len(masses), 1),
        "local_weight": sum(locals_) / max(len(locals_), 1),
    }
    return torch.stack(losses).mean(), stats
