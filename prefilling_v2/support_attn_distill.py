"""Dense attention distill on the student's fused support distribution.

Hard top-k, Q-Cal, and LMK are detached. KL uses this layer's content LoRA Q/K
and optional fusion weights; it is never attached to route_q.
Student:
  P(t) = w_local P_local(t) + w_remote P_remote(t)
on support = local window ∪ selected chunks. Teacher is dense softmax on the
same support, not 32-way chunk mass. QK is packed to those keys only.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _teacher_core(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def span_first_mask(flags: torch.Tensor) -> torch.Tensor:
    """Keep the first True of each contiguous run (gold-first / MASK span)."""

    previous = torch.zeros_like(flags)
    previous[:, 1:] = flags[:, :-1]
    return flags & ~previous


def last_k_mask(flags: torch.Tensor, cap: int) -> torch.Tensor:
    """Keep the last `cap` True positions (quiz/answer sits at the suffix)."""

    if cap <= 0:
        return flags
    flipped = flags.flip(-1)
    rank = flipped.long().cumsum(-1)
    return (flipped & (rank <= cap)).flip(-1)


def pack_query_positions(query_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = query_valid.sum(-1)
    if int(counts.min()) <= 0:
        raise ValueError("support attn distill needs MASK / answer-span queries")
    width = int(counts.max())
    batch, _ = query_valid.shape
    positions = query_valid.new_zeros((batch, width), dtype=torch.long)
    keep = query_valid.new_zeros((batch, width), dtype=torch.bool)
    for row, flags in enumerate(query_valid):
        ids = torch.where(flags)[0]
        positions[row, : ids.numel()] = ids
        keep[row, : ids.numel()] = True
    return positions, keep


def pack_support_indices(support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,Q,N] bool -> packed token indices [B,Q,S] and keep mask."""

    batch, n_query, _ = support.shape
    max_s = int(support.sum(-1).max().item()) if support.numel() else 0
    if max_s <= 0:
        idx = support.new_zeros((batch, n_query, 1), dtype=torch.long)
        keep = support.new_zeros((batch, n_query, 1), dtype=torch.bool)
        return idx, keep
    rank = support.long().cumsum(-1) - 1
    batch_ix, query_ix, token_ix = torch.where(support)
    idx = support.new_zeros((batch, n_query, max_s), dtype=torch.long)
    keep = support.new_zeros((batch, n_query, max_s), dtype=torch.bool)
    packed_rank = rank[batch_ix, query_ix, token_ix]
    idx[batch_ix, query_ix, packed_rank] = token_ix
    keep[batch_ix, query_ix, packed_rank] = True
    return idx, keep


def packed_token_scores(q, k, q_pos, k_idx, scale):
    """QK only on packed support keys [B,Q,S], not the full sequence."""

    batch = q_pos.shape[0]
    batch_q = torch.arange(batch, device=q.device)[:, None]
    query = q[batch_q, q_pos].float()
    batch_k = torch.arange(batch, device=k.device)[:, None, None]
    key = k[batch_k, k_idx].float()
    n_q_heads, n_kv_heads = query.shape[2], key.shape[3]
    groups = n_q_heads // n_kv_heads
    grouped = query.reshape(batch, query.shape[1], n_kv_heads, groups, query.shape[-1])
    scores = torch.einsum("bqhgd,bqshd->bqhgs", grouped, key) * scale
    return scores.reshape(batch, query.shape[1], n_q_heads, k_idx.shape[-1])


def branch_masks(
    positions: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    chunk_size: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local window and selected-chunk masks; indices are detached."""

    batch, seq_len = key_valid.shape
    n_query = positions.shape[1]
    device = key_valid.device
    batch_ix = torch.arange(batch, device=device)[:, None]
    selected = indices.detach()[batch_ix, positions]
    valid = selected >= 0
    safe = selected.clamp_min(0)
    n_chunks = seq_len // chunk_size
    chunk_hit = torch.zeros(
        batch, n_query, n_chunks, device=device, dtype=torch.bool
    )
    if bool(valid.any()):
        b_idx = torch.arange(batch, device=device)[:, None, None, None].expand_as(safe)
        q_idx = torch.arange(n_query, device=device)[None, :, None, None].expand_as(safe)
        chunk_hit[b_idx[valid], q_idx[valid], safe[valid]] = True
    remote = chunk_hit.repeat_interleave(chunk_size, dim=-1)
    tokens = torch.arange(seq_len, device=device)
    left = (positions - window).clamp_min(0)
    right = (positions + window).clamp_max(seq_len - 1)
    local = (tokens[None, None, :] >= left[:, :, None]) & (
        tokens[None, None, :] <= right[:, :, None]
    )
    landmark = torch.zeros(seq_len, dtype=torch.bool, device=device)
    landmark[chunk_size - 1 :: chunk_size] = True
    keep = key_valid[:, None, :] & ~landmark
    local = local & keep
    remote = remote & keep & ~local
    return local, remote


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = torch.finfo(scores.dtype).min
    logits = scores.masked_fill(~mask[:, :, None, :], fill)
    empty = ~mask.any(-1)
    probs = torch.softmax(logits, dim=-1)
    return probs.masked_fill(empty[:, :, None, None], 0.0)


def fused_support_kl(
    student_q: torch.Tensor,
    student_k: torch.Tensor,
    teacher_q: torch.Tensor,
    teacher_k: torch.Tensor,
    positions: torch.Tensor,
    query_keep: torch.Tensor,
    local_mask: torch.Tensor,
    remote_mask: torch.Tensor,
    local_weight: torch.Tensor,
    temperature: float,
    *,
    query_chunk: int = 16,
) -> torch.Tensor:
    del query_chunk
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    support = local_mask | remote_mask
    rows = query_keep & support.any(-1)
    if not bool(rows.any()):
        return student_q.float().sum() * 0.0
    k_idx, k_keep = pack_support_indices(support)
    local_p = local_mask.gather(-1, k_idx) & k_keep
    remote_p = remote_mask.gather(-1, k_idx) & k_keep
    scale = 1.0 / math.sqrt(student_q.shape[-1])
    student_scores = packed_token_scores(
        student_q, student_k, positions, k_idx, scale
    ) / temperature
    p_local = _masked_softmax(student_scores, local_p)
    p_remote = _masked_softmax(student_scores, remote_p)
    weight = local_weight.float().clamp(0.0, 1.0)
    has_local = local_p.any(-1)
    has_remote = remote_p.any(-1)
    mix = weight[:, :, :, None] * p_local + (1.0 - weight)[:, :, :, None] * p_remote
    mix = torch.where((has_local & ~has_remote)[:, :, None, None], p_local, mix)
    mix = torch.where((has_remote & ~has_local)[:, :, None, None], p_remote, mix)
    log_student = mix.clamp_min(1e-12).log()
    with torch.no_grad():
        teacher_scores = packed_token_scores(
            teacher_q, teacher_k, positions, k_idx, scale
        ) / temperature
        target = _masked_softmax(teacher_scores, k_keep)
    loss = F.kl_div(log_student, target, reduction="none").sum(-1)
    n_heads = student_q.shape[2]
    return (loss * rows[:, :, None]).sum() / (rows.sum() * n_heads).clamp_min(1)


def prepare_support_attn_distill(model, batch, args):
    from prefilling_v2.attention import KernelDreamFullRouteAttention

    core = _teacher_core(model)
    layers = {
        i: layer.self_attn
        for i, layer in enumerate(core.model.layers)
        if isinstance(layer.self_attn, KernelDreamFullRouteAttention)
    }
    missing = [
        i
        for i, attn in layers.items()
        if not hasattr(attn, "frozen_dense_q_weight")
    ]
    if missing:
        raise RuntimeError(
            "support attn distill requires snapshot_frozen_dense_qk; "
            f"missing on layers {missing[:8]}"
        )
    valid = batch["attention_mask"].bool().clone()
    chunk_size = int(args.chunk_size)
    valid[:, chunk_size - 1 :: chunk_size] = False
    mask_id = int(core.config.mask_token_id)
    masked = valid & batch["input_ids"].eq(mask_id)
    labeled = valid & batch["labels"].ne(-100)
    query_valid = last_k_mask(
        span_first_mask(masked | labeled),
        int(getattr(args, "route_support_attn_queries", 16) or 0),
    )
    positions, keep = pack_query_positions(query_valid)
    temperature = float(getattr(args, "route_support_attn_temperature", 1.5))
    query_chunk = int(getattr(args, "route_support_attn_query_chunk", 16))
    detach_gate = bool(getattr(args, "route_support_attn_detach_gate", False))
    for layer in layers.values():
        layer.support_attn_query_positions = positions
        layer.support_attn_query_keep = keep
        layer.support_attn_temperature = temperature
        layer.support_attn_query_chunk = query_chunk
        layer.support_attn_detach_gate = detach_gate
        layer.support_attn_kl = None
    return list(layers.values())


def attach_support_attn_kl(
    layer,
    hidden_states: torch.Tensor,
    student_q: torch.Tensor,
    student_k: torch.Tensor,
    key_valid: torch.Tensor,
    indices: torch.Tensor,
    selected_scores: torch.Tensor,
    prior_bias: torch.Tensor,
    local_lse: torch.Tensor,
    position_ids,
    position_embeddings,
) -> torch.Tensor:
    from prefilling_v2.attention import _route_weights_torch

    positions = layer.support_attn_query_positions
    keep = layer.support_attn_query_keep
    teacher_q, teacher_k = layer._project_qk_blhd_frozen_dense(
        hidden_states, position_ids, position_embeddings
    )
    local_mask, remote_mask = branch_masks(
        positions,
        indices,
        key_valid,
        chunk_size=int(layer.chunk_size),
        window=int(layer.local_window),
    )
    lse = local_lse.float().contiguous()
    if bool(getattr(layer, "support_attn_detach_gate", False)):
        lse = lse.detach()
    _, local_weight = _route_weights_torch(
        selected_scores.detach(),
        indices.detach(),
        prior_bias.detach().contiguous(),
        lse,
        temperature=float(layer.route_temperature),
    )
    batch_ix = torch.arange(positions.shape[0], device=positions.device)[:, None]
    gathered = local_weight[batch_ix, positions]
    if bool(getattr(layer, "support_attn_detach_gate", False)):
        gathered = gathered.detach()
    loss = fused_support_kl(
        student_q,
        student_k,
        teacher_q,
        teacher_k,
        positions,
        keep,
        local_mask,
        remote_mask,
        gathered,
        float(getattr(layer, "support_attn_temperature", 1.5)),
        query_chunk=int(getattr(layer, "support_attn_query_chunk", 16)),
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite support attention KL")
    return loss


def collect_support_attn_kl(layers):
    losses = [layer.support_attn_kl for layer in layers]
    if any(loss is None for loss in losses):
        raise RuntimeError("missing support attention KL from a route layer")
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite support attention KL")
    return loss
