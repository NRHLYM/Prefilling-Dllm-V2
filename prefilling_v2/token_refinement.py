"""Reference and dispatch helpers for HISA-lite remote-token refinement."""

from __future__ import annotations

import math

import torch


def candidate_token_scores_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Score every token in routed chunks, sharing choices over each GQA group.

    Returns FP32 scores with shape ``[B, Lq, Hkv, K, chunk_size]``. Invalid
    chunk slots, padding, and the final landmark slot in every chunk are
    represented by negative infinity.
    """

    if q.ndim != 4 or k.ndim != 4 or indices.ndim != 4:
        raise ValueError("q, k, and indices must be rank-four tensors")
    batch, query_len, h_q, dim = q.shape
    if k.shape[0] != batch or k.shape[-1] != dim:
        raise ValueError(
            f"incompatible q/k shapes {tuple(q.shape)} and {tuple(k.shape)}"
        )
    kv_len, h_kv = k.shape[1:3]
    if h_q % h_kv:
        raise ValueError(f"h_q={h_q} must be divisible by h_kv={h_kv}")
    if kv_len % chunk_size:
        raise ValueError(
            f"kv_len={kv_len} must be divisible by chunk_size={chunk_size}"
        )
    selected = indices.shape[-1]
    if indices.shape != (batch, query_len, h_kv, selected):
        raise ValueError(f"invalid indices shape {tuple(indices.shape)}")
    if key_valid.shape != (batch, kv_len):
        raise ValueError(f"invalid key_valid shape {tuple(key_valid.shape)}")

    groups = h_q // h_kv
    chunks = kv_len // chunk_size
    q_grouped = q.reshape(
        batch, query_len, h_kv, groups, dim
    )
    k_chunked = k.reshape(
        batch, chunks, chunk_size, h_kv, dim
    ).permute(0, 3, 1, 2, 4)
    idx = indices.permute(0, 2, 1, 3).long()
    safe_idx = idx.clamp(min=0, max=max(chunks - 1, 0))
    gather_idx = safe_idx[..., None, None].expand(
        batch,
        h_kv,
        query_len,
        selected,
        chunk_size,
        dim,
    )
    selected_k = torch.gather(
        k_chunked[:, :, None].expand(
            -1, -1, query_len, -1, -1, -1
        ),
        3,
        gather_idx,
    ).permute(0, 2, 1, 3, 4, 5)

    scores_by_head = torch.einsum(
        "blhgd,blhksd->blhkgs",
        q_grouped.float(),
        selected_k.float(),
    )
    scores = scores_by_head.amax(dim=-2) / math.sqrt(dim)

    valid_chunk = (indices >= 0) & (indices < chunks)
    valid_chunks = key_valid.bool().reshape(batch, chunks, chunk_size)
    valid_idx = safe_idx[..., None].expand(
        batch, h_kv, query_len, selected, chunk_size
    )
    selected_valid = torch.gather(
        valid_chunks[:, None, None].expand(
            -1, h_kv, query_len, -1, -1
        ),
        3,
        valid_idx,
    ).permute(0, 2, 1, 3, 4)
    selected_valid = selected_valid & valid_chunk[..., None]
    selected_valid[..., -1] = False
    return scores.masked_fill(~selected_valid, float("-inf"))


def _scores_with_gumbel(
    scores: torch.Tensor,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> torch.Tensor:
    if gumbel_scale < 0:
        raise ValueError(f"gumbel_scale must be non-negative, got {gumbel_scale}")
    if not (bool(training) and bool(use_gumbel) and gumbel_scale > 0):
        return scores
    eps = torch.finfo(torch.float32).eps
    uniform = torch.empty_like(scores, dtype=torch.float32).uniform_(eps, 1.0 - eps)
    noise = -torch.log(-torch.log(uniform)) * float(gumbel_scale)
    noisy = scores.float() + noise
    return noisy.masked_fill(~torch.isfinite(scores), float("-inf"))


def select_global_token_keep(
    scores: torch.Tensor,
    token_budget: int,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> torch.Tensor:
    """Return a byte keep mask for the global top budget over chunk/token slots."""

    if scores.ndim != 5:
        raise ValueError(
            "scores must have shape [B,L,Hkv,K,S], got "
            f"{tuple(scores.shape)}"
        )
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    ranking_scores = _scores_with_gumbel(
        scores,
        training=training,
        use_gumbel=use_gumbel,
        gumbel_scale=gumbel_scale,
    )
    flat = ranking_scores.reshape(*scores.shape[:3], -1)
    budget = min(token_budget, flat.shape[-1])
    top_values, top_indices = torch.topk(
        flat, k=budget, dim=-1, largest=True, sorted=False
    )
    selected = torch.isfinite(top_values)
    keep_flat = torch.zeros_like(flat, dtype=torch.uint8)
    keep_flat.scatter_(
        -1,
        top_indices,
        selected.to(dtype=torch.uint8),
    )
    return keep_flat.reshape(scores.shape)


def allocate_entropy_token_budget(
    entropy: torch.Tensor,
    capacity: torch.Tensor,
    token_budget: int,
    min_tokens_per_chunk: int = 1,
) -> torch.Tensor:
    """Apportion a fixed token budget using entropy effective support.

    ``exp(entropy)`` is the effective number of attended tokens in a chunk.
    Higher-entropy chunks therefore receive more slots, while every non-empty
    routed chunk keeps at least ``min_tokens_per_chunk`` when the budget allows.
    """

    if entropy.shape != capacity.shape:
        raise ValueError(
            f"entropy and capacity shapes differ: {tuple(entropy.shape)} "
            f"and {tuple(capacity.shape)}"
        )
    if entropy.ndim < 1:
        raise ValueError("entropy and capacity must have a chunk dimension")
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if min_tokens_per_chunk < 0:
        raise ValueError(
            "min_tokens_per_chunk must be non-negative, got "
            f"{min_tokens_per_chunk}"
        )
    if capacity.device.type == "cpu" and torch.any(capacity < 0):
        raise ValueError("capacity must be non-negative")

    capacity = capacity.to(dtype=torch.long)
    chunk_count = capacity.shape[-1]
    if token_budget < chunk_count * min_tokens_per_chunk:
        raise ValueError(
            "token_budget must cover the per-chunk minimum for every "
            f"candidate: {token_budget} < "
            f"{chunk_count * min_tokens_per_chunk}"
        )

    target = capacity.sum(dim=-1).clamp_max(token_budget)
    baseline = torch.minimum(
        capacity,
        torch.full_like(capacity, min_tokens_per_chunk),
    )
    allocation = baseline.clone()
    remaining = target - allocation.sum(dim=-1)

    max_entropy = math.log(max(token_budget, 1))
    support = torch.exp(
        torch.nan_to_num(
            entropy.float(),
            nan=0.0,
            posinf=max_entropy,
            neginf=0.0,
        ).clamp(min=0.0, max=max_entropy)
    )
    support = support.masked_fill(capacity <= 0, 0.0)
    rank = torch.arange(
        chunk_count, device=capacity.device
    ).view(*([1] * (capacity.ndim - 1)), chunk_count)

    # Capped largest-remainder apportionment. At least one slot is assigned
    # on every non-terminal iteration, so K+1 rounds cover all cap changes.
    for _ in range(chunk_count + 1):
        active = allocation < capacity
        active_support = support * active
        denominator = active_support.sum(dim=-1, keepdim=True)
        quota = torch.where(
            denominator > 0,
            remaining[..., None].float()
            * active_support
            / denominator.clamp_min(1e-12),
            torch.zeros_like(active_support),
        )
        floor_add = torch.minimum(
            torch.floor(quota).to(torch.long),
            capacity - allocation,
        )
        allocation = allocation + floor_add
        remaining = target - allocation.sum(dim=-1)

        active = allocation < capacity
        active_count = active.sum(dim=-1)
        bonus_count = torch.minimum(remaining, active_count)
        fractional = (quota - torch.floor(quota)).masked_fill(
            ~active, float("-inf")
        )
        order = torch.argsort(
            fractional, dim=-1, descending=True, stable=True
        )
        bonus_by_rank = rank < bonus_count[..., None]
        bonus = torch.zeros_like(allocation)
        bonus.scatter_(-1, order, bonus_by_rank.to(torch.long))
        allocation = torch.minimum(allocation + bonus, capacity)
        remaining = target - allocation.sum(dim=-1)

    if capacity.device.type == "cpu":
        if torch.any(allocation > capacity):
            raise RuntimeError("entropy allocation exceeded chunk capacity")
        if torch.any(allocation.sum(dim=-1) != target):
            raise RuntimeError("entropy allocation failed to exhaust the budget")
    return allocation


def gather_selected_chunk_entropy(
    entropy: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather and group-average entropy for routed chunk indices."""

    if entropy.ndim != 4 or indices.ndim != 4:
        raise ValueError(
            "expected entropy [B,C,Hkv,G] and indices [B,L,Hkv,K]"
        )
    batch, chunks, h_kv, groups = entropy.shape
    if indices.shape[0] != batch or indices.shape[2] != h_kv:
        raise ValueError(
            f"incompatible entropy/indices shapes {tuple(entropy.shape)} "
            f"and {tuple(indices.shape)}"
        )
    query_len, selected = indices.shape[1], indices.shape[-1]
    idx = indices.permute(0, 2, 1, 3).long()
    safe_idx = idx.clamp(min=0, max=max(chunks - 1, 0))
    source = entropy.permute(0, 2, 1, 3)
    gathered = torch.gather(
        source[:, :, None].expand(
            batch, h_kv, query_len, chunks, groups
        ),
        3,
        safe_idx[..., None].expand(
            batch, h_kv, query_len, selected, groups
        ),
    ).permute(0, 2, 1, 3, 4)
    gathered = gathered.mean(dim=-1)
    return gathered.masked_fill(indices < 0, 0.0)


def select_entropy_adaptive_token_keep(
    scores: torch.Tensor,
    selected_entropy: torch.Tensor,
    token_budget: int,
    min_tokens_per_chunk: int = 1,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select per-chunk QK winners under entropy-adaptive integer quotas."""

    if scores.ndim != 5:
        raise ValueError(
            "scores must have shape [B,L,Hkv,K,S], got "
            f"{tuple(scores.shape)}"
        )
    if selected_entropy.shape != scores.shape[:-1]:
        raise ValueError(
            f"selected_entropy shape {tuple(selected_entropy.shape)} must "
            f"equal {tuple(scores.shape[:-1])}"
        )
    capacity = torch.isfinite(scores).sum(dim=-1)
    allocation = allocate_entropy_token_budget(
        selected_entropy,
        capacity,
        token_budget,
        min_tokens_per_chunk,
    )
    ranking_scores = _scores_with_gumbel(
        scores,
        training=training,
        use_gumbel=use_gumbel,
        gumbel_scale=gumbel_scale,
    )
    order = torch.argsort(
        ranking_scores, dim=-1, descending=True, stable=True
    )
    rank = torch.arange(
        scores.shape[-1], device=scores.device
    ).view(*([1] * (scores.ndim - 1)), scores.shape[-1])
    ranked_keep = rank < allocation[..., None]
    keep = torch.zeros_like(scores, dtype=torch.uint8)
    keep.scatter_(-1, order, ranked_keep.to(torch.uint8))
    keep = keep.masked_fill(~torch.isfinite(scores), 0)
    return keep, allocation


def refine_remote_tokens_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    token_budget: int,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> torch.Tensor:
    scores = candidate_token_scores_reference(
        q, k, indices, key_valid, chunk_size
    )
    return select_global_token_keep(
        scores,
        token_budget,
        training=training,
        use_gumbel=use_gumbel,
        gumbel_scale=gumbel_scale,
    )


def refine_remote_tokens_g7(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    token_budget: int,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> torch.Tensor:
    """Run the compact candidate-score kernel and return an int8 keep mask."""

    if q.shape[2] != k.shape[2] * 7:
        raise ValueError(
            "Dream token refinement requires seven query heads per KV head"
        )
    from ops.token_score_gqa_tilelang import candidate_token_scores_gqa

    with torch.no_grad():
        scores = candidate_token_scores_gqa(
            q.detach().contiguous(),
            k.detach().contiguous(),
            indices.to(torch.int32).contiguous(),
            key_valid.contiguous(),
            chunk_size,
        )
        return select_global_token_keep(
            scores,
            token_budget,
            training=training,
            use_gumbel=use_gumbel,
            gumbel_scale=gumbel_scale,
        ).to(dtype=torch.int8)


def refine_remote_tokens_entropy_g7(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    entropy: torch.Tensor,
    chunk_size: int,
    token_budget: int,
    min_tokens_per_chunk: int = 1,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-score candidates and apply entropy-adaptive chunk quotas."""

    if q.shape[2] != k.shape[2] * 7:
        raise ValueError(
            "Dream token refinement requires seven query heads per KV head"
        )
    from ops.token_score_gqa_tilelang import candidate_token_scores_gqa

    with torch.no_grad():
        scores = candidate_token_scores_gqa(
            q.detach().contiguous(),
            k.detach().contiguous(),
            indices.to(torch.int32).contiguous(),
            key_valid.contiguous(),
            chunk_size,
        )
        selected_entropy = gather_selected_chunk_entropy(
            entropy, indices
        )
        keep, allocation = select_entropy_adaptive_token_keep(
            scores,
            selected_entropy,
            token_budget,
            min_tokens_per_chunk,
            training=training,
            use_gumbel=use_gumbel,
            gumbel_scale=gumbel_scale,
        )
        return keep.to(dtype=torch.int8), allocation
