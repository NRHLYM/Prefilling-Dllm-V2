"""Cache-aware partial attention for Dream+route Fast-dLLM inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prefilling_v2.attention import (
    KernelDreamFullRouteAttention,
    KernelDreamSlidingWindowAttention,
)
from prefilling_v2.fastdllm_cache import LayerKVCache
from prefilling_v2.dsa_attention import DreamDsaAttention
from prefilling_v2.nsa_attention import DreamNsaAttention
from prefilling_v2.kernel_utils import remote_drop_mask
from prefilling_v2.local_attention import cached_local_attention
from prefilling_v2.routing import (
    route_topk_g7,
    selected_attention_g7_cached,
)
from prefilling_v2.token_refinement import (
    refine_remote_tokens_entropy_g7,
    refine_remote_tokens_g7,
)


@dataclass(frozen=True)
class CachedAttentionStats:
    updated_positions: torch.Tensor
    refreshed_chunks: torch.Tensor
    routing_calls: int


def _validate_positions(
    positions: torch.Tensor,
    *,
    name: str,
    upper_bound: int,
    device: torch.device,
) -> torch.Tensor:
    positions = positions.to(device=device, dtype=torch.long)
    if positions.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError(f"{name} must be unique")
    if positions.numel() and bool(
        ((positions < 0) | (positions >= upper_bound)).any()
    ):
        raise ValueError(f"{name} must be in range")
    return positions


def _rows_for_positions(
    query_positions: torch.Tensor,
    owned_positions: torch.Tensor,
) -> torch.Tensor:
    if owned_positions.numel() == 0:
        return owned_positions
    rows = torch.searchsorted(query_positions, owned_positions)
    if bool((rows >= query_positions.numel()).any()) or not torch.equal(
        query_positions.index_select(0, rows),
        owned_positions,
    ):
        raise ValueError("all owned positions must also be query positions")
    return rows


@torch.inference_mode()
def cached_attention_forward(
    attention: KernelDreamSlidingWindowAttention
    | KernelDreamFullRouteAttention | DreamDsaAttention | DreamNsaAttention,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    query_positions: torch.Tensor,
    kv_update_positions: torch.Tensor,
    affected_chunks: torch.Tensor,
    cache: LayerKVCache,
) -> tuple[torch.Tensor, CachedAttentionStats]:
    """Run partial queries against a full layer cache and update owned K/V."""

    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("cached attention requires hidden states [1,M,D]")
    query_len = hidden_states.shape[1]
    query_positions = _validate_positions(
        query_positions,
        name="query_positions",
        upper_bound=cache.physical_length,
        device=hidden_states.device,
    )
    if query_positions.numel() != query_len:
        raise ValueError("query position count must match partial hidden length")
    if query_len > 1 and bool(
        (query_positions[1:] <= query_positions[:-1]).any()
    ):
        raise ValueError("query_positions must be sorted")
    kv_update_positions = _validate_positions(
        kv_update_positions,
        name="kv_update_positions",
        upper_bound=cache.physical_length,
        device=hidden_states.device,
    )
    update_rows = _rows_for_positions(
        query_positions,
        kv_update_positions,
    )
    chunk_size = getattr(attention, "chunk_size", None)
    chunk_count = (
        cache.physical_length // int(chunk_size)
        if chunk_size and int(chunk_size) > 1
        else 1
    )
    affected_chunks = _validate_positions(
        affected_chunks,
        name="affected_chunks",
        upper_bound=chunk_count,
        device=hidden_states.device,
    )

    if isinstance(attention, DreamNsaAttention):
        q_rope, q_raw, k_rope, k_raw, partial_v = attention._project_qkv_nsa(
            hidden_states,
            position_ids,
            position_embeddings,
        )
        cache.replace(
            kv_update_positions,
            k_rope.index_select(1, update_rows),
            partial_v.index_select(1, update_rows),
        )
        if cache.nsa_raw_key is None:
            raise ValueError("NSA cache is missing non-RoPE keys")
        cache.nsa_raw_key.index_copy_(
            1,
            kv_update_positions,
            k_raw.index_select(1, update_rows),
        )
        if affected_chunks.numel():
            raise ValueError("NSA layers do not cache landmark summaries")
        output = attention._nsa_output(
            hidden_states,
            q_rope,
            q_raw,
            cache.key,
            cache.nsa_raw_key,
            cache.value,
            cache.key_valid,
            allowed=None,
            query_positions=query_positions,
        ).reshape(1, query_len, attention.hidden_size)
        return attention.o_proj(output), CachedAttentionStats(
            updated_positions=kv_update_positions.clone(),
            refreshed_chunks=affected_chunks.clone(),
            routing_calls=1,
        )

    q, partial_k, partial_v = attention._project_qkv_blhd(
        hidden_states,
        position_ids,
        position_embeddings,
    )
    cache.replace(
        kv_update_positions,
        partial_k.index_select(1, update_rows),
        partial_v.index_select(1, update_rows),
    )
    if isinstance(attention, DreamDsaAttention):
        if affected_chunks.numel():
            raise ValueError("DSA layers do not have landmark summaries")
        if cache.dsa_index_keys is None:
            raise ValueError("DSA cache is missing indexer keys")
        index_q, index_k, index_w = attention.dsa_indexer._project(
            hidden_states, position_embeddings
        )
        cache.dsa_index_keys.index_copy_(
            1, kv_update_positions, index_k.index_select(1, update_rows)
        )
        outputs = []
        block_size = attention.dsa_indexer.query_block_size
        for start in range(0, query_len, block_size):
            end = min(start + block_size, query_len)
            scores = attention.dsa_indexer._score(
                index_q[:, start:end], cache.dsa_index_keys, index_w[:, start:end]
            )
            scores.masked_fill_(~cache.key_valid[:, None, :], float("-inf"))
            values, indices = torch.topk(
                scores, min(attention.topk, cache.physical_length),
                dim=-1, sorted=False,
            )
            valid = torch.isfinite(values)
            outputs.append(attention._selected_attention(
                q[:, start:end].clone(memory_format=torch.contiguous_format), cache.key, cache.value,
                indices.masked_fill(~valid, 0), valid,
            ))
        output = torch.cat(outputs, dim=1).reshape(1, query_len, attention.hidden_size)
        return attention.o_proj(output), CachedAttentionStats(
            updated_positions=kv_update_positions.clone(),
            refreshed_chunks=affected_chunks.clone(), routing_calls=1,
        )

    local_output, local_lse = cached_local_attention(
        q,
        cache.key,
        cache.value,
        cache.key_valid,
        query_positions,
        attention.local_window,
        int(attention.chunk_size),
        skip_inert_slots=getattr(attention, "skip_inert_slots", True),
    )

    if isinstance(attention, KernelDreamSlidingWindowAttention):
        if affected_chunks.numel():
            raise ValueError("sliding-window layers cannot refresh landmarks")
        output = local_output.reshape(1, query_len, attention.hidden_size)
        return attention.o_proj(output), CachedAttentionStats(
            updated_positions=kv_update_positions.clone(),
            refreshed_chunks=affected_chunks.clone(),
            routing_calls=0,
        )
    if not isinstance(attention, KernelDreamFullRouteAttention):
        raise TypeError(
            f"unsupported cached attention type {type(attention).__name__}"
        )
    if cache.landmark_keys is None or cache.prior_bias is None:
        raise ValueError("route layer cache is missing landmark summaries")

    from ops.chunk_attn_pool_gqa_tilelang import chunk_attn_pool_gqa
    from ops.bidir_output_fusion_tilelang import fuse_route_outputs
    from ops.bidir_route_weights_tilelang import route_weights

    route_q = attention._calibrated_query(hidden_states, q, position_ids, position_embeddings)
    chunk_size = int(attention.chunk_size)
    if affected_chunks.numel():
        landmark_positions = (
            affected_chunks * chunk_size + (chunk_size - 1)
        )
        landmark_rows = _rows_for_positions(
            query_positions,
            landmark_positions,
        )
        q_lmk = route_q.index_select(1, landmark_rows).reshape(
            1,
            affected_chunks.numel(),
            attention.num_key_value_heads,
            attention.num_key_value_groups,
            attention.head_dim,
        )
        chunk_offsets = torch.arange(
            chunk_size,
            device=hidden_states.device,
            dtype=torch.long,
        )
        chunk_positions = (
            affected_chunks[:, None] * chunk_size + chunk_offsets[None, :]
        )
        flat_chunk_positions = chunk_positions.reshape(-1)
        chunk_keys = cache.key.index_select(
            1, flat_chunk_positions
        ).reshape(
            1,
            affected_chunks.numel(),
            chunk_size,
            attention.num_key_value_heads,
            attention.head_dim,
        )
        chunk_valid = cache.key_valid.index_select(
            1, flat_chunk_positions
        ).reshape(1, affected_chunks.numel(), chunk_size)
        refreshed_landmarks, entropy = chunk_attn_pool_gqa(
            q_lmk.contiguous(),
            chunk_keys.contiguous(),
            chunk_valid.contiguous(),
        )
        refreshed_prior = entropy * attention.entropy_bias_scale.float().view(
            1,
            1,
            attention.num_key_value_heads,
            attention.num_key_value_groups,
        )
        cache.landmark_keys.index_copy_(
            1,
            affected_chunks,
            refreshed_landmarks,
        )
        cache.prior_bias.index_copy_(
            1,
            affected_chunks,
            refreshed_prior,
        )

    drop_mask = remote_drop_mask(
        cache.key_valid,
        attention.local_window,
        chunk_size,
    ).index_select(1, query_positions)
    indices, selected_scores = route_topk_g7(
        route_q,
        cache.landmark_keys,
        local_lse.reshape(
            1,
            query_len,
            attention.num_key_value_heads,
            attention.num_key_value_groups,
        ),
        cache.prior_bias,
        drop_mask,
        attention.topk,
        chunk_size,
        attention.local_window,
        training=False,
    )
    if attention.topk >= 32:
        from prefilling_v2.attention import _route_weights_torch
        remote_weights, local_weight = _route_weights_torch(
            selected_scores,
            indices,
            cache.prior_bias,
            local_lse.float().contiguous(),
            temperature=1.0,
        )
    else:
        remote_weights, local_weight = route_weights(
            selected_scores,
            indices,
            cache.prior_bias,
            local_lse.float().contiguous(),
        )
    token_keep = None
    if attention.token_budget > 0:
        if attention.token_policy == "global_qk":
            token_keep = refine_remote_tokens_g7(
                q,
                cache.key,
                indices,
                cache.key_valid,
                chunk_size,
                attention.token_budget,
                training=False,
            )
        else:
            if cache.prior_bias is None:
                raise ValueError("entropy token policy requires cached prior bias")
            entropy = cache.prior_bias / attention.entropy_bias_scale.float().view(
                1,
                1,
                attention.num_key_value_heads,
                attention.num_key_value_groups,
            )
            token_keep, _ = refine_remote_tokens_entropy_g7(
                q,
                cache.key,
                indices,
                cache.key_valid,
                entropy,
                chunk_size,
                attention.token_budget,
                attention.min_tokens_per_chunk,
                training=False,
            )
    remote_output = selected_attention_g7_cached(
        q,
        cache.key,
        cache.value,
        remote_weights,
        indices,
        cache.key_valid,
        chunk_size,
        token_keep=token_keep,
    )
    output = fuse_route_outputs(
        remote_output.contiguous(),
        local_output.contiguous(),
        local_weight.contiguous(),
    ).reshape(1, query_len, attention.hidden_size)
    return attention.o_proj(output), CachedAttentionStats(
        updated_positions=kv_update_positions.clone(),
        refreshed_chunks=affected_chunks.clone(),
        routing_calls=1,
    )
