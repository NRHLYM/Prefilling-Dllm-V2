"""Carrying a prefilled Prefilling-dLLM v2 request to another server in KV pool slots.

With prefill-decode disaggregation the prefill server runs a request's first
full forward and the decode server runs the cached forwards. SGLang's KV
sender moves the pool rows of a request, so everything the decode server
needs is written into rows of the request:

- rows ``[0, L)``: the keys and values of the ``L`` physical positions;
- rows ``[L, L + group * C)``: for each route layer, the ``C`` chunk
  summaries in that layer's key rows (one row per query head of a group) and
  the chunk priors, as raw float32 bits, in its first ``C`` value rows;
- the first of those rows in the key buffer of a sliding layer: the first
  block's answer tokens after the full forward, as raw int32.

Rows are bit patterns of BF16 tensors and travel unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.srt.dllm.prefilling_v2.stack import LayerCache, PrefillingV2Cache
from sglang.srt.dllm.prefilling_v2.weights import PrefillingV2Config

KVBuffers = List[Tuple[torch.Tensor, torch.Tensor]]


def side_slots(physical_length: int, chunk_size: int, group_size: int) -> int:
    """Rows a request needs beyond its physical positions."""

    return (physical_length // chunk_size) * group_size


def contiguous_views(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    slots: torch.Tensor,
    length: int,
) -> Optional[KVBuffers]:
    """Per-layer views over ``slots[:length]`` when they are one ascending run."""

    if slots.numel() < length:
        return None
    start = int(slots[0])
    run = torch.arange(start, start + length, dtype=slots.dtype, device=slots.device)
    if not torch.equal(slots[:length], run):
        return None
    stop = start + length
    return [(key[start:stop], value[start:stop]) for key, value in zip(keys, values)]


def _state_layer(config: PrefillingV2Config) -> int:
    return next(i for i in range(config.num_layers) if i not in config.route_layers)


def _as_rows(bits: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Pack int16 ``bits`` [n, m] into ``n`` zero-padded pool rows."""

    rows = torch.zeros(
        (bits.shape[0], like[0].numel()), dtype=torch.int16, device=like.device
    )
    if bits.shape[1] > rows.shape[1]:
        raise ValueError("handoff state does not fit a KV pool row")
    rows[:, : bits.shape[1]] = bits
    return rows.view(like.dtype).reshape(bits.shape[0], *like.shape[1:])


def _row_bits(rows: torch.Tensor, count: int) -> torch.Tensor:
    return rows.reshape(rows.shape[0], -1).view(torch.int16)[:, :count].contiguous()


def write_handoff(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    slots: torch.Tensor,
    cache: PrefillingV2Cache,
    answer_ids: torch.Tensor,
    config: PrefillingV2Config,
) -> None:
    """Write a block's cache and partly denoised answer into the request's rows."""

    length = cache.key_valid.shape[1]
    num_chunks = length // config.chunk_size
    group = config.group_size
    if slots.numel() < length + num_chunks * group:
        raise ValueError("the request has too few KV slots for a handoff")
    slots = slots.to(dtype=torch.long)
    position_slots = slots[:length]
    summary_slots = slots[length : length + num_chunks * group]
    for index, layer in enumerate(cache.layers):
        if layer.key.data_ptr() != keys[index][int(position_slots[0])].data_ptr():
            # The cache was kept outside the pool.
            keys[index].index_copy_(0, position_slots, layer.key[0])
            values[index].index_copy_(0, position_slots, layer.value[0])
        if layer.summary_keys is None:
            continue
        keys[index].index_copy_(
            0,
            summary_slots,
            layer.summary_keys[0]
            .permute(0, 2, 1, 3)
            .reshape(num_chunks * group, config.num_kv_heads, config.head_dim),
        )
        prior_bits = (
            layer.prior[0]
            .reshape(num_chunks, -1)
            .float()
            .contiguous()
            .view(torch.int16)
        )
        values[index].index_copy_(
            0, summary_slots[:num_chunks], _as_rows(prior_bits, values[index])
        )
    state = torch.cat((answer_ids.new_tensor([answer_ids.numel()]), answer_ids)).to(
        torch.int32
    )
    state_layer = _state_layer(config)
    keys[state_layer].index_copy_(
        0,
        summary_slots[:1],
        _as_rows(state.view(torch.int16).unsqueeze(0), keys[state_layer]),
    )


def read_handoff(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    slots: torch.Tensor,
    length: int,
    num_answer_ids: int,
    key_valid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: PrefillingV2Config,
) -> Optional[Tuple[PrefillingV2Cache, torch.Tensor]]:
    """Rebuild the cache and the answer tokens ``write_handoff`` stored.

    Returns None when the rows do not hold a handoff of this request.
    """

    num_chunks = length // config.chunk_size
    group = config.group_size
    if slots.numel() < length + num_chunks * group:
        return None
    slots = slots.to(dtype=torch.long)
    position_slots = slots[:length]
    summary_slots = slots[length : length + num_chunks * group]

    state_layer = _state_layer(config)
    state = _row_bits(
        keys[state_layer].index_select(0, summary_slots[:1]), 2 * (num_answer_ids + 1)
    ).view(torch.int32)[0]
    if int(state[0]) != num_answer_ids:
        return None
    answer_ids = state[1:].to(torch.long)

    views = contiguous_views(keys, values, position_slots, length)
    layers = []
    for index in range(config.num_layers):
        if views is not None:
            key, value = views[index]
        else:
            key = keys[index].index_select(0, position_slots)
            value = values[index].index_select(0, position_slots)
        layer = LayerCache(key.unsqueeze(0), value.unsqueeze(0))
        if index in config.route_layers:
            layer.summary_keys = (
                keys[index]
                .index_select(0, summary_slots)
                .reshape(num_chunks, group, config.num_kv_heads, config.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
                .unsqueeze(0)
            )
            prior_bits = _row_bits(
                values[index].index_select(0, summary_slots[:num_chunks]),
                2 * config.num_kv_heads * group,
            )
            layer.prior = (
                prior_bits.view(torch.float32)
                .reshape(1, num_chunks, config.num_kv_heads, group)
                .contiguous()
            )
        layers.append(layer)
    return PrefillingV2Cache(layers, key_valid, cos, sin), answer_ids
