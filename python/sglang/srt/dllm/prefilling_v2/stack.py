"""The Prefilling-dLLM v2 Dream transformer stack: one full forward, then cached partial forwards.

Seven of Dream's 28 layers are route layers: a query reads a bidirectional
local window plus the ``route_top_k`` chunks it scores highest from per-chunk
summaries. The other layers read a wider bidirectional window only. The first
forward of a block covers the whole sequence and keeps each layer's keys,
values and chunk summaries; later forwards recompute only the positions being
denoised against that cache.

Everything here takes one sequence at a time: the route kernels are written
for a batch of one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.dllm.prefilling_v2.layout import PrefillingV2Block
from sglang.srt.dllm.prefilling_v2.route_weights import route_weights_torch
from sglang.srt.dllm.prefilling_v2.weights import (
    LayerWeights,
    LowRank,
    PrefillingV2Config,
    PrefillingV2Weights,
)

# Rows per matrix product in the MLP, to bound its intermediate at long lengths.
_MLP_ROW_BLOCK = 16384
_ROUTE_QUERY_NORM_EPS = 1e-6


@dataclass
class LayerCache:
    key: torch.Tensor  # [1, L, kv_heads, head_dim], rotary applied
    value: torch.Tensor  # [1, L, kv_heads, head_dim]
    # Route layers only.
    summary_keys: Optional[torch.Tensor] = None  # [1, C, kv_heads, group, head_dim]
    prior: Optional[torch.Tensor] = None  # [1, C, kv_heads, group]


@dataclass
class PrefillingV2Cache:
    layers: List[LayerCache]
    key_valid: torch.Tensor  # [1, L]
    cos: torch.Tensor  # [1, L, head_dim]
    sin: torch.Tensor  # [1, L, head_dim]


def yarn_inverse_frequencies(
    head_dim: int, theta: float, factor: float, original_max_positions: int
) -> Tuple[torch.Tensor, float]:
    """YaRN frequencies and attention factor, as transformers computes them."""

    beta_fast, beta_slow = 32, 1

    def correction_dim(rotations: float) -> float:
        return (
            head_dim * math.log(original_max_positions / (rotations * 2 * math.pi))
        ) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    ramp = torch.clamp(
        (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    extrapolation_factor = 1 - ramp
    inv_freq = (
        interpolation * (1 - extrapolation_factor)
        + extrapolation * extrapolation_factor
    )
    attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return inv_freq, attention_factor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _invert_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) - (_rotate_half(x) * sin)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Dream normalizes in float32 and multiplies by the weight after the cast back.
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def _window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, radius: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional attention within ``radius``; returns output and log-sum-exp."""

    from flash_attn.cute.interface import flash_attn_func

    output, lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size=(int(radius), int(radius)),
        return_lse=True,
    )
    if lse.shape[1] == q.shape[2] and lse.shape[2] == q.shape[1]:
        lse = lse.transpose(1, 2)
    elif not (lse.shape[1] == q.shape[1] and lse.shape[2] == q.shape[2]):
        raise RuntimeError(
            f"window attention log-sum-exp {tuple(lse.shape)} does not match "
            f"queries {tuple(q.shape)}"
        )
    return output, lse.contiguous()


def _window_keys(
    key_valid: torch.Tensor, radius: int, chunk_size: int, zero_routing_slots: bool
) -> Tuple[torch.Tensor, int]:
    """Which keys a local window keeps, and its radius.

    Route layers zero the keys and values of routing slots and widen the
    window by one chunk less a slot; sliding layers do neither. A dropped key
    is zeroed, not masked: it keeps a logit of zero.
    """

    valid = key_valid.to(dtype=torch.bool)
    if not zero_routing_slots:
        return valid, int(radius)
    positions = torch.arange(valid.shape[-1], device=valid.device)
    return valid & ((positions + 1) % chunk_size != 0), int(radius + chunk_size - 1)


def _zero_dropped(k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor):
    if bool(valid.all()):
        return k, v
    scale = valid.to(dtype=v.dtype).unsqueeze(-1).unsqueeze(-1)
    return k * scale, v * scale


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: torch.Tensor,
    radius: int,
    chunk_size: int,
    zero_routing_slots: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid, radius = _window_keys(key_valid, radius, chunk_size, zero_routing_slots)
    k, v = _zero_dropped(k, v, valid)
    return _window_attention(q, k, v, radius)


def cached_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: torch.Tensor,
    query_positions: torch.Tensor,
    radius: int,
    chunk_size: int,
    zero_routing_slots: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local attention of a few query rows against cached keys and values.

    The queries are scattered into the span of positions they cover so the
    same windowed kernel serves them; rows between them are zero and unused.
    """

    valid, radius = _window_keys(key_valid, radius, chunk_size, zero_routing_slots)
    seq_len = k.shape[1]
    lo = max(0, int(query_positions.min()) - radius)
    hi = min(seq_len, int(query_positions.max()) + radius + 1)
    k_span, v_span = _zero_dropped(k[:, lo:hi], v[:, lo:hi], valid[:, lo:hi])
    rows = query_positions - lo
    q_span = q.new_zeros((q.shape[0], hi - lo, q.shape[2], q.shape[3]))
    q_span[:, rows] = q
    output, lse = _window_attention(q_span, k_span, v_span, radius)
    return output[:, rows].contiguous(), lse[:, rows].contiguous()


def remote_drop_rows(
    key_valid: torch.Tensor, radius: int, chunk_size: int, positions: torch.Tensor
) -> torch.Tensor:
    """For each query position, the chunks its route may not select (1 = drop).

    A chunk is dropped when it overlaps the query's local window or has no
    valid text key.
    """

    seq_len = key_valid.shape[1]
    num_chunks = seq_len // chunk_size
    device = key_valid.device
    all_positions = torch.arange(seq_len, device=device)
    text_key = (all_positions + 1).remainder(chunk_size) != 0
    chunk_valid = (
        (key_valid.bool() & text_key.unsqueeze(0))
        .reshape(key_valid.shape[0], num_chunks, chunk_size)
        .any(dim=-1)
    )
    left_chunk = torch.div(
        (positions - radius).clamp_min(0), chunk_size, rounding_mode="floor"
    )
    right_chunk = torch.div(
        (positions + radius).clamp_max(seq_len - 1), chunk_size, rounding_mode="floor"
    )
    chunk_ids = torch.arange(num_chunks, device=device)
    local = (chunk_ids[None, :] >= left_chunk[:, None]) & (
        chunk_ids[None, :] <= right_chunk[:, None]
    )
    return (local[None, :, :] | ~chunk_valid[:, None, :]).to(torch.int32)


def _rows_of(sorted_positions: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    rows = torch.searchsorted(sorted_positions, wanted)
    if bool((rows >= sorted_positions.numel()).any()) or not torch.equal(
        sorted_positions.index_select(0, rows), wanted
    ):
        raise ValueError("positions are missing from the query rows")
    return rows


class PrefillingV2Stack:
    def __init__(self, config: PrefillingV2Config, weights: PrefillingV2Weights):
        self.config = config
        self.weights = weights
        self.device = weights.embed.device
        self.dtype = weights.embed.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError("the Prefilling-dLLM v2 kernels require BF16 weights")
        if config.group_size != 7:
            raise ValueError(
                "the Prefilling-dLLM v2 route kernels are built for 7 query heads per KV head"
            )
        self._rotary_tables: Dict[float, Tuple[torch.Tensor, float]] = {}
        self.route_layers = frozenset(config.route_layers)
        self.q_size = config.num_heads * config.head_dim
        self.kv_size = config.num_kv_heads * config.head_dim

    # ---- shared pieces -------------------------------------------------

    def rope_factor_for(self, physical_length: int) -> float:
        """The YaRN factor a sequence of this length is read with.

        Up to the trained length it is the trained factor. Beyond, it doubles
        until the scaled context covers the sequence, which is the factor the
        length sweeps of the paper use at 32K, 64K and 128K.
        """

        cfg = self.config
        factor = cfg.rope_factor
        while factor * cfg.rope_original_max_positions < physical_length:
            factor *= 2
        return factor

    def rotary(
        self, position_ids: torch.Tensor, factor: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos and sin of shape [1, n, head_dim] in the model dtype."""

        cfg = self.config
        factor = float(cfg.rope_factor if factor is None else factor)
        if factor not in self._rotary_tables:
            inv_freq, scale = yarn_inverse_frequencies(
                cfg.head_dim, cfg.rope_theta, factor, cfg.rope_original_max_positions
            )
            self._rotary_tables[factor] = (inv_freq.to(self.device), scale)
        inv_freq, scale = self._rotary_tables[factor]
        freqs = position_ids.to(self.device).float()[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * scale).to(self.dtype)
        sin = (emb.sin() * scale).to(self.dtype)
        return cos.unsqueeze(0), sin.unsqueeze(0)

    def embed(
        self, input_ids: torch.Tensor, is_routing_slot: torch.Tensor
    ) -> torch.Tensor:
        """Token embeddings [1, n, hidden], with the role offset on routing slots."""

        hidden = F.embedding(input_ids.to(self.device), self.weights.embed)
        offset = self.weights.role_offset.to(self.dtype)
        hidden = (
            hidden + is_routing_slot.to(self.device, self.dtype).unsqueeze(-1) * offset
        )
        return hidden.unsqueeze(0)

    def _low_rank(
        self, base: torch.Tensor, x: torch.Tensor, pair: LowRank
    ) -> torch.Tensor:
        # The adapter runs in float32 and its sum with the base output is
        # rounded to the model dtype once.
        update = F.linear(F.linear(x.to(torch.float32), pair.down), pair.up)
        return (base + update * self.config.lora_scaling).to(base.dtype)

    def _project(self, layer: LayerWeights, x: torch.Tensor, cos, sin):
        """Rotated queries and keys and raw values, each [1, n, heads, head_dim]."""

        n = x.shape[1]
        cfg = self.config
        qkv = F.linear(x, layer.qkv_weight, layer.qkv_bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self._low_rank(q, x, layer.lora["q_proj"]).view(
            1, n, cfg.num_heads, cfg.head_dim
        )
        k = self._low_rank(k, x, layer.lora["k_proj"]).view(
            1, n, cfg.num_kv_heads, cfg.head_dim
        )
        v = self._low_rank(v, x, layer.lora["v_proj"]).view(
            1, n, cfg.num_kv_heads, cfg.head_dim
        )
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        return (
            _apply_rotary(q, cos, sin).contiguous(),
            _apply_rotary(k, cos, sin).contiguous(),
            v.contiguous(),
        )

    def _output(self, layer: LayerWeights, attended: torch.Tensor) -> torch.Tensor:
        attended = attended.reshape(1, attended.shape[1], self.config.hidden_size)
        return self._low_rank(
            F.linear(attended, layer.o_weight), attended, layer.lora["o_proj"]
        )

    def _route_query(
        self, layer: LayerWeights, x: torch.Tensor, q: torch.Tensor, cos, sin
    ):
        """The query that pools and scores chunks: q plus a low-rank residual.

        The residual is added before the rotary map and the sum is RMS
        normalized per head.
        """

        cfg = self.config
        rq = layer.route_query
        delta = F.linear(F.linear(x.to(torch.float32), rq.down), rq.up).to(q.dtype)
        delta = delta.view(1, x.shape[1], cfg.num_heads, cfg.head_dim)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        combined = (_invert_rotary(q, cos, sin) + delta).float()
        combined = combined * torch.rsqrt(
            combined.square().mean(dim=-1, keepdim=True) + _ROUTE_QUERY_NORM_EPS
        )
        combined = (rq.norm_weight.float() * combined).to(q.dtype)
        return _apply_rotary(combined, cos, sin).contiguous()

    def _mlp(self, layer: LayerWeights, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, x.shape[1], _MLP_ROW_BLOCK):
            gate, up = F.linear(
                x[:, start : start + _MLP_ROW_BLOCK], layer.gate_up_weight
            ).chunk(2, dim=-1)
            outputs.append(F.linear(F.silu(gate) * up, layer.down_weight))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

    def _logits(self, hidden: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        hidden = hidden[0].index_select(0, rows)
        hidden = _rms_norm(
            hidden, self.weights.final_norm_weight, self.config.rms_norm_eps
        )
        return F.linear(hidden, self.weights.lm_head)

    def _pool_summaries(self, route_q_slots, chunk_keys, chunk_valid):
        """Chunk summaries [1, n, kv_heads, group, head_dim] and their prior."""

        from sglang.srt.dllm.prefilling_v2.ops.chunk_attn_pool_gqa_tilelang import (
            chunk_attn_pool_gqa,
        )

        summaries, entropy = chunk_attn_pool_gqa(
            route_q_slots.contiguous(),
            chunk_keys.contiguous(),
            chunk_valid.contiguous(),
        )
        return summaries, entropy * self.config.entropy_prior_scale

    def _route(
        self,
        q: torch.Tensor,
        route_q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor,
        summary_keys: torch.Tensor,
        prior: torch.Tensor,
        local_output: torch.Tensor,
        local_lse: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        affine_gate: bool,
    ) -> torch.Tensor:
        """Select chunks for each query, read them, and mix in the local window."""

        from sglang.srt.dllm.prefilling_v2.ops.bidir_output_fusion_tilelang import (
            fuse_route_outputs,
        )
        from sglang.srt.dllm.prefilling_v2.routing import (
            route_topk_g7,
            selected_attention_g7_cached,
        )

        cfg = self.config
        n = q.shape[1]
        drop = remote_drop_rows(
            key_valid, cfg.route_local_radius, cfg.chunk_size, query_positions
        )
        indices, scores = route_topk_g7(
            route_q,
            summary_keys,
            local_lse.reshape(1, n, cfg.num_kv_heads, cfg.group_size),
            prior.contiguous(),
            drop,
            cfg.route_top_k,
            cfg.chunk_size,
            cfg.route_local_radius,
            training=False,
        )
        if affine_gate:
            # The full forward of the training tree normalizes through its
            # (untrained, identity) gate; the cached forward uses a softmax.
            # They agree up to rounding and each is kept as it was trained
            # and evaluated.
            remote_weights, local_weight, _ = route_weights_torch(
                scores,
                indices,
                prior.contiguous(),
                local_lse.float().contiguous(),
                temperature=1.0,
                return_gate_logit=True,
                gate_offset=torch.zeros((), dtype=torch.float32, device=q.device),
                gate_scale=torch.ones((), dtype=torch.float32, device=q.device),
            )
        else:
            remote_weights, local_weight = route_weights_torch(
                scores,
                indices,
                prior,
                local_lse.float().contiguous(),
                temperature=1.0,
            )
        remote_output = selected_attention_g7_cached(
            q, key, value, remote_weights, indices, key_valid, cfg.chunk_size
        )
        return fuse_route_outputs(
            remote_output.contiguous(),
            local_output.contiguous(),
            local_weight.contiguous(),
        )

    # ---- full forward --------------------------------------------------

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        key_valid: torch.Tensor,
        position_ids: torch.Tensor,
        routing_positions: torch.Tensor,
        logit_rows: torch.Tensor,
        *,
        rope_factor: Optional[float] = None,
        kv_buffers: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        keep_cache: bool = True,
        observer=None,
    ) -> Tuple[torch.Tensor, Optional[PrefillingV2Cache]]:
        """Forward the whole physical sequence.

        Returns the logits of ``logit_rows`` and, with ``keep_cache``, the
        per-layer cache the block's cached forwards read. ``kv_buffers``
        gives, per layer, a key and a value tensor of shape
        [seq_len, kv_heads, head_dim] to hold the cache in place of new
        tensors; the server passes views of its KV pool.
        """

        cfg = self.config
        seq_len = int(input_ids.numel())
        if seq_len % cfg.chunk_size:
            raise ValueError(
                f"sequence length {seq_len} is not a multiple of the chunk size"
            )
        num_chunks = seq_len // cfg.chunk_size
        device = self.device
        key_valid = key_valid.to(device).unsqueeze(0)
        is_routing = torch.zeros(seq_len, dtype=torch.bool, device=device)
        is_routing[routing_positions.to(device)] = True
        cos, sin = self.rotary(position_ids, rope_factor)
        hidden = self.embed(input_ids, is_routing)
        all_positions = torch.arange(seq_len, device=device)
        layer_caches: List[LayerCache] = []

        for index, layer in enumerate(self.weights.layers):
            x = _rms_norm(hidden, layer.input_norm_weight, cfg.rms_norm_eps)
            q, k, v = self._project(layer, x, cos, sin)
            if kv_buffers is not None:
                key_buffer, value_buffer = kv_buffers[index]
                key_buffer.copy_(k[0])
                value_buffer.copy_(v[0])
                k, v = key_buffer.unsqueeze(0), value_buffer.unsqueeze(0)
            if index in self.route_layers:
                local_output, local_lse = local_attention(
                    q, k, v, key_valid, cfg.route_local_radius, cfg.chunk_size, True
                )
                route_q = self._route_query(layer, x, q, cos, sin)
                summary_keys, prior = self._pool_summaries(
                    route_q[:, cfg.chunk_size - 1 :: cfg.chunk_size].reshape(
                        1, num_chunks, cfg.num_kv_heads, cfg.group_size, cfg.head_dim
                    ),
                    k.reshape(
                        1, num_chunks, cfg.chunk_size, cfg.num_kv_heads, cfg.head_dim
                    ),
                    key_valid.reshape(1, num_chunks, cfg.chunk_size),
                )
                attended = self._route(
                    q,
                    route_q,
                    k,
                    v,
                    key_valid,
                    summary_keys,
                    prior,
                    local_output,
                    local_lse,
                    all_positions,
                    affine_gate=True,
                )
                layer_caches.append(LayerCache(k, v, summary_keys, prior))
            else:
                attended, _ = local_attention(
                    q, k, v, key_valid, cfg.sliding_radius, cfg.chunk_size, False
                )
                layer_caches.append(LayerCache(k, v))
            if not keep_cache:
                layer_caches[-1] = None
            hidden = hidden + self._output(layer, attended)
            hidden = hidden + self._mlp(
                layer, _rms_norm(hidden, layer.post_norm_weight, cfg.rms_norm_eps)
            )
            if observer is not None:
                observer(index, hidden)

        logits = self._logits(hidden, logit_rows.to(device))
        if not keep_cache:
            return logits, None
        return logits, PrefillingV2Cache(layer_caches, key_valid, cos, sin)

    # ---- cached forward ------------------------------------------------

    @torch.inference_mode()
    def cached_forward(
        self,
        input_ids: torch.Tensor,
        routing_positions: torch.Tensor,
        block: PrefillingV2Block,
        cache: PrefillingV2Cache,
        *,
        observer=None,
    ) -> torch.Tensor:
        """Recompute the block's query positions against ``cache``.

        Replaces the cached keys and values of the block's answer and routing
        slots, refreshes the summaries of the chunks they belong to, and
        returns the logits of the block's predictor positions.
        """

        cfg = self.config
        device = self.device
        query_positions = block.query_positions.to(device)
        update_positions = block.kv_update_positions.to(device)
        affected_chunks = block.affected_chunks.to(device)
        update_rows = _rows_of(query_positions, update_positions)
        routing_positions = routing_positions.to(device)
        is_routing = (query_positions[:, None] == routing_positions[None, :]).any(dim=1)
        hidden = self.embed(
            input_ids.to(device).index_select(0, query_positions), is_routing
        )
        cos = cache.cos.index_select(1, query_positions)
        sin = cache.sin.index_select(1, query_positions)
        key_valid = cache.key_valid
        n = int(query_positions.numel())

        slot_positions = affected_chunks * cfg.chunk_size + (cfg.chunk_size - 1)
        slot_rows = _rows_of(query_positions, slot_positions)
        chunk_positions = (
            affected_chunks[:, None] * cfg.chunk_size
            + torch.arange(cfg.chunk_size, device=device)[None, :]
        ).reshape(-1)

        for index, layer in enumerate(self.weights.layers):
            layer_cache = cache.layers[index]
            x = _rms_norm(hidden, layer.input_norm_weight, cfg.rms_norm_eps)
            q, k, v = self._project(layer, x, cos, sin)
            layer_cache.key.index_copy_(
                1, update_positions, k.index_select(1, update_rows)
            )
            layer_cache.value.index_copy_(
                1, update_positions, v.index_select(1, update_rows)
            )
            if index in self.route_layers:
                local_output, local_lse = cached_local_attention(
                    q,
                    layer_cache.key,
                    layer_cache.value,
                    key_valid,
                    query_positions,
                    cfg.route_local_radius,
                    cfg.chunk_size,
                    True,
                )
                route_q = self._route_query(layer, x, q, cos, sin)
                summaries, prior = self._pool_summaries(
                    route_q.index_select(1, slot_rows).reshape(
                        1,
                        affected_chunks.numel(),
                        cfg.num_kv_heads,
                        cfg.group_size,
                        cfg.head_dim,
                    ),
                    layer_cache.key.index_select(1, chunk_positions).reshape(
                        1,
                        affected_chunks.numel(),
                        cfg.chunk_size,
                        cfg.num_kv_heads,
                        cfg.head_dim,
                    ),
                    key_valid.index_select(1, chunk_positions).reshape(
                        1, affected_chunks.numel(), cfg.chunk_size
                    ),
                )
                layer_cache.summary_keys.index_copy_(1, affected_chunks, summaries)
                layer_cache.prior.index_copy_(1, affected_chunks, prior)
                attended = self._route(
                    q,
                    route_q,
                    layer_cache.key,
                    layer_cache.value,
                    key_valid,
                    layer_cache.summary_keys,
                    layer_cache.prior,
                    local_output,
                    local_lse,
                    query_positions,
                    affine_gate=False,
                )
            else:
                attended, _ = cached_local_attention(
                    q,
                    layer_cache.key,
                    layer_cache.value,
                    key_valid,
                    query_positions,
                    cfg.sliding_radius,
                    cfg.chunk_size,
                    False,
                )
            hidden = hidden + self._output(
                layer, attended.reshape(1, n, cfg.hidden_size)
            )
            hidden = hidden + self._mlp(
                layer, _rms_norm(hidden, layer.post_norm_weight, cfg.rms_norm_eps)
            )
            if observer is not None:
                observer(index, hidden)

        predictor_rows = _rows_of(query_positions, block.predictor_positions.to(device))
        return self._logits(hidden, predictor_rows)
