"""Dream attention adapters for the route layers.

This module is intentionally a correctness-first PyTorch implementation. It
keeps Dream's projections and RoPE behavior, then replaces dense bidirectional
attention with either bidirectional sliding-window attention or a route-style
local-window plus per-query top-k chunk attention. It is useful for smoke tests
and LoRA experiments before wiring the optimized TileLang/FlashAttention kernels.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from prefilling_v2.fastdllm_cache import LayerKVCache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q heads, as in Dream/Llama GQA."""

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def invert_rotary_pos_emb(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) - (rotate_half(q) * sin)


@dataclass(frozen=True)
class SparseLayerPlan:
    route_layers: List[int]
    sliding_window_layers: List[int]
    dense_layers: List[int]


_ROUTE_RELAXATIONS = {
    "none",
    "gumbel_topk",
    "gumbel_softmax_topk",
    "gumbel_softmax_topk_st",
}
_TOKEN_RELAXATIONS = {"none", "gumbel_topk"}


def _validate_route_relaxation(
    route_relaxation: str,
    route_temperature: float,
    route_gumbel_scale: float,
) -> None:
    if route_relaxation not in _ROUTE_RELAXATIONS:
        raise ValueError(f"unsupported route_relaxation={route_relaxation}")
    if route_temperature <= 0:
        raise ValueError(
            f"route_temperature must be positive, got {route_temperature}"
        )
    if route_gumbel_scale < 0:
        raise ValueError(
            f"route_gumbel_scale must be non-negative, got {route_gumbel_scale}"
        )


def _validate_token_relaxation(
    token_relaxation: str,
    token_gumbel_scale: float,
) -> None:
    if token_relaxation not in _TOKEN_RELAXATIONS:
        raise ValueError(f"unsupported token_relaxation={token_relaxation}")
        if token_gumbel_scale < 0:
            raise ValueError(
                f"token_gumbel_scale must be non-negative, got {token_gumbel_scale}"
            )


def _sample_gumbel_like(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) <= 0:
        return torch.zeros(tensor.shape, device=tensor.device, dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps
    uniform = torch.empty(
        tensor.shape, device=tensor.device, dtype=torch.float32
    ).uniform_(eps, 1.0 - eps)
    return -torch.log(-torch.log(uniform)) * float(scale)


def _straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Forward equals hard; backward uses only the soft Jacobian."""
    return hard.detach() + (soft - soft.detach())


def _route_weights_torch(
    scores: torch.Tensor,
    indices: torch.Tensor,
    prior_bias: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    temperature: float,
    selected_gumbel: torch.Tensor | None = None,
    value_bonus: torch.Tensor | None = None,
    return_gate_logit: bool = False,
    gate_offset: torch.Tensor | None = None,
    gate_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize selected chunks with optional top-k-restricted Gumbel-Softmax."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if scores.ndim != 4 or indices.ndim != 4 or prior_bias.ndim != 4:
        raise ValueError("scores, indices, and prior_bias must be 4D")
    if local_lse.ndim != 3:
        raise ValueError("local_lse must be [B,L,Hq]")

    batch, seq_len, h_q, topk = scores.shape
    bias_batch, chunks, h_kv, groups = prior_bias.shape
    if chunks <= 0:
        raise ValueError("prior_bias must contain at least one chunk")
    if h_q != h_kv * groups:
        raise ValueError(
            f"invalid grouped-head layout Hq={h_q}, Hkv={h_kv}, G={groups}"
        )
    if bias_batch != batch:
        raise ValueError(f"invalid prior_bias shape {tuple(prior_bias.shape)}")
    if indices.shape != (batch, seq_len, h_kv, topk):
        raise ValueError(f"invalid indices shape {tuple(indices.shape)}")
    if local_lse.shape != (batch, seq_len, h_q):
        raise ValueError(f"invalid local_lse shape {tuple(local_lse.shape)}")
    if selected_gumbel is not None and selected_gumbel.shape != scores.shape:
        raise ValueError(
            f"invalid selected_gumbel shape {tuple(selected_gumbel.shape)}"
        )

    indices_hq = indices.to(torch.long).repeat_interleave(groups, dim=2)
    valid = (indices_hq >= 0) & (indices_hq < chunks)
    safe_indices = indices_hq.clamp_min(0).clamp_max(chunks - 1)
    bias_by_head = prior_bias.reshape(batch, chunks, h_q).permute(0, 2, 1)
    selected_bias = torch.gather(
        bias_by_head[:, None].expand(batch, seq_len, h_q, chunks),
        dim=-1,
        index=safe_indices,
    )

    remote_logits = scores.float() + selected_bias
    if value_bonus is not None:
        if value_bonus.shape != scores.shape:
            raise ValueError(
                f"invalid value_bonus shape {tuple(value_bonus.shape)}"
            )
        remote_logits = remote_logits + value_bonus.float()
    if selected_gumbel is not None:
        remote_logits = remote_logits + selected_gumbel.float()
    remote_logits = remote_logits.masked_fill(~valid, float("-inf"))
    local_logits = local_lse.float().unsqueeze(-1)
    use_affine = gate_scale is not None
    if gate_offset is not None and not use_affine:
        local_logits = local_logits - gate_offset.float().reshape(1, 1, 1, 1)
    logits = torch.cat((remote_logits, local_logits), dim=-1) / float(temperature)
    all_masked = torch.isneginf(logits).all(dim=-1, keepdim=True)
    logits = torch.where(all_masked, torch.zeros_like(logits), logits)
    remote_lse = torch.logsumexp(logits[..., :topk], dim=-1)
    local_logit = logits[..., topk]
    gate_logit = remote_lse - local_logit
    if use_affine:
        scale = gate_scale.float().reshape(())
        offset = (
            gate_offset.float().reshape(())
            if gate_offset is not None
            else logits.new_zeros(())
        )
        gate_logit = scale * gate_logit + offset
        remote_mass = torch.sigmoid(gate_logit)
        remote_internal = nn.functional.softmax(
            remote_logits / float(temperature), dim=-1, dtype=torch.float32
        )
        remote_internal = torch.where(
            (~valid).all(dim=-1, keepdim=True),
            torch.zeros_like(remote_internal),
            remote_internal,
        )
        remote_weights = (remote_internal * remote_mass.unsqueeze(-1)).to(scores.dtype)
        local_weight = (1.0 - remote_mass).to(scores.dtype)
        remote_weights = torch.where(
            all_masked, torch.zeros_like(remote_weights), remote_weights
        )
        local_weight = torch.where(
            all_masked.squeeze(-1), torch.zeros_like(local_weight), local_weight
        )
        gate_logit = torch.where(
            all_masked.squeeze(-1), torch.zeros_like(gate_logit), gate_logit
        )
        if not return_gate_logit:
            return remote_weights, local_weight
        return remote_weights, local_weight, gate_logit
    probs = nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    probs = torch.where(all_masked, torch.zeros_like(probs), probs)
    remote_weights = probs[..., :topk].to(scores.dtype)
    local_weight = probs[..., topk].to(scores.dtype)
    if not return_gate_logit:
        return remote_weights, local_weight
    gate_logit = torch.where(all_masked.squeeze(-1), torch.zeros_like(gate_logit), gate_logit)
    return remote_weights, local_weight, gate_logit


def _soft_route_residual_g7(
    route_q: torch.Tensor,
    value_states: torch.Tensor,
    landmark_keys: torch.Tensor,
    prior_bias: torch.Tensor,
    local_lse: torch.Tensor,
    drop_mask: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable all-chunk bypass around hard top-k routing."""
    batch, seq_len, h_q, dim = route_q.shape
    chunks = seq_len // chunk_size
    h_kv = value_states.shape[2]
    groups = h_q // h_kv
    grouped_q = route_q.float().reshape(batch, seq_len, h_kv, groups, dim)
    logits = torch.einsum(
        "blhgd,bchgd->blhgc", grouped_q, landmark_keys.float()
    ) / math.sqrt(dim)
    logits = logits + prior_bias.float().permute(0, 2, 3, 1).unsqueeze(1)
    real_valid = key_valid.reshape(batch, chunks, chunk_size)[:, :, :-1]
    chunk_valid = real_valid.any(dim=-1)
    allowed = (~drop_mask.bool()) & chunk_valid[:, None, :]
    logits = logits.masked_fill(~allowed[:, :, None, None, :], float("-inf"))
    local = local_lse.float().reshape(batch, seq_len, h_kv, groups).unsqueeze(-1)
    logits_all = torch.cat((logits, local), dim=-1) / float(temperature)
    all_masked = torch.isneginf(logits_all).all(dim=-1, keepdim=True)
    logits_all = torch.where(all_masked, torch.zeros_like(logits_all), logits_all)
    probs = torch.softmax(logits_all, dim=-1, dtype=torch.float32)
    probs = torch.where(all_masked, torch.zeros_like(probs), probs)
    remote_probs = probs[..., :chunks]
    values = value_states.reshape(batch, chunks, chunk_size, h_kv, dim)[:, :, :-1].float()
    valid_float = real_valid[:, :, :, None, None].to(values.dtype)
    chunk_values = (values * valid_float).sum(dim=2)
    chunk_values = chunk_values / valid_float.sum(dim=2).clamp_min(1.0)
    return torch.einsum("blhgc,bchd->blhgd", remote_probs, chunk_values).reshape(
        batch, seq_len, h_q, dim
    ).to(route_q.dtype)


class _NaiveDreamSparseAttentionBase(nn.Module):
    def __init__(self, source_attn: nn.Module, local_window: int, chunk_size: Optional[int] = None):
        super().__init__()
        self.config = source_attn.config
        self.layer_idx = source_attn.layer_idx
        self.hidden_size = source_attn.hidden_size
        self.num_heads = source_attn.num_heads
        self.head_dim = source_attn.head_dim
        self.num_key_value_heads = source_attn.num_key_value_heads
        self.num_key_value_groups = source_attn.num_key_value_groups
        self.attention_dropout = source_attn.attention_dropout
        self.local_window = int(local_window)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.is_causal = False

        self.q_proj = source_attn.q_proj
        self.k_proj = source_attn.k_proj
        self.v_proj = source_attn.v_proj
        self.o_proj = source_attn.o_proj
        self.rotary_emb = source_attn.rotary_emb
        self._prefill_capture = False
        self._captured_prefill_cache: LayerKVCache | None = None

    def begin_prefill_capture(self) -> None:
        if self._prefill_capture:
            raise RuntimeError("prefill capture is already active")
        self._prefill_capture = True
        self._captured_prefill_cache = None

    def _capture_prefill_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor,
        landmark_keys: torch.Tensor | None = None,
        prior_bias: torch.Tensor | None = None,
    ) -> None:
        if not self._prefill_capture:
            return
        self._captured_prefill_cache = LayerKVCache(
            key=key.detach().contiguous(),
            value=value.detach().contiguous(),
            key_valid=key_valid.detach().to(dtype=torch.bool).contiguous(),
            landmark_keys=(
                None
                if landmark_keys is None
                else landmark_keys.detach().contiguous()
            ),
            prior_bias=(
                None if prior_bias is None else prior_bias.detach().contiguous()
            ),
        )

    def end_prefill_capture(self) -> LayerKVCache:
        if not self._prefill_capture:
            raise RuntimeError("prefill capture is not active")
        self._prefill_capture = False
        captured = self._captured_prefill_cache
        self._captured_prefill_cache = None
        if captured is None:
            raise RuntimeError("attention forward did not capture a cache")
        return captured

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return query_states, key_states, value_states

    def _project_qkv_blhd(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states.transpose(1, 2), position_ids)
        else:
            cos, sin = position_embeddings
        q_bhld, k_bhld = apply_rotary_pos_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            cos,
            sin,
        )
        return q_bhld.transpose(1, 2).contiguous(), k_bhld.transpose(1, 2).contiguous(), value_states.contiguous()

    def _linear_without_lora(self, module, hidden_states):
        core = getattr(module, "base_layer", None)
        if core is None:
            core = getattr(module, "original_module", module)
        return core(hidden_states)

    def _project_qk_blhd_frozen_base(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        with torch.no_grad():
            query_states = self._linear_without_lora(self.q_proj, hidden_states)
            key_states = self._linear_without_lora(self.k_proj, hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            if position_embeddings is None:
                cos, sin = self.rotary_emb(query_states.transpose(1, 2), position_ids)
            else:
                cos, sin = position_embeddings
            q_bhld, k_bhld = apply_rotary_pos_emb(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                cos,
                sin,
            )
            return q_bhld.transpose(1, 2).contiguous(), k_bhld.transpose(1, 2).contiguous()

    def _project_qk_blhd_frozen_dense(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "frozen_dense_q_weight"):
            raise RuntimeError("frozen dense teacher Q/K snapshot is missing")
        bsz, q_len, _ = hidden_states.size()
        q_bias = getattr(self, "frozen_dense_q_bias", None)
        k_bias = getattr(self, "frozen_dense_k_bias", None)
        with torch.no_grad():
            query_states = torch.nn.functional.linear(
                hidden_states, self.frozen_dense_q_weight, q_bias
            )
            key_states = torch.nn.functional.linear(
                hidden_states, self.frozen_dense_k_weight, k_bias
            )
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            if position_embeddings is None:
                cos, sin = self.rotary_emb(query_states.transpose(1, 2), position_ids)
            else:
                cos, sin = position_embeddings
            q_bhld, k_bhld = apply_rotary_pos_emb(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                cos,
                sin,
            )
            return q_bhld.transpose(1, 2).contiguous(), k_bhld.transpose(1, 2).contiguous()

    def _qcal_delta(self, hidden_states, q, *, freeze_qcal: bool) -> torch.Tensor:
        batch, length, _, _ = q.shape
        if freeze_qcal:
            weight_in = self.qcal[0].weight.detach()
            weight_out = self.qcal[1].weight.detach()
            hidden = hidden_states.to(dtype=weight_in.dtype)
            delta = torch.nn.functional.linear(
                torch.nn.functional.linear(hidden, weight_in), weight_out
            )
            delta = delta.to(dtype=q.dtype)
        else:
            delta = self.qcal(
                hidden_states.to(dtype=next(self.qcal.parameters()).dtype)
            ).to(dtype=q.dtype)
        return (
            delta.view(batch, length, self.num_heads, self.head_dim)
            * float(getattr(self, "qcal_scale", 1.0))
        )

    def _finish_calibrated_query(
        self,
        q,
        delta,
        position_ids,
        position_embeddings,
        *,
        freeze_qcal: bool,
    ):
        if position_embeddings is None:
            cos, sin = self.rotary_emb(q.transpose(1, 2), position_ids)
        else:
            cos, sin = position_embeddings
        q_unrot = invert_rotary_pos_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
        combined = q_unrot + delta
        norm = getattr(self, "qcal_norm", None)
        if norm is not None:
            if freeze_qcal:
                from prefilling_v2.qcal import rms_norm

                combined = rms_norm(combined, norm.weight.detach(), norm.eps)
            else:
                combined = norm(combined)
        rotated, _ = apply_rotary_pos_emb(
            combined.transpose(1, 2), combined.transpose(1, 2), cos, sin
        )
        return rotated.transpose(1, 2).contiguous()

    def _calibrated_query(self, hidden_states, q, position_ids, position_embeddings):
        if not hasattr(self, "qcal"):
            return q
        delta = self._qcal_delta(hidden_states, q, freeze_qcal=False)
        return self._finish_calibrated_query(
            q, delta, position_ids, position_embeddings, freeze_qcal=False
        )

    def _calibrated_query_frozen_qcal(
        self, hidden_states, q, position_ids, position_embeddings
    ):
        """Same residual Q-Cal as `_calibrated_query`, but Q-Cal weights do not get grad."""
        if not hasattr(self, "qcal"):
            return q
        delta = self._qcal_delta(hidden_states, q, freeze_qcal=True)
        return self._finish_calibrated_query(
            q, delta, position_ids, position_embeddings, freeze_qcal=True
        )

    def _lmk_ce_ste_residual(
        self,
        *,
        hidden_states,
        q,
        k,
        v,
        key_valid,
        drop_mask,
        local_lse,
        position_ids,
        position_embeddings,
    ):
        """Hard-route forward; CE backward through a detached-Q soft mix into type embed."""
        from prefilling_v2.full_dense_teacher import ste_type_embed_hidden
        from ops.chunk_attn_pool_gqa_tilelang import chunk_attn_pool_gqa

        type_embed = getattr(self, "lmk_type_embed", None)
        if type_embed is None:
            raise RuntimeError("route_lmk_ce_ste requires bound lmk_type_embed")
        h_ste = ste_type_embed_hidden(hidden_states, self.chunk_size, type_embed)
        q_ste = self._calibrated_query_frozen_qcal(
            h_ste, q.detach(), position_ids, position_embeddings
        )
        batch, seq_len, _, dim = q.shape
        chunks = seq_len // self.chunk_size
        q_lmk = q_ste[:, self.chunk_size - 1 :: self.chunk_size].reshape(
            batch,
            chunks,
            self.num_key_value_heads,
            self.num_key_value_groups,
            dim,
        )
        k_chunked = k.detach().reshape(
            batch,
            chunks,
            self.chunk_size,
            self.num_key_value_heads,
            dim,
        )
        landmark_keys, entropy = chunk_attn_pool_gqa(
            q_lmk.contiguous(),
            k_chunked.contiguous(),
            key_valid.reshape(batch, chunks, self.chunk_size).contiguous(),
        )
        prior_bias = entropy * self.entropy_bias_scale.detach().float().view(
            1,
            1,
            self.num_key_value_heads,
            self.num_key_value_groups,
        )
        routing_q = self._routing_query(q_ste)
        return _soft_route_residual_g7(
            routing_q,
            v.detach(),
            landmark_keys,
            prior_bias,
            local_lse.detach(),
            drop_mask,
            key_valid.detach(),
            self.chunk_size,
            self.route_temperature,
        )

    def _external_allowed_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None or isinstance(attention_mask, str):
            return torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=device)

        if attention_mask.dim() == 2:
            valid = attention_mask.to(device=device).bool()
            return valid[:, None, :, None] & valid[:, None, None, :]

        if attention_mask.dim() != 4:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

        mask = attention_mask.to(device=device)
        if mask.dtype == torch.bool:
            allowed = mask
        else:
            allowed = mask > (torch.finfo(mask.dtype).min / 2)
        if allowed.shape[1] not in (1, num_heads):
            raise ValueError(f"Unsupported attention head dim in mask: {tuple(allowed.shape)}")
        return allowed

    def _key_valid_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None or isinstance(attention_mask, str):
            return torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        if attention_mask.dim() == 2:
            return attention_mask.to(device=device).bool()
        allowed = self._external_allowed_mask(attention_mask, bsz, 1, seq_len, device)
        return allowed.any(dim=-2).squeeze(1)

    def _local_allowed_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.local_window <= 0:
            return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        positions = torch.arange(seq_len, device=device)
        return (positions[:, None] - positions[None, :]).abs() <= self.local_window

    def _real_key_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if not self.chunk_size or self.chunk_size <= 1:
            return torch.ones(seq_len, dtype=torch.bool, device=device)
        positions = torch.arange(seq_len, device=device)
        return (positions + 1) % self.chunk_size != 0

    def _attention_output_and_lse(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        all_masked = ~allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        lse = torch.logsumexp(scores.float(), dim=-1)
        lse = torch.where(all_masked.squeeze(-1), torch.full_like(lse, float("-inf")), lse)
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        return torch.matmul(attn_weights, value_states), lse

    def _safe_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        allowed: torch.Tensor,
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, _, q_len, _ = query_states.shape
        attn_output, _ = self._attention_output_and_lse(query_states, key_states, value_states, allowed)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _finish(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        allowed: torch.Tensor,
        output_attentions: bool,
        past_key_value,
    ):
        attn_output, attn_weights = self._safe_attention(
            query_states, key_states, value_states, allowed, output_attentions
        )
        return attn_output, attn_weights, past_key_value


class NaiveDreamSlidingWindowAttention(_NaiveDreamSparseAttentionBase):
    """Bidirectional sliding-window attention for non-route layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError("Naive Dream sparse attention is training/prefill only for now.")

        query_states, key_states, value_states = self._project_qkv(
            hidden_states, position_ids, position_embeddings
        )
        bsz, num_heads, seq_len, _ = query_states.shape
        external = self._external_allowed_mask(attention_mask, bsz, num_heads, seq_len, query_states.device)
        local = self._local_allowed_mask(seq_len, query_states.device)[None, None, :, :]
        real_key = self._real_key_mask(seq_len, query_states.device)[None, None, None, :]
        return self._finish(
            query_states,
            key_states,
            value_states,
            external & local & real_key,
            output_attentions,
            past_key_value,
        )


class NaiveDreamRouteAttention(_NaiveDreamSparseAttentionBase):
    """route-style local-window plus per-query top-k chunk attention.

    Chunks are physical chunks of size ``chunk_size``. The collator pads real
    tokens to ``chunk_size - 1`` and appends one virtual landmark slot, so each
    physical chunk is ``chunk_size - 1`` real-token slots plus one landmark slot.
    """

    def __init__(self, source_attn: nn.Module, local_window: int, chunk_size: int, topk: int):
        super().__init__(source_attn, local_window, chunk_size)
        if chunk_size < 2:
            raise ValueError("chunk_size must be >= 2")
        self.chunk_size = int(chunk_size)
        self.topk = int(topk)
        weight = source_attn.q_proj.weight
        self.entropy_bias_scale = nn.Parameter(
            torch.ones(self.num_heads, dtype=torch.float32, device=weight.device)
        )

    def _remote_allowed_mask(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, num_heads, seq_len, _ = query_states.shape
        device = query_states.device
        num_chunks = seq_len // self.chunk_size
        if num_chunks == 0 or self.topk <= 0:
            return torch.zeros((bsz, num_heads, seq_len, seq_len), dtype=torch.bool, device=device)

        chunk_positions = torch.arange(num_chunks * self.chunk_size, device=device).view(num_chunks, self.chunk_size)
        real_pos = chunk_positions[:, : self.chunk_size - 1]
        lmk_pos = chunk_positions[:, self.chunk_size - 1]

        key_valid = self._key_valid_mask(attention_mask, bsz, seq_len, device)
        real_valid = key_valid[:, real_pos.reshape(-1)].view(bsz, num_chunks, self.chunk_size - 1)
        lmk_valid = key_valid[:, lmk_pos]
        chunk_valid = real_valid.any(dim=-1) & lmk_valid

        q_lmk = query_states.index_select(2, lmk_pos)
        k_chunk = key_states.index_select(2, real_pos.reshape(-1)).view(
            bsz, num_heads, num_chunks, self.chunk_size - 1, self.head_dim
        )

        pool_scores = (q_lmk.unsqueeze(-2) * k_chunk).sum(dim=-1) / math.sqrt(self.head_dim)
        pool_scores = pool_scores.masked_fill(~real_valid[:, None, :, :], torch.finfo(pool_scores.dtype).min)
        pool_probs = nn.functional.softmax(pool_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        pool_probs = pool_probs.masked_fill(~chunk_valid[:, None, :, None], 0)

        summary_k = (pool_probs.unsqueeze(-1) * k_chunk).sum(dim=-2)
        entropy_probs = pool_probs.float().clamp_min(1e-9)
        entropy = -(entropy_probs * entropy_probs.log()).sum(dim=-1)
        entropy = entropy.to(query_states.dtype) * self.entropy_bias_scale[None, :, None]

        chunk_scores = torch.einsum("bhld,bhcd->bhlc", query_states, summary_k) / math.sqrt(self.head_dim)
        chunk_scores = chunk_scores + entropy[:, :, None, :]
        chunk_scores = chunk_scores.masked_fill(~chunk_valid[:, None, None, :], torch.finfo(chunk_scores.dtype).min)
        external = self._external_allowed_mask(
            attention_mask, bsz, num_heads, seq_len, device
        )
        query_chunk_allowed = external.index_select(-1, lmk_pos)
        chunk_scores = chunk_scores.masked_fill(
            ~query_chunk_allowed,
            torch.finfo(chunk_scores.dtype).min,
        )

        k_eff = min(self.topk, num_chunks)
        selected = torch.topk(chunk_scores, k=k_eff, dim=-1).indices
        key_chunk_ids = torch.div(torch.arange(seq_len, device=device), self.chunk_size, rounding_mode="floor")
        remote = (selected[..., None] == key_chunk_ids[None, None, None, None, :]).any(dim=-2)

        real_key = torch.ones(seq_len, dtype=torch.bool, device=device)
        real_key[lmk_pos] = False
        return remote & real_key[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError("Naive Dream route attention is training/prefill only for now.")

        query_states, key_states, value_states = self._project_qkv(
            hidden_states, position_ids, position_embeddings
        )
        bsz, num_heads, seq_len, _ = query_states.shape
        external = self._external_allowed_mask(attention_mask, bsz, num_heads, seq_len, query_states.device)
        local = self._local_allowed_mask(seq_len, query_states.device)[None, None, :, :]
        remote = self._remote_allowed_mask(query_states, key_states, attention_mask)
        real_key = self._real_key_mask(seq_len, query_states.device)[None, None, None, :]
        return self._finish(
            query_states,
            key_states,
            value_states,
            external & (local | remote) & real_key,
            output_attentions,
            past_key_value,
        )


class KernelDreamRouteAttention(NaiveDreamRouteAttention):
    """Bidirectional route layer using the TileLang selected-chunk kernel in ``ops``.

    Routing and local-window mass are computed in PyTorch so we can keep the
    first prototype's bidirectional dLLM semantics. The selected chunk
    intra-attention is delegated to ``ops.hils_fwd_bwd_head.HiLS_block_M_head``.
    """

    def __init__(
        self,
        source_attn: nn.Module,
        local_window: int,
        chunk_size: int,
        topk: int,
        allow_fallback: bool = True,
        token_budget: int = 0,
        token_policy: str = "global_qk",
        min_tokens_per_chunk: int = 1,
        token_relaxation: str = "none",
        token_gumbel_scale: float = 1.0,
        route_relaxation: str = "none",
        route_temperature: float = 1.0,
        route_gumbel_scale: float = 1.0,
        route_selection_mode: str = "post_softmax",
        route_residual_weight: float = 0.0,
        detach_fusion_weights: bool = False,
        value_fusion_beta: float = 0.0,
        value_fusion_rank: int = 32,
        value_fusion_query_block: int = 128,
    ):
        super().__init__(source_attn, local_window, chunk_size, topk)
        _validate_route_relaxation(
            route_relaxation,
            float(route_temperature),
            float(route_gumbel_scale),
        )
        _validate_token_relaxation(
            str(token_relaxation),
            float(token_gumbel_scale),
        )
        self.allow_fallback = bool(allow_fallback)
        self.token_budget = int(token_budget)
        self.token_policy = str(token_policy)
        self.min_tokens_per_chunk = int(min_tokens_per_chunk)
        self.token_relaxation = str(token_relaxation)
        self.token_gumbel_scale = float(token_gumbel_scale)
        self.route_relaxation = str(route_relaxation)
        self.route_temperature = float(route_temperature)
        self.route_gumbel_scale = float(route_gumbel_scale)
        if route_selection_mode not in {"post_softmax", "pre_softmax"}:
            raise ValueError(f"unsupported route_selection_mode={route_selection_mode}")
        if route_residual_weight < 0:
            raise ValueError("route_residual_weight must be non-negative")
        self.route_selection_mode = str(route_selection_mode)
        self.route_residual_weight = float(route_residual_weight)
        self.detach_fusion_weights = bool(detach_fusion_weights)
        beta = float(value_fusion_beta)
        if beta < 0:
            raise ValueError("value_fusion_beta must be non-negative")
        rank = int(value_fusion_rank)
        if rank <= 0:
            raise ValueError("value_fusion_rank must be positive")
        self.value_fusion_beta = beta
        self.value_fusion_query_block = int(value_fusion_query_block)
        if beta > 0:
            self.value_fusion_q = nn.Linear(self.head_dim, rank, bias=False)
            self.value_fusion_v = nn.Linear(self.head_dim, rank, bias=False)
            nn.init.zeros_(self.value_fusion_v.weight)
        else:
            self.value_fusion_q = None
            self.value_fusion_v = None
        self.skip_inert_slots = True
        self.chunk_summary = "attn"
        self.entropy_prior = True
        self.route_query_positions: torch.Tensor | None = None
        self.force_remote_query_mask: torch.Tensor | None = None
        self.remote_ablation: str = "none"
        self.remote_ablation_seed: int = 0
        self.force_remote_oracle_route: bool = False
        self.force_remote_evidence_chunks: torch.Tensor | None = None
        self.supervise_fusion_gate: bool = False
        self.gate_ce_force: bool = False
        self.fusion_gate_offset = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.fusion_gate_scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.fusion_gate_bce: torch.Tensor | None = None
        self._last_labeled_w_remote: torch.Tensor | None = None
        self._last_token_allocation: torch.Tensor | None = None
        self._warned_kernel_fallback = False

    def _routing_query(self, q: torch.Tensor) -> torch.Tensor:
        positions = self.route_query_positions
        if positions is None:
            return q
        positions = positions.to(device=q.device, dtype=torch.long)
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError("route_query_positions must be a nonempty 1D tensor")
        if bool(((positions < 0) | (positions >= q.shape[1])).any()):
            raise ValueError("route_query_positions are outside the sequence")
        pooled = q.index_select(1, positions).mean(dim=1, keepdim=True)
        return pooled.expand(-1, q.shape[1], -1, -1).contiguous()

    def _fallback(self, reason: Exception | str, *args, **kwargs):
        if not self.allow_fallback:
            if isinstance(reason, Exception):
                raise reason
            raise RuntimeError(str(reason))
        if not self._warned_kernel_fallback:
            warnings.warn(f"Falling back to torch_bidir route attention: {reason}", RuntimeWarning)
            self._warned_kernel_fallback = True
        return super().forward(*args, **kwargs)

    def _expand_kv_for_kernel(self) -> bool:
        group = int(self.num_key_value_groups)
        return group not in (1, 2, 4, 8, 16)

    def _compute_chunk_scores_for_kernel(
        self,
        query_blhd: torch.Tensor,
        key_blhkvd: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h_q, _ = query_blhd.shape
        device = query_blhd.device
        num_chunks = seq_len // self.chunk_size
        if num_chunks == 0:
            empty_scores = torch.empty((bsz, seq_len, h_q, 0), dtype=query_blhd.dtype, device=device)
            empty_valid = torch.empty((bsz, 0), dtype=torch.bool, device=device)
            return empty_scores, empty_valid

        chunk_positions = torch.arange(num_chunks * self.chunk_size, device=device).view(num_chunks, self.chunk_size)
        real_pos = chunk_positions[:, : self.chunk_size - 1]
        lmk_pos = chunk_positions[:, self.chunk_size - 1]

        key_valid = self._key_valid_mask(attention_mask, bsz, seq_len, device)
        real_valid = key_valid[:, real_pos.reshape(-1)].view(bsz, num_chunks, self.chunk_size - 1)
        lmk_valid = key_valid[:, lmk_pos]
        chunk_valid = real_valid.any(dim=-1) & lmk_valid

        key_expanded = repeat_kv(key_blhkvd.transpose(1, 2), self.num_key_value_groups).transpose(1, 2).contiguous()
        q_lmk = query_blhd.index_select(1, lmk_pos)
        k_chunk = key_expanded.index_select(1, real_pos.reshape(-1)).view(
            bsz, num_chunks, self.chunk_size - 1, h_q, self.head_dim
        )

        pool_scores = (q_lmk.unsqueeze(2) * k_chunk).sum(dim=-1) / math.sqrt(self.head_dim)
        pool_scores = pool_scores.masked_fill(~real_valid[:, :, :, None], torch.finfo(pool_scores.dtype).min)
        pool_probs = nn.functional.softmax(pool_scores, dim=2, dtype=torch.float32).to(query_blhd.dtype)
        pool_probs = pool_probs.masked_fill(~chunk_valid[:, :, None, None], 0)

        summary_k = (pool_probs.unsqueeze(-1) * k_chunk).sum(dim=2)
        entropy_probs = pool_probs.float().clamp_min(1e-9)
        entropy = -(entropy_probs * entropy_probs.log()).sum(dim=2).to(query_blhd.dtype)
        entropy = entropy * self.entropy_bias_scale[None, None, :]
        entropy = entropy.transpose(1, 2).contiguous()

        chunk_scores = torch.einsum("blhd,bchd->blhc", query_blhd, summary_k) / math.sqrt(self.head_dim)
        chunk_scores = chunk_scores + entropy[:, None, :, :]
        chunk_scores = chunk_scores.masked_fill(~chunk_valid[:, None, None, :], torch.finfo(chunk_scores.dtype).min)
        return chunk_scores, chunk_valid

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError("Kernel Dream route attention is training/prefill only for now.")

        call_args = (
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
        )
        call_kwargs = kwargs

        try:
            from ops.hils_fwd_bwd_head import HiLS_block_M_head
        except Exception as exc:  # pragma: no cover - depends on TileLang env
            return self._fallback(exc, *call_args, **call_kwargs)

        if hidden_states.device.type != "cuda":
            return self._fallback("TileLang kernel requires CUDA tensors", *call_args, **call_kwargs)
        if hidden_states.dtype != torch.bfloat16:
            return self._fallback("TileLang kernel path currently expects bfloat16 hidden states", *call_args, **call_kwargs)
        if self.chunk_size < 32:
            return self._fallback("TileLang route chunk kernel requires chunk_size >= 32", *call_args, **call_kwargs)

        try:
            query_blhd, key_blhkvd, value_blhkvd = self._project_qkv_blhd(
                hidden_states, position_ids, position_embeddings
            )
            bsz, seq_len, h_q, _ = query_blhd.shape
            if seq_len % self.chunk_size != 0:
                return self._fallback(
                    f"kernel path expects seq_len multiple of chunk_size, got {seq_len} and {self.chunk_size}",
                    *call_args,
                    **call_kwargs,
                )

            key_expanded = repeat_kv(key_blhkvd.transpose(1, 2), self.num_key_value_groups)
            value_expanded = repeat_kv(value_blhkvd.transpose(1, 2), self.num_key_value_groups)
            query_bhld = query_blhd.transpose(1, 2).contiguous()

            external = self._external_allowed_mask(attention_mask, bsz, h_q, seq_len, hidden_states.device)
            local = self._local_allowed_mask(seq_len, hidden_states.device)[None, None, :, :]
            real_key = self._real_key_mask(seq_len, hidden_states.device)[None, None, None, :]
            local_o_bhld, local_lse_bhl = self._attention_output_and_lse(
                query_bhld,
                key_expanded,
                value_expanded,
                external & local & real_key,
            )

            chunk_scores, _ = self._compute_chunk_scores_for_kernel(query_blhd, key_blhkvd, attention_mask)
            num_chunks = chunk_scores.shape[-1]
            k_eff = min(self.topk, num_chunks)
            if k_eff <= 0:
                o_bhld = local_o_bhld
            else:
                h_kv = self.num_key_value_heads
                group = self.num_key_value_groups
                score_pool = chunk_scores.view(bsz, seq_len, h_kv, group, num_chunks).max(dim=3).values
                _, indices_kv = torch.topk(score_pool, k=k_eff, dim=-1)
                indices_hq = indices_kv.repeat_interleave(group, dim=2)
                selected_scores = chunk_scores.gather(dim=-1, index=indices_hq.long())

                cat_scores = torch.cat([selected_scores, local_lse_bhl.transpose(1, 2).unsqueeze(-1)], dim=-1)
                weights = nn.functional.softmax(cat_scores, dim=-1, dtype=torch.float32).to(query_blhd.dtype)
                sparse_w = weights[..., :k_eff].contiguous()
                local_w = weights[..., -1].transpose(1, 2).unsqueeze(-1)

                if k_eff < self.topk:
                    pad = self.topk - k_eff
                    sparse_w = nn.functional.pad(sparse_w, (0, pad), value=0.0).contiguous()
                    indices_kv = nn.functional.pad(indices_kv, (0, pad), value=0).contiguous()

                kernel_key = key_blhkvd.contiguous()
                kernel_value = value_blhkvd.contiguous()
                kernel_indices = indices_kv.to(torch.int32).contiguous()
                if self._expand_kv_for_kernel():
                    kernel_key = key_expanded.transpose(1, 2).contiguous()
                    kernel_value = value_expanded.transpose(1, 2).contiguous()
                    kernel_indices = indices_hq.to(torch.int32).contiguous()

                route_o_blhd = HiLS_block_M_head(
                    query_blhd.contiguous(),
                    kernel_key,
                    kernel_value,
                    sparse_w,
                    kernel_indices,
                    block_size=self.chunk_size,
                    sm_scale=self.head_dim ** -0.5,
                    block_M=None,
                    mask_last_token=True,
                    is_training=self.training,
                )
                o_bhld = route_o_blhd.transpose(1, 2).contiguous() + local_o_bhld * local_w

            attn_output = o_bhld.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None if not output_attentions else None, past_key_value
        except Exception as exc:  # pragma: no cover - kernel/runtime dependent
            return self._fallback(exc, *call_args, **call_kwargs)


class KernelDreamSlidingWindowAttention(NaiveDreamSlidingWindowAttention):
    """Compact-GQA bidirectional local attention (FlashAttention window)."""

    def __init__(
        self,
        source_attn: nn.Module,
        local_window: int,
        chunk_size: int,
        allow_fallback: bool = True,
        skip_inert_slots: bool = False,
    ):
        super().__init__(source_attn, local_window, chunk_size=chunk_size)
        self.allow_fallback = bool(allow_fallback)
        self.skip_inert_slots = bool(skip_inert_slots)
        self._warned_kernel_fallback = False

    def _fallback(self, reason: Exception | str, *args, **kwargs):
        if not self.allow_fallback:
            if isinstance(reason, Exception):
                raise reason
            raise RuntimeError(str(reason))
        if not self._warned_kernel_fallback:
            warnings.warn(
                f"Falling back to torch_bidir sliding attention: {reason}",
                RuntimeWarning,
            )
            self._warned_kernel_fallback = True
        return super().forward(*args, **kwargs)

    def _validate_kernel_call(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[Exception | str]:
        if hidden_states.device.type != "cuda":
            return "kernel_bidir requires CUDA hidden states"
        if hidden_states.dtype != torch.bfloat16:
            return "kernel_bidir requires BF16 hidden states"
        if attention_mask is not None and not isinstance(attention_mask, str):
            if attention_mask.ndim not in (2, 4):
                return ValueError(
                    "kernel_bidir requires a 2D valid mask or 4D allowed mask"
                )
        if self.num_key_value_groups != 7:
            return ValueError(
                "kernel_bidir requires Dream logical G=7; "
                f"got G={self.num_key_value_groups}"
            )
        seq_len = hidden_states.shape[1]
        if self.chunk_size is None or seq_len % self.chunk_size != 0:
            return ValueError(
                f"sequence length {seq_len} must be divisible by "
                f"chunk_size={self.chunk_size}"
            )
        if self.head_dim > 256:
            return ValueError(
                f"kernel_bidir requires head_dim <= 256, got {self.head_dim}"
            )
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError(
                "kernel_bidir is a training/prefill backend and has no KV cache"
            )
        call_args = (
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
        )
        reason = self._validate_kernel_call(hidden_states, attention_mask)
        if reason is not None:
            return self._fallback(reason, *call_args, **kwargs)
        try:
            from prefilling_v2.local_attention import bidir_local_attention
        except ImportError as exc:  # pragma: no cover - environment failure
            return self._fallback(exc, *call_args, **kwargs)

        q, k, v = self._project_qkv_blhd(
            hidden_states, position_ids, position_embeddings
        )
        batch, seq_len = hidden_states.shape[:2]
        key_valid = self._key_valid_mask(
            attention_mask, batch, seq_len, hidden_states.device
        )
        self._capture_prefill_cache(k, v, key_valid)
        output, _ = bidir_local_attention(
            q,
            k,
            v,
            key_valid,
            self.local_window,
            self.chunk_size,
            skip_inert_slots=getattr(self, "skip_inert_slots", False),
        )
        output = output.reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(output), None, past_key_value


class KernelDreamFullRouteAttention(KernelDreamRouteAttention):
    """Complete compact-GQA bidirectional route training backend."""

    def _validate_kernel_call(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[Exception | str]:
        if hidden_states.device.type != "cuda":
            return "kernel_bidir requires CUDA hidden states"
        if hidden_states.dtype != torch.bfloat16:
            return "kernel_bidir requires BF16 hidden states"
        if attention_mask is not None and not isinstance(attention_mask, str):
            if attention_mask.ndim not in (2, 4):
                return ValueError(
                    "kernel_bidir requires a 2D valid mask or 4D allowed mask"
                )
        if self.num_heads != 28 or self.num_key_value_heads != 4:
            return ValueError(
                "kernel_bidir currently requires Dream 28Q/4KV; "
                f"got {self.num_heads}Q/{self.num_key_value_heads}KV"
            )
        if self.num_key_value_groups != 7:
            return ValueError(
                "kernel_bidir requires Dream logical G=7; "
                f"got G={self.num_key_value_groups}"
            )
        seq_len = hidden_states.shape[1]
        if seq_len % self.chunk_size != 0:
            return ValueError(
                f"sequence length {seq_len} must be divisible by "
                f"chunk_size={self.chunk_size}"
            )
        if self.chunk_size < 16:
            return ValueError(
                f"kernel_bidir requires chunk_size >= 16, got {self.chunk_size}"
            )
        if self.topk <= 0 or self.topk > 32:
            return ValueError(
                f"kernel_bidir requires 1 <= topk <= 32, got {self.topk}"
            )
        if self.token_budget < 0:
            return ValueError(
                f"kernel_bidir requires token_budget >= 0, got "
                f"{self.token_budget}"
            )
        if self.token_budget > self.topk * self.chunk_size:
            return ValueError(
                f"token_budget={self.token_budget} exceeds routed capacity "
                f"{self.topk * self.chunk_size}"
            )
        if self.token_policy not in {"global_qk", "entropy_adaptive"}:
            return ValueError(
                f"unsupported token_policy={self.token_policy}"
            )
        if self.min_tokens_per_chunk < 0:
            return ValueError(
                "min_tokens_per_chunk must be non-negative"
            )
        if (
            self.token_budget > 0
            and self.token_budget
            < self.topk * self.min_tokens_per_chunk
        ):
            return ValueError(
                "token_budget must cover the minimum for every routed chunk"
            )
        if self.head_dim > 256:
            return ValueError(
                f"kernel_bidir requires head_dim <= 256, got {self.head_dim}"
            )
        return None

    def _gqa_mean_chunk_keys(
        self,
        k_chunked: torch.Tensor,
        chunk_key_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = chunk_key_valid[:, :, :-1].to(dtype=torch.float32)
        pooled = (k_chunked[:, :, :-1].float() * valid.unsqueeze(-1).unsqueeze(-1)).sum(
            dim=2
        )
        denom = valid.sum(dim=2).clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
        mean_k = (pooled / denom).to(dtype=k_chunked.dtype)
        return (
            mean_k.unsqueeze(3)
            .expand(-1, -1, -1, self.num_key_value_groups, -1)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError(
                "kernel_bidir is a training/prefill backend and has no KV cache"
            )
        call_args = (
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
        )
        reason = self._validate_kernel_call(hidden_states, attention_mask)
        if reason is not None:
            return self._fallback(reason, *call_args, **kwargs)
        try:
            from prefilling_v2.kernel_utils import remote_drop_mask
            from prefilling_v2.local_attention import bidir_local_attention
            from prefilling_v2.routing import (
                route_topk_g7,
                selected_attention_g7,
            )
            from prefilling_v2.token_refinement import (
                refine_remote_tokens_entropy_g7,
                refine_remote_tokens_g7,
            )
            from ops.chunk_attn_pool_gqa_tilelang import chunk_attn_pool_gqa
            from ops.bidir_output_fusion_tilelang import fuse_route_outputs
            from ops.bidir_route_weights_tilelang import route_weights
        except ImportError as exc:  # pragma: no cover - environment failure
            return self._fallback(exc, *call_args, **kwargs)

        q, k, v = self._project_qkv_blhd(
            hidden_states, position_ids, position_embeddings
        )
        batch, seq_len, _, dim = q.shape
        chunks = seq_len // self.chunk_size
        key_valid = self._key_valid_mask(
            attention_mask, batch, seq_len, hidden_states.device
        )
        packed_allowed = None
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            external = self._external_allowed_mask(
                attention_mask,
                batch,
                self.num_heads,
                seq_len,
                hidden_states.device,
            )
            if external.shape[1] != 1:
                external = external.all(dim=1, keepdim=True)
            packed_allowed = external[:, 0]

        local_output, local_lse = bidir_local_attention(
            q,
            k,
            v,
            key_valid,
            self.local_window,
            self.chunk_size,
            skip_inert_slots=getattr(self, "skip_inert_slots", True),
        )
        route_q = self._calibrated_query(hidden_states, q, position_ids, position_embeddings)
        q_lmk = route_q[:, self.chunk_size - 1 :: self.chunk_size].reshape(
            batch,
            chunks,
            self.num_key_value_heads,
            self.num_key_value_groups,
            dim,
        )
        k_chunked = k.reshape(
            batch,
            chunks,
            self.chunk_size,
            self.num_key_value_heads,
            dim,
        )
        chunk_key_valid = key_valid.reshape(
            batch, chunks, self.chunk_size
        )
        if str(getattr(self, "chunk_summary", "attn")) == "mean":
            landmark_keys = self._gqa_mean_chunk_keys(k_chunked, chunk_key_valid)
            entropy = torch.zeros(
                batch,
                chunks,
                self.num_key_value_heads,
                self.num_key_value_groups,
                dtype=torch.float32,
                device=k.device,
            )
        else:
            landmark_keys, entropy = chunk_attn_pool_gqa(
                q_lmk.contiguous(),
                k_chunked.contiguous(),
                chunk_key_valid.contiguous(),
            )
        if not bool(getattr(self, "entropy_prior", True)):
            entropy = entropy * 0
        prior_bias = entropy * self.entropy_bias_scale.float().view(
            1,
            1,
            self.num_key_value_heads,
            self.num_key_value_groups,
        )
        intervene = getattr(self, "_lmk_keys_intervene", None)
        if callable(intervene):
            landmark_keys, prior_bias = intervene(landmark_keys, prior_bias)
        hook = getattr(self, "_lmk_summary_hook", None)
        if callable(hook):
            hook(
                landmark_keys=landmark_keys,
                k_chunked=k_chunked,
                chunk_key_valid=chunk_key_valid,
            )
        self._capture_prefill_cache(
            k,
            v,
            key_valid,
            landmark_keys,
            prior_bias,
        )
        drop_mask = remote_drop_mask(
            key_valid,
            self.local_window,
            self.chunk_size,
            packed_allowed,
        )
        use_route_gumbel = (
            self.route_relaxation in {"gumbel_topk", "gumbel_softmax_topk"}
            and self.training
        )
        if self.training and getattr(self, "chunk_aux_queries", 0) > 0:
            from prefilling_v2.chunk_distillation import chunk_distillation_loss
            self.chunk_aux_loss = chunk_distillation_loss(
                q, k, landmark_keys, prior_bias, key_valid, drop_mask,
                self.chunk_size, self.chunk_aux_queries, packed_allowed,
            )
        routing_q = self._routing_query(route_q)
        if self.training and hasattr(self, "full_teacher_query_positions"):
            from prefilling_v2.full_dense_teacher import (
                attach_same_forward_teacher,
                student_router_kl,
            )
            source = str(getattr(self, "full_teacher_source", "same_forward"))
            if source == "base":
                q_t, k_t = self._project_qk_blhd_frozen_base(
                    hidden_states, position_ids, position_embeddings
                )
            elif source == "dense":
                q_t, k_t = self._project_qk_blhd_frozen_dense(
                    hidden_states, position_ids, position_embeddings
                )
            else:
                q_t, k_t = q, k
            attach_same_forward_teacher(self, q_t, k_t, key_valid)
            self.full_teacher_loss = student_router_kl(
                self, hidden_states, q, k, key_valid, drop_mask,
                position_ids, position_embeddings,
            )
        if self.training and hasattr(self, "evidence_route_query_mask"):
            from prefilling_v2.evidence_routing import evidence_route_loss

            self.evidence_route_loss = evidence_route_loss(
                routing_q,
                landmark_keys,
                prior_bias,
                local_lse.reshape(
                    batch,
                    seq_len,
                    self.num_key_value_heads,
                    self.num_key_value_groups,
                ),
                drop_mask,
                self.evidence_route_query_mask,
                self.evidence_route_chunks,
            )
        route_result = route_topk_g7(
            routing_q,
            landmark_keys,
            local_lse.reshape(
                batch,
                seq_len,
                self.num_key_value_heads,
                self.num_key_value_groups,
            ),
            prior_bias.contiguous(),
            drop_mask,
            self.topk,
            self.chunk_size,
            self.local_window,
            training=self.training,
            use_gumbel=use_route_gumbel,
            gumbel_scale=self.route_gumbel_scale,
            return_selected_gumbel=self.route_relaxation == "gumbel_softmax_topk",
            selection_mode=self.route_selection_mode,
        )
        if self.route_relaxation == "gumbel_softmax_topk":
            indices, selected_scores, selected_gumbel = route_result
        else:
            indices, selected_scores = route_result
            selected_gumbel = None
        force_mask = getattr(self, "force_remote_query_mask", None)
        ablation = str(getattr(self, "remote_ablation", "none"))
        if (
            bool(getattr(self, "force_remote_oracle_route", False))
            and ablation in {"none", "learned"}
            and force_mask is not None
            and getattr(self, "force_remote_evidence_chunks", None) is not None
        ):
            from prefilling_v2.force_remote import splice_oracle_indices

            indices, selected_scores = splice_oracle_indices(
                indices,
                selected_scores,
                self.force_remote_evidence_chunks,
                force_mask,
            )
        if self.training and getattr(self, "support_attn_query_positions", None) is not None:
            from prefilling_v2.support_attn_distill import attach_support_attn_kl

            # Content LoRA Q/K only. indices / selected_scores / LMK prior are
            # detached inside the KL so this is not live S1 routing CE.
            self.support_attn_kl = attach_support_attn_kl(
                self,
                hidden_states,
                q,
                k,
                key_valid,
                indices,
                selected_scores,
                prior_bias,
                local_lse,
                position_ids,
                position_embeddings,
            )
        # route_weights reserves one slot for the local branch and therefore
        # only supports topk <= 31.  The PyTorch implementation has the same
        # normalization semantics and keeps topk=32 usable for both training
        # and Fast-dLLM inference.
        gumbel_st = (
            self.training
            and self.route_relaxation == "gumbel_softmax_topk_st"
            and not self.detach_fusion_weights
        )
        use_torch_route_weights = (
            self.topk >= 32
            or (
                self.route_relaxation
                in {"gumbel_softmax_topk", "gumbel_softmax_topk_st"}
                and self.training
            )
        )
        def _fuse(value):
            if value is None:
                return None
            return value.detach() if self.detach_fusion_weights else value

        value_bonus = None
        if self.value_fusion_beta > 0 and self.value_fusion_q is not None:
            from prefilling_v2.value_aware_fusion import (
                selected_chunk_outputs,
                value_fusion_bonus,
            )

            chunk_out = selected_chunk_outputs(
                q,
                k,
                v,
                indices,
                key_valid,
                self.chunk_size,
                query_block=self.value_fusion_query_block,
            )
            value_bonus = self.value_fusion_beta * value_fusion_bonus(
                q,
                chunk_out,
                self.value_fusion_q,
                self.value_fusion_v,
            )
            if self.detach_fusion_weights:
                value_bonus = value_bonus.detach()

        if use_torch_route_weights:
            routed = _route_weights_torch(
                _fuse(selected_scores),
                indices,
                _fuse(prior_bias).contiguous(),
                _fuse(local_lse.float()).contiguous(),
                temperature=self.route_temperature,
                selected_gumbel=None if gumbel_st else _fuse(selected_gumbel),
                value_bonus=value_bonus,
                return_gate_logit=True,
                gate_offset=getattr(self, "fusion_gate_offset", None),
                gate_scale=getattr(self, "fusion_gate_scale", None),
            )
            remote_weights, local_weight, gate_logit = routed
            if gumbel_st:
                noise = _sample_gumbel_like(
                    selected_scores, self.route_gumbel_scale
                )
                remote_soft, local_soft, _ = _route_weights_torch(
                    selected_scores,
                    indices,
                    prior_bias.contiguous(),
                    local_lse.float().contiguous(),
                    temperature=self.route_temperature,
                    selected_gumbel=noise,
                    value_bonus=value_bonus,
                    return_gate_logit=True,
                    gate_offset=getattr(self, "fusion_gate_offset", None),
                    gate_scale=getattr(self, "fusion_gate_scale", None),
                )
                remote_weights = _straight_through(remote_weights, remote_soft)
                local_weight = _straight_through(local_weight, local_soft)
        else:
            remote_weights, local_weight = route_weights(
                _fuse(selected_scores),
                indices,
                _fuse(prior_bias).contiguous(),
                _fuse(local_lse.float()).contiguous(),
            )
            w_remote = remote_weights.float().sum(dim=-1).clamp(1e-6, 1.0 - 1e-6)
            w_local = local_weight.float().clamp(1e-6, 1.0 - 1e-6)
            gate_logit = w_remote.log() - w_local.log()
        force_mask = getattr(self, "force_remote_query_mask", None)
        ablation = str(getattr(self, "remote_ablation", "none"))
        supervise = bool(getattr(self, "supervise_fusion_gate", False))
        if force_mask is not None and supervise:
            from prefilling_v2.force_remote import (
                fusion_gate_bce,
                labeled_gate_vectors,
                labeled_remote_mass,
            )

            # BCE on the native gate logit, before any force override.
            if self.training:
                self.fusion_gate_bce = fusion_gate_bce(
                    gate_logit, force_mask, target_remote=True
                )
            self._last_labeled_w_remote = labeled_remote_mass(
                remote_weights.detach(), force_mask
            )
            logit_vec, mass_vec = labeled_gate_vectors(
                gate_logit.detach(), remote_weights.detach(), force_mask
            )
            self._last_labeled_gate_logit = logit_vec
            self._last_labeled_gate_mass = mass_vec
            from prefilling_v2.force_remote import local_control_query_mask

            control_mask = local_control_query_mask(force_mask, shift=64)
            control_logit, _ = labeled_gate_vectors(
                gate_logit.detach(), remote_weights.detach(), control_mask
            )
            self._last_control_gate_logit = control_logit
        apply_force = False
        force_ablation = ablation
        if force_mask is not None and ablation != "learned":
            if not self.training and ablation in {"none", "off", "shuffle"}:
                apply_force = True
            elif (
                self.training
                and supervise
                and bool(getattr(self, "gate_ce_force", False))
            ):
                # Train CE on forced-remote hidden; BCE already used the native gate.
                apply_force = True
                force_ablation = "none"
        if apply_force:
            from prefilling_v2.force_remote import apply_forced_remote_gate

            remote_weights, local_weight, indices = apply_forced_remote_gate(
                remote_weights,
                local_weight,
                indices,
                query_mask=force_mask,
                ablation=force_ablation,
                seed=int(getattr(self, "remote_ablation_seed", 0) or 0),
            )
        remote_weights = _fuse(remote_weights)
        local_weight = _fuse(local_weight)
        if self.training and getattr(self, "evidence_token_query_mask", None) is not None:
            from prefilling_v2.evidence_token_attn import evidence_token_attn_loss

            token_loss, token_stats = evidence_token_attn_loss(
                q,
                k,
                indices,
                self.evidence_token_query_mask,
                self.evidence_token_positions,
                key_valid,
                self.chunk_size,
                skip_inert_slots=bool(getattr(self, "skip_inert_slots", False)),
                local_weight=local_weight,
            )
            self.evidence_token_attn_loss = token_loss
            self._last_remote_needle_qk_mass = token_stats["remote_needle_qk_mass"]
            self._last_evidence_local_weight = token_stats["local_weight"]
        allchunk_positions = getattr(self, "allchunk_st_positions", None) if self.training else None
        if allchunk_positions is not None:
            from prefilling_v2.allchunk_gumbel import detach_sampled_gate
            remote_weights = detach_sampled_gate(remote_weights, allchunk_positions)
            local_weight = detach_sampled_gate(local_weight, allchunk_positions)
        token_keep = None
        self._last_token_allocation = None
        if self.token_budget > 0:
            if self.token_policy == "global_qk":
                token_keep = refine_remote_tokens_g7(
                    q,
                    k,
                    indices,
                    key_valid,
                    self.chunk_size,
                    self.token_budget,
                    training=self.training,
                    use_gumbel=self.token_relaxation == "gumbel_topk",
                    gumbel_scale=self.token_gumbel_scale,
                )
            else:
                token_keep, allocation = refine_remote_tokens_entropy_g7(
                    q,
                    k,
                    indices,
                    key_valid,
                    entropy,
                    self.chunk_size,
                    self.token_budget,
                    self.min_tokens_per_chunk,
                    training=self.training,
                    use_gumbel=self.token_relaxation == "gumbel_topk",
                    gumbel_scale=self.token_gumbel_scale,
                )
                self._last_token_allocation = allocation.detach()
        remote_output = selected_attention_g7(
            q,
            k,
            v,
            remote_weights,
            indices,
            key_valid,
            self.chunk_size,
            self.training,
            token_keep=token_keep,
        )
        output = fuse_route_outputs(
            remote_output.contiguous(),
            local_output.contiguous(),
            local_weight.contiguous(),
        )
        if allchunk_positions is not None:
            from prefilling_v2.allchunk_gumbel import attach_allchunk_st
            output = attach_allchunk_st(
                output,
                q,
                k,
                v,
                routing_q,
                landmark_keys,
                prior_bias,
                local_lse,
                drop_mask,
                key_valid,
                local_output,
                allchunk_positions,
                None,
                self.chunk_size,
                float(getattr(self, "allchunk_st_temperature", self.route_temperature)),
            )
        if self.training and bool(getattr(self, "lmk_ce_ste", False)):
            soft = self._lmk_ce_ste_residual(
                hidden_states=hidden_states,
                q=q,
                k=k,
                v=v,
                key_valid=key_valid,
                drop_mask=drop_mask,
                local_lse=local_lse,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            output = output + (soft - soft.detach())
        if self.route_residual_weight > 0:
            output = output + self.route_residual_weight * _soft_route_residual_g7(
                routing_q,
                v,
                landmark_keys,
                prior_bias,
                local_lse,
                drop_mask,
                key_valid,
                self.chunk_size,
                self.route_temperature,
            )
        output = output.reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(output), None, past_key_value


def route_layer_indices(num_layers: int, interleave: int) -> List[int]:
    if interleave <= 0:
        return list(range(num_layers))
    return [idx for idx in range(num_layers) if idx % interleave == interleave - 1]


def install_dream_sparse_attention(
    model: nn.Module,
    *,
    interleave: int = 4,
    local_window: int = 512,
    swa_local_window: Optional[int] = None,
    chunk_size: int = 64,
    topk: int = 16,
    token_budget: int = 0,
    token_policy: str = "global_qk",
    min_tokens_per_chunk: int = 1,
    token_relaxation: str = "none",
    token_gumbel_scale: float = 1.0,
    route_relaxation: str = "none",
    route_temperature: float = 1.0,
    route_gumbel_scale: float = 1.0,
    route_selection_mode: str = "post_softmax",
    route_residual_weight: float = 0.0,
    detach_fusion_weights: bool = False,
    value_fusion_beta: float = 0.0,
    value_fusion_rank: int = 32,
    value_fusion_query_block: int = 128,
    backend: str = "torch_bidir",
    allow_kernel_fallback: bool = True,
    non_route_attention: str = "sliding",
    chunk_summary: str = "attn",
    entropy_prior: bool = True,
) -> SparseLayerPlan:
    """Replace selected Dream layers with route and configure the rest."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a DreamModel-like object with model.layers")
    _validate_route_relaxation(
        str(route_relaxation),
        float(route_temperature),
        float(route_gumbel_scale),
    )
    _validate_token_relaxation(
        str(token_relaxation),
        float(token_gumbel_scale),
    )

    layers: Iterable[nn.Module] = model.model.layers
    num_layers = len(model.model.layers)
    route_layer_set = set(route_layer_indices(num_layers, interleave))
    sliding: List[int] = []
    route_list: List[int] = []
    dense: List[int] = []
    sliding_window = int(local_window if swa_local_window is None else swa_local_window)
    if non_route_attention not in {"sliding", "dense"}:
        raise ValueError(
            f"Unsupported non-route attention: {non_route_attention}"
        )
    if chunk_summary not in {"attn", "mean"}:
        raise ValueError(f"Unsupported chunk_summary: {chunk_summary}")
    use_entropy_prior = bool(entropy_prior)

    for layer_idx, layer in enumerate(layers):
        source_attn = layer.self_attn
        if layer_idx in route_layer_set:
            if backend == "torch_bidir":
                layer.self_attn = NaiveDreamRouteAttention(source_attn, local_window, chunk_size, topk)
            elif backend == "chunk_kernel_bidir":
                layer.self_attn = KernelDreamRouteAttention(
                    source_attn,
                    local_window,
                    chunk_size,
                    topk,
                    allow_fallback=allow_kernel_fallback,
                    token_budget=token_budget,
                    token_policy=token_policy,
                    min_tokens_per_chunk=min_tokens_per_chunk,
                    token_relaxation=token_relaxation,
                    token_gumbel_scale=token_gumbel_scale,
                    route_relaxation=route_relaxation,
                    route_temperature=route_temperature,
                    route_gumbel_scale=route_gumbel_scale,
                    route_selection_mode=route_selection_mode,
                    route_residual_weight=route_residual_weight,
                    detach_fusion_weights=detach_fusion_weights,
                    value_fusion_beta=value_fusion_beta,
                    value_fusion_rank=value_fusion_rank,
                    value_fusion_query_block=value_fusion_query_block,
                )
            elif backend == "kernel_bidir":
                layer.self_attn = KernelDreamFullRouteAttention(
                    source_attn,
                    local_window,
                    chunk_size,
                    topk,
                    allow_fallback=allow_kernel_fallback,
                    token_budget=token_budget,
                    token_policy=token_policy,
                    min_tokens_per_chunk=min_tokens_per_chunk,
                    token_relaxation=token_relaxation,
                    token_gumbel_scale=token_gumbel_scale,
                    route_relaxation=route_relaxation,
                    route_temperature=route_temperature,
                    route_gumbel_scale=route_gumbel_scale,
                    route_selection_mode=route_selection_mode,
                    route_residual_weight=route_residual_weight,
                    detach_fusion_weights=detach_fusion_weights,
                    value_fusion_beta=value_fusion_beta,
                    value_fusion_rank=value_fusion_rank,
                    value_fusion_query_block=value_fusion_query_block,
                )
            else:
                raise ValueError(f"Unsupported Dream route backend: {backend}")
            layer.self_attn.chunk_summary = str(chunk_summary)
            layer.self_attn.entropy_prior = use_entropy_prior
            route_list.append(layer_idx)
        else:
            if non_route_attention == "dense":
                dense.append(layer_idx)
                continue
            if backend == "kernel_bidir":
                layer.self_attn = KernelDreamSlidingWindowAttention(
                    source_attn,
                    sliding_window,
                    chunk_size,
                    allow_fallback=allow_kernel_fallback,
                    skip_inert_slots=False,
                )
            else:
                layer.self_attn = NaiveDreamSlidingWindowAttention(
                    source_attn, sliding_window, chunk_size=chunk_size
                )
            sliding.append(layer_idx)

    model.config.route_interleave = interleave
    model.config.route_local_window = local_window
    model.config.dream_swa_local_window = sliding_window
    model.config.route_chunk_size = chunk_size
    model.config.route_topk = topk
    model.config.route_token_budget = token_budget
    model.config.route_token_policy = token_policy
    model.config.route_min_tokens_per_chunk = min_tokens_per_chunk
    model.config.route_token_relaxation = token_relaxation
    model.config.route_token_gumbel_scale = token_gumbel_scale
    model.config.route_relaxation = route_relaxation
    model.config.route_temperature = route_temperature
    model.config.route_gumbel_scale = route_gumbel_scale
    model.config.route_detach_fusion_weights = bool(detach_fusion_weights)
    model.config.route_remote_ablation = "none"
    model.config.route_remote_ablation_seed = 0
    model.config.route_backend = backend
    model.config.route_allow_kernel_fallback = allow_kernel_fallback
    model.config.route_non_route_attention = non_route_attention
    model.config.route_chunk_summary = str(chunk_summary)
    model.config.route_entropy_prior = use_entropy_prior
    return SparseLayerPlan(
        route_layers=route_list,
        sliding_window_layers=sliding,
        dense_layers=dense,
    )


def set_route_fusion_detach(model: nn.Module, detach: bool) -> int:
    """Toggle fusion stop-grad on every route layer for the next forward."""

    enabled = bool(detach)
    updated = 0
    for module in model.modules():
        if hasattr(module, "detach_fusion_weights"):
            module.detach_fusion_weights = enabled
            updated += 1
    if hasattr(model, "config"):
        model.config.route_detach_fusion_weights = enabled
    return updated
