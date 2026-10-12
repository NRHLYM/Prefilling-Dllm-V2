"""Normalization of selected-chunk scores against the local window.

Copied unchanged from the attention module of the Prefilling-dLLM v2 training tree
(``_route_weights_torch``).
"""

from __future__ import annotations

import torch
from torch import nn


def route_weights_torch(
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


