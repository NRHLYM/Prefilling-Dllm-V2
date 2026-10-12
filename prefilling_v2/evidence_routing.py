"""Fixed-evidence supervision for Q-Cal chunk routing."""

from __future__ import annotations

import math

import torch


def evidence_route_loss(
    route_q: torch.Tensor,
    landmark_keys: torch.Tensor,
    prior_bias: torch.Tensor,
    local_lse: torch.Tensor,
    dropped: torch.Tensor,
    query_mask: torch.Tensor,
    evidence_chunks: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive CE over the chunk priorities consumed by top-k routing."""

    if route_q.ndim != 4 or landmark_keys.ndim != 5:
        raise ValueError("invalid route query or landmark layout")
    batch, length, query_heads, dim = route_q.shape
    if landmark_keys.shape[0] != batch or landmark_keys.shape[-1] != dim:
        raise ValueError("route query and landmark shapes are incompatible")
    chunks, kv_heads, groups = landmark_keys.shape[1:4]
    if query_heads != kv_heads * groups:
        raise ValueError("query heads do not match grouped landmark heads")
    if dropped.shape != (batch, length, chunks):
        raise ValueError("drop mask does not match route layout")
    if query_mask.shape != (batch, length):
        raise ValueError("query mask does not match route layout")
    if evidence_chunks.shape != (batch, chunks):
        raise ValueError("evidence chunks do not match route layout")
    if local_lse.shape != (batch, length, kv_heads, groups):
        raise ValueError("local LSE does not match grouped route layout")

    grouped_q = route_q.float().reshape(
        batch, length, kv_heads, groups, dim
    )
    batch_rows, query_positions = torch.where(query_mask.bool())
    if batch_rows.numel() == 0:
        return route_q.float().sum() * 0.0
    remote = ~dropped[batch_rows, query_positions].bool()
    positives = evidence_chunks.bool()[batch_rows] & remote
    valid_rows = positives.any(-1)
    if not bool(valid_rows.any()):
        return route_q.float().sum() * 0.0
    batch_rows = batch_rows[valid_rows]
    query_positions = query_positions[valid_rows]
    remote = remote[valid_rows]
    positives = positives[valid_rows]

    # Only Q-Cal receives auxiliary gradients. LMK pooling and entropy-prior
    # values are fixed features for this controlled experiment.
    logits = torch.einsum(
        "rhgd,rchgd->rhgc",
        grouped_q[batch_rows, query_positions],
        landmark_keys.detach().float()[batch_rows],
    ) * (1.0 / math.sqrt(dim))
    logits = logits + prior_bias.detach().float().permute(0, 2, 3, 1)[
        batch_rows
    ]

    # A route layer selects chunks per KV head after normalizing each query head with
    # both its local and remote attention mass, then max-pooling over the GQA
    # group. Supervise that exact deterministic priority rather than forcing
    # every query head to classify the evidence chunk independently.
    remote_lse = torch.logsumexp(logits, dim=-1)
    total_lse = torch.logaddexp(
        local_lse.detach().float()[batch_rows, query_positions], remote_lse
    )
    priorities = (logits - total_lse.unsqueeze(-1)).amax(dim=2)
    sentinel = torch.finfo(torch.float32).min
    log_probs = priorities.masked_fill(~remote[:, None, :], sentinel).log_softmax(-1)
    target = positives.float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    losses = -(log_probs * target[:, None, :]).sum(-1)
    return losses.mean()


def prepare_evidence_route_losses(
    model: torch.nn.Module,
    query_mask: torch.Tensor,
    evidence_chunks: torch.Tensor,
) -> list[torch.nn.Module]:
    from prefilling_v2.attention import KernelDreamFullRouteAttention

    layers = [
        module
        for module in model.modules()
        if isinstance(module, KernelDreamFullRouteAttention)
    ]
    if not layers:
        raise RuntimeError("evidence route supervision requires route layers")
    for layer in layers:
        layer.evidence_route_query_mask = query_mask
        layer.evidence_route_chunks = evidence_chunks
        layer.evidence_route_loss = None
    return layers


def collect_evidence_route_loss(layers: list[torch.nn.Module]) -> torch.Tensor:
    losses = [layer.evidence_route_loss for layer in layers]
    if any(loss is None for loss in losses):
        raise RuntimeError("missing evidence route loss from a route layer")
    return torch.stack(losses).mean()
