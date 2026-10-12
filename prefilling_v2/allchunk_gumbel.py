"""Sampled-query all-candidate straight-through routing gradients (S4).

Forward stays hard top-k. Sampled supervised queries replace only the gate
Jacobian with an all-legal-chunk soft mixture of exact per-chunk attention.
Pass noise=None for hard routing (no Gumbel). Do not stack with lmk_ce_ste.
"""
from __future__ import annotations

import math

import torch

_ANNEAL_SCHEDULES = {"cosine", "linear", "none"}


def allchunk_st_temperature(step, max_steps, start, end, schedule="cosine"):
    """ST-only softmax τ. step is 1-indexed; step 1 = start, step max_steps = end."""
    start = float(start)
    end = float(end)
    schedule = str(schedule)
    if start <= 0 or end <= 0:
        raise ValueError("all-chunk ST temperature must be positive")
    if schedule not in _ANNEAL_SCHEDULES:
        raise ValueError(f"unsupported all-chunk ST anneal={schedule}")
    if schedule == "none" or abs(start - end) <= 1e-12 or int(max_steps) <= 1:
        return start
    progress = (int(step) - 1) / max(int(max_steps) - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return start + (end - start) * progress
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def set_allchunk_st_temperature(core, temperature):
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    value = float(temperature)
    if value <= 0:
        raise ValueError("all-chunk ST temperature must be positive")
    for module in core.modules():
        if isinstance(module, KernelDreamFullRouteAttention):
            module.allchunk_st_temperature = value


def prepare_allchunk_queries(core, batch, count):
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    valid = batch["attention_mask"].bool() & batch["labels"].ne(-100)
    layers = [m for m in core.modules() if isinstance(m, KernelDreamFullRouteAttention)]
    count = min(count, int(valid.sum(-1).min()))
    if not layers or count <= 0:
        raise ValueError("all-chunk ST requires route and supervised query positions")
    devices = [valid.device.index] if valid.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        for layer in layers:
            layer.allchunk_st_positions = torch.stack([
                (ids := torch.where(row)[0])[torch.randperm(ids.numel(), device=valid.device)[:count]]
                for row in valid])


def detach_sampled_gate(weights, positions):
    """Keep the hard gate values and content gradients, replace only its gate Jacobian."""
    mask = torch.zeros(weights.shape[:2], device=weights.device, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    mask = mask.reshape(*mask.shape, *([1] * (weights.ndim - 2)))
    return torch.where(mask, weights.detach(), weights)


@torch.no_grad()
def exact_chunk_outputs(q, k, v, key_valid, positions, chunk_size):
    """Actual per-chunk token attention, not a mean-V proxy; bounded query blocks."""
    b, n, h, d = k.shape
    groups = q.shape[2] // h
    chunks = n // chunk_size
    bi = torch.arange(b, device=q.device)[:, None]
    body = key_valid.reshape(b, chunks, chunk_size).bool().clone()
    body[:, :, -1] = False
    nonempty = body.any(-1)
    with torch.autocast(q.device.type, enabled=False):
        keys = k.detach().float().reshape(b, chunks, chunk_size, h, d)
        values = v.detach().float().reshape_as(keys)
        outputs = []
        for start in range(0, positions.shape[1], 4):
            queries = q[bi, positions[:, start:start+4]].detach().float().reshape(b, -1, h, groups, d)
            scores = torch.einsum("brhgd,bcshd->brhgcs", queries, keys) / math.sqrt(d)
            scores = scores.masked_fill(~body[:, None, None, None], -torch.inf)
            scores = torch.where(nonempty[:, None, None, None, :, None], scores, 0.)
            probs = scores.softmax(-1) * body[:, None, None, None]
            attended = torch.einsum("brhgcs,bcshd->brhgcd", probs, values)
            outputs.append(attended.reshape(b, -1, h * groups, chunks, d).to(q.dtype))
    return torch.cat(outputs, dim=1)


def allchunk_soft_output(route_q, landmarks, prior, local_lse, dropped, local_output,
                         chunk_outputs, positions, noise, temperature):
    b, chunks, h, groups, d = landmarks.shape
    bi = torch.arange(b, device=route_q.device)[:, None]
    queries = route_q[bi, positions].reshape(b, -1, h, groups, d)
    with torch.autocast(route_q.device.type, enabled=False):
        logits = torch.einsum("brhgd,bchgd->brhgc", queries.float(), landmarks.float()) / math.sqrt(d)
        logits = logits + prior.float().permute(0, 2, 3, 1)[:, None]
        logits = logits.reshape(b, -1, h * groups, chunks)
        if noise is not None:
            logits = logits + noise[..., :chunks].float()
        logits = logits.masked_fill(dropped[bi, positions, None].bool(), -torch.inf)
        local = local_lse[bi, positions].float().unsqueeze(-1)
        all_logits = torch.cat((logits, local), -1) / temperature
        empty = torch.isneginf(all_logits).all(-1, keepdim=True)
        probs = torch.where(empty, 0., torch.where(empty, 0., all_logits).softmax(-1))
        result = torch.einsum("brhc,brhcd->brhd", probs[..., :chunks], chunk_outputs.detach().float())
        result = result + probs[..., -1:] * local_output[bi, positions].detach().float()
    return result, logits


def attach_allchunk_st(output, q, k, v, route_q, landmarks, prior, local_lse, dropped,
                       key_valid, local_output, positions, noise, chunk_size, temperature,
                       observer=None):
    responses = exact_chunk_outputs(q, k, v, key_valid, positions, chunk_size)
    soft, logits = allchunk_soft_output(route_q, landmarks, prior, local_lse, dropped,
        local_output, responses, positions, noise, temperature)
    if observer is not None:
        observer(logits, soft, responses)
    correction = (soft - soft.detach()).to(output.dtype)
    index = positions[:, :, None, None].expand_as(correction)
    delta = torch.zeros_like(output).scatter_add(1, index, correction)
    return output + delta
