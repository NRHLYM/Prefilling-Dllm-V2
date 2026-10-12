"""Sampled dense-to-chunk KL on the router's remote candidate domain."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def verify_yarn(model, args):
    import math
    rope = args.model_rope_scaling
    assert rope['original_max_position_embeddings'] == 2048 and rope['factor'] == 16
    assert args.model_max_position_embeddings == 32768
    found = 0
    for module in model.modules():
        if not hasattr(module, 'inv_freq') or not hasattr(module, 'rope_type'):
            continue
        assert module.rope_type == 'yarn'
        inv = module.inv_freq.detach().float().cpu()
        dim, base = inv.numel() * 2, args.model_rope_theta
        low = max(math.floor(dim * math.log(2048 / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
        high = min(math.ceil(dim * math.log(2048 / (2 * math.pi)) / (2 * math.log(base))), dim - 1)
        ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
        original = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))
        expected = original / 16 * ramp + original * (1 - ramp)
        torch.testing.assert_close(inv, expected, rtol=1e-5, atol=1e-8)
        assert abs(float(module.attention_scaling) - (1 + 0.1 * math.log(16))) < 1e-6
        found += 1
    if not found:
        raise RuntimeError('no runtime rotary module found')
    print(f'[yarn-verified] original=2048 max=32768 factor=16 modules={found}', flush=True)


def chunk_distillation_loss(q, k, landmark_keys, prior_bias, key_valid,
                            dropped, chunk_size, sample_count, packed_allowed=None,
                            positions=None):
    batch, length, heads, dim = q.shape
    chunks, kv_heads, groups = landmark_keys.shape[1:4]
    if length != chunks * chunk_size or heads != kv_heads * groups:
        raise ValueError("incompatible chunk distillation layouts")
    text_valid = key_valid.bool() & ((torch.arange(length, device=q.device) + 1) % chunk_size != 0)[None]
    if positions is None:
        count = min(sample_count, int(text_valid.sum(-1).min()))
        if count <= 0:
            return (landmark_keys.float().sum() + prior_bias.float().sum()) * 0
        # Sampling must not change denoising/dropout/Gumbel RNG in the paired arm.
        devices = [q.device.index] if q.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            positions = torch.stack([
                (ids := torch.where(valid)[0])[torch.randperm(ids.numel(), device=q.device)[:count]]
                for valid in text_valid
            ])
    bi = torch.arange(batch, device=q.device)[:, None]
    sampled_q = q[bi, positions].reshape(batch, positions.shape[1], kv_heads, groups, dim)
    remote = ~dropped[bi, positions].bool()
    allowed = text_valid[:, None, :] & remote.repeat_interleave(chunk_size, -1)
    if packed_allowed is not None:
        allowed = allowed & packed_allowed[bi, positions].bool()
    valid_chunks = allowed.reshape(batch, positions.shape[1], chunks, chunk_size).any(-1)
    valid_rows = valid_chunks.any(-1)
    with torch.no_grad():
        teacher_scores = torch.einsum("brhgd,bnhd->brhgn", sampled_q.detach().float(), k.detach().float()) * dim**-0.5
        teacher_scores = teacher_scores.masked_fill(~allowed[:, :, None, None], float("-inf"))
        teacher_scores = torch.where(valid_rows[:, :, None, None, None], teacher_scores, torch.zeros_like(teacher_scores))
        teacher = teacher_scores.softmax(-1).reshape(batch, positions.shape[1], kv_heads, groups, chunks, chunk_size).sum(-1)
        teacher = teacher * valid_chunks[:, :, None, None]
    student = torch.einsum("brhgd,bchgd->brhgc", sampled_q.float(), landmark_keys.float()) * dim**-0.5
    student = student + prior_bias.float().permute(0, 2, 3, 1)[:, None]
    # A finite sentinel avoids 0 * inf in KL for masked chunks.
    student = student.masked_fill(~valid_chunks[:, :, None, None], torch.finfo(torch.float32).min)
    losses = F.kl_div(student.log_softmax(-1), teacher, reduction="none").sum(-1)
    return (losses * valid_rows[:, :, None, None]).sum() / (valid_rows.sum() * heads).clamp_min(1)


def prepare_chunk_losses(model, sample_count):
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    layers = [m for m in model.modules() if isinstance(m, KernelDreamFullRouteAttention)]
    if not layers:
        raise RuntimeError("chunk distillation requires route kernel attention layers")
    for layer in layers:
        layer.chunk_aux_queries = sample_count
        layer.chunk_aux_loss = None
    return layers


def collect_chunk_loss(layers):
    losses = [m.chunk_aux_loss for m in layers]
    if any(loss is None for loss in losses):
        raise RuntimeError("missing chunk distillation loss from a route layer")
    return torch.stack(losses).mean()
