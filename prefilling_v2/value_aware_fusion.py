"""Value-aware residual for 33-way route fusion.

Top-k is unchanged.  Remote fusion logits become QK + entropy prior + beta * s^V
where s^V scores each selected chunk's intra-chunk output against the query.
Local still uses local LSE.  beta=0 skips this path and matches s2.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def selected_chunk_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    *,
    query_block: int = 128,
) -> torch.Tensor:
    """Intra-chunk attention outputs for selected chunks.  [B, L, Hq, K, D]."""
    batch, seq_len, h_q, dim = q.shape
    h_kv = k.shape[2]
    groups = h_q // h_kv
    selected = indices.shape[-1]
    device = q.device
    key_ids = torch.arange(int(chunk_size), device=device)
    real_tok = (key_ids + 1).remainder(int(chunk_size)) != 0
    chunks = seq_len // int(chunk_size)
    outputs = q.new_zeros(batch, seq_len, h_q, selected, dim)
    scale = 1.0 / math.sqrt(dim)
    query_block = max(1, int(query_block))
    b_ix = torch.arange(batch, device=device)[:, None, None, None, None]
    h_ix = torch.arange(h_kv, device=device)[None, None, :, None, None]
    with torch.no_grad():
        for start in range(0, seq_len, query_block):
            stop = min(seq_len, start + query_block)
            idx = indices[:, start:stop].to(torch.long)
            invalid = (idx < 0) | (idx >= chunks)
            safe = idx.clamp(min=0, max=max(chunks - 1, 0))
            token_pos = safe.unsqueeze(-1) * int(chunk_size) + key_ids
            token_pos = token_pos.clamp(0, seq_len - 1)
            k_sel = k[b_ix, token_pos, h_ix]
            v_sel = v[b_ix, token_pos, h_ix]
            valid = key_valid[b_ix, token_pos] & real_tok & ~invalid.unsqueeze(-1)
            bq = stop - start
            queries = q[:, start:stop].reshape(batch, bq, h_kv, groups, dim).float()
            logits = torch.einsum("brhgd,brhksd->brhgks", queries, k_sel.float()) * scale
            logits = logits.masked_fill(~valid[:, :, :, None], torch.finfo(logits.dtype).min)
            empty = ~valid.any(-1)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.masked_fill(empty[:, :, :, None, :, None], 0.0)
            attended = torch.einsum("brhgks,brhksd->brhgkd", probs, v_sel.float())
            outputs[:, start:stop] = attended.reshape(batch, bq, h_q, selected, dim).to(
                q.dtype
            )
    return outputs


def value_fusion_bonus(
    q: torch.Tensor,
    chunk_out: torch.Tensor,
    query_proj: nn.Linear,
    value_proj: nn.Linear,
) -> torch.Tensor:
    """s^V = (W_q q)^T (W_v h_c) / sqrt(r).  Shape [B, L, Hq, K]."""
    rank = max(int(query_proj.out_features), 1)
    q_score = query_proj(q.float())
    h_score = value_proj(chunk_out.float())
    return torch.einsum("blhd,blhkd->blhk", q_score, h_score) / math.sqrt(rank)
