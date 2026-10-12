"""Shared bidirectional window attention for route / NSA / DSA / SWA.

Prefill and Fast-dLLM decode both use FlashAttention `window_size=(W, W)`.
Callers pass a 2D `key_valid` mask only — never a dense L×L `allowed_mask`.
"""

from __future__ import annotations

import inspect
from typing import Tuple

import torch


def _validate_layout(q, k, v, key_valid) -> None:
    if q.device.type != "cuda":
        raise ValueError("local attention requires CUDA tensors")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("local attention requires BF16 q, k, and v")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have BLHD layout")
    batch, seq_len, h_q, dim = q.shape
    if k.shape[0] != batch or v.shape != k.shape:
        raise ValueError(
            f"q/k/v batch mismatch: {q.shape}, {k.shape}, {v.shape}"
        )
    if k.shape[-1] != dim or h_q % k.shape[2] != 0:
        raise ValueError("query heads must be divisible by KV heads with equal head_dim")
    if key_valid.shape[0] != batch or key_valid.shape[-1] != k.shape[1]:
        raise ValueError(
            f"key_valid must have shape {(batch, k.shape[1])}, got {tuple(key_valid.shape)}"
        )


def _window_key_valid(
    key_valid: torch.Tensor,
    *,
    chunk_size: int,
    skip_inert_slots: bool,
) -> torch.Tensor:
    valid = key_valid.to(dtype=torch.bool)
    if not skip_inert_slots:
        return valid
    seq_len = valid.shape[-1]
    positions = torch.arange(seq_len, device=valid.device)
    is_real_key = (positions + 1) % chunk_size != 0
    return valid & is_real_key


def _apply_key_valid(k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor):
    if bool(valid.all()):
        return k, v
    scale = valid.to(dtype=v.dtype).unsqueeze(-1).unsqueeze(-1)
    return k * scale, v * scale


def _flash_window(
    window: int,
    skip_inert_slots: bool,
    chunk_size: int,
) -> int:
    if skip_inert_slots:
        return int(window + chunk_size - 1)
    return int(window)


def _flash_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flash_attn import flash_attn_func

    kwargs = {
        "dropout_p": 0.0,
        "softmax_scale": q.shape[-1] ** -0.5,
        "causal": False,
        "window_size": (int(window), int(window)),
    }
    signature = inspect.signature(flash_attn_func)
    if "window_size" not in signature.parameters:
        raise RuntimeError("flash_attn_func does not support window_size")
    if "return_softmax_lse" in signature.parameters:
        kwargs["return_softmax_lse"] = True
        result = flash_attn_func(q, k, v, **kwargs)
    elif "return_attn_probs" in signature.parameters:
        kwargs["return_attn_probs"] = True
        result = flash_attn_func(q, k, v, **kwargs)
    else:
        raise RuntimeError(
            "flash_attn_func cannot return LSE; install a newer flash-attn"
        )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("flash_attn_func did not return (output, lse)")
    output, lse = result[0], result[1]
    if lse.ndim != 3:
        raise RuntimeError(f"unexpected flash-attn LSE shape {tuple(lse.shape)}")
    if lse.shape[1] == q.shape[2] and lse.shape[2] == q.shape[1]:
        lse = lse.transpose(1, 2).contiguous()
    elif lse.shape[1] == q.shape[1] and lse.shape[2] == q.shape[2]:
        lse = lse.contiguous()
    else:
        raise RuntimeError(
            f"flash-attn LSE {tuple(lse.shape)} does not match q {tuple(q.shape)}"
        )
    return output, lse


def bidir_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: torch.Tensor,
    window: int,
    chunk_size: int,
    allowed_mask: torch.Tensor | None = None,
    skip_inert_slots: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional local GQA attention and natural-log LSE."""

    _validate_layout(q, k, v, key_valid)
    seq_len = q.shape[1]
    if k.shape[1] != seq_len:
        raise ValueError("prefill local attention requires equal Q/K lengths")
    if window < 0 or chunk_size <= 0 or seq_len % chunk_size != 0:
        raise ValueError(
            f"invalid window/chunk layout: L={seq_len}, W={window}, S={chunk_size}"
        )
    if allowed_mask is not None:
        raise ValueError(
            "bidir_local_attention only accepts 2D key_valid; "
            "do not pass a dense L×L allowed_mask"
        )
    valid = _window_key_valid(
        key_valid, chunk_size=chunk_size, skip_inert_slots=skip_inert_slots
    )
    masked_k, masked_v = _apply_key_valid(k, v, valid)
    return _flash_window_attention(
        q,
        masked_k,
        masked_v,
        _flash_window(window, skip_inert_slots, chunk_size),
    )


def cached_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: torch.Tensor,
    query_positions: torch.Tensor,
    window: int,
    chunk_size: int,
    skip_inert_slots: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attend Fast-dLLM query rows to a windowed K/V cache with the same FA kernel."""

    _validate_layout(q, k, v, key_valid)
    batch, query_len, h_q, dim = q.shape
    seq_len = k.shape[1]
    if query_positions.shape != (query_len,):
        raise ValueError(
            f"query_positions must have shape {(query_len,)}, "
            f"got {tuple(query_positions.shape)}"
        )
    if query_positions.device != q.device or query_positions.dtype != torch.long:
        raise ValueError("query_positions must be CUDA torch.long")
    if query_len == 0:
        empty_lse = q.new_empty((batch, 0, h_q), dtype=torch.float32)
        return q.new_empty((batch, 0, h_q, dim)), empty_lse
    if torch.unique(query_positions).numel() != query_len:
        raise ValueError("query_positions must be unique")
    if bool(((query_positions < 0) | (query_positions >= seq_len)).any()):
        raise ValueError("query_positions must be in cache range")
    if window < 0 or chunk_size <= 0 or seq_len % chunk_size:
        raise ValueError(
            f"invalid window/chunk layout: L={seq_len}, W={window}, S={chunk_size}"
        )

    flash_window = _flash_window(window, skip_inert_slots, chunk_size)
    valid = _window_key_valid(
        key_valid, chunk_size=chunk_size, skip_inert_slots=skip_inert_slots
    )
    masked_k, masked_v = _apply_key_valid(k, v, valid)
    pos_min = int(query_positions.min().item())
    pos_max = int(query_positions.max().item())
    kv_lo = max(0, pos_min - flash_window)
    kv_hi = min(seq_len, pos_max + flash_window + 1)
    span = kv_hi - kv_lo
    rows = query_positions - kv_lo
    q_span = q.new_zeros((batch, span, h_q, dim))
    q_span[:, rows] = q
    output_span, lse_span = _flash_window_attention(
        q_span,
        masked_k[:, kv_lo:kv_hi],
        masked_v[:, kv_lo:kv_hi],
        flash_window,
    )
    return output_span[:, rows].contiguous(), lse_span[:, rows].contiguous()
