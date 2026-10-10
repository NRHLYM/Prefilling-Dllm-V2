"""Tensor layout and chunk-mask helpers for the Dream HiLS kernel backend."""

from __future__ import annotations

from typing import Tuple

import torch


def _validate_group_layout(
    x: torch.Tensor,
    h_kv: int,
    logical_g: int,
    physical_g: int,
) -> None:
    if x.ndim < 2:
        raise ValueError(f"expected at least two dimensions, got {tuple(x.shape)}")
    if h_kv <= 0 or logical_g <= 0 or physical_g < logical_g:
        raise ValueError(
            "expected h_kv > 0 and physical_g >= logical_g > 0; "
            f"got h_kv={h_kv}, logical_g={logical_g}, physical_g={physical_g}"
        )
    expected_heads = h_kv * logical_g
    if x.shape[-2] != expected_heads:
        raise ValueError(
            f"head dimension must be h_kv * logical_g = {expected_heads}, "
            f"got {x.shape[-2]} for shape {tuple(x.shape)}"
        )


def group_pad_to(
    x: torch.Tensor,
    h_kv: int,
    logical_g: int,
    physical_g: int,
    value: float = 0.0,
) -> torch.Tensor:
    """Pad the per-KV-head query-group axis without expanding K/V tensors."""

    _validate_group_layout(x, h_kv, logical_g, physical_g)
    if physical_g == logical_g:
        return x

    prefix = x.shape[:-2]
    feature_dim = x.shape[-1]
    grouped = x.reshape(*prefix, h_kv, logical_g, feature_dim)
    pad_shape = (*prefix, h_kv, physical_g - logical_g, feature_dim)
    padding = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((grouped, padding), dim=-2).reshape(
        *prefix, h_kv * physical_g, feature_dim
    )


def group_unpad(
    x: torch.Tensor,
    h_kv: int,
    logical_g: int,
    physical_g: int,
) -> torch.Tensor:
    """Remove physical query-group padding and restore logical head layout."""

    if x.ndim < 2 or x.shape[-2] != h_kv * physical_g:
        raise ValueError(
            f"head dimension must be h_kv * physical_g = {h_kv * physical_g}, "
            f"got shape {tuple(x.shape)}"
        )
    prefix = x.shape[:-2]
    feature_dim = x.shape[-1]
    grouped = x.reshape(*prefix, h_kv, physical_g, feature_dim)
    return grouped[..., :logical_g, :].reshape(
        *prefix, h_kv * logical_g, feature_dim
    )


def chunk_aligned_local_bounds(
    seq_len: int,
    window: int,
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive-left/exclusive-right local bounds for every query."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    if chunk_size <= 0 or seq_len % chunk_size != 0:
        raise ValueError(
            f"seq_len={seq_len} must be divisible by chunk_size={chunk_size}"
        )

    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    raw_left = (positions - window).clamp_min(0)
    raw_right = (positions + window).clamp_max(seq_len - 1)
    left = torch.div(raw_left, chunk_size, rounding_mode="floor") * chunk_size
    right = (
        torch.div(raw_right, chunk_size, rounding_mode="floor") + 1
    ) * chunk_size
    return left, right.clamp_max(seq_len)


def remote_drop_mask(
    valid_mask: torch.Tensor,
    window: int,
    chunk_size: int,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the fused router's 1=drop mask for local or invalid chunks."""

    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must have shape [B, L], got {tuple(valid_mask.shape)}"
        )
    batch, seq_len = valid_mask.shape
    if chunk_size <= 0 or seq_len % chunk_size != 0:
        raise ValueError(
            f"sequence length {seq_len} must be divisible by chunk_size={chunk_size}"
        )
    if allowed_mask is not None and allowed_mask.shape != (
        batch,
        seq_len,
        seq_len,
    ):
        raise ValueError(
            "allowed_mask must have shape "
            f"{(batch, seq_len, seq_len)}, got {tuple(allowed_mask.shape)}"
        )

    device = valid_mask.device
    num_chunks = seq_len // chunk_size
    positions = torch.arange(seq_len, device=device)
    real_key = (positions + 1).remainder(chunk_size) != 0
    chunk_valid = (
        valid_mask.bool()
        .logical_and(real_key.unsqueeze(0))
        .reshape(batch, num_chunks, chunk_size)
        .any(dim=-1)
    )

    left, right = chunk_aligned_local_bounds(
        seq_len, window, chunk_size, device
    )
    left_chunk = torch.div(left, chunk_size, rounding_mode="floor")
    right_chunk = torch.div(right - 1, chunk_size, rounding_mode="floor")
    chunk_ids = torch.arange(num_chunks, device=device)
    local = (chunk_ids[None, :] >= left_chunk[:, None]) & (
        chunk_ids[None, :] <= right_chunk[:, None]
    )
    drop = local[None, :, :] | ~chunk_valid[:, None, :]
    if allowed_mask is not None:
        landmark_positions = torch.arange(
            chunk_size - 1,
            seq_len,
            chunk_size,
            device=device,
        )
        query_chunk_allowed = allowed_mask.to(dtype=torch.bool).index_select(
            -1, landmark_positions
        )
        drop = drop | ~query_chunk_allowed
    return drop.to(torch.int32)
