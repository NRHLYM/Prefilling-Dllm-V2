"""HoPE in-range mask on an already-built Dream RoPE module.

YaRN stretches RoPE frequencies to 16k. HoPE instead zeros rotary components
whose period exceeds the 2k training window, so those dims become NoPE at 16k.
"""

from __future__ import annotations

import math

import torch


def apply_hope_inrange(
    model: torch.nn.Module,
    *,
    context_length: int,
    period_multiplier: float = 1.0,
) -> int:
    if int(context_length) <= 0:
        raise ValueError("HoPE context_length must be positive")
    if float(period_multiplier) <= 0:
        raise ValueError("HoPE period_multiplier must be positive")
    threshold = 2.0 * math.pi * float(period_multiplier) / float(context_length)
    patched = 0
    for module in model.modules():
        inv = getattr(module, "inv_freq", None)
        if inv is None or not torch.is_tensor(inv):
            continue
        masked = torch.where(inv >= threshold, inv, torch.zeros_like(inv))
        inv.copy_(masked)
        original = getattr(module, "original_inv_freq", None)
        if original is not None and torch.is_tensor(original):
            original.copy_(masked)
        patched += 1
    if patched == 0:
        raise RuntimeError("HoPE found no inv_freq buffers on the Dream model")
    return patched
