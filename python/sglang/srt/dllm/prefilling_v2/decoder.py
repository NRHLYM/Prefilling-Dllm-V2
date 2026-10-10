"""Confidence-threshold denoising of an answer canvas with the Prefilling-dLLM v2 stack.

The answer is denoised block by block. A block starts with one full forward of
the sequence, which commits every answer token whose confidence reaches the
threshold (and always the most confident one). The remaining tokens are
committed the same way from cached forwards that recompute only the block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.dllm.prefilling_v2.layout import build_blocks, build_layout
from sglang.srt.dllm.prefilling_v2.stack import PrefillingV2Stack


@dataclass(frozen=True)
class PrefillingV2GenerationStats:
    full_forwards: int
    cached_forwards: int
    physical_length: int
    rope_factor: float
    kv_in_buffers: bool


def select_confident(
    logits: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    threshold: float,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``active_rows`` to commit, and the token for each."""

    candidates = logits.index_select(0, active_rows).float().clone()
    candidates[:, mask_token_id] = float("-inf")
    confidence, token_ids = torch.softmax(candidates, dim=-1).max(dim=-1)
    selected = confidence >= threshold
    selected[torch.argmax(confidence)] = True
    return active_rows[selected], token_ids[selected]


@torch.inference_mode()
def generate(
    stack: PrefillingV2Stack,
    token_ids: torch.Tensor,
    *,
    answer_start: int,
    block_size: int,
    threshold: float,
    rope_factor: Optional[float] = None,
    kv_buffers_for: Optional[
        Callable[[int], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]
    ] = None,
) -> Tuple[torch.Tensor, PrefillingV2GenerationStats]:
    """Denoise ``token_ids[answer_start:]``, which must all be mask tokens.

    ``rope_factor`` fixes the YaRN factor; by default it follows the sequence
    length (``PrefillingV2Stack.rope_factor_for``). ``kv_buffers_for`` is asked, with
    the physical sequence length, for per-layer key and value buffers to keep
    the cache in; it may return None. Returns the answer tokens and the number
    of forwards spent.
    """

    cfg = stack.config
    device = stack.device
    layout = build_layout(
        token_ids,
        chunk_size=cfg.chunk_size,
        mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id,
    )
    blocks = build_blocks(
        layout,
        answer_start=answer_start,
        block_size=block_size,
        chunk_size=cfg.chunk_size,
    )
    input_ids = layout.input_ids.to(device)
    answer_positions = layout.text_positions[answer_start:].to(device)
    if bool((input_ids[answer_positions] != cfg.mask_token_id).any()):
        raise ValueError("the answer canvas must be fully masked")

    if rope_factor is None:
        rope_factor = stack.rope_factor_for(int(input_ids.numel()))
    kv_buffers = None
    if kv_buffers_for is not None:
        kv_buffers = kv_buffers_for(int(input_ids.numel()))
    full_forwards = cached_forwards = 0
    for block in blocks:
        answer = block.answer_positions.to(device)
        logits, cache = stack.prefill(
            input_ids,
            layout.key_valid,
            layout.position_ids,
            layout.routing_positions,
            block.predictor_positions,
            rope_factor=rope_factor,
            kv_buffers=kv_buffers,
        )
        full_forwards += 1
        rows, tokens = select_confident(
            logits,
            torch.arange(answer.numel(), device=device),
            threshold=threshold,
            mask_token_id=cfg.mask_token_id,
        )
        input_ids[answer[rows]] = tokens
        while True:
            active = torch.where(input_ids[answer] == cfg.mask_token_id)[0]
            if active.numel() == 0:
                break
            logits = stack.cached_forward(
                input_ids, layout.routing_positions, block, cache
            )
            cached_forwards += 1
            rows, tokens = select_confident(
                logits,
                active,
                threshold=threshold,
                mask_token_id=cfg.mask_token_id,
            )
            input_ids[answer[rows]] = tokens
        del cache

    return input_ids[answer_positions], PrefillingV2GenerationStats(
        full_forwards=full_forwards,
        cached_forwards=cached_forwards,
        physical_length=int(input_ids.numel()),
        rope_factor=float(rope_factor),
        kv_in_buffers=kv_buffers is not None,
    )
