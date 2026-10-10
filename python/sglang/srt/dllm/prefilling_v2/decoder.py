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

from sglang.srt.dllm.prefilling_v2.layout import (
    PrefillingV2Block,
    PrefillingV2Layout,
    build_blocks,
    build_layout,
)
from sglang.srt.dllm.prefilling_v2.stack import PrefillingV2Cache, PrefillingV2Stack


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


@dataclass
class PrefillingV2Plan:
    """A request laid out for the stack: where its tokens sit and its blocks."""

    layout: PrefillingV2Layout
    blocks: List[PrefillingV2Block]
    # Physical token ids on the stack's device; denoising fills them in place.
    input_ids: torch.Tensor
    answer_positions: torch.Tensor
    rope_factor: float

    @property
    def physical_length(self) -> int:
        return int(self.input_ids.numel())


def plan_request(
    stack: PrefillingV2Stack,
    token_ids: torch.Tensor,
    *,
    answer_start: int,
    block_size: int,
    rope_factor: Optional[float] = None,
) -> PrefillingV2Plan:
    """Lay out ``token_ids``, whose tokens from ``answer_start`` on are masks.

    ``rope_factor`` fixes the YaRN factor; by default it follows the sequence
    length (``PrefillingV2Stack.rope_factor_for``).
    """

    cfg = stack.config
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
    input_ids = layout.input_ids.to(stack.device)
    answer_positions = layout.text_positions[answer_start:].to(stack.device)
    if bool((input_ids[answer_positions] != cfg.mask_token_id).any()):
        raise ValueError("the answer canvas must be fully masked")
    if rope_factor is None:
        rope_factor = stack.rope_factor_for(int(input_ids.numel()))
    return PrefillingV2Plan(
        layout, blocks, input_ids, answer_positions, float(rope_factor)
    )


@torch.inference_mode()
def start_block(
    stack: PrefillingV2Stack,
    plan: PrefillingV2Plan,
    block: PrefillingV2Block,
    *,
    threshold: float,
    kv_buffers: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> PrefillingV2Cache:
    """Forward the whole sequence and commit the block's confident tokens."""

    answer = block.answer_positions.to(stack.device)
    logits, cache = stack.prefill(
        plan.input_ids,
        plan.layout.key_valid,
        plan.layout.position_ids,
        plan.layout.routing_positions,
        block.predictor_positions,
        rope_factor=plan.rope_factor,
        kv_buffers=kv_buffers,
    )
    rows, tokens = select_confident(
        logits,
        torch.arange(answer.numel(), device=stack.device),
        threshold=threshold,
        mask_token_id=stack.config.mask_token_id,
    )
    plan.input_ids[answer[rows]] = tokens
    return cache


@torch.inference_mode()
def finish_block(
    stack: PrefillingV2Stack,
    plan: PrefillingV2Plan,
    block: PrefillingV2Block,
    cache: PrefillingV2Cache,
    *,
    threshold: float,
) -> int:
    """Run cached forwards until the block has no mask left; returns how many."""

    mask_token_id = stack.config.mask_token_id
    answer = block.answer_positions.to(stack.device)
    cached_forwards = 0
    while True:
        active = torch.where(plan.input_ids[answer] == mask_token_id)[0]
        if active.numel() == 0:
            return cached_forwards
        logits = stack.cached_forward(
            plan.input_ids, plan.layout.routing_positions, block, cache
        )
        cached_forwards += 1
        rows, tokens = select_confident(
            logits, active, threshold=threshold, mask_token_id=mask_token_id
        )
        plan.input_ids[answer[rows]] = tokens


def close_at_stop_token(
    stack: PrefillingV2Stack, plan: PrefillingV2Plan, block: PrefillingV2Block
) -> bool:
    """End the answer after ``block`` if the block holds the end-of-sequence token.

    The blocks after it are filled with that token instead of being denoised,
    which saves each of them a full forward. Returns whether the answer ended.
    """

    eos = stack.config.eos_token_id
    answer = block.answer_positions.to(stack.device)
    if not bool((plan.input_ids[answer] == eos).any()):
        return False
    rest = plan.answer_positions[plan.answer_positions > answer[-1]]
    plan.input_ids[rest] = eos
    return True


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
    stop_at_eos: bool = True,
) -> Tuple[torch.Tensor, PrefillingV2GenerationStats]:
    """Denoise ``token_ids[answer_start:]``, which must all be mask tokens.

    ``kv_buffers_for`` is asked, with the physical sequence length, for
    per-layer key and value buffers to keep the cache in; it may return None.
    With ``stop_at_eos`` a block that holds the end-of-sequence token ends the
    answer (``close_at_stop_token``). Returns the answer tokens and the number
    of forwards spent.
    """

    plan = plan_request(
        stack,
        token_ids,
        answer_start=answer_start,
        block_size=block_size,
        rope_factor=rope_factor,
    )
    kv_buffers = None
    if kv_buffers_for is not None:
        kv_buffers = kv_buffers_for(plan.physical_length)
    full_forwards = cached_forwards = 0
    for block in plan.blocks:
        cache = start_block(
            stack, plan, block, threshold=threshold, kv_buffers=kv_buffers
        )
        full_forwards += 1
        cached_forwards += finish_block(stack, plan, block, cache, threshold=threshold)
        del cache
        if stop_at_eos and close_at_stop_token(stack, plan, block):
            break

    return plan.input_ids[plan.answer_positions], PrefillingV2GenerationStats(
        full_forwards=full_forwards,
        cached_forwards=cached_forwards,
        physical_length=plan.physical_length,
        rope_factor=plan.rope_factor,
        kv_in_buffers=kv_buffers is not None,
    )
