"""Physical token layout for Prefilling-dLLM v2.

A request is a run of text tokens: the prompt followed by the masked answer
canvas. Prefilling-dLLM v2 reads it in chunks of ``chunk_size`` positions. A chunk holds
``chunk_size - 1`` text slots and ends with one routing slot, which stores the
model's mask token. Text positions are numbered consecutively; a routing slot
takes the position one past its chunk's last text slot, which is also the
position of the next chunk's first text slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillingV2Layout:
    # All tensors are 1-D over the physical sequence unless noted.
    input_ids: torch.Tensor
    key_valid: torch.Tensor
    position_ids: torch.Tensor
    # Physical index of each text token, [num_text_tokens].
    text_positions: torch.Tensor
    # Physical index of each chunk's routing slot, [num_chunks].
    routing_positions: torch.Tensor


@dataclass(frozen=True)
class PrefillingV2Block:
    """One block of answer tokens that is denoised against a prefilled cache."""

    # Physical positions of the block's answer tokens.
    answer_positions: torch.Tensor
    # Physical position whose output predicts each answer token.
    predictor_positions: torch.Tensor
    # Chunks that hold an answer token, and their routing slots.
    affected_chunks: torch.Tensor
    # Sorted positions recomputed by each cached forward.
    query_positions: torch.Tensor
    # Sorted positions whose keys and values each cached forward replaces.
    kv_update_positions: torch.Tensor


def physical_text_positions(num_text_slots: int, chunk_size: int) -> torch.Tensor:
    slots = torch.arange(num_text_slots, dtype=torch.long)
    return slots + torch.div(slots, chunk_size - 1, rounding_mode="floor")


def build_layout(
    token_ids: torch.Tensor,
    *,
    chunk_size: int,
    mask_token_id: int,
    pad_token_id: int,
) -> PrefillingV2Layout:
    """Lay ``token_ids`` out in the fewest chunks that hold them."""

    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError("token_ids must be a nonempty 1-D tensor")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be at least 2, got {chunk_size}")

    text_per_chunk = chunk_size - 1
    num_tokens = int(token_ids.numel())
    num_chunks = -(-num_tokens // text_per_chunk)
    text_slots = num_chunks * text_per_chunk

    text_ids = torch.full((text_slots,), int(pad_token_id), dtype=torch.long)
    text_ids[:num_tokens] = token_ids.to(dtype=torch.long, device="cpu")
    valid = torch.zeros(text_slots, dtype=torch.bool)
    valid[:num_tokens] = True

    text_chunks = text_ids.view(num_chunks, text_per_chunk)
    valid_chunks = valid.view(num_chunks, text_per_chunk)
    routing_ids = torch.full((num_chunks, 1), int(mask_token_id), dtype=torch.long)
    input_ids = torch.cat((text_chunks, routing_ids), dim=-1).reshape(-1)
    key_valid = torch.cat(
        (valid_chunks, valid_chunks.any(dim=-1, keepdim=True)), dim=-1
    ).reshape(-1)

    text_position_ids = torch.arange(text_slots, dtype=torch.long).view(
        num_chunks, text_per_chunk
    )
    position_ids = torch.cat(
        (text_position_ids, text_position_ids[:, -1:] + 1), dim=-1
    ).reshape(-1)

    return PrefillingV2Layout(
        input_ids=input_ids,
        key_valid=key_valid,
        position_ids=position_ids,
        text_positions=physical_text_positions(text_slots, chunk_size)[:num_tokens],
        routing_positions=torch.arange(
            chunk_size - 1, num_chunks * chunk_size, chunk_size, dtype=torch.long
        ),
    )


def build_blocks(
    layout: PrefillingV2Layout,
    *,
    answer_start: int,
    block_size: int,
    chunk_size: int,
) -> list[PrefillingV2Block]:
    """Split the text tokens from ``answer_start`` on into denoising blocks."""

    num_tokens = int(layout.text_positions.numel())
    if not 0 < answer_start < num_tokens:
        raise ValueError(
            "the answer needs a preceding prompt token and at least one slot; "
            f"got answer_start={answer_start} of {num_tokens} tokens"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    blocks = []
    for start in range(answer_start, num_tokens, block_size):
        stop = min(start + block_size, num_tokens)
        answer = layout.text_positions[start:stop]
        predictor = layout.text_positions[start - 1 : stop - 1]
        chunks = torch.unique(
            torch.div(answer, chunk_size, rounding_mode="floor"), sorted=True
        )
        routing = chunks * chunk_size + (chunk_size - 1)
        blocks.append(
            PrefillingV2Block(
                answer_positions=answer.clone(),
                predictor_positions=predictor.clone(),
                affected_chunks=chunks,
                query_positions=torch.unique(
                    torch.cat((answer, predictor, routing)), sorted=True
                ),
                kv_update_positions=torch.unique(
                    torch.cat((answer, routing)), sorted=True
                ),
            )
        )
    return blocks
