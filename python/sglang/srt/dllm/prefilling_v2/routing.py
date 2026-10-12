"""Dream G=7 adapters for the chunk-selection and selected-chunk kernels in ``ops``."""

from __future__ import annotations

import torch

from sglang.srt.dllm.prefilling_v2.kernel_utils import group_pad_to, group_unpad
from sglang.srt.dllm.prefilling_v2.ops.hils_fwd_bwd_head import HiLS_block_M_head
from sglang.srt.dllm.prefilling_v2.ops.topk_head_softmax import (
    online_softmax_topk_head,
    ref_softmax_topk_max_pooling,
)


_LOGICAL_G = 7
_PHYSICAL_G = 8
_DUMMY_BIAS = -1.0e9


def _pad_group_scalar(
    x: torch.Tensor,
    h_kv: int,
    value: float,
) -> torch.Tensor:
    return group_pad_to(
        x.reshape(*x.shape[:-2], h_kv * _LOGICAL_G, 1),
        h_kv,
        _LOGICAL_G,
        _PHYSICAL_G,
        value=value,
    ).squeeze(-1)


def _validate_g7_qkv(q: torch.Tensor, k: torch.Tensor) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"q and k must be BLHD tensors, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    batch, seq_len, h_q, dim = q.shape
    if k.shape[:2] != (batch, seq_len) or k.shape[-1] != dim:
        raise ValueError(
            f"incompatible q/k shapes {tuple(q.shape)} and {tuple(k.shape)}"
        )
    h_kv = k.shape[2]
    if h_q != h_kv * _LOGICAL_G:
        raise ValueError(
            f"Dream G7 requires h_q == 7 * h_kv, got h_q={h_q}, h_kv={h_kv}"
        )
    return batch, seq_len, h_q, h_kv, dim


def route_topk_g7(
    q: torch.Tensor,
    lmks: torch.Tensor,
    local_lse: torch.Tensor,
    prior_bias: torch.Tensor,
    drop_mask: torch.Tensor,
    topk: int,
    chunk_size: int,
    window: int,
    *,
    training: bool = True,
    use_gumbel: bool = False,
    gumbel_scale: float = 1.0,
    return_selected_gumbel: bool = False,
    selection_mode: str = "post_softmax",
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Route logical G=7 query heads through the official physical-G=8 kernel."""

    if q.ndim != 4 or lmks.ndim != 5:
        raise ValueError(
            f"expected q [B,L,Hq,D] and lmks [B,C,Hkv,7,D], got "
            f"{tuple(q.shape)} and {tuple(lmks.shape)}"
        )
    batch, seq_len, h_q, dim = q.shape
    if lmks.shape[0] != batch or lmks.shape[3:] != (_LOGICAL_G, dim):
        raise ValueError(f"invalid landmark shape {tuple(lmks.shape)} for q {tuple(q.shape)}")
    chunks, h_kv = lmks.shape[1:3]
    if h_q != h_kv * _LOGICAL_G:
        raise ValueError(f"expected {h_kv * _LOGICAL_G} query heads, got {h_q}")
    if local_lse.shape != (batch, seq_len, h_kv, _LOGICAL_G):
        raise ValueError(f"invalid local_lse shape {tuple(local_lse.shape)}")
    if prior_bias.shape != (batch, chunks, h_kv, _LOGICAL_G):
        raise ValueError(f"invalid prior_bias shape {tuple(prior_bias.shape)}")
    if drop_mask.shape != (batch, seq_len, chunks):
        raise ValueError(f"invalid drop_mask shape {tuple(drop_mask.shape)}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if gumbel_scale < 0:
        raise ValueError(f"gumbel_scale must be non-negative, got {gumbel_scale}")
    if selection_mode not in {"post_softmax", "pre_softmax"}:
        raise ValueError(f"unsupported selection_mode={selection_mode}")

    if selection_mode == "pre_softmax":
        logical_q = q.reshape(batch, seq_len, h_kv, _LOGICAL_G, dim)
        logical_lmks = lmks.reshape(batch, chunks, h_q, dim)
        gumbel_noise = None
        if bool(use_gumbel) and bool(training) and gumbel_scale > 0:
            eps = torch.finfo(torch.float32).eps
            uniform = torch.empty(
                1, 1, h_q, chunks, device=q.device, dtype=torch.float32
            ).uniform_(eps, 1.0 - eps)
            gumbel_noise = -torch.log(-torch.log(uniform)) * float(gumbel_scale)
        indices, score_view = ref_softmax_topk_max_pooling(
            logical_q,
            logical_lmks,
            local_lse,
            topk,
            chunk_size,
            window,
            is_causal=False,
            drop_mask=drop_mask,
            bias=prior_bias,
            gumbel_noise=gumbel_noise,
            selection_mode=selection_mode,
        )
        scores = score_view.reshape(batch, seq_len, h_q, topk).to(q.dtype)
        if not return_selected_gumbel:
            return indices, scores
        selected_gumbel = None
        if gumbel_noise is not None:
            indices_hq = indices.repeat_interleave(_LOGICAL_G, dim=2)
            safe_indices = indices_hq.clamp_min(0).clamp_max(chunks - 1)
            selected_gumbel = torch.gather(
                gumbel_noise.expand(batch, seq_len, h_q, chunks),
                -1,
                safe_indices,
            )
            selected_gumbel = selected_gumbel.masked_fill(indices_hq < 0, 0.0)
        return indices, scores, selected_gumbel

    if not training:
        # The dynamic TileLang route kernel is nondeterministic on H20. Keep it
        # for training, but use the exact reference scoring path for inference.
        reference_indices = []
        reference_scores = []
        query_block = 1024
        logical_lmks = lmks.reshape(batch, chunks, h_q, dim)
        for start in range(0, seq_len, query_block):
            end = min(start + query_block, seq_len)
            indices_block, scores_block = ref_softmax_topk_max_pooling(
                q[:, start:end].reshape(
                    batch,
                    end - start,
                    h_kv,
                    _LOGICAL_G,
                    dim,
                ),
                logical_lmks,
                local_lse[:, start:end],
                topk,
                chunk_size,
                window,
                is_causal=False,
                drop_mask=drop_mask[:, start:end],
                bias=prior_bias,
            )
            reference_indices.append(indices_block)
            reference_scores.append(
                scores_block.reshape(
                    batch,
                    end - start,
                    h_q,
                    topk,
                ).to(q.dtype)
            )
        indices = torch.cat(reference_indices, dim=1)
        scores = torch.cat(reference_scores, dim=1)
        if return_selected_gumbel:
            return indices, scores, None
        return indices, scores

    q_padded = group_pad_to(q, h_kv, _LOGICAL_G, _PHYSICAL_G)
    lmk_heads = lmks.reshape(batch, chunks, h_q, dim)
    lmk_padded = group_pad_to(
        lmk_heads, h_kv, _LOGICAL_G, _PHYSICAL_G
    )
    lse_padded = _pad_group_scalar(local_lse, h_kv, value=0.0)
    bias_padded = _pad_group_scalar(prior_bias, h_kv, value=_DUMMY_BIAS)

    physical_chunks = max(chunks, topk)
    if physical_chunks > chunks:
        chunk_pad = physical_chunks - chunks
        lmk_padded = torch.cat(
            (
                lmk_padded,
                torch.zeros(
                    batch,
                    chunk_pad,
                    h_kv * _PHYSICAL_G,
                    dim,
                    device=lmks.device,
                    dtype=lmks.dtype,
                ),
            ),
            dim=1,
        )
        bias_padded = torch.cat(
            (
                bias_padded,
                torch.full(
                    (batch, chunk_pad, h_kv * _PHYSICAL_G),
                    _DUMMY_BIAS,
                    device=prior_bias.device,
                    dtype=prior_bias.dtype,
                ),
            ),
            dim=1,
        )
        drop_mask = torch.cat(
            (
                drop_mask,
                torch.ones(
                    batch,
                    seq_len,
                    chunk_pad,
                    device=drop_mask.device,
                    dtype=drop_mask.dtype,
                ),
            ),
            dim=-1,
        )

    gumbel_noise = None
    use_gumbel_now = bool(use_gumbel) and bool(training) and gumbel_scale > 0
    if use_gumbel_now:
        eps = torch.finfo(torch.float32).eps
        uniform = torch.empty(
            1,
            1,
            h_kv * _PHYSICAL_G,
            physical_chunks,
            device=q.device,
            dtype=torch.float32,
        ).uniform_(eps, 1.0 - eps)
        gumbel_noise = -torch.log(-torch.log(uniform)) * float(gumbel_scale)

    indices, physical_scores = online_softmax_topk_head(
        q_padded.contiguous(),
        lmk_padded.contiguous(),
        lse_padded.contiguous(),
        topk,
        chunk_size,
        window,
        is_causal=False,
        is_training=training,
        drop_mask=drop_mask.to(torch.int32).contiguous(),
        bias=bias_padded.contiguous(),
        use_gumbel=use_gumbel_now,
        gumbel_noise=gumbel_noise,
        G=_PHYSICAL_G,
    )
    scores = group_unpad(
        physical_scores,
        h_kv,
        _LOGICAL_G,
        _PHYSICAL_G,
    )
    invalid = indices >= chunks
    if invalid.any():
        indices = indices.masked_fill(invalid, -1)
        scores = scores.masked_fill(
            invalid.repeat_interleave(_LOGICAL_G, dim=2),
            float("-inf"),
        )
    if not return_selected_gumbel:
        return indices, scores

    selected_gumbel = None
    if gumbel_noise is not None:
        noise_view = gumbel_noise.view(
            1,
            1,
            h_kv,
            _PHYSICAL_G,
            physical_chunks,
        ).expand(batch, seq_len, h_kv, _PHYSICAL_G, physical_chunks)
        safe_indices = indices.clamp_min(0).clamp_max(physical_chunks - 1)
        gather_index = safe_indices.unsqueeze(3).expand(
            batch,
            seq_len,
            h_kv,
            _PHYSICAL_G,
            topk,
        )
        physical_gumbel = torch.gather(noise_view, -1, gather_index).reshape(
            batch,
            seq_len,
            h_kv * _PHYSICAL_G,
            topk,
        )
        selected_gumbel = group_unpad(
            physical_gumbel,
            h_kv,
            _LOGICAL_G,
            _PHYSICAL_G,
        )
        selected_gumbel = selected_gumbel.masked_fill(
            invalid.repeat_interleave(_LOGICAL_G, dim=2),
            0.0,
        )
    return indices, scores, selected_gumbel


def selected_attention_g7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    training: bool,
    token_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run selected-chunk attention with compact KV and padded physical G=8 Q."""

    return _selected_attention_g7_rectangular(
        q,
        k,
        v,
        weights,
        indices,
        key_valid,
        chunk_size,
        training=training,
        token_keep=token_keep,
    )


def selected_attention_g7_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    token_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run partial queries against a full compact-GQA selected-chunk cache."""

    return _selected_attention_g7_rectangular(
        q,
        k,
        v,
        weights,
        indices,
        key_valid,
        chunk_size,
        training=False,
        token_keep=token_keep,
    )


def _selected_attention_g7_rectangular(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
    *,
    training: bool,
    token_keep: torch.Tensor | None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"q and k must be BLHD tensors, got {tuple(q.shape)} and "
            f"{tuple(k.shape)}"
        )
    batch, query_len, h_q, dim = q.shape
    if k.shape[0] != batch or k.shape[-1] != dim:
        raise ValueError(
            f"incompatible q/k shapes {tuple(q.shape)} and {tuple(k.shape)}"
        )
    kv_len, h_kv = k.shape[1:3]
    if h_q != h_kv * _LOGICAL_G:
        raise ValueError(
            f"Dream G7 requires h_q == 7 * h_kv, got h_q={h_q}, "
            f"h_kv={h_kv}"
        )
    if v.shape != k.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
    selected = indices.shape[-1]
    if indices.shape != (batch, query_len, h_kv, selected):
        raise ValueError(f"invalid indices shape {tuple(indices.shape)}")
    if weights.shape != (batch, query_len, h_q, selected):
        raise ValueError(f"invalid weights shape {tuple(weights.shape)}")
    if key_valid.shape != (batch, kv_len):
        raise ValueError(f"invalid key_valid shape {tuple(key_valid.shape)}")
    if token_keep is not None and token_keep.shape != (
        batch,
        query_len,
        h_kv,
        selected,
        chunk_size,
    ):
        raise ValueError(
            "token_keep must have shape "
            f"{(batch, query_len, h_kv, selected, chunk_size)}, got "
            f"{tuple(token_keep.shape)}"
        )
    if kv_len % chunk_size != 0:
        raise ValueError(
            f"cache length {kv_len} must be divisible by "
            f"chunk_size={chunk_size}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("selected attention requires BF16 q, k, and v")
    if q.device != k.device or v.device != q.device:
        raise ValueError("q, k, and v must share a device")
    if weights.device != q.device or indices.device != q.device:
        raise ValueError("weights and indices must share the q device")
    if key_valid.device != q.device:
        raise ValueError("key_valid must share the q device")
    if token_keep is not None and token_keep.device != q.device:
        raise ValueError("token_keep must share the q device")

    kernel_weights = weights
    kernel_indices = indices
    kernel_token_keep = token_keep
    if selected <= 0:
        raise ValueError("selected attention requires at least one chunk slot")
    # TileLang needs aligned selected-block strides, including for cached Q.
    physical_selected = max(4, 1 << (selected - 1).bit_length())
    if physical_selected != selected:
        pad = physical_selected - selected
        kernel_weights = torch.nn.functional.pad(
            kernel_weights,
            (0, pad),
            value=0.0,
        )
        kernel_indices = torch.nn.functional.pad(
            kernel_indices,
            (0, pad),
            value=-1,
        )
        if kernel_token_keep is not None:
            kernel_token_keep = torch.nn.functional.pad(
                kernel_token_keep,
                (0, 0, 0, pad),
                value=0,
            )

    q_padded = group_pad_to(q, h_kv, _LOGICAL_G, _PHYSICAL_G)
    weights_padded = group_pad_to(
        kernel_weights, h_kv, _LOGICAL_G, _PHYSICAL_G
    )
    physical_output = HiLS_block_M_head(
        q_padded.contiguous(),
        k.contiguous(),
        v.contiguous(),
        weights_padded.contiguous(),
        kernel_indices.to(torch.int32).contiguous(),
        block_size=chunk_size,
        mask_last_token=True,
        is_training=training,
        key_valid=key_valid.to(torch.int32).contiguous(),
        token_keep=(
            None
            if kernel_token_keep is None
            else kernel_token_keep.to(torch.int8).contiguous()
        ),
    )
    return group_unpad(
        physical_output,
        h_kv,
        _LOGICAL_G,
        _PHYSICAL_G,
    )
