"""Native Sparse Attention on Dream, aligned to Yuan et al. 2025.

Paper compute graph kept:
- independent K/V maps for cmp / slc / win
- φ is a block-flatten MLP with intra-block PE: Linear(lD,lD)→ReLU→Linear(lD,D)
- overlapping compression (l=32, d=16)
- p = softmax(q_raw · K̃_cmp); selection scores via Eq. 9 overlap-sum then Eq. 10
  per-GQA-group head-sum and independent top-n
- selected tokens are the full selection block
- sliding window is a separate branch; selection is not locally dropped
- compression uses non-RoPE Q/K; selection and window use RoPE Q/K

Dream packing only: bidirectional softmax and Fast-dLLM cache. No MASK landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefilling_v2.attention import (
    KernelDreamSlidingWindowAttention,
    _NaiveDreamSparseAttentionBase,
    apply_rotary_pos_emb,
)
from prefilling_v2.fastdllm_cache import LayerKVCache
from prefilling_v2.local_attention import bidir_local_attention


@dataclass(frozen=True)
class NsaLayerPlan:
    nsa_layers: list[int]
    sliding_window_layers: list[int]
    dense_layers: list[int]


def _real_key_mask(length: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    return (positions + 1).remainder(chunk_size) != 0


def _identity_linear(dim: int, device, dtype) -> nn.Linear:
    layer = nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)
    nn.init.eye_(layer.weight)
    return layer


class DreamNsaAttention(_NaiveDreamSparseAttentionBase):
    """Three-branch native sparse attention with TileLang selected attention."""

    def __init__(
        self,
        source_attn: nn.Module,
        *,
        local_window: int,
        chunk_size: int,
        block_count: int,
        compress_block: int = 32,
        compress_stride: int = 16,
        select_block: int | None = None,
        backend: str = "tilelang",
        cmp_query_block: int = 512,
        skip_inert_slots: bool = False,
    ) -> None:
        super().__init__(source_attn, local_window, chunk_size=chunk_size)
        if backend != "tilelang":
            raise ValueError("Dream NSA production backend is tilelang")
        if block_count <= 0:
            raise ValueError("nsa block_count must be positive")
        if chunk_size < 2:
            raise ValueError("NSA chunk_size must leave one text token")
        if compress_block <= 0 or compress_stride <= 0:
            raise ValueError("NSA compression block and stride must be positive")
        if compress_stride > compress_block:
            raise ValueError("NSA compression stride must be <= block length")
        select_block = int(chunk_size if select_block is None else select_block)
        if select_block <= 0 or select_block % int(chunk_size) != 0:
            raise ValueError("NSA select_block must be a positive multiple of chunk_size")
        self.block_count = int(block_count)
        self.compress_block = int(compress_block)
        self.compress_stride = int(compress_stride)
        self.select_block = select_block
        self.backend = str(backend)
        self.cmp_query_block = int(cmp_query_block)
        self.skip_inert_slots = bool(skip_inert_slots)
        device = source_attn.q_proj.weight.device
        dtype = source_attn.q_proj.weight.dtype
        hidden = self.hidden_size
        dim = self.head_dim
        phi_in = self.compress_block * dim
        self.nsa_gate = nn.Linear(
            hidden,
            self.num_heads * 3,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.zeros_(self.nsa_gate.weight)
        self.nsa_cmp_k = _identity_linear(dim, device, dtype)
        self.nsa_cmp_v = _identity_linear(dim, device, dtype)
        self.nsa_slc_k = _identity_linear(dim, device, dtype)
        self.nsa_slc_v = _identity_linear(dim, device, dtype)
        self.nsa_win_k = _identity_linear(dim, device, dtype)
        self.nsa_win_v = _identity_linear(dim, device, dtype)
        self.nsa_compress_pe_k = nn.Parameter(
            torch.zeros(
                self.num_key_value_heads,
                self.compress_block,
                dim,
                device=device,
                dtype=dtype,
            )
        )
        self.nsa_compress_pe_v = nn.Parameter(
            torch.zeros_like(self.nsa_compress_pe_k)
        )
        self.nsa_phi_k_up = nn.Linear(phi_in, phi_in, device=device, dtype=dtype)
        self.nsa_phi_k_down = nn.Linear(phi_in, dim, device=device, dtype=dtype)
        self.nsa_phi_v_up = nn.Linear(phi_in, phi_in, device=device, dtype=dtype)
        self.nsa_phi_v_down = nn.Linear(phi_in, dim, device=device, dtype=dtype)

    def _project_qkv_nsa(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden_states.size()
        query_raw = self.q_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        )
        key_raw = self.k_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value.transpose(1, 2), position_ids)
        else:
            cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(
            query_raw.transpose(1, 2),
            key_raw.transpose(1, 2),
            cos,
            sin,
        )
        return (
            q_rope.transpose(1, 2).contiguous(),
            query_raw.contiguous(),
            k_rope.transpose(1, 2).contiguous(),
            key_raw.contiguous(),
            value.contiguous(),
        )

    def _compress_starts(self, length: int, device: torch.device) -> torch.Tensor:
        if length < self.compress_block:
            raise ValueError("sequence shorter than NSA compression block")
        if (length - self.compress_block) % self.compress_stride:
            raise ValueError("NSA compression windows must tile the sequence")
        return torch.arange(
            0,
            length - self.compress_block + 1,
            self.compress_stride,
            device=device,
            dtype=torch.long,
        )

    def _phi(
        self,
        blocks: torch.Tensor,
        pe: torch.Tensor,
        up: nn.Linear,
        down: nn.Linear,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, windows, block, kv_heads, dim = blocks.shape
        shifted = blocks + pe.permute(1, 0, 2).reshape(1, 1, block, kv_heads, dim)
        shifted = shifted.masked_fill(~token_valid[..., None, None], 0)
        flat = shifted.permute(0, 1, 3, 2, 4).reshape(batch, windows, kv_heads, block * dim)
        return down(F.relu(up(flat.to(dtype=up.weight.dtype)))).to(dtype=blocks.dtype)

    def _compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, kv_heads, dim = key.shape
        starts = self._compress_starts(length, key.device)
        offsets = torch.arange(self.compress_block, device=key.device, dtype=torch.long)
        positions = starts[:, None] + offsets[None, :]
        valid = key_valid.bool()
        if self.skip_inert_slots:
            valid = valid & _real_key_mask(
                length, int(self.chunk_size), key.device
            ).unsqueeze(0)
        token_valid = valid[:, positions]
        cmp_key = self.nsa_cmp_k(key)
        cmp_value = self.nsa_cmp_v(value)
        key_blocks = cmp_key[:, positions]
        value_blocks = cmp_value[:, positions]
        compress_key = self._phi(
            key_blocks,
            self.nsa_compress_pe_k,
            self.nsa_phi_k_up,
            self.nsa_phi_k_down,
            token_valid,
        )
        compress_value = self._phi(
            value_blocks,
            self.nsa_compress_pe_v,
            self.nsa_phi_v_up,
            self.nsa_phi_v_down,
            token_valid,
        )
        window_valid = token_valid.any(dim=-1)
        return compress_key, compress_value, window_valid

    def _compressed_attention(
        self,
        query: torch.Tensor,
        compress_key: torch.Tensor,
        compress_value: torch.Tensor,
        window_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = self.num_key_value_groups
        expanded_key = compress_key.repeat_interleave(groups, dim=2)
        expanded_value = compress_value.repeat_interleave(groups, dim=2)
        scale = self.head_dim ** -0.5
        outputs = []
        probs = []
        query_len = query.shape[1]
        block = max(1, int(self.cmp_query_block))
        valid = window_valid[:, None, None, :]
        for start in range(0, query_len, block):
            end = min(start + block, query_len)
            logits = torch.einsum(
                "blhd,bchd->blhc",
                query[:, start:end].float(),
                expanded_key.float(),
            ) * scale
            logits = logits.masked_fill(~valid, float("-inf"))
            all_masked = torch.isneginf(logits).all(dim=-1, keepdim=True)
            logits = torch.where(all_masked, torch.zeros_like(logits), logits)
            weight = torch.softmax(logits, dim=-1)
            weight = torch.where(all_masked, torch.zeros_like(weight), weight)
            outputs.append(
                torch.einsum(
                    "blhc,bchd->blhd",
                    weight,
                    expanded_value.float(),
                ).to(dtype=query.dtype)
            )
            probs.append(weight)
        return torch.cat(outputs, dim=1).contiguous(), torch.cat(probs, dim=1)

    def _select_token_offsets(self, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(self.select_block, device=device, dtype=torch.long)
        if self.skip_inert_slots:
            real = (offsets + 1).remainder(int(self.chunk_size)) != 0
            return offsets[real]
        return offsets

    def _selected_indices(
        self,
        compress_probs: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_len, _, compress_count = compress_probs.shape
        length = key_valid.shape[1]
        if length % self.select_block:
            raise ValueError("NSA selection requires select-block-aligned length")
        select_count = length // self.select_block
        starts = self._compress_starts(length, compress_probs.device)
        if starts.numel() != compress_count:
            raise ValueError("compressed probability width does not match φ windows")
        overlap = self._compress_to_select_overlap(starts, select_count).to(
            dtype=compress_probs.dtype
        )
        select_mass = torch.einsum("blhc,cs->blhs", compress_probs, overlap)
        groups = self.num_key_value_groups
        grouped = select_mass.view(
            batch,
            query_len,
            self.num_key_value_heads,
            groups,
            select_count,
        ).sum(dim=3)
        select_valid = key_valid.bool()
        if self.skip_inert_slots:
            select_valid = select_valid & _real_key_mask(
                length, int(self.chunk_size), compress_probs.device
            ).unsqueeze(0)
        select_valid = select_valid.view(
            batch, select_count, self.select_block
        ).any(dim=-1)
        grouped = grouped.masked_fill(~select_valid[:, None, None, :], float("-inf"))
        count = min(self.block_count, select_count)
        values, block_ids = torch.topk(grouped, k=count, dim=-1, sorted=False)
        block_valid = torch.isfinite(values)
        offsets = self._select_token_offsets(compress_probs.device)
        indices = (
            block_ids[..., None] * self.select_block + offsets
        ).reshape(batch, query_len, self.num_key_value_heads, -1)
        valid = block_valid[..., None].expand(
            -1, -1, -1, -1, offsets.numel()
        ).reshape(batch, query_len, self.num_key_value_heads, -1)
        return indices, valid

    def _compress_to_select_overlap(
        self,
        compress_starts: torch.Tensor,
        select_count: int,
    ) -> torch.Tensor:
        starts = compress_starts[:, None]
        ends = starts + self.compress_block
        select_starts = torch.arange(
            select_count, device=compress_starts.device, dtype=torch.long
        ) * self.select_block
        select_ends = select_starts + self.select_block
        return (starts < select_ends) & (ends > select_starts)

    def _selected_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        from ops.dsa_selected_attention_tilelang import (
            selected_token_attention_tilelang,
        )

        groups = self.num_key_value_groups
        outputs = []
        for kv_head in range(self.num_key_value_heads):
            q_start = kv_head * groups
            outputs.append(
                selected_token_attention_tilelang(
                    query[:, :, q_start : q_start + groups].contiguous(),
                    key[:, :, kv_head : kv_head + 1].contiguous(),
                    value[:, :, kv_head : kv_head + 1].contiguous(),
                    indices[:, :, kv_head].contiguous(),
                    valid[:, :, kv_head].contiguous(),
                    sm_scale=self.head_dim**-0.5,
                )
            )
        return torch.cat(outputs, dim=2)

    def _nsa_output(
        self,
        hidden_states: torch.Tensor,
        query_rope: torch.Tensor,
        query_raw: torch.Tensor,
        key_rope: torch.Tensor,
        key_raw: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor,
        allowed: torch.Tensor | None,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        compress_key, compress_value, window_valid = self._compress_kv(
            key_raw, value, key_valid
        )
        slc_k = self.nsa_slc_k(key_rope)
        slc_v = self.nsa_slc_v(value)
        window_k = self.nsa_win_k(key_rope)
        window_v = self.nsa_win_v(value)
        if query_positions is None or query_rope.shape[1] == key_rope.shape[1]:
            local, _ = bidir_local_attention(
                query_rope,
                window_k,
                window_v,
                key_valid,
                self.local_window,
                int(self.chunk_size),
                skip_inert_slots=self.skip_inert_slots,
            )
        else:
            from prefilling_v2.local_attention import cached_local_attention
            if allowed is not None:
                raise ValueError("cached NSA local attention does not take allowed masks")
            local, _ = cached_local_attention(
                query_rope,
                window_k,
                window_v,
                key_valid,
                query_positions,
                self.local_window,
                int(self.chunk_size),
                skip_inert_slots=self.skip_inert_slots,
            )
        # Prefill Q is length L; materializing L×#compress probs OOMs at 128k.
        # Top-k is per query row, so tile the cmp/select branches.
        query_len = query_rope.shape[1]
        tile = max(1, int(self.cmp_query_block))
        parts = []
        for start in range(0, query_len, tile):
            end = min(start + tile, query_len)
            compressed, probs = self._compressed_attention(
                query_raw[:, start:end],
                compress_key,
                compress_value,
                window_valid,
            )
            indices, selected_valid = self._selected_indices(probs, key_valid)
            del probs
            selected = self._selected_attention(
                query_rope[:, start:end],
                slc_k,
                slc_v,
                indices,
                selected_valid,
            )
            del indices, selected_valid
            hidden = hidden_states[:, start:end]
            gates = torch.sigmoid(
                self.nsa_gate(hidden).view(
                    hidden.shape[0], hidden.shape[1], self.num_heads, 3
                )
            ).to(dtype=query_rope.dtype)
            parts.append(
                compressed.to(dtype=query_rope.dtype) * gates[..., 0:1]
                + selected * gates[..., 1:2]
                + local[:, start:end] * gates[..., 2:3]
            )
        return torch.cat(parts, dim=1)

    def _capture_prefill_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor,
        landmark_keys: torch.Tensor | None = None,
        prior_bias: torch.Tensor | None = None,
        nsa_raw_key: torch.Tensor | None = None,
    ) -> None:
        if not self._prefill_capture:
            return
        self._captured_prefill_cache = LayerKVCache(
            key=key.detach().contiguous(),
            value=value.detach().contiguous(),
            key_valid=key_valid.detach().to(dtype=torch.bool).contiguous(),
            landmark_keys=(
                None if landmark_keys is None else landmark_keys.detach().contiguous()
            ),
            prior_bias=None if prior_bias is None else prior_bias.detach().contiguous(),
            nsa_raw_key=(
                None if nsa_raw_key is None else nsa_raw_key.detach().contiguous()
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        del cache_position, kwargs
        if use_cache or past_key_value is not None:
            raise NotImplementedError("Dream NSA uses the Fast-dLLM cache path")
        query_rope, query_raw, key_rope, key_raw, value = self._project_qkv_nsa(
            hidden_states, position_ids, position_embeddings
        )
        batch, length = hidden_states.shape[:2]
        key_valid = self._key_valid_mask(
            attention_mask, batch, length, hidden_states.device
        )
        if self.skip_inert_slots:
            key_valid = key_valid & self._real_key_mask(
                length, hidden_states.device
            )[None, :]
        self._capture_prefill_cache(
            key_rope, value, key_valid, nsa_raw_key=key_raw
        )
        output = self._nsa_output(
            hidden_states,
            query_rope,
            query_raw,
            key_rope,
            key_raw,
            value,
            key_valid,
            None,
        )
        return self.o_proj(output.reshape(batch, length, self.hidden_size)), None, past_key_value


def install_dream_nsa_attention(
    model: nn.Module,
    *,
    interleave: int,
    local_window: int,
    chunk_size: int,
    block_count: int,
    non_nsa_attention: str,
    compress_block: int = 32,
    compress_stride: int = 16,
    select_block: int | None = None,
    skip_inert_slots: bool = False,
    swa_local_window: int | None = None,
) -> NsaLayerPlan:
    if non_nsa_attention not in {"sliding", "dense"}:
        raise ValueError("non_nsa_attention must be sliding or dense")
    layers: Iterable[nn.Module] = model.model.layers
    nsa_layers: list[int] = []
    sliding_layers: list[int] = []
    dense_layers: list[int] = []
    select_block = int(chunk_size if select_block is None else select_block)
    sliding_window = int(local_window if swa_local_window is None else swa_local_window)
    for layer_idx, layer in enumerate(layers):
        if interleave <= 0 or layer_idx % interleave == interleave - 1:
            layer.self_attn = DreamNsaAttention(
                layer.self_attn,
                local_window=local_window,
                chunk_size=chunk_size,
                block_count=block_count,
                compress_block=compress_block,
                compress_stride=compress_stride,
                select_block=select_block,
                skip_inert_slots=skip_inert_slots,
            )
            nsa_layers.append(layer_idx)
        elif non_nsa_attention == "sliding":
            layer.self_attn = KernelDreamSlidingWindowAttention(
                layer.self_attn,
                sliding_window,
                chunk_size,
                allow_fallback=False,
                skip_inert_slots=False,
            )
            sliding_layers.append(layer_idx)
        else:
            dense_layers.append(layer_idx)
    model.config.dream_nsa_interleave = int(interleave)
    model.config.dream_nsa_block_size = int(select_block)
    model.config.dream_nsa_compress_block = int(compress_block)
    model.config.dream_nsa_compress_stride = int(compress_stride)
    model.config.dream_nsa_block_count = int(block_count)
    model.config.dream_nsa_local_window = int(local_window)
    model.config.dream_swa_local_window = int(sliding_window)
    model.config.dream_nsa_backend = "tilelang"
    model.config.dream_nsa_score = "compress_softmax_mass_eq9_gqa"
    model.config.dream_nsa_skip_inert_slots = bool(skip_inert_slots)
    return NsaLayerPlan(nsa_layers, sliding_layers, dense_layers)


__all__ = ["DreamNsaAttention", "NsaLayerPlan", "install_dream_nsa_attention"]
