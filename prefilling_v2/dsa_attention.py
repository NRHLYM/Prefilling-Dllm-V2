"""DSA-style direct-token sparse attention adapted to bidirectional Dream.

The DeepSeek indexer is retained at the algorithmic level: a lightweight,
query-dependent projection ranks individual key tokens and hard top-k indices
drive exact QK attention.  Dream differs from a causal AR decoder, so routing
respects the supplied bidirectional/packed-document mask instead of imposing a
causal prefix.  The implementation is correctness-first and query-blocked;
the selected-attention contract is kept separate so an optimized kernel can
replace it without changing the trained indexer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from prefilling_v2.attention import (
    _NaiveDreamSparseAttentionBase,
    apply_rotary_pos_emb,
    route_layer_indices,
)


# Both backends implement this contract. They are numerically close, not
# bit-exact, because TileLang and cuBLAS reduce FP32 values in different orders.
DSA_SELECTED_ATTENTION_SEMANTICS = "fp32_scores_softmax_prob_to_v_dtype_v1"


class DsaRmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)


class DreamDsaIndexer(nn.Module):
    """Lightning-indexer-style per-query token scorer for Dream."""

    def __init__(
        self,
        source_attn: nn.Module,
        *,
        num_index_heads: int,
        index_head_dim: int,
        topk: int,
        query_block_size: int,
    ) -> None:
        super().__init__()
        if num_index_heads <= 0:
            raise ValueError("num_index_heads must be positive")
        if index_head_dim != int(source_attn.head_dim):
            raise ValueError(
                "Dream DSA currently requires index_head_dim == attention "
                f"head_dim ({source_attn.head_dim}), got {index_head_dim}"
            )
        if topk <= 0 or query_block_size <= 0:
            raise ValueError("topk and query_block_size must be positive")

        self.hidden_size = int(source_attn.hidden_size)
        self.num_index_heads = int(num_index_heads)
        self.index_head_dim = int(index_head_dim)
        self.topk = int(topk)
        self.query_block_size = int(query_block_size)
        eps = float(getattr(source_attn.config, "rms_norm_eps", 1e-6))
        weight = source_attn.q_proj.weight

        # Deliberately avoid q_proj/k_proj names: PEFT should adapt the actual
        # attention projections, while these full indexer weights train via KL.
        self.index_q = nn.Linear(
            self.hidden_size,
            self.num_index_heads * self.index_head_dim,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.index_k = nn.Linear(
            self.hidden_size,
            self.index_head_dim,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.index_w = nn.Linear(
            self.hidden_size,
            self.num_index_heads,
            bias=False,
            device=weight.device,
            # DeepSeek keeps the query-dependent index-head weights in FP32.
            # They are signed coefficients in Equation (1), not probabilities.
            dtype=torch.float32,
        )
        # Keep the tiny norm scales in FP32. With plain AdamW there is no
        # separate master copy, and BF16 cannot represent sub-0.0078 updates
        # around their initial value of one.
        self.q_norm = DsaRmsNorm(self.index_head_dim, eps).to(
            device=weight.device
        )
        self.k_norm = DsaRmsNorm(self.index_head_dim, eps).to(
            device=weight.device
        )
        self._initialize_from_attention(source_attn)

    @torch.no_grad()
    def _initialize_from_attention(self, source_attn: nn.Module) -> None:
        """Warm-start routing from existing Dream Q/K geometry."""

        q_weight = source_attn.q_proj.weight.view(
            source_attn.num_heads,
            source_attn.head_dim,
            source_attn.hidden_size,
        )
        selected_heads = torch.linspace(
            0,
            source_attn.num_heads - 1,
            steps=self.num_index_heads,
            device=q_weight.device,
        ).round().long()
        initialized_q = q_weight.index_select(0, selected_heads)
        self.index_q.weight.copy_(initialized_q.reshape_as(self.index_q.weight))

        k_weight = source_attn.k_proj.weight.view(
            source_attn.num_key_value_heads,
            source_attn.head_dim,
            source_attn.hidden_size,
        )
        self.index_k.weight.copy_(k_weight.mean(dim=0))

    def _project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.index_q(hidden_states).view(
            batch, seq_len, self.num_index_heads, self.index_head_dim
        )).transpose(1, 2)
        k = self.k_norm(self.index_k(hidden_states).view(
            batch, seq_len, 1, self.index_head_dim
        )).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # DeepSeek-V3.2 Eq. (1): I_{t,s} = Σ_j w_{t,j} ReLU(q_{t,j} · k_s).
        # No 1/√d or 1/√H^I; those would only change softmax(I) in the KL term.
        weights = self.index_w(hidden_states.float())
        return q.transpose(1, 2), k[:, 0], weights

    def _score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        per_head = torch.einsum("bqhd,bkd->bqhk", query, key)
        return (
            torch.relu(per_head.float())
            * weights.float().unsqueeze(-1)
        ).sum(dim=2)

    @torch.no_grad()
    def select(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        allowed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared per-query token indices and a validity mask."""

        candidate_count = hidden_states.shape[1]
        if allowed.ndim == 2:
            if allowed.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"allowed must be [B,L], got {tuple(allowed.shape)}"
                )
            allowed_keys = allowed
            allowed_pairs = None
        elif allowed.shape == hidden_states.shape[:2] + (hidden_states.shape[1],):
            allowed_keys = None
            allowed_pairs = allowed
        else:
            raise ValueError(
                f"allowed must be [B,L] or [B,L,L], got {tuple(allowed.shape)}"
            )
        query, key, weights = self._project(hidden_states, position_embeddings)
        selected_indices: list[torch.Tensor] = []
        selected_valid: list[torch.Tensor] = []
        effective_topk = min(self.topk, candidate_count)
        for start in range(0, candidate_count, self.query_block_size):
            end = min(start + self.query_block_size, candidate_count)
            scores = self._score(
                query[:, start:end], key, weights[:, start:end]
            )
            if allowed_pairs is None:
                scores.masked_fill_(~allowed_keys[:, None, :], float("-inf"))
            else:
                scores.masked_fill_(~allowed_pairs[:, start:end], float("-inf"))
            values, indices = torch.topk(
                scores, k=effective_topk, dim=-1, sorted=False
            )
            valid = torch.isfinite(values)
            selected_indices.append(indices.masked_fill(~valid, 0))
            selected_valid.append(valid)
        return torch.cat(selected_indices, dim=1), torch.cat(selected_valid, dim=1)

    def _kl_for_positions(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        teacher_q: torch.Tensor,
        teacher_k: torch.Tensor,
        allowed: torch.Tensor,
        allowed_pairs: torch.Tensor | None,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        loss_scope: str,
    ) -> torch.Tensor:
        """Mean KL over a [B, Q] query-index block. Does not materialize LxL."""

        batch, count = positions.shape
        seq_len = hidden_states.shape[1]
        gather_hidden = positions[..., None].expand(batch, count, self.hidden_size)
        sampled_hidden = torch.gather(hidden_states, 1, gather_hidden)
        cos, sin = position_embeddings
        gather_rope = positions[..., None].expand(batch, count, cos.shape[-1])
        sampled_position_embeddings = (
            torch.gather(cos, 1, gather_rope),
            torch.gather(sin, 1, gather_rope),
        )
        index_q, _, index_weights = self._project(
            sampled_hidden, sampled_position_embeddings
        )
        index_scores = self._score(index_q, index_k, index_weights)
        if allowed_pairs is None:
            sampled_allowed = allowed[:, None, :].expand(batch, count, seq_len)
        else:
            gather_allowed = positions[..., None].expand(batch, count, seq_len)
            sampled_allowed = torch.gather(allowed_pairs, 1, gather_allowed)
        index_scores = index_scores.masked_fill(~sampled_allowed, float("-inf"))

        if loss_scope == "selected":
            effective_topk = min(self.topk, seq_len)
            selected_scores, selected_indices = torch.topk(
                index_scores,
                k=effective_topk,
                dim=-1,
                sorted=False,
            )
            selected_valid = torch.isfinite(selected_scores)
            safe_indices = selected_indices.masked_fill(~selected_valid, 0)
            index_scores_for_loss = selected_scores
        else:
            selected_indices = None
            selected_valid = None
            index_scores_for_loss = index_scores

        # Teacher QK is detached (frozen backbone). Keep its L×H tables out of
        # autograd so all-query tiling does not retain ~256 score maps.
        with torch.no_grad():
            teacher_distribution, teacher_log = self._teacher_key_mass(
                teacher_q,
                teacher_k,
                positions,
                sampled_allowed,
                selected_indices=selected_indices,
                selected_valid=selected_valid,
            )
        log_index_distribution = torch.log_softmax(
            index_scores_for_loss.float(), dim=-1
        )
        positive_teacher = teacher_distribution > 0
        per_token_kl = torch.where(
            positive_teacher,
            teacher_distribution * (teacher_log - log_index_distribution),
            torch.zeros_like(teacher_distribution),
        )
        return per_token_kl.sum(dim=-1).mean()

    def _teacher_key_mass(
        self,
        teacher_q: torch.Tensor,
        teacher_k: torch.Tensor,
        positions: torch.Tensor,
        sampled_allowed: torch.Tensor,
        *,
        selected_indices: torch.Tensor | None,
        selected_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = positions.shape
        h_kv = teacher_k.shape[2]
        groups = teacher_q.shape[2] // h_kv
        dim = teacher_q.shape[-1]
        gather_teacher = positions[..., None, None].expand(
            batch, count, teacher_q.shape[2], dim
        )
        sampled_teacher_q = torch.gather(
            teacher_q.detach(), 1, gather_teacher
        ).view(batch, count, h_kv, groups, dim)
        if selected_indices is not None:
            # B=2, Q=64, K=2560, Hkv=4, D=128 float32 gather is 320MiB.
            batch_index = torch.arange(batch, device=teacher_q.device)[:, None, None]
            q_f = sampled_teacher_q.float()
            key_chunks = []
            chunk = 128
            n_sel = int(selected_indices.shape[-1])
            for start in range(0, n_sel, chunk):
                end = min(start + chunk, n_sel)
                selected_teacher_k = teacher_k.detach()[
                    batch_index, selected_indices[:, :, start:end]
                ]
                scores = torch.einsum(
                    "bqhgd,bqkhd->bqhgk",
                    q_f,
                    selected_teacher_k.float(),
                ) * (dim**-0.5)
                scores = scores.masked_fill(
                    ~selected_valid[:, :, start:end].unsqueeze(2).unsqueeze(3),
                    float("-inf"),
                )
                key_chunks.append(scores)
            teacher_scores = torch.cat(key_chunks, dim=-1)
        else:
            teacher_scores = torch.einsum(
                "bqhgd,bkhd->bqhgk",
                sampled_teacher_q.float(),
                teacher_k.detach().float(),
            ) * (dim**-0.5)
            teacher_scores = teacher_scores.masked_fill(
                ~sampled_allowed[:, :, None, None], float("-inf")
            )
        teacher_probs = torch.softmax(teacher_scores, dim=-1)
        teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0)
        teacher_distribution = teacher_probs.sum(dim=(2, 3))
        teacher_distribution = teacher_distribution / teacher_distribution.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        positive_teacher = teacher_distribution > 0
        teacher_log = torch.where(
            positive_teacher,
            teacher_distribution.clamp_min(1e-30).log(),
            torch.zeros_like(teacher_distribution),
        )
        return teacher_distribution.clone(), teacher_log.clone()

    def _unused_indexer_anchor(self, loss: torch.Tensor) -> torch.Tensor:
        """Keep every indexer weight in the DDP graph (Dolma no_sync + RULER)."""

        for parameter in self.parameters():
            if parameter.requires_grad:
                loss = loss + parameter.sum() * 0
        return loss

    def distillation_loss(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        teacher_q: torch.Tensor,
        teacher_k: torch.Tensor,
        allowed: torch.Tensor,
        sample_count: int,
        loss_scope: str = "full",
    ) -> torch.Tensor:
        """KL-match indexer rows to dense Dream attention mass.

        sample_count > 0 randomly samples that many queries (old recipe).
        sample_count <= 0 walks every query in query_block_size tiles,
        matching DeepSeek's sum_t without allocating a full LxL table.
        """

        if loss_scope not in {"full", "selected"}:
            raise ValueError(f"unsupported DSA auxiliary loss scope: {loss_scope}")

        batch, seq_len, _ = hidden_states.shape
        if allowed.ndim == 2:
            if allowed.shape != (batch, seq_len):
                raise ValueError(
                    f"allowed must be [B,L], got {tuple(allowed.shape)}"
                )
            query_valid = allowed
            allowed_pairs = None
        elif allowed.shape == (batch, seq_len, seq_len):
            query_valid = allowed.any(dim=-1)
            allowed_pairs = allowed
        else:
            raise ValueError(
                f"allowed must be [B,L] or [B,L,L], got {tuple(allowed.shape)}"
            )
        if int(query_valid.sum().item()) <= 0:
            return self._unused_indexer_anchor(
                hidden_states.new_zeros((), dtype=torch.float32)
            )

        _, index_k, _ = self._project(hidden_states, position_embeddings)
        if sample_count > 0:
            counts = query_valid.sum(dim=-1)
            count = min(int(counts.min().item()), int(sample_count))
            if count <= 0:
                return self._unused_indexer_anchor(
                    hidden_states.new_zeros((), dtype=torch.float32)
                )
            sample_positions = []
            for batch_idx in range(batch):
                candidates = torch.where(query_valid[batch_idx])[0]
                order = torch.randperm(candidates.numel(), device=candidates.device)
                sample_positions.append(candidates[order[:count]])
            positions = torch.stack(sample_positions)
            return self._kl_for_positions(
                hidden_states,
                position_embeddings,
                teacher_q,
                teacher_k,
                allowed if allowed_pairs is None else allowed,
                allowed_pairs,
                index_k,
                positions,
                loss_scope,
            )

        block = int(self.query_block_size)
        kl_sum = hidden_states.new_zeros((), dtype=torch.float32)
        blocks = 0
        cos, sin = position_embeddings
        allowed_tensor = allowed if allowed_pairs is None else allowed_pairs

        def _tile_kl(
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            teacher_q: torch.Tensor,
            teacher_k: torch.Tensor,
            allowed_tensor: torch.Tensor,
            index_k: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            pairs = None if allowed_tensor.ndim == 2 else allowed_tensor
            return self._kl_for_positions(
                hidden_states,
                (cos, sin),
                teacher_q,
                teacher_k,
                allowed_tensor,
                pairs,
                index_k,
                positions,
                loss_scope,
            )

        for start in range(0, seq_len, block):
            end = min(start + block, seq_len)
            if not bool(query_valid[:, start:end].any()):
                continue
            positions = torch.arange(
                start, end, device=hidden_states.device
            )[None, :].expand(batch, -1)
            tile_args = (
                hidden_states,
                cos,
                sin,
                teacher_q,
                teacher_k,
                allowed_tensor,
                index_k,
                positions,
            )
            # Full-L tiles only. Selected KL must not nest checkpoint with the
            # layer wrapper: recompute mixed 16384 indexer scores vs 2560 topk.
            if (
                loss_scope == "full"
                and torch.is_grad_enabled()
                and any(
                    torch.is_tensor(tensor) and tensor.requires_grad
                    for tensor in tile_args
                )
            ):
                kl_sum = kl_sum + checkpoint(
                    _tile_kl, *tile_args, use_reentrant=False
                )
            else:
                kl_sum = kl_sum + _tile_kl(*tile_args)
            blocks += 1
        if blocks <= 0:
            return hidden_states.new_zeros((), dtype=torch.float32)
        return kl_sum / blocks


def selected_token_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Exact GQA attention over shared selected token indices."""

    batch, query_len, h_q, dim = q.shape
    h_kv = k.shape[2]
    if h_q % h_kv:
        raise ValueError("query heads must be divisible by KV heads")
    if indices.shape != valid.shape or indices.shape[:2] != (batch, query_len):
        raise ValueError("indices/valid must have shape [B,Lq,K]")
    groups = h_q // h_kv
    safe = indices.clamp(min=0, max=k.shape[1] - 1)
    batch_index = torch.arange(batch, device=q.device)[:, None, None]
    selected_k = k[batch_index, safe]
    selected_v = v[batch_index, safe]
    grouped_q = q.view(batch, query_len, h_kv, groups, dim)
    # The caller trains under BF16 autocast. Without this guard, autocast
    # silently downcasts the explicit q.float()/k.float() einsum and changes
    # the hard-routing trajectory relative to the FP32-score TileLang kernel.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = torch.einsum(
            "bqhgd,bqkhd->bqhgk", grouped_q.float(), selected_k.float()
        ) * (dim**-0.5)
        scores = scores.masked_fill(
            ~valid[:, :, None, None], float("-inf")
        )
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        probabilities = torch.nn.functional.dropout(
            probabilities, p=dropout_p, training=training
        )
        output = torch.einsum(
            "bqhgk,bqkhd->bqhgd", probabilities.to(v.dtype), selected_v
        )
    return output.reshape(batch, query_len, h_q, dim)


class DreamDsaAttention(_NaiveDreamSparseAttentionBase):
    """Bidirectional Dream attention with DSA direct-token retrieval."""

    def __init__(
        self,
        source_attn: nn.Module,
        *,
        topk: int,
        num_index_heads: int,
        index_head_dim: int,
        query_block_size: int,
        attention_query_block_size: int,
        aux_queries: int,
        aux_loss_scope: str = "full",
        backend: str = "torch",
    ) -> None:
        # Contiguous tokens only. Do not inherit route landmark/chunk holes.
        super().__init__(source_attn, local_window=0, chunk_size=None)
        if attention_query_block_size <= 0:
            raise ValueError("attention_query_block_size must be positive")
        if backend not in {"torch", "tilelang"}:
            raise ValueError(f"unsupported Dream DSA backend: {backend}")
        self.dsa_indexer = DreamDsaIndexer(
            source_attn,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            topk=topk,
            query_block_size=query_block_size,
        )
        self.topk = int(topk)
        self.attention_query_block_size = int(attention_query_block_size)
        self.backend = str(backend)
        self.aux_queries = int(aux_queries)
        if aux_loss_scope not in {"full", "selected"}:
            raise ValueError(
                f"unsupported DSA auxiliary loss scope: {aux_loss_scope}"
            )
        self.aux_loss_scope = str(aux_loss_scope)
        self.warmup_dense = False
        self.index_loss: torch.Tensor | None = None
        self._collect_index_loss = False
        self._index_loss_ctx: tuple | None = None

    def _dense_bidirectional_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Full bidirectional GQA attention for indexer warm-up (no top-k)."""

        groups = q.shape[2] // k.shape[2]
        k_exp = k.repeat_interleave(groups, dim=2)
        v_exp = v.repeat_interleave(groups, dim=2)
        qh = q.transpose(1, 2)
        kh = k_exp.transpose(1, 2)
        vh = v_exp.transpose(1, 2)
        # Additive mask: invalid keys get -inf. Avoid bool-mask True/False flip.
        attn_mask = torch.zeros(
            key_valid.shape[0],
            1,
            1,
            key_valid.shape[1],
            device=q.device,
            dtype=qh.dtype,
        )
        attn_mask = attn_mask.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        output = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        return output.transpose(1, 2).contiguous()

    def prepare_index_loss(self) -> None:
        self.index_loss = None
        self._collect_index_loss = True
        self._index_loss_ctx = None

    def materialize_index_loss(self) -> torch.Tensor:
        """Run indexer KL after the layer checkpoint has returned.

        Computing it inside ``forward`` makes non-reentrant checkpoint see a
        different saved-tensor count on recompute (2339 vs 52 at 16k).
        """

        if self.index_loss is not None:
            return self.index_loss
        ctx = self._index_loss_ctx
        if ctx is None:
            raise RuntimeError("Dream DSA index loss was not captured this step")
        hidden_states, position_embeddings, teacher_q, teacher_k, allowed = ctx
        # Sparse (and post-warmup resume) must not build 16k teacher maps.
        # Full-key KL already ran in the dense warmup phase.
        loss_scope = "selected"
        self.index_loss = self.dsa_indexer.distillation_loss(
            hidden_states,
            position_embeddings,
            teacher_q,
            teacher_k,
            allowed,
            self.aux_queries,
            loss_scope,
        )
        self.index_loss = self.dsa_indexer._unused_indexer_anchor(self.index_loss)
        self._index_loss_ctx = None
        return self.index_loss

    def _selected_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.backend == "tilelang":
            if self.training and self.attention_dropout:
                raise ValueError(
                    "Dream DSA TileLang attention does not support dropout"
                )
            from ops.dsa_selected_attention_tilelang import (
                selected_token_attention_tilelang,
            )

            return selected_token_attention_tilelang(
                q,
                k,
                v,
                indices,
                valid,
                sm_scale=self.head_dim**-0.5,
            )

        outputs: list[torch.Tensor] = []
        for start in range(0, q.shape[1], self.attention_query_block_size):
            end = min(start + self.attention_query_block_size, q.shape[1])

            def block_attention(
                query_block: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                index_block: torch.Tensor,
                valid_block: torch.Tensor,
            ) -> torch.Tensor:
                return selected_token_attention_reference(
                    query_block,
                    key,
                    value,
                    index_block,
                    valid_block,
                    dropout_p=self.attention_dropout,
                    training=self.training,
                )

            block_args = (
                q[:, start:end],
                k,
                v,
                indices[:, start:end],
                valid[:, start:end],
            )
            if self.training and any(tensor.requires_grad for tensor in block_args[:3]):
                output = checkpoint(
                    block_attention, *block_args, use_reentrant=False
                )
            else:
                output = block_attention(*block_args)
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        del cache_position, kwargs
        if use_cache or past_key_value is not None:
            raise NotImplementedError(
                "Dream DSA is a full denoising-step backend and has no AR KV cache"
            )
        if position_embeddings is None:
            value_for_rope = self.v_proj(hidden_states).view(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            ).transpose(1, 2)
            position_embeddings = self.rotary_emb(value_for_rope, position_ids)

        q, k, v = self._project_qkv_blhd(
            hidden_states, position_ids, position_embeddings
        )
        batch, seq_len = hidden_states.shape[:2]
        key_valid = self._key_valid_mask(
            attention_mask, batch, seq_len, hidden_states.device
        )
        real_key = self._real_key_mask(seq_len, hidden_states.device)
        key_valid = key_valid & real_key[None, :]
        allowed = key_valid

        if self.training and self._collect_index_loss:
            # Stash only. Distillation is materialized after the checkpointed
            # layer returns so recompute does not change the saved-tensor set.
            self._index_loss_ctx = (
                hidden_states.detach(),
                tuple(item.detach() for item in position_embeddings),
                q.detach(),
                k.detach(),
                allowed.detach(),
            )

        if self.warmup_dense:
            output = self._dense_bidirectional_attention(q, k, v, allowed)
            output = output.reshape(batch, seq_len, self.hidden_size)
            return (
                self.o_proj(output),
                None if not output_attentions else None,
                past_key_value,
            )

        indices, valid = self.dsa_indexer.select(
            hidden_states.detach(),
            tuple(item.detach() for item in position_embeddings),
            allowed,
        )
        if self._prefill_capture:
            from prefilling_v2.fastdllm_cache import LayerKVCache

            with torch.no_grad():
                _, index_keys, _ = self.dsa_indexer._project(
                    hidden_states, position_embeddings
                )
            self._captured_prefill_cache = LayerKVCache(
                key=k.detach().contiguous(),
                value=v.detach().contiguous(),
                key_valid=allowed.detach().contiguous(),
                dsa_index_keys=index_keys.detach().contiguous(),
            )
        output = self._selected_attention(q, k, v, indices, valid)
        output = output.reshape(batch, seq_len, self.hidden_size)
        return self.o_proj(output), None if not output_attentions else None, past_key_value


@dataclass(frozen=True)
class DsaLayerPlan:
    dsa_layers: list[int]
    sliding_window_layers: list[int]
    dense_layers: list[int]


def install_dream_dsa_attention(
    model: nn.Module,
    *,
    interleave: int,
    local_window: int,
    chunk_size: int,
    topk: int,
    num_index_heads: int,
    index_head_dim: int,
    query_block_size: int,
    attention_query_block_size: int,
    aux_queries: int,
    non_dsa_attention: str,
    aux_loss_scope: str = "full",
    backend: str = "torch",
) -> DsaLayerPlan:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("expected a DreamModel-like object with model.layers")
    if non_dsa_attention not in {"sliding", "dense"}:
        raise ValueError(f"unsupported non-DSA attention: {non_dsa_attention}")

    from prefilling_v2.attention import KernelDreamSlidingWindowAttention

    dsa_set = set(route_layer_indices(len(model.model.layers), interleave))
    dsa_layers: list[int] = []
    sliding_layers: list[int] = []
    dense_layers: list[int] = []
    for layer_idx, layer in enumerate(model.model.layers):
        source_attn = layer.self_attn
        if layer_idx in dsa_set:
            layer.self_attn = DreamDsaAttention(
                source_attn,
                topk=topk,
                num_index_heads=num_index_heads,
                index_head_dim=index_head_dim,
                query_block_size=query_block_size,
                attention_query_block_size=attention_query_block_size,
                aux_queries=aux_queries,
                aux_loss_scope=aux_loss_scope,
                backend=backend,
            )
            dsa_layers.append(layer_idx)
        elif non_dsa_attention == "sliding":
            import inspect

            swa_kwargs = {"allow_fallback": False}
            if (
                "skip_inert_slots"
                in inspect.signature(
                    KernelDreamSlidingWindowAttention.__init__
                ).parameters
            ):
                swa_kwargs["skip_inert_slots"] = False
            layer.self_attn = KernelDreamSlidingWindowAttention(
                source_attn,
                local_window,
                chunk_size,
                **swa_kwargs,
            )
            sliding_layers.append(layer_idx)
        else:
            dense_layers.append(layer_idx)

    model.config.dream_dsa_interleave = int(interleave)
    model.config.dream_dsa_topk = int(topk)
    model.config.dream_dsa_num_index_heads = int(num_index_heads)
    model.config.dream_dsa_index_head_dim = int(index_head_dim)
    model.config.dream_dsa_query_block_size = int(query_block_size)
    model.config.dream_dsa_attention_query_block_size = int(
        attention_query_block_size
    )
    model.config.dream_dsa_aux_queries = int(aux_queries)
    model.config.dream_dsa_aux_loss_scope = str(aux_loss_scope)
    model.config.dream_dsa_backend = str(backend)
    model.config.dream_dsa_chunk_size = int(chunk_size)
    model.config.dream_dsa_non_dsa_attention = str(non_dsa_attention)
    model.config.dream_swa_local_window = int(local_window)
    return DsaLayerPlan(dsa_layers, sliding_layers, dense_layers)


def prepare_dsa_index_losses(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, DreamDsaAttention):
            module.prepare_index_loss()
            count += 1
    return count


def collect_dsa_index_loss(model: nn.Module) -> torch.Tensor:
    modules = [
        module
        for module in model.modules()
        if isinstance(module, DreamDsaAttention)
    ]
    losses = [module.materialize_index_loss() for module in modules]
    if not losses or any(loss is None for loss in losses):
        raise RuntimeError("one or more Dream DSA layers did not produce index loss")
    return torch.stack([loss.float() for loss in losses if loss is not None]).mean()


def dsa_modules(model: nn.Module) -> Iterable[DreamDsaAttention]:
    return (
        module for module in model.modules() if isinstance(module, DreamDsaAttention)
    )


def set_dsa_official_stage(
    model: nn.Module,
    *,
    warmup: bool,
    aux_queries: int,
    sparse_scope: str = "selected",
) -> None:
    """Switch 7 DSA layers between dense indexer warm-up and sparse LoRA."""

    for module in dsa_modules(model):
        module.warmup_dense = bool(warmup)
        module.aux_queries = int(aux_queries)
        module.aux_loss_scope = str(sparse_scope)
