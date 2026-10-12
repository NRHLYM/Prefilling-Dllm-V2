"""Compact-GQA candidate token scores for HISA-lite refinement."""

import math

import tilelang
import tilelang.language as T
import torch


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[4], pass_configs=_PASS_CONFIGS)
def _token_score_fwd(
    batch,
    query_len,
    kv_len,
    kv_heads,
    groups,
    head_dim,
    selected_chunks,
    chunk_size,
    sm_scale,
    threads=128,
):
    dtype = "bfloat16"
    accum_dtype = "float"
    M = 16
    D = head_dim
    S = chunk_size
    BK = min(256, tilelang.math.next_power_of_2(D))

    q_shape = [batch, query_len, kv_heads * groups, D]
    k_shape = [batch, kv_len, kv_heads, D]
    index_shape = [batch, query_len, kv_heads, selected_chunks]
    valid_shape = [batch, kv_len]
    score_shape = [
        batch,
        query_len,
        kv_heads,
        selected_chunks,
        S,
    ]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(index_shape, "int32"),
        KeyValid: T.Tensor(valid_shape, "int32"),
        Scores: T.Tensor(score_shape, accum_dtype),
    ):
        with T.Kernel(
            batch, query_len, kv_heads, threads=threads
        ) as (i_b, i_q, i_h):
            Q_shared = T.alloc_shared([M, BK], dtype)
            K_shared = T.alloc_shared([S, BK], dtype)
            score_frag = T.alloc_fragment([M, S], accum_dtype)
            score_max = T.alloc_fragment([S], accum_dtype)

            T.fill(Q_shared, 0)
            for g, d in T.Parallel(groups, D):
                Q_shared[g, d] = Q[i_b, i_q, i_h * groups + g, d]

            for i_sel in T.serial(selected_chunks):
                chunk_idx = Indices[i_b, i_q, i_h, i_sel]
                chunk_start = chunk_idx * S

                T.fill(K_shared, 0)
                if chunk_idx >= 0:
                    T.copy(
                        K[
                            i_b,
                            chunk_start:chunk_start + S,
                            i_h,
                            :,
                        ],
                        K_shared,
                    )

                T.clear(score_frag)
                T.gemm(
                    Q_shared,
                    K_shared,
                    score_frag,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for m, s in T.Parallel(M, S):
                    score_frag[m, s] = T.if_then_else(
                        m < groups,
                        score_frag[m, s],
                        -T.infinity(accum_dtype),
                    )
                T.fill(score_max, -T.infinity(accum_dtype))
                T.reduce_max(score_frag, score_max, dim=0, clear=True)

                for s in T.Parallel(S):
                    valid = (
                        (chunk_idx >= 0)
                        and (s < S - 1)
                        and (KeyValid[i_b, chunk_start + s] != 0)
                    )
                    Scores[i_b, i_q, i_h, i_sel, s] = T.if_then_else(
                        valid,
                        score_max[s] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

    return main


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int, int]:
    if q.device.type != "cuda":
        raise ValueError("candidate_token_scores_gqa requires CUDA tensors")
    if not (q.device == k.device == indices.device == key_valid.device):
        raise ValueError("all token-score tensors must share one CUDA device")
    if q.ndim != 4 or k.ndim != 4 or indices.ndim != 4:
        raise ValueError("q, k, and indices must be rank-four tensors")
    batch, query_len, query_heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[-1] != head_dim:
        raise ValueError(
            f"incompatible q/k shapes {tuple(q.shape)} and {tuple(k.shape)}"
        )
    kv_len, kv_heads = k.shape[1:3]
    if query_heads % kv_heads:
        raise ValueError(
            f"query_heads={query_heads} must divide kv_heads={kv_heads}"
        )
    groups = query_heads // kv_heads
    selected = indices.shape[-1]
    if indices.shape != (batch, query_len, kv_heads, selected):
        raise ValueError(f"invalid indices shape {tuple(indices.shape)}")
    if key_valid.shape != (batch, kv_len):
        raise ValueError(f"invalid key_valid shape {tuple(key_valid.shape)}")
    if chunk_size < 16 or kv_len % chunk_size:
        raise ValueError(
            f"kv_len={kv_len} must be divisible by chunk_size >= 16"
        )
    if head_dim > 256:
        raise ValueError(f"head_dim must be <= 256, got {head_dim}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("candidate_token_scores_gqa requires BF16 q and k")
    if indices.dtype != torch.int32:
        raise ValueError("indices must be int32")
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and indices.is_contiguous()
        and key_valid.is_contiguous()
    ):
        raise ValueError("candidate_token_scores_gqa requires contiguous tensors")
    return (
        batch,
        query_len,
        kv_len,
        kv_heads,
        groups,
        head_dim,
        selected,
    )


def candidate_token_scores_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    key_valid: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Return FP32 max-over-query-group scores for routed candidate tokens."""

    (
        batch,
        query_len,
        kv_len,
        kv_heads,
        groups,
        head_dim,
        selected,
    ) = _validate(q, k, indices, key_valid, chunk_size)
    kernel = _token_score_fwd(
        batch,
        query_len,
        kv_len,
        kv_heads,
        groups,
        head_dim,
        selected,
        chunk_size,
        1.0 / math.sqrt(head_dim),
    )
    return kernel(
        q,
        k,
        indices,
        key_valid.to(dtype=torch.int32),
    )
