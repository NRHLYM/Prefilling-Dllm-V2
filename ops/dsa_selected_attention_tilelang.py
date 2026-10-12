"""TileLang selected-token GQA attention for bidirectional Dream DSA.

The forward/backward structure follows TileLang's official DeepSeek-V3.2
sparse MLA example, but this kernel targets Dream's ordinary GQA layout:
separate K/V tensors, shared token indices, and an explicit validity mask.
"""

import math

import torch


_TILELANG_IMPORT_ERROR: Exception | None = None
try:
    import tilelang
    import tilelang.language as T
except Exception as error:  # pragma: no cover - exercised without TileLang
    tilelang = None
    T = None
    _TILELANG_IMPORT_ERROR = error


if tilelang is not None:
    _PASS_CONFIGS = {
        # DSA's hard top-k makes small hidden-state errors discontinuous in
        # later layers. Keep accurate exp/log lowering for training parity.
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

    @tilelang.jit(out_idx=[5, 6], pass_configs=_PASS_CONFIGS)
    def _selected_attention_fwd(
        batch: int,
        query_len: int,
        kv_len: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        topk: int,
        sm_scale: float,
        block_k: int = 64,
        threads: int = 128,
    ):
        groups = query_heads // kv_heads
        padded_groups = max(tilelang.math.next_power_of_2(groups), 16)
        scale_log2 = sm_scale * 1.4426950408889634

        q_shape = [batch, query_len, query_heads, head_dim]
        kv_shape = [batch, kv_len, kv_heads, head_dim]
        index_shape = [batch, query_len, topk]
        output_shape = q_shape
        lse_shape = [batch, query_len, query_heads]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "bfloat16"),
            K: T.Tensor(kv_shape, "bfloat16"),
            V: T.Tensor(kv_shape, "bfloat16"),
            Indices: T.Tensor(index_shape, "int32"),
            Valid: T.Tensor(index_shape, "int8"),
            Output: T.Tensor(output_shape, "bfloat16"),
            Lse: T.Tensor(lse_shape, "float"),
        ):
            with T.Kernel(
                query_len, batch, kv_heads, threads=threads
            ) as (i_q, i_b, i_kvh):
                q_shared = T.alloc_shared(
                    [padded_groups, head_dim], "bfloat16"
                )
                k_shared = T.alloc_shared(
                    [block_k, head_dim], "bfloat16"
                )
                v_shared = T.alloc_shared(
                    [block_k, head_dim], "bfloat16"
                )
                p_shared = T.alloc_shared(
                    [padded_groups, block_k], "bfloat16"
                )
                output_shared = T.alloc_shared(
                    [padded_groups, head_dim], "bfloat16"
                )

                scores = T.alloc_fragment(
                    [padded_groups, block_k], "float"
                )
                output = T.alloc_fragment(
                    [padded_groups, head_dim], "float"
                )
                row_max = T.alloc_fragment([padded_groups], "float")
                previous_max = T.alloc_fragment(
                    [padded_groups], "float"
                )
                block_sum = T.alloc_fragment([padded_groups], "float")
                row_sum = T.alloc_fragment([padded_groups], "float")
                alpha = T.alloc_fragment([padded_groups], "float")

                T.fill(q_shared, 0.0)
                T.copy(
                    Q[
                        i_b,
                        i_q,
                        i_kvh * groups : (i_kvh + 1) * groups,
                        :,
                    ],
                    q_shared[:groups, :],
                )
                T.fill(row_sum, 0.0)
                T.fill(row_max, -(2**30))

                # Pass 1 computes the FP32 softmax normalizer over the complete
                # selected set.  This matches torch.softmax's cast order: the
                # normalized probabilities, rather than block-local exp values,
                # are rounded to BF16 before the P @ V matmul.
                for i_block in T.serial(tilelang.cdiv(topk, block_k)):
                    for i_token, i_dim in T.Parallel(block_k, head_dim):
                        selected = Indices[
                            i_b, i_q, i_block * block_k + i_token
                        ]
                        k_shared[i_token, i_dim] = K[
                            i_b, selected, i_kvh, i_dim
                        ]

                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_head, i_token in T.Parallel(
                        padded_groups, block_k
                    ):
                        scores[i_head, i_token] = T.if_then_else(
                            Valid[
                                i_b,
                                i_q,
                                i_block * block_k + i_token,
                            ]
                            != 0,
                            scores[i_head, i_token],
                            -T.infinity("float"),
                        )

                    T.copy(row_max, previous_max)
                    T.reduce_max(scores, row_max, dim=1, clear=False)
                    for i_head in T.Parallel(padded_groups):
                        row_max[i_head] = T.max(
                            row_max[i_head], previous_max[i_head]
                        )
                        alpha[i_head] = T.exp2(
                            (previous_max[i_head] - row_max[i_head])
                            * scale_log2
                        )
                    for i_head, i_token in T.Parallel(
                        padded_groups, block_k
                    ):
                        scores[i_head, i_token] = T.if_then_else(
                            Valid[
                                i_b,
                                i_q,
                                i_block * block_k + i_token,
                            ]
                            != 0,
                            T.exp2(
                                (scores[i_head, i_token] - row_max[i_head])
                                * scale_log2
                            ),
                            0.0,
                        )
                    T.reduce_sum(scores, block_sum, dim=1, clear=True)
                    for i_head in T.Parallel(padded_groups):
                        row_sum[i_head] = (
                            row_sum[i_head] * alpha[i_head]
                            + block_sum[i_head]
                        )

                T.fill(output, 0.0)
                for i_block in T.Pipelined(
                    tilelang.cdiv(topk, block_k), num_stages=1
                ):
                    for i_token, i_dim in T.Parallel(block_k, head_dim):
                        selected = Indices[
                            i_b, i_q, i_block * block_k + i_token
                        ]
                        k_shared[i_token, i_dim] = K[
                            i_b, selected, i_kvh, i_dim
                        ]
                        v_shared[i_token, i_dim] = V[
                            i_b, selected, i_kvh, i_dim
                        ]

                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i_head, i_token in T.Parallel(
                        padded_groups, block_k
                    ):
                        is_valid = (
                            Valid[
                                i_b,
                                i_q,
                                i_block * block_k + i_token,
                            ]
                            != 0
                        )
                        scores[i_head, i_token] = T.if_then_else(
                            is_valid,
                            T.exp2(
                                (scores[i_head, i_token] - row_max[i_head])
                                * scale_log2
                            )
                            / row_sum[i_head],
                            0.0,
                        )

                    T.copy(scores, p_shared)
                    T.gemm(
                        p_shared,
                        v_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                T.copy(output, output_shared)
                T.copy(
                    output_shared[:groups, :],
                    Output[
                        i_b,
                        i_q,
                        i_kvh * groups : (i_kvh + 1) * groups,
                        :,
                    ],
                )
                for i_head in T.Parallel(groups):
                    Lse[
                        i_b, i_q, i_kvh * groups + i_head
                    ] = T.if_then_else(
                        row_sum[i_head] > 0.0,
                        T.log2(row_sum[i_head])
                        + row_max[i_head] * scale_log2,
                        -T.infinity("float"),
                    )

        return main

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _selected_attention_bwd(
        batch: int,
        query_len: int,
        kv_len: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        topk: int,
        sm_scale: float,
        block_k: int = 32,
        threads: int = 128,
    ):
        groups = query_heads // kv_heads
        padded_groups = max(tilelang.math.next_power_of_2(groups), 16)
        scale_log2 = sm_scale * 1.4426950408889634

        q_shape = [batch, query_len, query_heads, head_dim]
        kv_shape = [batch, kv_len, kv_heads, head_dim]
        index_shape = [batch, query_len, topk]
        lse_shape = [batch, query_len, query_heads]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "bfloat16"),
            K: T.Tensor(kv_shape, "bfloat16"),
            V: T.Tensor(kv_shape, "bfloat16"),
            DOutput: T.Tensor(q_shape, "bfloat16"),
            Indices: T.Tensor(index_shape, "int32"),
            Valid: T.Tensor(index_shape, "int8"),
            Lse: T.Tensor(lse_shape, "float"),
            Delta: T.Tensor(lse_shape, "float"),
            DQ: T.Tensor(q_shape, "bfloat16"),
            DK: T.Tensor(kv_shape, "float"),
            DV: T.Tensor(kv_shape, "float"),
        ):
            with T.Kernel(
                query_len, batch, kv_heads, threads=threads
            ) as (i_q, i_b, i_kvh):
                q_shared = T.alloc_shared(
                    [padded_groups, head_dim], "bfloat16"
                )
                do_shared = T.alloc_shared(
                    [padded_groups, head_dim], "bfloat16"
                )
                k_shared = T.alloc_shared(
                    [block_k, head_dim], "bfloat16"
                )
                v_shared = T.alloc_shared(
                    [block_k, head_dim], "bfloat16"
                )
                p_shared = T.alloc_shared(
                    [padded_groups, block_k], "bfloat16"
                )
                ds_shared = T.alloc_shared(
                    [padded_groups, block_k], "bfloat16"
                )
                dq_shared = T.alloc_shared(
                    [padded_groups, head_dim], "bfloat16"
                )

                scores = T.alloc_fragment(
                    [padded_groups, block_k], "float"
                )
                dp = T.alloc_fragment(
                    [padded_groups, block_k], "float"
                )
                dq = T.alloc_fragment(
                    [padded_groups, head_dim], "float"
                )
                dk = T.alloc_fragment([block_k, head_dim], "float")
                dv = T.alloc_fragment([block_k, head_dim], "float")
                lse = T.alloc_fragment([padded_groups], "float")
                delta = T.alloc_fragment([padded_groups], "float")

                T.fill(q_shared, 0.0)
                T.fill(do_shared, 0.0)
                T.copy(
                    Q[
                        i_b,
                        i_q,
                        i_kvh * groups : (i_kvh + 1) * groups,
                        :,
                    ],
                    q_shared[:groups, :],
                )
                T.copy(
                    DOutput[
                        i_b,
                        i_q,
                        i_kvh * groups : (i_kvh + 1) * groups,
                        :,
                    ],
                    do_shared[:groups, :],
                )
                T.fill(lse, 0.0)
                T.fill(delta, 0.0)
                for i_head in T.Parallel(groups):
                    lse[i_head] = Lse[
                        i_b, i_q, i_kvh * groups + i_head
                    ]
                    delta[i_head] = Delta[
                        i_b, i_q, i_kvh * groups + i_head
                    ]
                T.fill(dq, 0.0)

                for i_block in T.Pipelined(
                    tilelang.cdiv(topk, block_k), num_stages=0
                ):
                    for i_token, i_dim in T.Parallel(block_k, head_dim):
                        selected = Indices[
                            i_b, i_q, i_block * block_k + i_token
                        ]
                        k_shared[i_token, i_dim] = K[
                            i_b, selected, i_kvh, i_dim
                        ]
                        v_shared[i_token, i_dim] = V[
                            i_b, selected, i_kvh, i_dim
                        ]

                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        do_shared,
                        v_shared,
                        dp,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for i_head, i_token in T.Parallel(
                        padded_groups, block_k
                    ):
                        is_valid = (
                            Valid[
                                i_b,
                                i_q,
                                i_block * block_k + i_token,
                            ]
                            != 0
                        )
                        probability = T.if_then_else(
                            is_valid,
                            T.exp2(
                                scores[i_head, i_token] * scale_log2
                                - lse[i_head]
                            ),
                            0.0,
                        )
                        scores[i_head, i_token] = probability
                        dp[i_head, i_token] = (
                            probability
                            * (dp[i_head, i_token] - delta[i_head])
                            * sm_scale
                        )

                    T.copy(scores, p_shared)
                    T.copy(dp, ds_shared)
                    T.gemm(
                        ds_shared,
                        k_shared,
                        dq,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        ds_shared,
                        q_shared,
                        dk,
                        transpose_A=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        p_shared,
                        do_shared,
                        dv,
                        transpose_A=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for i_token, i_dim in T.Parallel(block_k, head_dim):
                        selected = Indices[
                            i_b, i_q, i_block * block_k + i_token
                        ]
                        if Valid[
                            i_b, i_q, i_block * block_k + i_token
                        ] != 0:
                            T.atomic_add(
                                DK[i_b, selected, i_kvh, i_dim],
                                dk[i_token, i_dim],
                            )
                            T.atomic_add(
                                DV[i_b, selected, i_kvh, i_dim],
                                dv[i_token, i_dim],
                            )

                T.copy(dq, dq_shared)
                T.copy(
                    dq_shared[:groups, :],
                    DQ[
                        i_b,
                        i_q,
                        i_kvh * groups : (i_kvh + 1) * groups,
                        :,
                    ],
                )

        return main


def tilelang_is_available() -> bool:
    return tilelang is not None


def _require_tilelang() -> None:
    if tilelang is None:
        raise RuntimeError(
            "Dream DSA TileLang backend requires the pinned tilelang package"
        ) from _TILELANG_IMPORT_ERROR


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    if q.device.type != "cuda":
        raise ValueError("Dream DSA TileLang attention requires CUDA tensors")
    if not (q.device == k.device == v.device == indices.device == valid.device):
        raise ValueError("all Dream DSA TileLang tensors must share one device")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("q/k/v must have [B,L,H,D] GQA layouts")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Dream DSA requires matching Q/K batch and head dims")
    if q.shape[0:2] != indices.shape[0:2] or valid.shape != indices.shape:
        raise ValueError("indices/valid must have shape [B,L,topk]")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if q.shape[-1] != 128:
        raise ValueError(
            f"Dream DSA TileLang kernel currently requires head_dim=128, got {q.shape[-1]}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Dream DSA TileLang attention requires BF16 q/k/v")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Dream DSA TileLang q/k/v tensors must be contiguous")


def _prepare_indices(
    indices: torch.Tensor,
    valid: torch.Tensor,
    kv_len: int,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = indices.shape[-1]
    padded_topk = math.ceil(topk / block) * block
    safe = indices.clamp(min=0, max=kv_len - 1).to(torch.int32)
    keep = valid.to(torch.int8)
    if padded_topk != topk:
        pad_shape = indices.shape[:-1] + (padded_topk - topk,)
        safe = torch.cat(
            (safe, torch.zeros(pad_shape, dtype=torch.int32, device=safe.device)),
            dim=-1,
        )
        keep = torch.cat(
            (keep, torch.zeros(pad_shape, dtype=torch.int8, device=keep.device)),
            dim=-1,
        )
    return safe.contiguous(), keep.contiguous()


class _SelectedTokenAttentionTileLang(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, valid, sm_scale):
        _require_tilelang()
        _validate_inputs(q, k, v, indices, valid)
        safe, keep = _prepare_indices(indices, valid, k.shape[1], block=64)
        batch, query_len, query_heads, head_dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        kernel = _selected_attention_fwd(
            batch,
            query_len,
            kv_len,
            query_heads,
            kv_heads,
            head_dim,
            safe.shape[-1],
            sm_scale,
        )
        output, lse = kernel(q, k, v, safe, keep)
        ctx.save_for_backward(q, k, v, safe, keep, output, lse)
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, indices, valid, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        delta = (output.float() * grad_output.float()).sum(dim=-1)
        batch, query_len, query_heads, head_dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        kernel = _selected_attention_bwd(
            batch,
            query_len,
            kv_len,
            query_heads,
            kv_heads,
            head_dim,
            indices.shape[-1],
            ctx.sm_scale,
        )
        kernel(
            q,
            k,
            v,
            grad_output,
            indices,
            valid,
            lse,
            delta.contiguous(),
            dq,
            dk,
            dv,
        )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


def selected_token_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute exact selected-set GQA attention with TileLang kernels."""

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return _SelectedTokenAttentionTileLang.apply(
        q, k, v, indices, valid, float(sm_scale)
    )


__all__ = ["selected_token_attention_tilelang", "tilelang_is_available"]
