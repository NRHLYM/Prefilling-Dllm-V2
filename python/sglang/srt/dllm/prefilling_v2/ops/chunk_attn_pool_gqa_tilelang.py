"""Compact native-GQA landmark pooling for the Dream HiLS backend."""

import math

import tilelang
import tilelang.language as T
import torch


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def _compact_pool_torch(mu_q, k_chunked, key_valid, sm_scale):
    scores = torch.einsum(
        "bnkgd,bnskd->bnkgs", mu_q.float(), k_chunked.float()
    ) * sm_scale
    valid = key_valid.bool().clone()
    valid[..., -1] = False
    scores = scores.masked_fill(
        ~valid[:, :, None, None, :], float("-inf")
    )
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    pooled = torch.einsum(
        "bnkgs,bnskd->bnkgd", probs, k_chunked.float()
    )
    log_probs = torch.where(
        probs > 0,
        probs.clamp_min(1e-30).log(),
        torch.zeros_like(probs),
    )
    entropy = -(probs * log_probs).sum(dim=-1)
    return pooled.to(k_chunked.dtype), entropy


_COMPILED_COMPACT_POOL = torch.compile(
    _compact_pool_torch,
    fullgraph=True,
    dynamic=False,
)


@tilelang.jit(out_idx=[3, 4, 5], pass_configs=_PASS_CONFIGS)
def _pool_gqa_fwd(
    batch_chunks,
    chunk_size,
    kv_heads,
    groups,
    head_dim,
    sm_scale,
    threads=128,
):
    dtype = "bfloat16"
    accum_dtype = "float"
    M = 16
    S = chunk_size
    D = head_dim
    LN2 = 0.6931471805599453

    q_shape = [batch_chunks, kv_heads, groups, D]
    k_shape = [batch_chunks, S, kv_heads, D]
    valid_shape = [batch_chunks, S]
    out_k_shape = [batch_chunks, kv_heads, groups, D]
    out_b_shape = [batch_chunks, kv_heads, groups]
    out_p_shape = [batch_chunks, S, kv_heads, groups]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        KeyValid: T.Tensor(valid_shape, "int32"),
        LmkK: T.Tensor(out_k_shape, dtype),
        LmkB: T.Tensor(out_b_shape, accum_dtype),
        POut: T.Tensor(out_p_shape, accum_dtype),
    ):
        with T.Kernel(batch_chunks, kv_heads, threads=threads) as (i_bn, i_kv):
            Q_shared = T.alloc_shared([M, D], dtype)
            K_shared = T.alloc_shared([S, D], dtype)
            P_shared = T.alloc_shared([M, S], dtype)
            O_shared = T.alloc_shared([M, D], dtype)
            chunk_valid = T.alloc_shared([1], "int32")

            logits = T.alloc_fragment([M, S], accum_dtype)
            output = T.alloc_fragment([M, D], accum_dtype)
            scores_max = T.alloc_fragment([M], accum_dtype)
            scores_sum = T.alloc_fragment([M], accum_dtype)
            entropy_terms = T.alloc_fragment([M, S], accum_dtype)
            entropy_sum = T.alloc_fragment([M], accum_dtype)

            T.fill(chunk_valid, 0)
            if T.get_thread_binding() == 0:
                for s in T.serial(S - 1):
                    if KeyValid[i_bn, s] != 0:
                        chunk_valid[0] = 1
            T.sync_threads()

            for m, d in T.Parallel(M, D):
                Q_shared[m, d] = T.if_then_else(
                    m < groups,
                    MuQ[i_bn, i_kv, T.if_then_else(m < groups, m, 0), d],
                    T.cast(0, dtype),
                )
            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, i_kv, d]

            T.clear(logits)
            T.gemm(
                Q_shared,
                K_shared,
                logits,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for m, s in T.Parallel(M, S):
                is_valid_score = (
                    (m < groups)
                    and (chunk_valid[0] != 0)
                    and (s < S - 1)
                    and (KeyValid[i_bn, s] != 0)
                )
                logits[m, s] = T.if_then_else(
                    is_valid_score,
                    logits[m, s] * sm_scale,
                    T.if_then_else(s == 0, T.cast(0, accum_dtype), -T.infinity(accum_dtype)),
                )

            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(logits, scores_max, dim=1, clear=True)
            for m, s in T.Parallel(M, S):
                logits[m, s] = T.exp2(
                    (logits[m, s] - scores_max[m]) * 1.4426950408889634
                )
            T.fill(scores_sum, 0.0)
            T.reduce_sum(logits, scores_sum, dim=1, clear=True)
            for m, s in T.Parallel(M, S):
                logits[m, s] = logits[m, s] / scores_sum[m]

            for m, s in T.Parallel(M, S):
                entropy_terms[m, s] = T.if_then_else(
                    logits[m, s] > 0,
                    logits[m, s] * T.log2(logits[m, s]) * LN2,
                    T.cast(0, accum_dtype),
                )
            T.fill(entropy_sum, 0.0)
            T.reduce_sum(entropy_terms, entropy_sum, dim=1, clear=True)

            T.copy(logits, P_shared)
            T.clear(output)
            T.gemm(
                P_shared,
                K_shared,
                output,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(output, O_shared)

            for g, s in T.Parallel(groups, S):
                POut[i_bn, s, i_kv, g] = T.if_then_else(
                    chunk_valid[0] != 0,
                    logits[g, s],
                    T.cast(0, accum_dtype),
                )
            for g in T.Parallel(groups):
                LmkB[i_bn, i_kv, g] = T.if_then_else(
                    chunk_valid[0] != 0,
                    -entropy_sum[g],
                    T.cast(0, accum_dtype),
                )
            for g, d in T.Parallel(groups, D):
                LmkK[i_bn, i_kv, g, d] = T.if_then_else(
                    chunk_valid[0] != 0,
                    O_shared[g, d],
                    T.cast(0, dtype),
                )

    return main


@tilelang.jit(out_idx=[6, 7], pass_configs=_PASS_CONFIGS)
def _pool_gqa_bwd(
    batch_chunks,
    chunk_size,
    kv_heads,
    groups,
    head_dim,
    sm_scale,
    threads=128,
):
    dtype = "bfloat16"
    accum_dtype = "float"
    M = 16
    S = chunk_size
    D = head_dim
    LN2 = 0.6931471805599453

    q_shape = [batch_chunks, kv_heads, groups, D]
    k_shape = [batch_chunks, S, kv_heads, D]
    p_shape = [batch_chunks, S, kv_heads, groups]
    b_shape = [batch_chunks, kv_heads, groups]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        PSaved: T.Tensor(p_shape, accum_dtype),
        EntropySaved: T.Tensor(b_shape, accum_dtype),
        GradLmkK: T.Tensor(q_shape, dtype),
        GradEntropy: T.Tensor(b_shape, accum_dtype),
        DMuQ: T.Tensor(q_shape, accum_dtype),
        DK: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(batch_chunks, kv_heads, threads=threads) as (i_bn, i_kv):
            Q_shared = T.alloc_shared([M, D], dtype)
            K_shared = T.alloc_shared([S, D], dtype)
            GradK_shared = T.alloc_shared([M, D], dtype)
            P_shared = T.alloc_shared([M, S], dtype)
            DP_shared = T.alloc_shared([M, S], accum_dtype)
            DLogits_shared = T.alloc_shared([M, S], dtype)
            DMu_shared = T.alloc_shared([M, D], dtype)
            DK_shared = T.alloc_shared([S, D], dtype)

            d_p = T.alloc_fragment([M, S], accum_dtype)
            d_mu = T.alloc_fragment([M, D], accum_dtype)
            d_k = T.alloc_fragment([S, D], accum_dtype)
            d_k_direct = T.alloc_fragment([S, D], accum_dtype)
            p_delta_terms = T.alloc_fragment([M, S], accum_dtype)
            p_delta = T.alloc_fragment([M], accum_dtype)

            for m, d in T.Parallel(M, D):
                Q_shared[m, d] = T.if_then_else(
                    m < groups,
                    MuQ[i_bn, i_kv, T.if_then_else(m < groups, m, 0), d],
                    T.cast(0, dtype),
                )
                GradK_shared[m, d] = T.if_then_else(
                    m < groups,
                    GradLmkK[i_bn, i_kv, T.if_then_else(m < groups, m, 0), d],
                    T.cast(0, dtype),
                )
            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, i_kv, d]
            for m, s in T.Parallel(M, S):
                P_shared[m, s] = T.if_then_else(
                    m < groups,
                    T.cast(PSaved[i_bn, s, i_kv, T.if_then_else(m < groups, m, 0)], dtype),
                    T.cast(0, dtype),
                )

            T.clear(d_p)
            T.gemm(
                GradK_shared,
                K_shared,
                d_p,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(d_p, DP_shared)

            for m, s in T.Parallel(M, S):
                p_delta_terms[m, s] = T.cast(P_shared[m, s], accum_dtype) * DP_shared[m, s]
            T.fill(p_delta, 0.0)
            T.reduce_sum(p_delta_terms, p_delta, dim=1, clear=True)

            for m, s in T.Parallel(M, S):
                p_val = T.cast(P_shared[m, s], accum_dtype)
                grad_b = T.if_then_else(
                    m < groups,
                    GradEntropy[i_bn, i_kv, T.if_then_else(m < groups, m, 0)],
                    T.cast(0, accum_dtype),
                )
                entropy = T.if_then_else(
                    m < groups,
                    EntropySaved[i_bn, i_kv, T.if_then_else(m < groups, m, 0)],
                    T.cast(0, accum_dtype),
                )
                log_p = T.if_then_else(
                    p_val > 0,
                    T.log2(p_val) * LN2,
                    T.cast(0, accum_dtype),
                )
                d_logit = T.if_then_else(
                    (m < groups) and (p_val > 0),
                    p_val * ((DP_shared[m, s] - p_delta[m]) - grad_b * (log_p + entropy)),
                    T.cast(0, accum_dtype),
                )
                DLogits_shared[m, s] = T.cast(d_logit * sm_scale, dtype)

            T.clear(d_mu)
            T.gemm(
                DLogits_shared,
                K_shared,
                d_mu,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(d_mu, DMu_shared)

            T.clear(d_k)
            T.gemm(
                DLogits_shared,
                Q_shared,
                d_k,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.clear(d_k_direct)
            T.gemm(
                P_shared,
                GradK_shared,
                d_k_direct,
                transpose_A=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for s, d in T.Parallel(S, D):
                DK_shared[s, d] = T.cast(d_k[s, d] + d_k_direct[s, d], dtype)

            for g, d in T.Parallel(groups, D):
                DMuQ[i_bn, i_kv, g, d] = T.cast(DMu_shared[g, d], accum_dtype)
            for s, d in T.Parallel(S, D):
                DK[i_bn, s, i_kv, d] = T.cast(DK_shared[s, d], accum_dtype)

    return main


class _ChunkAttnPoolGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_q, k_chunked, key_valid, sm_scale):
        batch, chunks, kv_heads, groups, head_dim = mu_q.shape
        chunk_size = k_chunked.shape[2]
        batch_chunks = batch * chunks
        mu_flat = mu_q.reshape(batch_chunks, kv_heads, groups, head_dim).contiguous()
        k_flat = k_chunked.reshape(batch_chunks, chunk_size, kv_heads, head_dim).contiguous()
        valid_flat = key_valid.reshape(batch_chunks, chunk_size).to(torch.int32).contiguous()

        kernel = _pool_gqa_fwd(
            batch_chunks,
            chunk_size,
            kv_heads,
            groups,
            head_dim,
            sm_scale,
        )
        with torch.cuda.device(mu_flat.device):
            lmk_k, entropy, probs = kernel(mu_flat, k_flat, valid_flat)
        ctx.save_for_backward(mu_flat, k_flat, probs, entropy)
        ctx.shape = (batch, chunks, kv_heads, groups, head_dim, chunk_size)
        ctx.sm_scale = sm_scale
        return (
            lmk_k.reshape(batch, chunks, kv_heads, groups, head_dim),
            entropy.reshape(batch, chunks, kv_heads, groups),
        )

    @staticmethod
    def backward(ctx, grad_lmk_k, grad_entropy):
        mu_flat, k_flat, probs, entropy = ctx.saved_tensors
        batch, chunks, kv_heads, groups, head_dim, chunk_size = ctx.shape
        batch_chunks = batch * chunks
        if grad_lmk_k is None:
            grad_lmk_k = torch.zeros(
                batch, chunks, kv_heads, groups, head_dim,
                device=mu_flat.device, dtype=mu_flat.dtype,
            )
        if grad_entropy is None:
            grad_entropy = torch.zeros(
                batch, chunks, kv_heads, groups,
                device=mu_flat.device, dtype=torch.float32,
            )
        grad_k_flat = grad_lmk_k.reshape(
            batch_chunks, kv_heads, groups, head_dim
        ).contiguous()
        grad_b_flat = grad_entropy.reshape(batch_chunks, kv_heads, groups).float().contiguous()
        kernel = _pool_gqa_bwd(
            batch_chunks,
            chunk_size,
            kv_heads,
            groups,
            head_dim,
            ctx.sm_scale,
        )
        with torch.cuda.device(mu_flat.device):
            d_mu, d_k = kernel(
                mu_flat,
                k_flat,
                probs,
                entropy,
                grad_k_flat,
                grad_b_flat,
            )
        return (
            d_mu.to(mu_flat.dtype).reshape(
                batch, chunks, kv_heads, groups, head_dim
            ),
            d_k.to(k_flat.dtype).reshape(
                batch, chunks, chunk_size, kv_heads, head_dim
            ),
            None,
            None,
        )


def _validate(mu_q, k_chunked, key_valid):
    if mu_q.device.type != "cuda":
        raise ValueError("chunk_attn_pool_gqa requires CUDA tensors")
    if mu_q.dtype != torch.bfloat16 or k_chunked.dtype != torch.bfloat16:
        raise ValueError("chunk_attn_pool_gqa requires BF16 mu_q and K")
    if mu_q.ndim != 5 or k_chunked.ndim != 5:
        raise ValueError("mu_q and k_chunked must be 5D")
    batch, chunks, kv_heads, groups, head_dim = mu_q.shape
    expected_k = (batch, chunks, k_chunked.shape[2], kv_heads, head_dim)
    if k_chunked.shape != expected_k:
        raise ValueError(
            f"expected compact K shape {expected_k}, got {tuple(k_chunked.shape)}"
        )
    if key_valid.shape != (batch, chunks, k_chunked.shape[2]):
        raise ValueError(
            f"invalid key_valid shape {tuple(key_valid.shape)}"
        )
    if groups > 16:
        raise ValueError(f"groups must be <= 16, got {groups}")


def chunk_attn_pool_gqa(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    key_valid: torch.Tensor,
    sm_scale: float | None = None,
):
    """Pool compact per-KV-head chunks into per-query-head summaries."""

    _validate(mu_q, k_chunked, key_valid)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(mu_q.shape[-1])
    if mu_q.shape[3] == 1:
        return _COMPILED_COMPACT_POOL(
            mu_q.contiguous(),
            k_chunked.contiguous(),
            key_valid.to(device=mu_q.device, dtype=torch.bool).contiguous(),
            float(sm_scale),
        )
    return _ChunkAttnPoolGQA.apply(
        mu_q.contiguous(),
        k_chunked.contiguous(),
        key_valid.to(device=mu_q.device, dtype=torch.bool).contiguous(),
        float(sm_scale),
    )
