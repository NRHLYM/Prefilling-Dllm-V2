"""Fused K-remote plus local branch normalization for bidirectional route."""

import tilelang
import tilelang.language as T
import torch


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[4, 5, 6], pass_configs=_PASS_CONFIGS)
def _route_weights_fwd(
    batch,
    seq_len,
    chunks,
    kv_heads,
    groups,
    topk,
    threads=64,
):
    h_q = kv_heads * groups
    padded_k = 8 if topk + 1 <= 8 else 16 if topk + 1 <= 16 else 32
    score_shape = [batch, seq_len, h_q, topk]
    index_shape = [batch, seq_len, kv_heads, topk]
    bias_shape = [batch, chunks, kv_heads, groups]
    local_shape = [batch, seq_len, h_q]
    probs_shape = [batch, seq_len, h_q, padded_k]

    @T.prim_func
    def main(
        Scores: T.Tensor(score_shape, "bfloat16"),
        Indices: T.Tensor(index_shape, "int32"),
        PriorBias: T.Tensor(bias_shape, "float"),
        LocalLSE: T.Tensor(local_shape, "float"),
        RemoteWeights: T.Tensor(score_shape, "bfloat16"),
        LocalWeight: T.Tensor(local_shape, "bfloat16"),
        Probs: T.Tensor(probs_shape, "float"),
    ):
        # seq_len on grid.x: CUDA grid.y/z max out at 65535.
        with T.Kernel(seq_len, h_q, batch, threads=threads) as (i_l, i_h, i_b):
            logits = T.alloc_fragment([1, padded_k], "float")
            row_max = T.alloc_fragment([1], "float")
            row_sum = T.alloc_fragment([1], "float")
            i_kv = i_h // groups
            i_g = i_h % groups

            for j in T.Parallel(padded_k):
                if j < topk:
                    chunk_idx = Indices[i_b, i_l, i_kv, j]
                    safe_idx = T.if_then_else(
                        (chunk_idx >= 0) and (chunk_idx < chunks),
                        chunk_idx,
                        0,
                    )
                    logits[0, j] = T.if_then_else(
                        (chunk_idx >= 0) and (chunk_idx < chunks),
                        T.cast(Scores[i_b, i_l, i_h, j], "float")
                        + PriorBias[i_b, safe_idx, i_kv, i_g],
                        -T.infinity("float"),
                    )
                elif j == topk:
                    logits[0, j] = LocalLSE[i_b, i_l, i_h]
                else:
                    logits[0, j] = -T.infinity("float")

            T.fill(row_max, -T.infinity("float"))
            T.reduce_max(logits, row_max, dim=1, clear=True)
            for j in T.Parallel(padded_k):
                logits[0, j] = T.exp2(
                    (logits[0, j] - row_max[0]) * 1.4426950408889634
                )
            T.fill(row_sum, 0.0)
            T.reduce_sum(logits, row_sum, dim=1, clear=True)
            for j in T.Parallel(padded_k):
                prob = T.if_then_else(
                    row_sum[0] > 0,
                    logits[0, j] / row_sum[0],
                    T.cast(0, "float"),
                )
                Probs[i_b, i_l, i_h, j] = prob
                if j < topk:
                    RemoteWeights[i_b, i_l, i_h, j] = T.cast(
                        prob, "bfloat16"
                    )
                elif j == topk:
                    LocalWeight[i_b, i_l, i_h] = T.cast(
                        prob, "bfloat16"
                    )

    return main


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _route_weights_bwd(
    batch,
    seq_len,
    chunks,
    kv_heads,
    groups,
    topk,
    threads=64,
):
    h_q = kv_heads * groups
    padded_k = 8 if topk + 1 <= 8 else 16 if topk + 1 <= 16 else 32
    score_shape = [batch, seq_len, h_q, topk]
    index_shape = [batch, seq_len, kv_heads, topk]
    bias_shape = [batch, chunks, kv_heads, groups]
    local_shape = [batch, seq_len, h_q]
    probs_shape = [batch, seq_len, h_q, padded_k]

    @T.prim_func
    def main(
        Probs: T.Tensor(probs_shape, "float"),
        Indices: T.Tensor(index_shape, "int32"),
        GradRemote: T.Tensor(score_shape, "bfloat16"),
        GradLocal: T.Tensor(local_shape, "bfloat16"),
        DScores: T.Tensor(score_shape, "float"),
        DBias: T.Tensor(bias_shape, "float"),
        DLocal: T.Tensor(local_shape, "float"),
    ):
        with T.Kernel(seq_len, h_q, batch, threads=threads) as (i_l, i_h, i_b):
            delta_terms = T.alloc_fragment([1, padded_k], "float")
            delta = T.alloc_fragment([1], "float")
            grad_probs = T.alloc_fragment([1, padded_k], "float")
            i_kv = i_h // groups
            i_g = i_h % groups

            for j in T.Parallel(padded_k):
                if j < topk:
                    grad_probs[0, j] = T.cast(
                        GradRemote[i_b, i_l, i_h, j], "float"
                    )
                elif j == topk:
                    grad_probs[0, j] = T.cast(
                        GradLocal[i_b, i_l, i_h], "float"
                    )
                else:
                    grad_probs[0, j] = 0.0
                delta_terms[0, j] = (
                    Probs[i_b, i_l, i_h, j] * grad_probs[0, j]
                )

            T.fill(delta, 0.0)
            T.reduce_sum(delta_terms, delta, dim=1, clear=True)
            for j in T.Parallel(padded_k):
                d_logit = Probs[i_b, i_l, i_h, j] * (
                    grad_probs[0, j] - delta[0]
                )
                if j < topk:
                    chunk_idx = Indices[i_b, i_l, i_kv, j]
                    valid = (chunk_idx >= 0) and (chunk_idx < chunks)
                    DScores[i_b, i_l, i_h, j] = T.if_then_else(
                        valid, d_logit, T.cast(0, "float")
                    )
                    if valid:
                        T.atomic_add(
                            DBias[i_b, chunk_idx, i_kv, i_g], d_logit
                        )
                elif j == topk:
                    DLocal[i_b, i_l, i_h] = d_logit

    return main


class _RouteWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, indices, prior_bias, local_lse):
        batch, seq_len, h_q, topk = scores.shape
        chunks, kv_heads, groups = prior_bias.shape[1:]
        kernel = _route_weights_fwd(
            batch, seq_len, chunks, kv_heads, groups, topk
        )
        remote, local, probs = kernel(
            scores, indices, prior_bias, local_lse
        )
        ctx.save_for_backward(probs, indices)
        ctx.shape = (batch, seq_len, chunks, kv_heads, groups, topk)
        ctx.score_dtype = scores.dtype
        return remote, local

    @staticmethod
    def backward(ctx, grad_remote, grad_local):
        probs, indices = ctx.saved_tensors
        batch, seq_len, chunks, kv_heads, groups, topk = ctx.shape
        h_q = kv_heads * groups
        if grad_remote is None:
            grad_remote = torch.zeros(
                batch,
                seq_len,
                h_q,
                topk,
                device=probs.device,
                dtype=torch.bfloat16,
            )
        if grad_local is None:
            grad_local = torch.zeros(
                batch,
                seq_len,
                h_q,
                device=probs.device,
                dtype=torch.bfloat16,
            )
        d_scores = torch.empty(
            batch,
            seq_len,
            h_q,
            topk,
            device=probs.device,
            dtype=torch.float32,
        )
        d_bias = torch.zeros(
            batch,
            chunks,
            kv_heads,
            groups,
            device=probs.device,
            dtype=torch.float32,
        )
        d_local = torch.empty(
            batch,
            seq_len,
            h_q,
            device=probs.device,
            dtype=torch.float32,
        )
        kernel = _route_weights_bwd(
            batch, seq_len, chunks, kv_heads, groups, topk
        )
        kernel(
            probs,
            indices,
            grad_remote.to(torch.bfloat16).contiguous(),
            grad_local.to(torch.bfloat16).contiguous(),
            d_scores,
            d_bias,
            d_local,
        )
        return d_scores.to(ctx.score_dtype), None, d_bias, d_local


def _validate(scores, indices, prior_bias, local_lse):
    if scores.device.type != "cuda":
        raise ValueError("route_weights requires CUDA tensors")
    if not (
        scores.device == indices.device == prior_bias.device == local_lse.device
    ):
        raise ValueError("all route_weights tensors must share one CUDA device")
    if scores.ndim != 4 or indices.ndim != 4 or prior_bias.ndim != 4:
        raise ValueError("scores, indices, and prior_bias must be 4D")
    if local_lse.ndim != 3:
        raise ValueError("local_lse must be [B,L,Hq]")
    batch, seq_len, h_q, topk = scores.shape
    bias_batch, chunks, kv_heads, groups = prior_bias.shape
    if topk <= 0 or topk + 1 > 32:
        raise ValueError(f"route_weights requires 1 <= topk <= 31, got {topk}")
    if groups <= 0 or groups > 16 or h_q != kv_heads * groups:
        raise ValueError(
            f"invalid grouped-head layout Hq={h_q}, Hkv={kv_heads}, G={groups}"
        )
    if bias_batch != batch or chunks <= 0:
        raise ValueError(f"invalid prior_bias shape {tuple(prior_bias.shape)}")
    if indices.shape != (batch, seq_len, kv_heads, topk):
        raise ValueError(f"invalid indices shape {tuple(indices.shape)}")
    if local_lse.shape != (batch, seq_len, h_q):
        raise ValueError(f"invalid local_lse shape {tuple(local_lse.shape)}")
    if scores.dtype != torch.bfloat16:
        raise ValueError("route_weights requires BF16 scores")
    if prior_bias.dtype != torch.float32 or local_lse.dtype != torch.float32:
        raise ValueError("prior_bias and local_lse must be FP32")


def route_weights(
    scores: torch.Tensor,
    indices: torch.Tensor,
    prior_bias: torch.Tensor,
    local_lse: torch.Tensor,
):
    """Normalize selected remote branches together with the local branch."""

    _validate(scores, indices, prior_bias, local_lse)
    return _RouteWeights.apply(
        scores.contiguous(),
        indices.to(torch.int32).contiguous(),
        prior_bias.contiguous(),
        local_lse.contiguous(),
    )
