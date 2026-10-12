"""Fused local/remote output combine for bidirectional HiLS attention."""

import tilelang
import tilelang.language as T
import torch


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[3], pass_configs=_PASS_CONFIGS)
def _fusion_fwd(batch, seq_len, heads, head_dim, threads=128):
    output_shape = [batch, seq_len, heads, head_dim]
    weight_shape = [batch, seq_len, heads]

    @T.prim_func
    def main(
        Remote: T.Tensor(output_shape, "bfloat16"),
        Local: T.Tensor(output_shape, "bfloat16"),
        LocalWeight: T.Tensor(weight_shape, "bfloat16"),
        Output: T.Tensor(output_shape, "bfloat16"),
    ):
        # seq_len on grid.x: CUDA grid.y/z max out at 65535.
        with T.Kernel(seq_len, heads, batch, threads=threads) as (i_l, i_h, i_b):
            for d in T.Parallel(head_dim):
                Output[i_b, i_l, i_h, d] = T.cast(
                    T.cast(Remote[i_b, i_l, i_h, d], "float")
                    + T.cast(LocalWeight[i_b, i_l, i_h], "float")
                    * T.cast(Local[i_b, i_l, i_h, d], "float"),
                    "bfloat16",
                )

    return main


@tilelang.jit(out_idx=[3, 4, 5], pass_configs=_PASS_CONFIGS)
def _fusion_bwd(batch, seq_len, heads, head_dim, threads=128):
    output_shape = [batch, seq_len, heads, head_dim]
    weight_shape = [batch, seq_len, heads]

    @T.prim_func
    def main(
        Local: T.Tensor(output_shape, "bfloat16"),
        LocalWeight: T.Tensor(weight_shape, "bfloat16"),
        GradOutput: T.Tensor(output_shape, "bfloat16"),
        DRemote: T.Tensor(output_shape, "bfloat16"),
        DLocal: T.Tensor(output_shape, "bfloat16"),
        DLocalWeight: T.Tensor(weight_shape, "bfloat16"),
    ):
        with T.Kernel(seq_len, heads, batch, threads=threads) as (i_l, i_h, i_b):
            weight_terms = T.alloc_fragment([1, head_dim], "float")
            weight_sum = T.alloc_fragment([1], "float")
            local_weight = T.cast(
                LocalWeight[i_b, i_l, i_h], "float"
            )

            for d in T.Parallel(head_dim):
                grad = T.cast(GradOutput[i_b, i_l, i_h, d], "float")
                DRemote[i_b, i_l, i_h, d] = T.cast(grad, "bfloat16")
                DLocal[i_b, i_l, i_h, d] = T.cast(
                    grad * local_weight, "bfloat16"
                )
                weight_terms[0, d] = (
                    grad * T.cast(Local[i_b, i_l, i_h, d], "float")
                )

            T.fill(weight_sum, 0.0)
            T.reduce_sum(weight_terms, weight_sum, dim=1, clear=True)
            DLocalWeight[i_b, i_l, i_h] = T.cast(
                weight_sum[0], "bfloat16"
            )

    return main


class _FuseHiLSOutputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, remote_output, local_output, local_weight):
        batch, seq_len, heads, head_dim = remote_output.shape
        kernel = _fusion_fwd(batch, seq_len, heads, head_dim)
        output = kernel(remote_output, local_output, local_weight)
        ctx.save_for_backward(local_output, local_weight)
        ctx.shape = (batch, seq_len, heads, head_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        local_output, local_weight = ctx.saved_tensors
        batch, seq_len, heads, head_dim = ctx.shape
        kernel = _fusion_bwd(batch, seq_len, heads, head_dim)
        d_remote, d_local, d_weight = kernel(
            local_output,
            local_weight,
            grad_output.contiguous(),
        )
        return d_remote, d_local, d_weight


def _validate(remote_output, local_output, local_weight):
    if remote_output.device.type != "cuda":
        raise ValueError("fuse_hils_outputs requires CUDA tensors")
    if not (
        remote_output.device == local_output.device == local_weight.device
    ):
        raise ValueError("all fusion tensors must share one CUDA device")
    if remote_output.ndim != 4 or local_output.shape != remote_output.shape:
        raise ValueError(
            "remote_output and local_output must have the same [B,L,H,D] shape"
        )
    if local_weight.shape != remote_output.shape[:-1]:
        raise ValueError(
            f"local_weight shape {tuple(local_weight.shape)} must equal "
            f"{tuple(remote_output.shape[:-1])}"
        )
    if remote_output.shape[-1] > 256:
        raise ValueError(
            f"head_dim must be <= 256, got {remote_output.shape[-1]}"
        )
    if not (
        remote_output.dtype
        == local_output.dtype
        == local_weight.dtype
        == torch.bfloat16
    ):
        raise ValueError("fuse_hils_outputs requires BF16 tensors")
    if not (
        remote_output.is_contiguous()
        and local_output.is_contiguous()
        and local_weight.is_contiguous()
    ):
        raise ValueError("fuse_hils_outputs requires contiguous tensors")


def fuse_hils_outputs(
    remote_output: torch.Tensor,
    local_output: torch.Tensor,
    local_weight: torch.Tensor,
) -> torch.Tensor:
    """Add the weighted local branch to the already weighted remote branch."""

    _validate(remote_output, local_output, local_weight)
    return _FuseHiLSOutputs.apply(
        remote_output, local_output, local_weight
    )
