"""Residual routing-query calibration: random low-rank residual plus score RMSNorm."""
import math

import torch
from torch import nn


class QCalRMSNorm(nn.Module):
    """Per-head RMSNorm on the last dim, as in the query normalization of the upstream kernels."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError("qcal RMSNorm dim must be positive")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim), dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden, self.weight, self.eps)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = hidden.dtype
    x = hidden.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + float(eps))
    return (weight.float() * x).to(orig)


def _qcal_init_std(model) -> float:
    config = getattr(model, "config", None)
    std = getattr(config, "initializer_range", 0.02) if config is not None else 0.02
    std = 0.02 if std is None else float(std)
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"qcal initializer_range must be positive and finite, got {std}")
    return std


def install_qcal(model, rank):
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    if rank == 0:
        return
    std = _qcal_init_std(model)
    for module in model.modules():
        if not isinstance(module, KernelDreamFullRouteAttention):
            continue
        # Do not perturb initialization of the shared LoRA/LMK or data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() + module.layer_idx)
            adapter = nn.Sequential(
                nn.Linear(module.hidden_size, rank, bias=False),
                nn.Linear(rank, module.num_heads * module.head_dim, bias=False),
            )
            nn.init.normal_(adapter[0].weight, mean=0.0, std=std)
            nn.init.normal_(adapter[1].weight, mean=0.0, std=std)
        device = module.q_proj.weight.device
        module.qcal = adapter.to(device=device, dtype=module.q_proj.weight.dtype)
        module.qcal_norm = QCalRMSNorm(module.head_dim).to(device=device)
        module.qcal_scale = 1.0
    model.config.route_qcal_rank = rank
    model.config.route_qcal_init_std = std
    model.config.route_qcal_version = "residual-random-lowrank-rmsnorm-v1"


def set_qcal_scale(model, scale):
    """Scale an installed Q-Cal residual without changing its checkpoint."""
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("qcal scale must be finite")
    count = 0
    for module in model.modules():
        if not hasattr(module, "qcal"):
            continue
        module.qcal_scale = scale
        count += 1
    if scale != 1.0 and count == 0:
        raise ValueError("non-unit qcal scale requires an installed Q-Cal adapter")
    return count
