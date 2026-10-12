"""Small DDP helpers for globally normalized Dream denoising training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


_GLOO_GROUP = None


def init_distributed() -> DistributedContext:
    global _GLOO_GROUP
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        init_kwargs = {"backend": backend, "init_method": "env://"}
        if backend == "nccl":
            try:
                dist.init_process_group(device_id=device, **init_kwargs)
            except TypeError:
                dist.init_process_group(**init_kwargs)
        else:
            dist.init_process_group(**init_kwargs)
        initialized_here = True
        if backend == "nccl":
            _GLOO_GROUP = dist.new_group(backend="gloo")
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if (
            world_size > 1
            and dist.get_backend() == "nccl"
            and _GLOO_GROUP is None
        ):
            _GLOO_GROUP = dist.new_group(backend="gloo")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def destroy_distributed(context: DistributedContext) -> None:
    global _GLOO_GROUP
    _GLOO_GROUP = None
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _cpu_barrier_group():
    """Gloo group for rank rendezvous that must not sit in NCCL during PVC IO."""
    if not dist.is_initialized():
        return None
    if dist.get_backend() != "nccl":
        return None
    return _GLOO_GROUP


def barrier() -> None:
    if not dist.is_initialized():
        return
    group = _cpu_barrier_group()
    if group is not None:
        dist.barrier(group=group)
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()


def all_gather_object(value):
    if not dist.is_initialized():
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    group = _cpu_barrier_group()
    if group is not None:
        dist.all_gather_object(gathered, value, group=group)
    else:
        dist.all_gather_object(gathered, value)
    return gathered


def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def global_target_count(
    local_micro_counts: Iterable[torch.Tensor | int],
    *,
    device: torch.device,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    local_total = sum(
        float(count.item()) if isinstance(count, torch.Tensor) else float(count)
        for count in local_micro_counts
    )
    count = torch.tensor(local_total, dtype=torch.float64, device=device)
    reducer = all_reduce_sum or _distributed_sum
    reduced = reducer(count)
    if reduced is not count:
        count = reduced
    return count


def scale_local_loss_for_ddp(
    local_loss_sum: torch.Tensor,
    *,
    global_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if float(global_count.item()) <= 0:
        raise ValueError("global target count must be positive")
    return local_loss_sum * (float(world_size) / global_count.to(local_loss_sum.dtype))


def all_reduce_detached_sum(value: torch.Tensor) -> torch.Tensor:
    reduced = value.detach().clone()
    return _distributed_sum(reduced)
