"""Rank-safe, exact-resume checkpoints for Dream route LoRA training."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from prefilling_v2.distributed import all_gather_object, barrier


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
        state["cuda_device"] = torch.cuda.current_device()
    else:
        state["cuda"] = None
        state["cuda_device"] = None
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def _current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    unwrapped = _unwrap_model(model)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }


def _validate_resume_contract(
    checkpoint_dir: Path,
    expected_resume_contract: dict[str, object] | None,
) -> None:
    """Reject exact resumes whose execution/numerical contract changed."""

    if not expected_resume_contract:
        return
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_resume_contract = manifest.get("resume_contract")
    if saved_resume_contract is None:
        raise ValueError(
            "checkpoint has no resume_contract and predates numerical-semantics "
            "tracking; use initialize_from for an intentional weights-only "
            "migration instead of exact resume"
        )
    if saved_resume_contract != expected_resume_contract:
        raise ValueError(
            "checkpoint resume_contract mismatch: "
            f"saved={saved_resume_contract}, expected={expected_resume_contract}; "
            "use initialize_from for an intentional weights-only migration"
        )


def _alias_dense_source_attn_keys(
    saved_trainables: dict[str, torch.Tensor],
    current_names: set[str],
) -> tuple[dict[str, torch.Tensor], int]:
    """Map dense wrapper LoRA onto native/sparse `self_attn.{q,k,v,o}_proj` names.

    DenseDreamAttentionAdapter stores trainables under `self_attn.source_attn.*`.
    DSA/NSA/SWA/route replace that wrapper, so the same LoRA lives at `self_attn.*`.
    """

    aliased = dict(saved_trainables)
    n_aliased = 0
    marker = ".self_attn.source_attn."
    replacement = ".self_attn."
    for name, tensor in saved_trainables.items():
        if marker not in name:
            continue
        alias = name.replace(marker, replacement, 1)
        if alias not in current_names or alias in aliased:
            continue
        aliased[alias] = tensor
        n_aliased += 1
    return aliased, n_aliased


def load_trainable_checkpoint(
    *,
    checkpoint_dir: Path | str,
    model: torch.nn.Module,
    allow_partial: bool = False,
) -> list[str]:
    """Load trainable weights without restoring trainer or optimizer state."""

    checkpoint_dir = Path(checkpoint_dir)
    unwrapped = _unwrap_model(model)
    saved_trainables = torch.load(
        checkpoint_dir / "trainable_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    current_trainables = {
        name: parameter
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    saved_trainables, n_aliased = _alias_dense_source_attn_keys(
        saved_trainables,
        set(current_trainables),
    )
    if n_aliased:
        print(
            f"initialize_from aliased {n_aliased} dense source_attn keys onto self_attn",
            flush=True,
        )
    missing = sorted(set(current_trainables) - set(saved_trainables))
    unexpected = sorted(set(saved_trainables) - set(current_trainables))
    overlap = sorted(set(current_trainables) & set(saved_trainables))
    if missing or unexpected:
        if not allow_partial or not overlap:
            raise ValueError(
                f"trainable checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        print(
            "partial initialize_from: "
            f"copied={len(overlap)} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    with torch.no_grad():
        for name in overlap:
            parameter = current_trainables[name]
            source = saved_trainables[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"trainable checkpoint shape mismatch for {name}: "
                    f"saved={tuple(source.shape)} current={tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(parameter.device, parameter.dtype))
    return overlap


def overlay_checkpoint_parameters(
    *,
    checkpoint_dir: Path | str,
    model: torch.nn.Module,
    name_predicate,
) -> list[str]:
    """Copy matching tensors from a checkpoint regardless of requires_grad."""

    checkpoint_dir = Path(checkpoint_dir)
    saved = torch.load(
        checkpoint_dir / "trainable_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(saved, dict):
        raise ValueError(f"unexpected trainable state type: {type(saved)}")
    unwrapped = _unwrap_model(model)
    copied: list[str] = []
    with torch.no_grad():
        for name, parameter in unwrapped.named_parameters():
            if not name_predicate(name) or name not in saved:
                continue
            source = saved[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"overlay shape mismatch for {name}: "
                    f"saved={tuple(source.shape)} current={tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(parameter.device, parameter.dtype))
            copied.append(name)
    if not copied:
        raise ValueError(f"overlay from {checkpoint_dir} copied no parameters")
    return copied


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path | str,
    step: int,
    sampler_epoch: int,
    batches_in_epoch: int,
    config: dict[str, object],
    dataset_manifest_hash: str,
    rank: int,
    world_size: int,
    barrier_fn: Callable[[], None] = barrier,
    gather_object_fn: Callable[[object], list[object]] = all_gather_object,
    git_commit: str | None = None,
    resume_contract: dict[str, object] | None = None,
) -> Path | None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    local_rng = capture_rng_state()
    local_rng["rank"] = rank
    rng_states = gather_object_fn(local_rng)
    barrier_fn()

    checkpoint_path: Path | None = None
    if rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"step-{step}"
        temporary = output_dir / f"step-{step}.incomplete"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        unwrapped = _unwrap_model(model)
        trainables = _trainable_state(unwrapped)
        if not trainables:
            raise ValueError("refusing to checkpoint a model with no trainable parameters")
        torch.save(trainables, temporary / "trainable_state.pt")
        route_extra = {
            name: tensor
            for name, tensor in trainables.items()
            if "entropy_bias_scale" in name
        }
        torch.save(route_extra, temporary / "route_extra_trainable.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")

        serialized_resume_contract = dict(resume_contract or {})
        trainer_state = {
            "step": int(step),
            "sampler_epoch": int(sampler_epoch),
            "batches_in_epoch": int(batches_in_epoch),
            "world_size": int(world_size),
            "rng_states": rng_states,
            "config": dict(config),
            "dataset_manifest_hash": dataset_manifest_hash,
            "resume_contract": serialized_resume_contract,
        }
        torch.save(trainer_state, temporary / "trainer_state.pt")

        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(temporary / "adapter")
        if tokenizer is not None:
            tokenizer.save_pretrained(temporary / "tokenizer")

        manifest = {
            "format_version": 2,
            "step": int(step),
            "sampler_epoch": int(sampler_epoch),
            "batches_in_epoch": int(batches_in_epoch),
            "world_size": int(world_size),
            "git_commit": git_commit or _current_git_commit(),
            "dataset_manifest_hash": dataset_manifest_hash,
            "trainable_parameter_names": sorted(trainables),
            "route_extra_parameter_names": sorted(route_extra),
            "resume_contract": serialized_resume_contract,
        }
        (temporary / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, checkpoint_path)

    barrier_fn()
    return checkpoint_path


def load_training_checkpoint(
    *,
    checkpoint_dir: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    rank: int,
    restore_rng: bool = True,
    expected_resume_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint_dir = Path(checkpoint_dir)
    _validate_resume_contract(checkpoint_dir, expected_resume_contract)
    load_trainable_checkpoint(checkpoint_dir=checkpoint_dir, model=model)

    optimizer.load_state_dict(
        torch.load(
            checkpoint_dir / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    scheduler.load_state_dict(
        torch.load(
            checkpoint_dir / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    trainer_state = torch.load(
        checkpoint_dir / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved_world = int(trainer_state["world_size"])
    if restore_rng and rank < saved_world:
        restore_rng_state(trainer_state["rng_states"][rank])
    return trainer_state
