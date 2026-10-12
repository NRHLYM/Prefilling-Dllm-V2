"""Force the route fusion gate onto the remote branch for a 1-token unit test."""

from __future__ import annotations

from typing import Iterable

import torch


def apply_forced_remote_gate(
    remote_weights: torch.Tensor,
    local_weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    query_mask: torch.Tensor | None,
    ablation: str = "none",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Override fusion on supervised queries.

    ``none``: set local mass to 0 and renormalize selected remote weights to 1.
    ``off``: local mass 1, remote mass 0.
    ``shuffle``: same forced-remote mix, but permute selected chunk slots.
    """

    mode = str(ablation or "none")
    if mode not in {"none", "off", "shuffle", "learned"}:
        raise ValueError(f"unsupported remote ablation={mode}")
    if query_mask is None or mode == "learned":
        return remote_weights, local_weight, indices
    if query_mask.shape[:2] != local_weight.shape[:2]:
        raise ValueError(
            f"force-remote mask {tuple(query_mask.shape)} does not match "
            f"local_weight {tuple(local_weight.shape)}"
        )
    forced = query_mask.bool().unsqueeze(-1)
    if mode == "off":
        zeros = remote_weights.new_zeros(remote_weights.shape)
        ones = local_weight.new_ones(local_weight.shape)
        remote_weights = torch.where(forced.unsqueeze(-1), zeros, remote_weights)
        local_weight = torch.where(forced, ones, local_weight)
        return remote_weights, local_weight, indices

    mass = remote_weights.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    renormalized = (remote_weights.float() / mass).to(dtype=remote_weights.dtype)
    remote_weights = torch.where(forced.unsqueeze(-1), renormalized, remote_weights)
    local_weight = torch.where(
        forced, torch.zeros_like(local_weight), local_weight
    )
    if mode == "shuffle":
        topk = int(indices.shape[-1])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(topk, generator=generator).to(device=indices.device)
        indices = indices.index_select(-1, perm)
    return remote_weights, local_weight, indices


def splice_oracle_indices(
    indices: torch.Tensor,
    selected_scores: torch.Tensor,
    evidence_chunks: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put the first evidence chunk into top-k slot 0 on supervised queries."""

    if query_mask is None or evidence_chunks is None:
        return indices, selected_scores
    if evidence_chunks.ndim != 2:
        raise ValueError(
            f"evidence_chunks must be [B, C], got {tuple(evidence_chunks.shape)}"
        )
    if indices.shape[0] != evidence_chunks.shape[0]:
        raise ValueError(
            f"indices batch {indices.shape[0]} != evidence {evidence_chunks.shape[0]}"
        )
    if query_mask.shape[:2] != indices.shape[:2]:
        raise ValueError(
            f"query mask {tuple(query_mask.shape)} does not match indices "
            f"{tuple(indices.shape[:2])}"
        )
    present = evidence_chunks.bool().any(dim=-1)
    if not bool(present.all()):
        raise ValueError("oracle route needs an evidence chunk in every batch row")
    evidence_id = evidence_chunks.to(dtype=torch.long).argmax(dim=-1)
    indices = indices.clone()
    slot = indices[..., 0]
    fill = evidence_id.to(device=indices.device, dtype=indices.dtype)
    while fill.ndim < slot.ndim:
        fill = fill.unsqueeze(-1)
    qmask = query_mask.bool()
    while qmask.ndim < slot.ndim:
        qmask = qmask.unsqueeze(-1)
    indices[..., 0] = torch.where(qmask.expand_as(slot), fill.expand_as(slot), slot)
    # Keep the original slot-0 mix weight. Rewriting scores with amax()+1
    # puts every routed head on the backward of a max, which overflowed the
    # shared landmark type offset on the first train step.
    return indices, selected_scores


def fusion_gate_bce(
    gate_logit: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    target_remote: bool = True,
) -> torch.Tensor:
    """BCEWithLogits on native fusion gate_logit = log(w_remote) - log(w_local).

    ``target_remote=True`` (y=1) is the current one-token remote probe.
    A paired local control would pass ``target_remote=False`` (y=0).
    """

    if query_mask.shape[:2] != gate_logit.shape[:2]:
        raise ValueError(
            f"gate mask {tuple(query_mask.shape)} does not match "
            f"gate_logit {tuple(gate_logit.shape[:2])}"
        )
    mask = query_mask.bool()
    while mask.ndim < gate_logit.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(gate_logit)
    if not bool(mask.any()):
        return gate_logit.float().sum() * 0.0
    target = torch.ones_like(gate_logit, dtype=torch.float32)
    if not bool(target_remote):
        target = torch.zeros_like(gate_logit, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        gate_logit.float(),
        target,
        reduction="none",
    )
    return loss.masked_select(mask).mean()


def labeled_remote_mass(
    remote_weights: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    """Mean w_remote = remote_weights.sum(-1) on labeled queries."""

    w_remote = remote_weights.float().sum(dim=-1)
    mask = query_mask.bool()
    while mask.ndim < w_remote.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(w_remote)
    if not bool(mask.any()):
        return w_remote.new_zeros(())
    return w_remote.masked_select(mask).mean()


def collect_fusion_gate_bce(model: torch.nn.Module) -> torch.Tensor:
    losses = [
        module.fusion_gate_bce
        for module in iter_force_remote_layers(model)
        if getattr(module, "fusion_gate_bce", None) is not None
    ]
    if not losses:
        raise RuntimeError("missing fusion gate BCE from route layers")
    return torch.stack(losses).mean()


def labeled_gate_vectors(
    gate_logit: torch.Tensor,
    remote_weights: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten native gate_logit and w_remote on labeled queries."""

    logit = gate_logit.float()
    mass = remote_weights.float().sum(dim=-1)
    while mass.ndim < logit.ndim:
        mass = mass.unsqueeze(-1)
    mass = mass.expand_as(logit)
    mask = query_mask.bool()
    while mask.ndim < logit.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(logit)
    if not bool(mask.any()):
        empty = logit.new_empty((0,))
        return empty, empty
    return logit.masked_select(mask), mass.masked_select(mask)


def local_control_query_mask(
    query_mask: torch.Tensor, *, shift: int = 64
) -> torch.Tensor:
    """One in-window control position before each labeled needle query."""

    if query_mask.ndim < 2:
        raise ValueError("query_mask must be at least [B, S]")
    if shift <= 0:
        raise ValueError(f"control shift must be positive, got {shift}")
    needle = query_mask.bool()
    control = torch.zeros_like(needle)
    batch, seq_len = needle.shape[:2]
    for row in range(batch):
        positions = needle[row].nonzero(as_tuple=False).flatten()
        for position in positions.tolist():
            control_pos = max(0, int(position) - int(shift))
            if control_pos != int(position) and control_pos < seq_len:
                control[row, control_pos] = True
    return control & ~needle


def pairwise_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """P(pos > neg) + 0.5 P(pos == neg)."""

    if int(positive.numel()) == 0 or int(negative.numel()) == 0:
        return float("nan")
    pos = positive.detach().float().reshape(-1).cpu()
    neg = negative.detach().float().reshape(-1).cpu()
    greater = pos[:, None] > neg[None, :]
    equal = pos[:, None] == neg[None, :]
    return float(greater.float().mean().item() + 0.5 * equal.float().mean().item())


def two_class_logit_metrics(
    positive: torch.Tensor, negative: torch.Tensor
) -> dict[str, float]:
    """Needle (y=1) vs local-control (y=0) gate-logit diagnostics."""

    nan = float("nan")
    pos = (
        positive.detach().float().reshape(-1).cpu()
        if positive is not None
        else torch.zeros((0,))
    )
    neg = (
        negative.detach().float().reshape(-1).cpu()
        if negative is not None
        else torch.zeros((0,))
    )
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    pos_mean = float(pos.mean().item()) if n_pos else nan
    neg_mean = float(neg.mean().item()) if n_neg else nan
    pos_prob = torch.sigmoid(pos) if n_pos else pos
    neg_prob = torch.sigmoid(neg) if n_neg else neg
    return {
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "pos_frac": float(n_pos / max(n_pos + n_neg, 1)),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "delta_pos_minus_neg": (
            pos_mean - neg_mean if n_pos and n_neg else nan
        ),
        "pos_std": float(pos.std(unbiased=False).item()) if n_pos > 1 else 0.0,
        "neg_std": float(neg.std(unbiased=False).item()) if n_neg > 1 else 0.0,
        "auc": pairwise_auc(pos, neg),
        "bce_grad_y1": float((pos_prob - 1.0).mean().item()) if n_pos else nan,
        "bce_grad_y0": float(neg_prob.mean().item()) if n_neg else nan,
        "sign_ok": (
            1.0 if n_pos and n_neg and pos_mean > neg_mean else 0.0
        ),
    }


def summarize_gate_vectors(logit: torch.Tensor, mass: torch.Tensor) -> dict[str, float]:
    """Distribution of remote_lse - local_lse and remote mass on labeled queries."""

    nan = float("nan")
    if logit is None or mass is None or int(logit.numel()) == 0:
        return {
            "gate_logit_mean": nan,
            "gate_logit_std": nan,
            "gate_logit_p50": nan,
            "gate_logit_p90": nan,
            "live_w_remote": nan,
            "live_w_remote_p50": nan,
            "live_frac_w_gt_0.8": nan,
            "live_gate_bce": nan,
        }
    logit = logit.detach().float().reshape(-1).cpu()
    mass = mass.detach().float().reshape(-1).cpu()
    ones = torch.ones_like(logit)
    std = float(logit.std(unbiased=False).item()) if logit.numel() > 1 else 0.0
    return {
        "gate_logit_mean": float(logit.mean().item()),
        "gate_logit_std": std,
        "gate_logit_p50": float(logit.quantile(0.5).item()),
        "gate_logit_p90": float(logit.quantile(0.9).item()),
        "live_w_remote": float(mass.mean().item()),
        "live_w_remote_p50": float(mass.quantile(0.5).item()),
        "live_frac_w_gt_0.8": float((mass > 0.8).float().mean().item()),
        "live_gate_bce": float(
            torch.nn.functional.binary_cross_entropy_with_logits(logit, ones).item()
        ),
    }


def collect_labeled_gate_vectors(model: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    packed = collect_labeled_gate_layer_vectors(model)
    if not packed:
        return None
    return {
        "logit": torch.cat([item["logit"] for item in packed]),
        "mass": torch.cat([item["mass"] for item in packed]),
        "layers": packed,
    }


def collect_labeled_gate_layer_vectors(
    model: torch.nn.Module,
) -> list[dict[str, torch.Tensor]]:
    layers = []
    for module in iter_force_remote_layers(model):
        logit = getattr(module, "_last_labeled_gate_logit", None)
        mass = getattr(module, "_last_labeled_gate_mass", None)
        if logit is None or mass is None or not torch.is_tensor(logit):
            continue
        if int(logit.numel()) == 0:
            continue
        control = getattr(module, "_last_control_gate_logit", None)
        layers.append(
            {
                "logit": logit.detach().float().reshape(-1).cpu(),
                "mass": mass.detach().float().reshape(-1).cpu(),
                "control_logit": (
                    control.detach().float().reshape(-1).cpu()
                    if torch.is_tensor(control) and int(control.numel()) > 0
                    else logit.new_empty((0,)).cpu()
                ),
            }
        )
    return layers


def layer_gate_metrics(layers: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, packed in enumerate(layers):
        stats = summarize_gate_vectors(packed["logit"], packed["mass"])
        for key, value in stats.items():
            metrics[f"l{index}_{key}"] = value
        two_class = two_class_logit_metrics(
            packed["logit"], packed.get("control_logit", packed["logit"].new_empty((0,)))
        )
        for key, value in two_class.items():
            metrics[f"l{index}_{key}"] = value
    return metrics


def collect_labeled_w_remote(model: torch.nn.Module) -> float:
    values = []
    for module in iter_force_remote_layers(model):
        value = getattr(module, "_last_labeled_w_remote", None)
        if value is None:
            continue
        if torch.is_tensor(value):
            values.append(float(value.detach().float().mean().cpu().item()))
        else:
            values.append(float(value))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def collect_fusion_gate_offsets(model: torch.nn.Module) -> list[float]:
    return collect_named_scalar_params(model, "fusion_gate_offset")


def collect_fusion_gate_scales(model: torch.nn.Module) -> list[float]:
    return collect_named_scalar_params(model, "fusion_gate_scale")


def collect_named_scalar_params(model: torch.nn.Module, suffix: str) -> list[float]:
    return [
        float(parameter.detach().float().cpu().item())
        for name, parameter in model.named_parameters()
        if name.endswith(suffix)
    ]


def iter_force_remote_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    for module in model.modules():
        if hasattr(module, "force_remote_query_mask"):
            yield module


def prepare_force_remote(
    model: torch.nn.Module,
    labels: torch.Tensor,
    *,
    ablation: str = "none",
    seed: int = 0,
    evidence_chunks: torch.Tensor | None = None,
    oracle_route: bool = False,
    supervise_gate: bool = False,
    gate_ce_force: bool = False,
) -> int:
    mask = labels.ne(-100)
    if bool(oracle_route) and evidence_chunks is None:
        raise ValueError("oracle route requires route_evidence_chunks")
    updated = 0
    for module in iter_force_remote_layers(model):
        module.force_remote_query_mask = mask
        module.remote_ablation = str(ablation)
        module.remote_ablation_seed = int(seed)
        module.force_remote_oracle_route = bool(oracle_route)
        module.force_remote_evidence_chunks = evidence_chunks
        module.supervise_fusion_gate = bool(supervise_gate)
        module.gate_ce_force = bool(gate_ce_force)
        module.fusion_gate_bce = None
        updated += 1
    if updated <= 0:
        raise RuntimeError("force-remote unit found no route layers")
    return updated


def clear_force_remote(model: torch.nn.Module) -> None:
    for module in iter_force_remote_layers(model):
        module.force_remote_query_mask = None
        module.remote_ablation = "none"
        module.force_remote_oracle_route = False
        module.force_remote_evidence_chunks = None
        module.supervise_fusion_gate = False
        module.gate_ce_force = False
        module.fusion_gate_bce = None
