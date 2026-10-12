"""route routing targets from detached Q/K on the sparse-residual hidden.

`same_forward` uses the live student Q/K. `base` uses frozen no-LoRA Q/K.
`dense` uses a snapshot of the initialize_from (dense-500) merged Q/K, frozen.
Scores are detached; Q-Cal and the landmark type embed see the KL.
LoRA / residual QK stay detached. No parallel dense rollout.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from prefilling_v2.dense_reference import dense_attention, project


RULES = ("lse", "max", "top4_lse", "half_entropy", "weighted", "attn_mass")
TOKEN_SCORE_QUERY_BATCH = 16


def chunk_scores(scores, rule):
    """Reduce [..., chunk, token] FP32 QK; masked tokens are negative infinity."""
    if rule not in RULES:
        raise ValueError(f"unknown teacher rule: {rule}")
    valid = torch.isfinite(scores)
    nonempty = valid.any(-1)
    safe = torch.where(nonempty[..., None], scores.float(), 0.0)
    lse = torch.logsumexp(safe, -1)
    if rule == "lse":
        out = lse
    elif rule == "max":
        out = safe.max(-1).values
    elif rule == "top4_lse":
        out = torch.logsumexp(safe.topk(min(4, safe.shape[-1]), dim=-1).values, -1)
    elif rule == "attn_mass":
        # Global softmax over legal keys, then sum mass per chunk (naive BSA Z_c).
        fill = torch.finfo(torch.float32).min
        padded = scores.float().masked_fill(~valid, fill)
        mass = torch.softmax(padded.reshape(*padded.shape[:-2], -1), dim=-1).reshape_as(padded).sum(-1)
        out = mass.clamp_min(1e-30).log()
    else:
        probs = safe.softmax(-1)
        weighted = (probs * torch.where(valid, safe, 0.0)).sum(-1)
        out = weighted if rule == "weighted" else 0.5 * (weighted + lse)
    return out.masked_fill(~nonempty, -torch.inf)


def candidate_kl(student, teacher, eligible, temperature=1.0):
    """Per-query-head KL on all legal remote chunks, including unselected ones."""
    mask = eligible[:, :, None, None, :]
    rows = eligible.any(-1)
    finite_min = torch.finfo(torch.float32).min
    log_student = (student.float() / temperature).masked_fill(~mask, finite_min).log_softmax(-1)
    target = (teacher.detach().float() / temperature).masked_fill(~mask, finite_min).softmax(-1)
    target = target * mask
    loss = F.kl_div(log_student, target, reduction="none").sum(-1)
    heads = student.shape[2] * student.shape[3]
    return (loss * rows[:, :, None, None]).sum() / (rows.sum() * heads).clamp_min(1)


def token_scores(q, k, positions, valid, segments, chunk_size):
    if positions.shape[1] > TOKEN_SCORE_QUERY_BATCH:
        return torch.cat(
            [
                token_scores(
                    q, k, positions[:, start:start + TOKEN_SCORE_QUERY_BATCH],
                    valid, segments, chunk_size,
                )
                for start in range(0, positions.shape[1], TOKEN_SCORE_QUERY_BATCH)
            ],
            dim=1,
        )
    b, n, h, d = k.shape
    bi = torch.arange(b, device=q.device)[:, None]
    qs = q[bi, positions].reshape(b, positions.shape[1], h, -1, d)
    with torch.autocast(q.device.type, enabled=False):
        scores = torch.einsum("brhgd,bnhd->brhgn", qs.float(), k.float()) / math.sqrt(d)
    allowed = valid[:, None, :] & (segments[bi, positions, None] == segments[:, None, :])
    return scores.masked_fill(~allowed[:, :, None, None], -torch.inf).reshape(
        b, positions.shape[1], h, -1, n // chunk_size, chunk_size)


def sampled_candidates(positions, valid, segments, chunk_size, window):
    b, n = valid.shape
    c = n // chunk_size
    chunk_ids = torch.arange(c, device=valid.device)
    left = (positions - window).clamp_min(0) // chunk_size
    right = (positions + window).clamp_max(n - 1) // chunk_size
    local = (chunk_ids >= left[..., None]) & (chunk_ids <= right[..., None])
    bi = torch.arange(b, device=valid.device)[:, None]
    chunk_segments = segments[:, chunk_size - 1::chunk_size]
    same_segment = segments[bi, positions, None] == chunk_segments[:, None, :]
    eligible = valid.reshape(b, c, chunk_size).any(-1)[:, None] & same_segment & ~local
    return eligible, local & same_segment


def segment_dense_attention(q, k, v, valid, segments):
    """Flash dense inside each packed document; never mix distinct documents."""
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        for segment in torch.unique(segments[b, valid[b]]).tolist():
            ids = torch.where(segments[b] == segment)[0]
            result = dense_attention(q[b:b+1, ids], k[b:b+1, ids], v[b:b+1, ids], valid[b:b+1, ids])
            out[b, ids] = result[0]
    return out


@contextlib.contextmanager
def deterministic_teacher(model):
    states = [(module, module.training) for module in model.modules()]
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    try:
        model.eval()
        device_type = next(model.parameters()).device.type
        with torch.random.fork_rng(devices=devices), torch.no_grad(), torch.autocast(
            device_type, enabled=torch.is_autocast_enabled(device_type),
            dtype=torch.get_autocast_dtype(device_type), cache_enabled=False,
        ):
            yield
    finally:
        for module, training in states:
            module.training = training


@contextlib.contextmanager
def base_dream_adapter(model):
    """Drop LoRA so dense Q/K/V are the pretrained Dream projections."""
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("base dense teacher requires PEFT disable_adapter")
    with model.disable_adapter():
        yield


def _teacher_core(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _merged_linear_weight_bias(module):
    """Clone W (+ bias) including the current LoRA delta, then detach."""
    if hasattr(module, "get_base_layer"):
        base = module.get_base_layer()
    else:
        base = getattr(module, "base_layer", module)
    weight = base.weight.detach().clone()
    bias = None if getattr(base, "bias", None) is None else base.bias.detach().clone()
    lora_a = getattr(module, "lora_A", None)
    if lora_a is None:
        return weight, bias
    adapters = getattr(module, "active_adapters", None)
    if not adapters:
        adapter = getattr(module, "active_adapter", None)
        adapters = [adapter] if adapter else list(lora_a.keys())
    delta = torch.zeros_like(weight)
    for name in adapters:
        scale = module.scaling[name] if not isinstance(module.scaling, (int, float)) else module.scaling
        delta = delta + (module.lora_B[name].weight @ lora_a[name].weight) * scale
    return weight + delta, bias


def snapshot_frozen_dense_qk(model):
    """Freeze merged dense-500 Q/K on route layers. Call after initialize_from."""
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    core = _teacher_core(model)
    count = 0
    for layer in core.model.layers:
        attn = layer.self_attn
        if not isinstance(attn, KernelDreamFullRouteAttention):
            continue
        q_weight, q_bias = _merged_linear_weight_bias(attn.q_proj)
        k_weight, k_bias = _merged_linear_weight_bias(attn.k_proj)
        attn.register_buffer("frozen_dense_q_weight", q_weight.contiguous(), persistent=False)
        attn.register_buffer("frozen_dense_k_weight", k_weight.contiguous(), persistent=False)
        if q_bias is None:
            if hasattr(attn, "frozen_dense_q_bias"):
                delattr(attn, "frozen_dense_q_bias")
        else:
            attn.register_buffer("frozen_dense_q_bias", q_bias.contiguous(), persistent=False)
        if k_bias is None:
            if hasattr(attn, "frozen_dense_k_bias"):
                delattr(attn, "frozen_dense_k_bias")
        else:
            attn.register_buffer("frozen_dense_k_bias", k_bias.contiguous(), persistent=False)
        count += 1
    if count == 0:
        raise ValueError("frozen dense teacher requires route layers")
    return count


def dense_pass(core, batch, observer, chunk_size=64):
    """One all-layer dense forward; excludes external LMK keys."""
    ids = batch["input_ids"].clone()
    ids[:, chunk_size - 1::chunk_size] = int(core.config.mask_token_id)
    valid = batch["attention_mask"].bool().clone()
    valid[:, chunk_size - 1::chunk_size] = False
    segments = batch.get("segment_ids", batch["attention_mask"].long())
    with deterministic_teacher(core):
        hidden = core.model.embed_tokens(ids)
        embeddings = core.model.rotary_emb(hidden, batch["position_ids"])
        for index, layer in enumerate(core.model.layers):
            q, k, v = project(layer.self_attn, layer.input_layernorm(hidden), embeddings)
            observer(index, q, k, valid, segments)
            attended = segment_dense_attention(q, k, v, valid, segments).reshape_as(hidden)
            hidden = hidden + layer.self_attn.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return core.model.norm(hidden)


def prepare_dense_teacher(model, batch, args):
    """Sample query rows and bind them onto route layers. Scores are filled later
    from that layer's Q/K (live or frozen base), not from a parallel dense pass."""
    from prefilling_v2.attention import KernelDreamFullRouteAttention
    source = str(getattr(args, "route_dense_teacher_source", "same_forward"))
    if source == "online":
        source = "same_forward"
    if source not in {"same_forward", "base", "dense"}:
        raise ValueError(f"unsupported route_dense_teacher_source: {source}")
    if source == "dense":
        missing = [
            index for index, layer in enumerate(_teacher_core(model).model.layers)
            if isinstance(layer.self_attn, KernelDreamFullRouteAttention)
            and not hasattr(layer.self_attn, "frozen_dense_q_weight")
        ]
        if missing:
            raise RuntimeError(
                "route_dense_teacher_source=dense requires snapshot_frozen_dense_qk "
                f"before training; missing on layers {missing[:8]}"
            )
    core = _teacher_core(model)
    layers = {i: layer.self_attn for i, layer in enumerate(core.model.layers)
              if isinstance(layer.self_attn, KernelDreamFullRouteAttention)}
    valid = batch["attention_mask"].bool().clone()
    valid[:, args.chunk_size - 1::args.chunk_size] = False
    query_valid = valid & batch["labels"].ne(-100)
    labeled = int(query_valid.sum(-1).min())
    requested = int(args.route_dense_teacher_queries)
    count = labeled if requested <= 0 else min(requested, labeled)
    if not layers or count <= 0:
        raise ValueError("dense teacher requires route layers and supervised queries")
    segments = batch.get("segment_ids", batch["attention_mask"].long())
    devices = [valid.device.index] if valid.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        positions = {i: torch.stack([
            (ids := torch.where(row)[0])[torch.randperm(ids.numel(), device=valid.device)[:count]]
            for row in query_valid]) for i in layers}
    for index, layer in layers.items():
        layer.full_teacher_query_positions = positions[index]
        layer.full_teacher_segments = segments
        layer.full_teacher_rule = args.route_dense_teacher_rule
        layer.full_teacher_temperature = args.route_dense_teacher_temperature
        layer.full_teacher_source = source
        layer.full_teacher_loss = None
        if hasattr(layer, "full_teacher_target"):
            delattr(layer, "full_teacher_target")
    core._last_dense_teacher_layers = len(layers)
    core._last_dense_teacher_queries = count
    core._last_dense_teacher_source = source
    return list(layers.values())


def attach_same_forward_teacher(layer, q, k, key_valid):
    """Fill teacher chunk scores from detached Q/K.

    Eligible chunks are all remote (non-local) chunks in-document, including
    chunks the hard router did not pick. Teacher Q/K are detached so CE still
    cannot train a second QK; only Q-Cal receives the KL.
    """
    positions = layer.full_teacher_query_positions
    segments = layer.full_teacher_segments
    scores = token_scores(
        q.detach(), k.detach(), positions, key_valid, segments, int(layer.chunk_size)
    )
    eligible, _ = sampled_candidates(
        positions, key_valid, segments, int(layer.chunk_size), int(layer.local_window)
    )
    target = chunk_scores(scores, layer.full_teacher_rule)
    eligible_scores = target.masked_select(eligible[:, :, None, None].expand_as(target))
    if eligible_scores.numel() and not torch.isfinite(eligible_scores).all():
        raise FloatingPointError("non-finite eligible teacher chunk score")
    layer.full_teacher_target = (positions, target.detach(), eligible)
    layer._probe_q = q.detach()
    layer._probe_k = k.detach()
    layer._probe_key_valid = key_valid.detach()


def ste_type_embed_hidden(hidden, chunk_size, type_embed):
    """Detach the residual stream; STE the landmark type vector onto LMK slots.

    Forward equals `hidden.detach()`. Backward updates `type_embed` only.
    """
    routed = hidden.detach()
    if type_embed is None:
        return routed
    is_lmk = torch.zeros(
        hidden.shape[:2], device=hidden.device, dtype=routed.dtype
    )
    is_lmk[:, int(chunk_size) - 1 :: int(chunk_size)] = 1
    live = type_embed.to(device=routed.device, dtype=routed.dtype)
    return routed + is_lmk.unsqueeze(-1) * (live - live.detach())


def router_hidden_for_kl(hidden, rows, chunk_size, type_embed=None):
    """Detach the residual stream; STE the landmark type vector onto LMK rows.

    Forward matches `hidden.detach()`. Backward updates `type_embed` only, not LoRA.
    """
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    routed = hidden[batch, rows].detach()
    if type_embed is None:
        return routed
    landmarks = torch.arange(
        int(chunk_size) - 1, hidden.shape[1], int(chunk_size), device=hidden.device
    )
    is_lmk = (rows.unsqueeze(-1) == landmarks).any(-1).unsqueeze(-1).to(routed.dtype)
    live = type_embed.to(device=routed.device, dtype=routed.dtype)
    return routed + is_lmk * (live - live.detach())


def student_router_kl(layer, hidden, q, k, key_valid, dropped, position_ids, position_embeddings):
    """Recompute sampled/LMK queries with detached QK: KL -> Q-Cal and LMK type embed."""
    from ops.chunk_attn_pool_gqa_tilelang import chunk_attn_pool_gqa
    positions, teacher, eligible = layer.full_teacher_target
    b, n, h, d = k.shape
    bi = torch.arange(b, device=k.device)[:, None]
    if not torch.equal(eligible, ~dropped[bi, positions].bool()):
        raise AssertionError("teacher/student remote candidate domains differ")
    lmk = torch.arange(layer.chunk_size - 1, n, layer.chunk_size, device=k.device)
    rows = torch.cat((positions, lmk[None].expand(b, -1)), -1)
    if position_embeddings is None:
        raise ValueError("dense teacher KL requires explicit shared rotary embeddings")
    rope = tuple(x[bi, rows].detach() for x in position_embeddings)
    type_embed = (
        getattr(layer, "lmk_type_embed", None)
        if bool(getattr(layer, "lmk_kl_ste", True))
        else None
    )
    route_hidden = router_hidden_for_kl(
        hidden, rows, layer.chunk_size, type_embed
    )
    route_q = layer._calibrated_query(route_hidden, q[bi, rows].detach(),
                                      position_ids[bi, rows], rope)
    groups = q.shape[2] // h
    queries = route_q[:, :positions.shape[1]].reshape(b, positions.shape[1], h, groups, d)
    lmk_q = route_q[:, positions.shape[1]:].reshape(b, n // layer.chunk_size, h, groups, d)
    keys, entropy = chunk_attn_pool_gqa(
        lmk_q.contiguous(), k.detach().reshape(b, n // layer.chunk_size, layer.chunk_size, h, d).contiguous(),
        key_valid.reshape(b, n // layer.chunk_size, layer.chunk_size).contiguous())
    with torch.autocast(q.device.type, enabled=False):
        logits = torch.einsum("brhgd,bchgd->brhgc", queries.float(), keys.float()) / math.sqrt(d)
        prior = entropy * layer.entropy_bias_scale.detach().float().view(1, 1, h, groups)
        logits = logits + prior.permute(0, 2, 3, 1)[:, None]
        return candidate_kl(logits, teacher, eligible, layer.full_teacher_temperature)


def collect_dense_teacher_loss(layers):
    losses = [layer.full_teacher_loss for layer in layers]
    if any(loss is None for loss in losses):
        raise RuntimeError("missing dense teacher KL from a route layer")
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite full dense teacher KL")
    return loss
