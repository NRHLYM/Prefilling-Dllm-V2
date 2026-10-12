"""Independent all-layer dense diagnostics, without LMK keys or routing."""
from __future__ import annotations

import importlib
import math
from types import MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from prefilling_v2.lse_calibration import exact_chunk_lse, group_priority, top_indices
from prefilling_v2.lse_oracle import evidence_recall, retained_mass


LAYERS = (3, 7, 11, 15, 19, 23, 27)


def body_key_mask(valid, chunk_size=64):
    positions = torch.arange(valid.shape[-1], device=valid.device)
    return valid.bool() & ((positions + 1) % chunk_size != 0)


def local_chunks(query_positions, length, window=512, chunk_size=64):
    left = (query_positions - window).clamp_min(0) // chunk_size
    right = (query_positions + window).clamp_max(length - 1) // chunk_size
    chunks = torch.arange(length // chunk_size, device=query_positions.device)
    return (chunks[None] >= left[:, None]) & (chunks[None] <= right[:, None])


def summarize_exact(exact, local, dropped, values, facts, topk=16):
    group_idx = top_indices(group_priority(exact, local, dropped), topk)
    early_idx = top_indices(group_priority(exact, local, dropped, eligible_denominator=True), topk)
    head_idx = top_indices(exact.masked_fill(dropped[:, None, None], -torch.inf), topk)
    head_idx = head_idx.flatten(1, 2)
    probabilities = torch.softmax(exact, -1)
    evidence = torch.unique(values.to(exact.device))
    remote = probabilities.masked_fill(dropped[:, None, None], 0).sum(-1)
    remote_p = torch.softmax(exact.masked_fill(dropped[:, None, None], -torch.inf), -1)
    mask = (~dropped[:, evidence])[:, None, None]
    return dict(
        value=evidence_recall(group_idx, dropped, values),
        fact=evidence_recall(group_idx, dropped, facts),
        early_value=evidence_recall(early_idx, dropped, values),
        per_head_value=evidence_recall(head_idx, dropped, values),
        evidence_all_attention_mass=float(probabilities[..., evidence].sum(-1).mean()),
        evidence_remote_attention_mass=float((remote_p[..., evidence] * mask).sum(-1).mean()),
        remote_attention_mass=float(remote.mean()),
        retained_remote_mass=retained_mass(exact, dropped, group_idx),
    ), group_idx


def dense_attention(q, k, v, key_valid, *, backend=SDPBackend.FLASH_ATTENTION):
    """BLHD GQA, all valid body keys, noncausal; no quadratic score allocation."""
    if q.shape[0] != 1 or key_valid.shape != k.shape[:2]:
        raise ValueError("reference supports a single sequence")
    rows = torch.where(key_valid[0])[0]
    if not rows.numel():
        raise ValueError("no valid body keys")
    groups = q.shape[2] // k.shape[2]
    kk = k.index_select(1, rows).transpose(1, 2).repeat_interleave(groups, dim=1)
    vv = v.index_select(1, rows).transpose(1, 2).repeat_interleave(groups, dim=1)
    with sdpa_kernel(backend):
        out = scaled_dot_product_attention(q.transpose(1, 2), kk, vv, dropout_p=0, is_causal=False)
    return out.transpose(1, 2).contiguous()


def project(attn, hidden, position_embeddings):
    from prefilling_v2.attention import apply_rotary_pos_emb
    b, n, _ = hidden.shape
    q = attn.q_proj(hidden).view(b, n, attn.num_heads, attn.head_dim)
    k = attn.k_proj(hidden).view(b, n, attn.num_key_value_heads, attn.head_dim)
    v = attn.v_proj(hidden).view_as(k)
    q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *position_embeddings)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v


@torch.inference_mode()
def dense_forward(core, ids, valid, positions, predictors, observer=None):
    hidden = core.model.embed_tokens(ids)
    position_embeddings = core.model.rotary_emb(hidden, positions)
    for layer_id, layer in enumerate(core.model.layers):
        q, k, v = project(layer.self_attn, layer.input_layernorm(hidden), position_embeddings)
        if observer is not None:
            observer(layer_id, q, k)
        attended = dense_attention(q, k, v, valid).reshape_as(hidden)
        hidden = hidden + layer.self_attn.o_proj(attended)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    hidden = core.model.norm(hidden.index_select(1, predictors))
    return core.lm_head(hidden)


@torch.inference_mode()
def validate_native(core, ids, valid, positions):
    """Compare a long-position, short-sequence forward against native Dream SDPA."""
    from prefilling_v2.train_fulltext import DenseDreamAttentionAdapter
    from scripts.prefilling_v2.diagnose_retrieval import compare
    native_cls = importlib.import_module(type(core).__module__).DreamSdpaAttention
    # Masked-out queries cannot affect body keys in any layer. Compact both
    # references to isolate the layer-loop implementation from backend rounding.
    kept = torch.where(valid[0])[0]
    ids, positions = ids[:, kept], positions[:, kept]
    valid = torch.ones_like(ids, dtype=torch.bool)
    predictors = torch.arange(ids.shape[1] - 32, ids.shape[1], device=ids.device)
    ours = dense_forward(core, ids, valid, positions, predictors)
    originals = []
    try:
        for layer in core.model.layers:
            attn = layer.self_attn
            originals.append((layer, attn, attn.forward))
            attn.forward = MethodType(native_cls.forward, attn)
            layer.self_attn = DenseDreamAttentionAdapter(attn)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            hidden = core.model(input_ids=ids, attention_mask=None,
                                position_ids=positions, use_cache=False).last_hidden_state
            reference = core.lm_head(hidden.index_select(1, predictors))
    finally:
        for layer, attn, forward in originals:
            layer.self_attn = attn
            attn.forward = forward
    result = compare(ours, reference)
    if result["relative_l2"] > .003 or result["cosine"] < .99999:
        raise AssertionError(f"dense native validation failed: {result}")
    result["backend"] = "native and diagnostic both Flash SDPA; compact valid body tokens, original positions"
    return result


class DenseEvidenceProbe:
    def __init__(self, layout, values, facts, old_record, old_arrays):
        self.sample = layout.predictor_positions[
            torch.linspace(0, len(layout.predictor_positions) - 1, 16).long()].cuda()
        self.valid = layout.attention_mask.cuda().bool()
        self.values, self.facts = values.cuda(), facts.cuda()
        self.old = {o["layer"]: o for o in old_record["observations"]}
        self.old_arrays = old_arrays
        self.records, self.arrays, self.visited = [], {}, []

    def __call__(self, layer, q, k):
        self.visited.append(layer)
        if layer not in LAYERS:
            return
        h, d = k.shape[2:]
        query = q[0, self.sample].reshape(len(self.sample), h, -1, d)
        with torch.autocast("cuda", enabled=False):
            exact = exact_chunk_lse(query, k[0], self.valid, 64)
            local_mask = local_chunks(self.sample, k.shape[1])
            local = torch.logsumexp(exact.masked_fill(~local_mask[:, None, None], -torch.inf), -1)
            eligible = body_key_mask(self.valid).reshape(-1, 64).any(-1)
            dropped = local_mask | ~eligible[None]
            old_drop = ~torch.from_numpy(self.old_arrays[f"layer{layer}_eligible"][:, 0, 0]).cuda()
            if not torch.equal(dropped, old_drop):
                raise AssertionError("candidate eligibility differs from sparse audit")
            if self.sample.tolist() != self.old[layer]["query_positions"]:
                raise AssertionError("query positions differ from sparse audit")
            stats, indices = summarize_exact(exact, local, dropped, self.values, self.facts)
        stats.update(layer=layer, query_positions=self.sample.tolist())
        self.records.append(stats)
        self.arrays.update({f"layer{layer}_{name}": array.cpu().numpy() for name, array in
                           dict(exact_token=exact, local_lse=local, indices=indices, dropped=dropped).items()})
