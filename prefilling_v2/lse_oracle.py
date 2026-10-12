"""Oracle diagnostics on active answer predictors; never use labels to route."""
from __future__ import annotations

import math
from contextlib import ExitStack
from unittest.mock import patch

import torch

from prefilling_v2.lse_calibration import exact_chunk_lse, group_priority, top_indices, overlap


def evidence_recall(indices, dropped, evidence_chunks):
    """Denominator: query x KV head x distinct eligible remote evidence chunk."""
    evidence = torch.unique(evidence_chunks.to(indices.device).long())
    if not evidence.numel():
        raise ValueError("missing evidence labels")
    eligible = ~dropped.index_select(-1, evidence).bool()
    hits = (indices[..., None] == evidence).any(-2)
    eligible = eligible[:, None].expand_as(hits)
    count = int(eligible.sum())
    hit_count = int((hits & eligible).sum())
    union_eligible = eligible.any((0, 1))
    union_hits = (hits & eligible).any((0, 1))
    per_evidence = (hits & eligible).sum((0, 1))
    return dict(hits=hit_count, units=count, recall=hit_count / count if count else None,
                evidence_chunks=evidence.tolist(), hits_per_evidence=per_evidence.tolist(),
                units_per_evidence=eligible.sum((0, 1)).tolist(),
                any_hit=int(hit_count > 0), union_hits=int(union_hits.sum()),
                union_units=int(union_eligible.sum()),
                union_recall=float(union_hits.sum() / union_eligible.sum()) if union_eligible.any() else None)


def full_gate(z, local, indices):
    gather = indices[:, :, None].expand(-1, -1, z.shape[2], -1)
    selected = z.gather(-1, gather.clamp_min(0)).masked_fill(gather < 0, -torch.inf)
    return torch.softmax(torch.cat((selected.float(), local.float()[..., None]), -1), -1)


def canonical_indices(priority, topk):
    idx = top_indices(priority, topk)
    sentinel = priority.shape[-1]
    sorted_idx = idx.masked_fill(idx < 0, sentinel).sort(-1).values
    return sorted_idx.masked_fill(sorted_idx == sentinel, -1)


def choose_intervention(variant, estimated, exact, local, dropped, native, frozen_support=None):
    if variant not in {"baseline", "select_exact", "fusion_exact"}:
        raise ValueError(variant)
    if variant == "select_exact":
        selected = canonical_indices(group_priority(exact, local, dropped), native.shape[-1]).to(native.dtype)
    elif variant == "fusion_exact" and frozen_support is not None:
        if frozen_support.shape != native.shape:
            raise ValueError("frozen support shape mismatch")
        selected = frozen_support.clone()
    else:
        selected = native.clone()
    # Selection-only retains the original surrogate fusion rule on its new support.
    gate_scores = exact if variant == "fusion_exact" else estimated
    return selected, full_gate(gate_scores, local, selected)


def retained_mass(exact, dropped, indices):
    masked = exact.masked_fill(dropped[:, None, None], -torch.inf)
    p = torch.softmax(masked, -1)
    gather = indices[:, :, None].expand(-1, -1, exact.shape[2], -1)
    selected = p.gather(-1, gather.clamp_min(0)).masked_fill(gather < 0, 0)
    return float(selected.sum(-1).mean())


class LSEOracleProbe:
    def __init__(self, decoder):
        self.decoder = decoder
        self.stack = ExitStack()
        self.enabled = False
        self.pending = self.component = None
        self.records = []

    def begin(self, case, layout, variant, phase):
        from prefilling_v2.diagnostic_helpers import physical_positions
        self.positions_to_change = layout.predictor_positions
        if self.positions_to_change.numel() != 32:
            raise ValueError("oracle expects 32 answer predictor positions")
        self.values = torch.unique(physical_positions(torch.tensor(sum(case.evidence_values, [])), 64) // 64)
        self.facts = torch.unique(physical_positions(torch.tensor(sum(case.evidence_facts, [])), 64) // 64)
        self.variant, self.phase = variant, phase
        if variant == "baseline" and phase == "prefill":
            self.baseline_support = {}
        self.enabled = True
        self.records = []

    def enter_layer(self, module, positions, layer):
        self.module, self.positions, self.layer = module, positions, layer
        self.pending = self.component = None

    def rows(self):
        return torch.where(torch.isin(self.positions, self.positions_to_change.to(self.positions.device)))[0]

    def __enter__(self):
        from prefilling_v2 import routing, fastdllm_attention, fastdllm_v1
        from ops import bidir_output_fusion_tilelang as fusion
        ids = {id(layer.self_attn): i for i, layer in enumerate(self.decoder._core().model.layers)}
        for layer in self.decoder._core().model.layers:
            def before(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.enter_layer(module, torch.arange(hidden.shape[1], device=hidden.device), ids[id(module)])
            handle = layer.self_attn.register_forward_pre_hook(before, with_kwargs=True)
            self.stack.callback(handle.remove)
        native_cached = fastdllm_v1.cached_attention_forward
        def cached(module, hidden, **kwargs):
            self.enter_layer(module, kwargs["query_positions"], ids[id(module)])
            return native_cached(module, hidden, **kwargs)
        self.stack.enter_context(patch.object(fastdllm_v1, "cached_attention_forward", cached))
        native_route = routing.route_topk_g7
        def route(*args, **kwargs):
            result = native_route(*args, **kwargs)
            if self.enabled:
                rows = self.rows()
                if rows.numel():
                    q, lmks, local, prior, dropped = args[:5]
                    if kwargs.get("selection_mode", "post_softmax") != "post_softmax":
                        raise ValueError("oracle is defined for the native post-softmax selector")
                    h, g, d = lmks.shape[2:]
                    raw = torch.einsum("blhgd,bshgd->blhgs",
                        q[:, rows].float().reshape(1, len(rows), h, g, d), lmks.float()) * (1 / math.sqrt(d))
                    estimated = raw.float()[0] + prior.float()[0].permute(1, 2, 0)[None]
                    self.component = dict(rows=rows, estimated=estimated, local=local[0, rows].float(),
                                          dropped=dropped[0, rows].bool(), h=h, g=g, d=d)
            return result
        for mod in (routing, fastdllm_attention):
            self.stack.enter_context(patch.object(mod, "route_topk_g7", route))
        for mod, name in ((routing, "selected_attention_g7"), (fastdllm_attention, "selected_attention_g7_cached")):
            original = getattr(mod, name)
            def selected(*args, _original=original, **kwargs):
                native_output = _original(*args, **kwargs)
                if not self.enabled or self.component is None:
                    return native_output
                q, k, v, weights, indices, valid, size = args[:7]
                if kwargs.get("token_keep") is not None or self.module.token_budget != 0:
                    raise ValueError("oracle requires no token eviction")
                c = self.component
                rows = c["rows"]
                with torch.autocast(q.device.type, enabled=False):
                    exact = exact_chunk_lse(q[0, rows].reshape(-1, c["h"], c["g"], c["d"]), k[0], valid[0], size)
                    native_idx = indices[0, rows]
                    key = (self.phase, self.layer)
                    if self.variant == "baseline":
                        self.baseline_support[key] = native_idx.clone()
                    reference = self.baseline_support[key]
                    reconstructed = canonical_indices(group_priority(c["estimated"], c["local"], c["dropped"]), native_idx.shape[-1])
                    reconstruction_overlap = overlap(reconstructed, native_idx)
                    if reconstruction_overlap < .99:
                        raise AssertionError(f"native score reconstruction failed: {reconstruction_overlap}")
                    chosen, gate = choose_intervention(self.variant, c["estimated"], exact, c["local"], c["dropped"], native_idx,
                                                        frozen_support=reference)
                    shadow_idx, shadow_gate = choose_intervention("select_exact", c["estimated"], exact, c["local"], c["dropped"], native_idx)
                    fixed_exact = full_gate(exact, c["local"], native_idx)
                    estimated_native = full_gate(c["estimated"], c["local"], native_idx)
                if not torch.isfinite(gate).all():
                    raise AssertionError("non-finite oracle gate")
                if self.variant != "baseline":
                    changed_indices = indices.clone()
                    changed_indices[0, rows] = chosen
                    changed_weights = weights.clone()
                    changed_weights[0, rows] = gate[..., :-1].flatten(1, 2).to(weights.dtype)
                    modified = (*args[:3], changed_weights, changed_indices, *args[5:])
                    output = _original(*modified, **kwargs)
                else:
                    output = native_output
                # Record the same 16 sampled predictors as the earlier calibration audit.
                sample = torch.linspace(0, len(rows) - 1, 16, device=rows.device).long()
                dr = c["dropped"][sample]
                record = dict(layer=self.layer, phase=self.phase, variant=self.variant,
                    intervened_query_count=0 if self.variant == "baseline" else len(rows),
                    query_positions=self.positions[rows[sample]].tolist(),
                    native_selector_reconstruction_overlap=reconstruction_overlap,
                    support_overlap=overlap(chosen[sample], native_idx[sample]),
                    support_unchanged=bool(torch.equal(chosen, native_idx)),
                    support_matches_frozen_baseline=bool(torch.equal(chosen, reference)),
                    baseline_support_overlap=overlap(chosen[sample], reference[sample]),
                    native_indices=native_idx[sample].tolist(), actual_indices=chosen[sample].tolist(),
                    native_value=evidence_recall(native_idx[sample], dr, self.values),
                    actual_value=evidence_recall(chosen[sample], dr, self.values),
                    shadow_select_value=evidence_recall(shadow_idx[sample], dr, self.values),
                    native_fact=evidence_recall(native_idx[sample], dr, self.facts),
                    actual_fact=evidence_recall(chosen[sample], dr, self.facts),
                    shadow_select_fact=evidence_recall(shadow_idx[sample], dr, self.facts),
                    native_remote_mass=retained_mass(exact[sample], dr, native_idx[sample]),
                    actual_remote_mass=retained_mass(exact[sample], dr, chosen[sample]),
                    estimated_native_local=float(estimated_native[sample, ..., -1].mean()),
                    fixed_exact_local=float(fixed_exact[sample, ..., -1].mean()),
                    shadow_select_local=float(shadow_gate[sample, ..., -1].mean()))
                if self.variant == "fusion_exact" and not record["support_matches_frozen_baseline"]:
                    raise AssertionError("fusion-only changed the frozen baseline support")
                self.records.append(record)
                self.pending = dict(record=record, rows=rows, sample=sample, gate=gate,
                                    native_remote=native_output[0, rows].float().clone())
                self.component = None
                return output
            self.stack.enter_context(patch.object(mod, name, selected))
        native_fuse = fusion.fuse_route_outputs
        def fuse(remote, local, weight):
            if self.enabled and self.pending is not None:
                p = self.pending
                rows, sample = p["rows"], p["sample"]
                old_weight = weight[0, rows].float()
                p["record"]["native_local_weight"] = float(old_weight[sample].mean())
                if self.variant != "baseline":
                    weight = weight.clone()
                    weight[0, rows] = p["gate"][..., -1].flatten(1, 2).to(weight.dtype)
                result = native_fuse(remote, local, weight)
                p["record"]["actual_local_weight"] = float(weight[0, rows[sample]].float().mean())
                native_fused = p["native_remote"] + local[0, rows].float() * old_weight[..., None]
                actual_fused = result[0, rows].float()
                p["record"]["layer_output_relative_l2_vs_same_state_native"] = float(
                    (actual_fused[sample] - native_fused[sample]).norm() / native_fused[sample].norm().clamp_min(1e-8))
                self.pending = None
                return result
            return native_fuse(remote, local, weight)
        self.stack.enter_context(patch.object(fusion, "fuse_route_outputs", fuse))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
