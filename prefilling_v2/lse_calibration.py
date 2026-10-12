"""Read-only, same-forward audits of landmark chunk-LSE calibration."""
from __future__ import annotations

import math
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import rankdata


def exact_chunk_lse(q, k, valid, chunk_size):
    """q [Q,Hkv,G,D], k [N,Hkv,D]; invalid and LMK keys are excluded."""
    n, h, d = k.shape
    if n % chunk_size or q.shape[1] != h or q.shape[-1] != d:
        raise ValueError("incompatible Q/K/chunk layout")
    body = valid.bool() & (torch.arange(n, device=k.device) % chunk_size != chunk_size - 1)
    with torch.autocast(q.device.type, enabled=False):
        scores = torch.einsum("qhgd,nhd->qhgn", q.float(), k.float()) / math.sqrt(d)
        scores = scores.masked_fill(~body[None, None, None], -torch.inf)
        return torch.logsumexp(scores.reshape(*q.shape[:3], n // chunk_size, chunk_size), -1)


def group_priority(z, local, dropped, mode="post_softmax", eligible_denominator=False):
    """Match the native late-drop normalization, or explicitly audit early-drop."""
    if eligible_denominator:
        z = z.masked_fill(dropped[:, None, None], -torch.inf)
    if mode == "post_softmax":
        total = torch.logaddexp(local.float(), torch.logsumexp(z, -1))
        scores = z - total[..., None]
    elif mode == "pre_softmax":
        scores = z
    else:
        raise ValueError(mode)
    return scores.amax(2).masked_fill(dropped[:, None], -torch.inf)


def top_indices(priority, topk):
    value, index = priority.topk(min(topk, priority.shape[-1]), dim=-1)
    return index.masked_fill(~torch.isfinite(value), -1)


def overlap(a, b):
    matches = (a[..., :, None] == b[..., None, :]) & (a[..., :, None] >= 0)
    denom = (b >= 0).sum(-1)
    rates = matches.any(-2).sum(-1).float() / denom.clamp_min(1)
    return float(rates[denom > 0].mean()) if (denom > 0).any() else None


def local_gate(z, local, indices, temperature=1.0):
    gather = indices[:, :, None].expand(-1, -1, z.shape[2], -1)
    selected = z.gather(-1, gather.clamp_min(0)).masked_fill(gather < 0, -torch.inf)
    return torch.softmax(torch.cat((selected, local[..., None]), -1) / temperature, -1)[..., -1]


def distribution(x):
    a = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"mean": None, "p05": None, "p50": None, "p95": None, "count": 0}
    return dict(mean=float(a.mean()), p05=float(np.quantile(a, .05)),
                p50=float(np.median(a)), p95=float(np.quantile(a, .95)), count=int(a.size))


def score_metrics(estimate, reference, eligible, topk=16):
    """Row-wise ranks and sampled pair order, never ranks pooled across heads."""
    shape = estimate.shape
    a = estimate.detach().double().cpu().numpy().reshape(-1, shape[-1])
    b = reference.detach().double().cpu().numpy().reshape(a.shape)
    mask = torch.broadcast_to(eligible, shape).detach().cpu().numpy().reshape(a.shape)
    errors, centered, spearman, pearson, order, overlaps = [], [], [], [], [], []
    ties, pair_count, rows = 0, 0, 0
    rng = np.random.default_rng(20260916)
    for aa, bb, mm in zip(a, b, mask):
        good = mm & np.isfinite(aa) & np.isfinite(bb)
        aa, bb = aa[good], bb[good]
        if len(aa) < 2:
            continue
        rows += 1
        error = aa - bb
        errors.append(error)
        centered.append(error - error.mean())
        ra, rb = rankdata(aa), rankdata(bb)
        if ra.std() > 0 and rb.std() > 0:
            spearman.append(float(np.corrcoef(ra, rb)[0, 1]))
        if aa.std() > 0 and bb.std() > 0:
            pearson.append(float(np.corrcoef(aa, bb)[0, 1]))
        # Fixed deterministic sample, with replacement, excluding identical chunks.
        left = rng.integers(0, len(aa), 4096)
        right = (left + rng.integers(1, len(aa), 4096)) % len(aa)
        da, db = aa[left] - aa[right], bb[left] - bb[right]
        non_tied_ref = np.abs(db) > 1e-6
        eq = np.abs(da) <= 1e-6
        ties += int((~non_tied_ref).sum())
        pair_count += len(left)
        if non_tied_ref.any():
            agreement = np.where(eq, .5, (np.sign(da) == np.sign(db)).astype(float))
            order.append(float(agreement[non_tied_ref].mean()))
        k = min(topk, len(aa))
        ia, ib = np.argsort(-aa, kind="stable")[:k], np.argsort(-bb, kind="stable")[:k]
        overlaps.append(len(set(ia) & set(ib)) / k)
    if not errors:
        return {"rows": 0, "values": 0}
    err, cen = np.concatenate(errors), np.concatenate(centered)
    return dict(rows=rows, values=int(err.size), bias=float(err.mean()), mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt(np.mean(err ** 2))), p95_abs=float(np.quantile(np.abs(err), .95)),
                centered_rmse=float(np.sqrt(np.mean(cen ** 2))),
                underestimated_fraction=float((err < 0).mean()),
                spearman=distribution(spearman), pearson=distribution(pearson),
                pair_order_agreement=distribution(order),
                reference_pair_tie_fraction=ties / max(pair_count, 1),
                topk_overlap=distribution(overlaps))


def audit_tensors(route_q, token_q, k, lmks, prior, local, dropped, indices, valid,
                  chunk_size=64, mode="post_softmax", temperature=1.0, native_z=None,
                  selected_scores=None):
    h, g, d = lmks.shape[1:]
    route_q = route_q.reshape(-1, h, g, d)
    token_q = token_q.reshape_as(route_q)
    with torch.autocast(k.device.type, enabled=False):
        z_fp32 = torch.einsum("qhgd,chgd->qhgc", route_q.float(), lmks.float()) / math.sqrt(d)
        z_fp32 += prior.float().permute(1, 2, 0)[None]
        z = z_fp32 if native_z is None else native_z.float()
        route = exact_chunk_lse(route_q, k, valid, chunk_size)
        token = exact_chunk_lse(token_q, k, valid, chunk_size)
        local = local.float()
        eligible = ~dropped[:, None, None]
        for exact in (route, token):
            if not torch.isfinite(exact.masked_select(eligible.expand_as(exact))).all():
                raise AssertionError("selectable chunk has no finite body-key LSE")
        native_ref = top_indices(group_priority(z, local, dropped, mode), indices.shape[-1])
        fp32_ref = top_indices(group_priority(z_fp32, local, dropped, mode), indices.shape[-1])
        base_gate = local_gate(z, local, indices, temperature)
        result = {
            "route_reference": score_metrics(z, route, eligible),
            "token_reference": score_metrics(z, token, eligible),
            "query_branch_gap": score_metrics(route, token, eligible),
            "native_selector_reconstruction_overlap": overlap(native_ref, indices),
            "fp32_selector_native_overlap": overlap(fp32_ref, indices),
            "fp32_route_reference": score_metrics(z_fp32, route, eligible),
            "fp32_token_reference": score_metrics(z_fp32, token, eligible),
            "autocast_minus_fp32": score_metrics(z, z_fp32, eligible),
            "estimated_local_weight": distribution(base_gate),
            "local_lse": distribution(local),
            "eligible_chunks_per_query": distribution((~dropped).sum(-1)),
        }
        arrays = {"estimated": z, "estimated_fp32": z_fp32, "exact_route": route, "exact_token": token,
                  "eligible": eligible, "local_lse": local, "native_indices": indices}
        approximate_priority = group_priority(z, local, dropped, mode)
        for name, exact in (("route", route), ("token", token)):
            priority = group_priority(exact, local, dropped, mode)
            oracle_idx = top_indices(priority, indices.shape[-1])
            early_idx = top_indices(group_priority(exact, local, dropped, mode, True), indices.shape[-1])
            fixed = local_gate(exact, local, indices, temperature)
            reranked = local_gate(exact, local, oracle_idx, temperature)
            result[name + "_group_rank"] = score_metrics(approximate_priority, priority, ~dropped[:, None])
            result[name + "_selector_overlap"] = overlap(indices, oracle_idx)
            result[name + "_eligible_denominator_selector_overlap"] = overlap(indices, early_idx)
            result[name + "_fixed_support_local_weight"] = distribution(fixed)
            result[name + "_fixed_support_local_delta"] = distribution(fixed - base_gate)
            result[name + "_fixed_support_local_abs_delta"] = distribution((fixed - base_gate).abs())
            result[name + "_reranked_local_weight"] = distribution(reranked)
            arrays[name + "_fixed_gate"] = fixed
        arrays["estimated_gate"] = base_gate
        if selected_scores is not None:
            gather = indices[:, :, None].expand(-1, -1, g, -1)
            bias = prior.float().permute(1, 2, 0)[None].expand(z.shape[0], -1, -1, -1)
            selected_z = selected_scores.float().reshape_as(gather) + bias.gather(-1, gather.clamp_min(0))
            selected_z = selected_z.masked_fill(gather < 0, -torch.inf)
            arrays["kernel_gate_reference"] = torch.softmax(
                torch.cat((selected_z, local[..., None]), -1) / temperature, -1)[..., -1]
        return result, arrays


class LSECalibrationProbe:
    """Passthrough hooks: no tensor edits, no changes to routing or fusion."""

    def __init__(self, decoder, max_queries=16):
        self.decoder, self.max_queries = decoder, max_queries
        self.stack = ExitStack()
        self.records, self.arrays = [], []
        self.enabled = False
        self.pending = None
        self.component = None

    def begin(self, layout, phase):
        self.active = layout.predictor_positions
        self.phase = phase
        self.records, self.arrays = [], []
        self.enabled = True

    def enter_layer(self, module, positions):
        self.module, self.positions = module, positions
        self.pending, self.component = None, None

    def rows(self):
        rows = torch.where(torch.isin(self.positions, self.active.to(self.positions.device)))[0]
        if rows.numel() > self.max_queries:
            rows = rows[torch.linspace(0, rows.numel() - 1, self.max_queries, device=rows.device).long()]
        return rows

    def __enter__(self):
        from prefilling_v2 import routing, fastdllm_attention, fastdllm_v1
        from ops import bidir_output_fusion_tilelang as fusion
        for i, layer in enumerate(self.decoder._core().model.layers):
            attn = layer.self_attn
            def before(module, args, kwargs, layer_id=i):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.layer = layer_id
                self.enter_layer(module, torch.arange(hidden.shape[1], device=hidden.device))
            handle = attn.register_forward_pre_hook(before, with_kwargs=True)
            self.stack.callback(handle.remove)
        original_cached = fastdllm_v1.cached_attention_forward
        layer_ids = {id(layer.self_attn): i for i, layer in enumerate(self.decoder._core().model.layers)}
        def cached(module, hidden, **kwargs):
            self.layer = layer_ids[id(module)]
            self.enter_layer(module, kwargs["query_positions"])
            return original_cached(module, hidden, **kwargs)
        self.stack.enter_context(patch.object(fastdllm_v1, "cached_attention_forward", cached))
        original_route = routing.route_topk_g7
        def route(*args, **kwargs):
            result = original_route(*args, **kwargs)
            if self.enabled:
                rows = self.rows()
                if rows.numel():
                    q, lmks, local, prior, dropped = args[:5]
                    h, g, d = lmks.shape[2:]
                    # Preserve the native autocast/scaling sequence, independently of FP32 exact LSE.
                    raw = torch.einsum("blhgd,bshgd->blhgs",
                        q[:, rows].float().reshape(1, len(rows), h, g, d), lmks.float())
                    native_z = raw * (1.0 / math.sqrt(d)) + prior.float().permute(0, 2, 3, 1)[:, None]
                    self.component = dict(rows=rows, route_q=q[0, rows], lmks=lmks[0],
                                          local=local[0, rows], prior=prior[0],
                                          dropped=dropped[0, rows].bool(),
                                          native_z=native_z[0], selected_scores=result[1][0, rows],
                                          mode=kwargs.get("selection_mode", "post_softmax"))
            return result
        for mod in (routing, fastdllm_attention):
            self.stack.enter_context(patch.object(mod, "route_topk_g7", route))
        for mod, name in ((routing, "selected_attention_g7"), (fastdllm_attention, "selected_attention_g7_cached")):
            original = getattr(mod, name)
            def selected(*args, _original=original, **kwargs):
                output = _original(*args, **kwargs)
                if self.enabled and self.component is not None:
                    q, k, v, weights, indices, valid, size = args[:7]
                    c = self.component
                    rows = c["rows"]
                    metrics, arrays = audit_tensors(c["route_q"], q[0, rows], k[0], c["lmks"],
                        c["prior"], c["local"], c["dropped"], indices[0, rows], valid[0], size,
                        c["mode"], self.module.route_temperature, c["native_z"], c["selected_scores"])
                    metrics.update(layer=self.layer, phase=self.phase, queries=len(rows),
                                   query_positions=self.positions[rows].tolist(), selection_mode=c["mode"],
                                   token_budget=self.module.token_budget)
                    self.records.append(metrics)
                    self.arrays.append({key: value.detach().cpu().numpy() for key, value in arrays.items()})
                    self.pending = (metrics, rows, arrays["kernel_gate_reference"])
                    self.component = None
                return output
            self.stack.enter_context(patch.object(mod, name, selected))
        original_fuse = fusion.fuse_route_outputs
        def fuse(remote, local, weight):
            output = original_fuse(remote, local, weight)
            if self.enabled and self.pending is not None:
                metrics, rows, expected = self.pending
                actual = weight[0, rows].reshape_as(expected).float()
                metrics["native_local_weight"] = distribution(actual)
                metrics["gate_reconstruction_abs_error"] = distribution((actual - expected).abs())
                self.pending = None
            return output
        self.stack.enter_context(patch.object(fusion, "fuse_route_outputs", fuse))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
