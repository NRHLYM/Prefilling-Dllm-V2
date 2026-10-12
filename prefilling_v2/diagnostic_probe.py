"""Process-local instrumentation; never patches files in the live experiment."""
from __future__ import annotations

import math
from contextlib import ExitStack
from unittest.mock import patch

import torch

from prefilling_v2.diagnostic_helpers import (
    coverage_metrics, dense_reference, force_chunks, force_tokens, physical_positions, support_mask,
)
from prefilling_v2.kernel_utils import chunk_aligned_local_bounds
from prefilling_v2.token_refinement import candidate_token_scores_reference
from prefilling_v2.rank_audit import evidence_rank_audit
from prefilling_v2.routing_score_decomposition import decompose_chunk_scores


def summarize_query_route_collapse(
    queries: torch.Tensor,
    indices: torch.Tensor,
    evidence_chunks: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    """Summarize similarity across query positions without retaining tensors."""

    if queries.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "expected queries [Q,H,D] and indices [Q,Hkv,K], got "
            f"{tuple(queries.shape)} and {tuple(indices.shape)}"
        )
    if queries.shape[0] != indices.shape[0] or queries.shape[0] < 2:
        raise ValueError("route-collapse metrics require at least two query positions")

    pairs = torch.triu_indices(
        queries.shape[0], queries.shape[0], offset=1, device=queries.device
    )
    normalized = torch.nn.functional.normalize(queries.float(), dim=-1)
    same_head_cosine = (
        normalized[pairs[0]] * normalized[pairs[1]]
    ).sum(dim=-1).flatten()
    flattened = torch.nn.functional.normalize(
        queries.float().flatten(1), dim=-1
    )
    flattened_cosine = (
        flattened[pairs[0]] * flattened[pairs[1]]
    ).sum(dim=-1)

    selected = indices.detach().cpu()
    jaccards: list[float] = []
    exact_matches = 0
    comparisons = 0
    for left, right in zip(pairs[0].cpu().tolist(), pairs[1].cpu().tolist()):
        for head in range(selected.shape[1]):
            a = {int(value) for value in selected[left, head].tolist() if value >= 0}
            b = {int(value) for value in selected[right, head].tolist() if value >= 0}
            union = a | b
            jaccards.append(len(a & b) / len(union) if union else 1.0)
            exact_matches += int(a == b)
            comparisons += 1

    unique_per_head = []
    for head in range(selected.shape[1]):
        union = {
            int(value)
            for row in range(selected.shape[0])
            for value in selected[row, head].tolist()
            if value >= 0
        }
        unique_per_head.append(float(len(union)))

    jaccard_tensor = torch.tensor(jaccards, dtype=torch.float32)
    unique_tensor = torch.tensor(unique_per_head, dtype=torch.float32)
    result = {
        "route_query_same_head_cosine_mean": float(same_head_cosine.mean()),
        "route_query_same_head_cosine_p10": float(
            torch.quantile(same_head_cosine, 0.1)
        ),
        "route_query_same_head_cosine_p90": float(
            torch.quantile(same_head_cosine, 0.9)
        ),
        "route_query_flat_cosine_mean": float(flattened_cosine.mean()),
        "route_query_flat_cosine_p10": float(
            torch.quantile(flattened_cosine, 0.1)
        ),
        "route_query_flat_cosine_p90": float(
            torch.quantile(flattened_cosine, 0.9)
        ),
        "route_topk_between_query_jaccard_mean": float(jaccard_tensor.mean()),
        "route_topk_between_query_jaccard_p10": float(
            torch.quantile(jaccard_tensor, 0.1)
        ),
        "route_topk_between_query_jaccard_p90": float(
            torch.quantile(jaccard_tensor, 0.9)
        ),
        "route_topk_between_query_exact_fraction": exact_matches / comparisons,
        "route_unique_chunks_per_kv_head_mean": float(unique_tensor.mean()),
        "route_unique_chunks_per_kv_head_min": float(unique_tensor.min()),
        "route_unique_chunks_per_kv_head_max": float(unique_tensor.max()),
    }
    if evidence_chunks is None:
        return result

    evidence = {
        int(value)
        for value in evidence_chunks.detach().cpu().flatten().tolist()
        if value >= 0
    }
    if not evidence:
        raise ValueError("evidence-aware route metrics require evidence chunks")

    query_head_sets = [
        [
            {
                int(value)
                for value in selected[row, head].tolist()
                if value >= 0
            }
            for head in range(selected.shape[1])
        ]
        for row in range(selected.shape[0])
    ]
    query_sets = [set().union(*heads) for heads in query_head_sets]
    evidence_list = sorted(evidence)
    query_head_chunk_hits = [
        [
            [int(chunk in query_head_sets[row][head]) for chunk in evidence_list]
            for head in range(selected.shape[1])
        ]
        for row in range(selected.shape[0])
    ]
    query_chunk_hits = [
        [
            int(any(query_head_chunk_hits[row][head][chunk_index]
                    for head in range(selected.shape[1])))
            for chunk_index in range(len(evidence_list))
        ]
        for row in range(selected.shape[0])
    ]
    strict_hits = sum(
        hit
        for rows in query_head_chunk_hits
        for heads in rows
        for hit in heads
    )
    strict_units = selected.shape[0] * selected.shape[1] * len(evidence_list)
    query_chunk_hit_count = sum(hit for row in query_chunk_hits for hit in row)
    query_chunk_units = selected.shape[0] * len(evidence_list)
    query_any_hits = [int(any(row)) for row in query_chunk_hits]
    query_all_hits = [int(all(row)) for row in query_chunk_hits]
    union_chunk_hits = [
        int(any(query_chunk_hits[row][chunk_index]
                for row in range(selected.shape[0])))
        for chunk_index in range(len(evidence_list))
    ]

    missed_query_pairs = []
    missed_rows = [row for row, hit in enumerate(query_any_hits) if not hit]
    for left_index, left in enumerate(missed_rows):
        for right in missed_rows[left_index + 1:]:
            a = query_sets[left] - evidence
            b = query_sets[right] - evidence
            union = a | b
            missed_query_pairs.append(len(a & b) / len(union) if union else 1.0)

    missed_same_head_pairs = []
    for head in range(selected.shape[1]):
        missed = [
            row
            for row in range(selected.shape[0])
            if not (query_head_sets[row][head] & evidence)
        ]
        for left_index, left in enumerate(missed):
            for right in missed[left_index + 1:]:
                a = query_head_sets[left][head] - evidence
                b = query_head_sets[right][head] - evidence
                union = a | b
                missed_same_head_pairs.append(
                    len(a & b) / len(union) if union else 1.0
                )

    result.update(
        {
            "route_evidence_chunk_count": len(evidence_list),
            "route_evidence_query_kv_hits": strict_hits,
            "route_evidence_query_kv_units": strict_units,
            "route_evidence_query_kv_hit_rate": strict_hits / strict_units,
            "route_evidence_query_chunk_hits": query_chunk_hit_count,
            "route_evidence_query_chunk_units": query_chunk_units,
            "route_evidence_query_chunk_recall": (
                query_chunk_hit_count / query_chunk_units
            ),
            "route_evidence_query_any_chunk_hits": sum(query_any_hits),
            "route_evidence_query_count": selected.shape[0],
            "route_evidence_query_any_chunk_hit_rate": (
                sum(query_any_hits) / selected.shape[0]
            ),
            "route_evidence_query_all_chunks_hits": sum(query_all_hits),
            "route_evidence_query_all_chunks_hit_rate": (
                sum(query_all_hits) / selected.shape[0]
            ),
            "route_evidence_union_chunk_hits": sum(union_chunk_hits),
            "route_evidence_union_chunk_units": len(evidence_list),
            "route_evidence_union_chunk_recall": (
                sum(union_chunk_hits) / len(evidence_list)
            ),
            "route_evidence_union_all_chunks_hit": int(all(union_chunk_hits)),
            "route_evidence_all_queries_miss": int(not any(query_any_hits)),
            "route_evidence_missed_query_pair_wrong_jaccard_sum": sum(
                missed_query_pairs
            ),
            "route_evidence_missed_query_pair_count": len(missed_query_pairs),
            "route_evidence_missed_query_pair_wrong_jaccard_mean": (
                sum(missed_query_pairs) / len(missed_query_pairs)
                if missed_query_pairs
                else None
            ),
            "route_evidence_missed_same_head_pair_wrong_jaccard_sum": sum(
                missed_same_head_pairs
            ),
            "route_evidence_missed_same_head_pair_count": len(
                missed_same_head_pairs
            ),
            "route_evidence_missed_same_head_pair_wrong_jaccard_mean": (
                sum(missed_same_head_pairs) / len(missed_same_head_pairs)
                if missed_same_head_pairs
                else None
            ),
        }
    )
    return result


class RoutingProbe:
    def __init__(self, decoder, *, max_queries=16):
        self.decoder = decoder
        self.max_queries = max_queries
        self.records = []
        self.step = -1
        self.variant = "baseline"
        self.observation_steps = {0, 1, 4, 8, 16, 31}
        self.stack = ExitStack()

    def begin(self, case, layout, metadata, variant="baseline", observe=True):
        self.metadata = metadata
        self.variant = variant
        self.observe = observe
        self.step = -1
        self.records = []
        self.last_routes = {}
        self.predictors = layout.predictor_positions
        self.answers = layout.answer_positions
        self.key_valid = layout.attention_mask.bool().clone()
        if layout.landmark_positions.numel():
            self.key_valid[layout.landmark_positions] = False
            self.evidence = [physical_positions(torch.tensor(v), 64) for v in case.evidence_values]
            self.facts = [physical_positions(torch.tensor(v), 64) for v in case.evidence_facts]
        else:
            self.evidence = [torch.as_tensor(v, dtype=torch.long) for v in case.evidence_values]
            self.facts = [torch.as_tensor(v, dtype=torch.long) for v in case.evidence_facts]
        self.pending = None
        self.route_summary = {}
        self.chunk_priority = None
        self.route_components = None
        self.gate_update = None
        self.interventions = {"changed_chunk_slots": 0, "changed_token_slots": 0}

    def _enter_ids(self, ids):
        self.step += 1
        mask = ids[0, self.answers.to(ids.device)] == self.decoder.mask_token_id
        self.active = self.predictors.to(ids.device)[mask]

    def _enter_layer(self, module, positions):
        self.layer = module._diagnostic_layer_id
        self.module = module
        self.positions = positions
        self.block_offset = 0
        self.pending = None
        self.gate_update = None
        self.route_summary = {}
        self.chunk_priority = None
        self.route_components = None

    def _rows(self, positions=None, *, sampled=False):
        positions = self.positions if positions is None else positions
        rows = torch.where(torch.isin(positions, self.active))[0]
        if sampled and rows.numel() > self.max_queries:
            select = torch.linspace(0, rows.numel() - 1, self.max_queries, device=rows.device).long()
            rows = rows[select]
        return rows

    def _observe_now(self):
        return self.observe and self.step in self.observation_steps

    def __enter__(self):
        from prefilling_v2 import routing, token_refinement, fastdllm_attention, fastdllm_v1
        from prefilling_v2.dsa_attention import DreamDsaAttention
        from prefilling_v2.nsa_attention import DreamNsaAttention
        from ops import bidir_output_fusion_tilelang as fusion
        for i, layer in enumerate(self.decoder._core().model.layers):
            module = layer.self_attn
            module._diagnostic_layer_id = i
            def prehook(attn, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                self._enter_layer(attn, torch.arange(hidden.shape[1], device=hidden.device))
            handle = module.register_forward_pre_hook(prehook, with_kwargs=True)
            self.stack.callback(handle.remove)
        for name in ("prefill", "cached_forward"):
            original = getattr(self.decoder, name)
            def call(ids, *args, _original=original, **kwargs):
                self._enter_ids(ids)
                return _original(ids, *args, **kwargs)
            self.stack.enter_context(patch.object(self.decoder, name, call))
        original_cached = fastdllm_v1.cached_attention_forward
        def cached(module, hidden, **kwargs):
            self._enter_layer(module, kwargs["query_positions"])
            return original_cached(module, hidden, **kwargs)
        self.stack.enter_context(patch.object(fastdllm_v1, "cached_attention_forward", cached))
        original_route = routing.route_topk_g7
        def route(*args, **kwargs):
            return self._route(original_route, args, kwargs)
        for mod in (routing, fastdllm_attention):
            self.stack.enter_context(patch.object(mod, "route_topk_g7", route))
        original_tokens = token_refinement.refine_remote_tokens_g7
        def tokens(*args, **kwargs):
            return self._tokens(original_tokens, args, kwargs)
        for mod in (token_refinement, fastdllm_attention):
            self.stack.enter_context(patch.object(mod, "refine_remote_tokens_g7", tokens))
        for mod, name in ((routing, "selected_attention_g7"), (fastdllm_attention, "selected_attention_g7_cached")):
            original = getattr(mod, name)
            def selected(*args, _original=original, **kwargs):
                return self._route_selected(_original, args, kwargs)
            self.stack.enter_context(patch.object(mod, name, selected))
        original_fuse = fusion.fuse_route_outputs
        def fuse(remote, local, weight):
            raw_weight = weight
            if self.gate_update is not None:
                rows, updated = self.gate_update
                weight = weight.clone()
                weight.index_copy_(1, rows, updated.to(weight.dtype))
            output = original_fuse(remote, local, weight)
            if self.pending is not None:
                record, rows, teacher = self.pending
                actual = output[0].index_select(0, rows).float()
                record["attention_output_relative_l2"] = float((actual - teacher).norm() / teacher.norm().clamp_min(1e-8))
                record["remote_gate_mean"] = float(1 - weight[0, rows].float().mean())
                record["original_remote_gate_mean"] = float(1 - raw_weight[0, rows].float().mean())
                self.pending = None
            return output
        self.stack.enter_context(patch.object(fusion, "fuse_route_outputs", fuse))
        original_dsa = DreamDsaAttention._selected_attention
        def dsa(attn, q, k, v, indices, valid):
            return self._dsa_selected(original_dsa, attn, q, k, v, indices, valid)
        self.stack.enter_context(patch.object(DreamDsaAttention, "_selected_attention", dsa))
        original_nsa = DreamNsaAttention._selected_attention
        def nsa(attn, q, k, v, indices, valid):
            return self._nsa_selected(original_nsa, attn, q, k, v, indices, valid)
        self.stack.enter_context(patch.object(DreamNsaAttention, "_selected_attention", nsa))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _route(self, original, args, kwargs):
        result = original(*args, **kwargs)
        q, lmks, local_lse, bias, dropped = args[:5]
        indices, scores = result[:2]
        if self.variant == "lmk_only_route":
            rows = self._rows()
            if rows.numel():
                h, g = lmks.shape[2:4]
                raw = torch.einsum("brhgd,bchgd->brhgc", q[:, rows].float().reshape(1, rows.numel(), h, g, q.shape[-1]), lmks.float()) / math.sqrt(q.shape[-1])
                lmk_lse = torch.logaddexp(local_lse[0, rows].float(), torch.logsumexp(raw[0], dim=-1))
                priority = (raw[0] - lmk_lse[..., None]).amax(2).masked_fill(dropped[0, rows, None].bool(), float("-inf"))
                selected = torch.topk(priority, k=indices.shape[-1], dim=-1, sorted=False).indices.to(indices.dtype)
                valid = torch.isfinite(torch.gather(priority, -1, selected.long()))
                selected = selected.masked_fill(~valid, -1)
                gather = selected.long().clamp_min(0)[:, :, None].expand(-1, -1, g, -1)
                selected_raw = torch.gather(raw[0], -1, gather).reshape(rows.numel(), h * g, -1)
                selected_raw.masked_fill_(selected.repeat_interleave(g, 1) < 0, float("-inf"))
                self.interventions["changed_chunk_slots"] += int((selected != indices[0, rows]).sum())
                indices[0, rows] = selected
                scores[0, rows] = selected_raw.to(scores.dtype)
        if self.variant in {"oracle_chunk", "oracle_both"}:
            rows = self._rows()
            if rows.numel():
                h, g = lmks.shape[2:4]
                raw = torch.einsum("brhgd,bchgd->brhgc", q[:, rows].float().reshape(1, rows.numel(), h, g, q.shape[-1]), lmks.float()) / math.sqrt(q.shape[-1])
                logits = raw[0] + bias[0].permute(1, 2, 0)[None]
                total = torch.logaddexp(torch.logsumexp(logits, -1), local_lse[0, rows].float())
                priority = (logits - total[..., None]).amax(2)
                required = torch.unique(torch.cat(self.facts) // 64).tolist()
                changed = force_chunks(indices[0, rows], priority, required, dropped[0, rows].bool())
                self.interventions["changed_chunk_slots"] += int((changed != indices[0, rows]).sum())
                indices[0, rows] = changed
                gather = changed.long().clamp_min(0)[:, :, None].expand(-1, -1, g, -1)
                changed_scores = torch.gather(raw[0], -1, gather).reshape(rows.numel(), h * g, -1)
                changed_scores.masked_fill_(changed.repeat_interleave(g, 1) < 0, float("-inf"))
                scores[0, rows] = changed_scores.to(scores.dtype)
        if self._observe_now():
            rows = self._rows(sampled=True)
            if rows.numel():
                if self.step == 0:
                    h, g = lmks.shape[2:4]
                    raw_rank = torch.einsum("brhgd,bchgd->brhgc", q[:, rows].float().reshape(1, rows.numel(), h, g, q.shape[-1]), lmks.float()) / math.sqrt(q.shape[-1])
                    rank_logits = raw_rank[0] + bias[0].permute(1, 2, 0)[None]
                    total = torch.logaddexp(torch.logsumexp(rank_logits, -1), local_lse[0, rows].float())
                    self.chunk_priority = (rank_logits - total[..., None]).amax(2).masked_fill(dropped[0, rows, None].bool(), float("-inf"))
                    self.route_components = dict(
                        rows=rows,
                        q=q[0, rows],
                        lmks=lmks[0],
                        local_lse=local_lse[0, rows],
                        bias=bias[0],
                        dropped=dropped[0, rows].bool(),
                    )
                self.route_summary = {
                    "landmark_norm_mean": float(lmks.float().norm(dim=-1).mean()),
                    "landmark_norm_max": float(lmks.float().norm(dim=-1).max()),
                    "prior_bias_abs_max": float(bias.float().abs().max()),
                    "finite_selected_score_fraction": float(torch.isfinite(scores[0, rows]).float().mean()),
                    "route_collapse_query_count": int(rows.numel()),
                }
                if rows.numel() >= 2:
                    self.route_summary.update(
                        summarize_query_route_collapse(
                            q[0, rows],
                            indices[0, rows],
                            torch.unique(torch.cat(self.evidence) // 64),
                        )
                    )
                overlaps = []
                for row in rows.tolist():
                    position = int(self.positions[row])
                    current = indices[0, row].detach().cpu()
                    key = (self.layer, position)
                    previous = self.last_routes.get(key)
                    if previous is not None:
                        for a, b in zip(current.tolist(), previous.tolist()):
                            sa, sb = {x for x in a if x >= 0}, {x for x in b if x >= 0}
                            overlaps.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
                    self.last_routes[key] = current
                self.route_summary["route_jaccard_previous_observation"] = sum(overlaps) / len(overlaps) if overlaps else None
        return result

    def _tokens(self, original, args, kwargs):
        keep = original(*args, **kwargs)
        if self.variant not in {"oracle_token", "oracle_both"}:
            return keep
        rows = self._rows()
        if rows.numel():
            q, k, indices, valid, size = args[:5]
            scores = candidate_token_scores_reference(q[:, rows], k, indices[:, rows], valid, size)
            pos = indices[:, rows].long()[..., None] * size + torch.arange(size, device=q.device)
            fixed = force_tokens(keep[:, rows], scores, pos, torch.cat(self.facts).to(q.device))
            self.interventions["changed_token_slots"] += int((fixed != keep[:, rows]).sum())
            keep[0, rows] = fixed[0]
        return keep

    def _local(self, positions, n, heads, valid):
        left, right = chunk_aligned_local_bounds(n, self.module.local_window, 64, positions.device)
        keys = torch.arange(n, device=positions.device)
        mask = (keys[None] >= left[positions, None]) & (keys[None] < right[positions, None]) & valid[None]
        if getattr(self.module, "skip_inert_slots", True):
            mask &= keys[None].remainder(64) != 63
        return mask[:, None].expand(-1, heads, -1)

    def _record(self, q, k, v, positions, before, after, local, valid, *, kind, route_rows=None):
        scores, probs = dense_reference(q, k, valid)
        values = [x.to(q.device) for x in self.evidence]
        record = dict(self.metadata, layer=self.layer, forward_index=self.step, model_kind=kind,
                      variant=self.variant, query_count=q.shape[0], query_positions=positions.tolist())
        record.update(coverage_metrics(probs, before, after, local, values))
        for name, mask in (("before", before | local), ("after", after | local)):
            hits = torch.stack([mask.index_select(-1, p.to(q.device)).all(-1) for p in self.facts])
            record[f"complete_fact_recall_{name}"] = float(hits.float().mean())
            record[f"all_complete_facts_{name}"] = float(hits.all(0).float().mean())
        record.update(self.route_summary)
        if kind == "route" and self.step == 0:
            record["rank_audit"] = evidence_rank_audit(
                scores.amax(2), before, after, local, values, getattr(self, "chunk_priority", None),
                budget=getattr(self.module, "token_budget", 512), chunk_size=64)
            record["rank_audit"]["rank_source"] = "FP32 reference priority; missed flags use actual native support"
            for item in record["rank_audit"]["examples"]:
                item["query_position"] = int(positions[item["query_row"]])
            components = getattr(self, "route_components", None)
            if components is not None:
                if route_rows is None or not torch.equal(components["rows"], route_rows):
                    raise AssertionError("routing-component rows do not match selected-attention rows")
                record["score_decomposition"] = decompose_chunk_scores(
                    components["q"], components["lmks"], components["local_lse"], components["bias"],
                    components["dropped"], k, valid, self.facts, chunk_size=64, token_q=q)
        retained = after | local
        teacher = torch.einsum("qhgn,nhd->qhgd", probs, v.float()).flatten(1, 2)
        restricted = probs * retained[:, :, None]
        restricted /= restricted.sum(-1, keepdim=True).clamp_min(1e-20)
        flat = torch.einsum("qhgn,nhd->qhgd", restricted, v.float()).flatten(1, 2)
        record["flat_selected_output_relative_l2"] = float((flat - teacher).norm() / teacher.norm().clamp_min(1e-8))
        record["kv_group_token_pairs_mean"] = float(retained.sum((-1, -2)).float().mean())
        record["ideal_retained_remote_gate_mean"] = float((restricted * (after & ~local)[:, :, None]).sum(-1).mean())
        self.records.append(record)
        return record, teacher

    def _route_selected(self, original, args, kwargs):
        q, k, v, weights, indices, valid, size = args[:7]
        text_valid = valid[0].bool() & (torch.arange(k.shape[1], device=k.device).remainder(size) != size - 1)
        keep = kwargs.get("token_keep")
        rows = self._rows()
        if self.variant == "flat_gate" and rows.numel():
            local = self._local(self.positions[rows], k.shape[1], k.shape[2], text_valid)
            logits, _ = dense_reference(q[0, rows], k[0], text_valid)
            pos = indices[0, rows].long()[..., None] * size + torch.arange(size, device=q.device)
            allowed = (indices[0, rows, :, :, None] >= 0) & valid[0, pos.clamp(0, k.shape[1] - 1)].bool()
            allowed &= pos.remainder(size) != size - 1
            if keep is not None:
                allowed &= keep[0, rows].bool()
            gathered = torch.gather(logits, -1, pos.clamp(0, k.shape[1] - 1).flatten(-2)[:, :, None].expand(-1, -1, logits.shape[2], -1))
            gathered = gathered.reshape(*logits.shape[:3], indices.shape[-1], size)
            remote_lse = torch.logsumexp(gathered.masked_fill(~allowed[:, :, None], float("-inf")), -1)
            local_lse = torch.logsumexp(logits.masked_fill(~local[:, :, None], float("-inf")), -1)
            gate = torch.softmax(torch.cat((remote_lse, local_lse[..., None]), -1), -1)
            weights = weights.clone()
            weights[0, rows] = gate[..., :-1].flatten(1, 2).to(weights.dtype)
            self.gate_update = (rows, gate[..., -1].flatten(1, 2)[None])
            args = (*args[:3], weights, *args[4:])
        output = original(*args, **kwargs)
        if self._observe_now():
            rows = self._rows(sampled=True)
            if rows.numel():
                before = support_mask(indices[0, rows], text_valid, size)
                after = support_mask(indices[0, rows], text_valid, size, keep[0, rows] if keep is not None else None)
                local = self._local(self.positions[rows], k.shape[1], k.shape[2], text_valid)
                record, teacher = self._record(
                    q[0, rows], k[0], v[0], self.positions[rows], before, after, local, text_valid,
                    kind="route", route_rows=rows,
                )
                self.pending = (record, rows, teacher)
        return output

    def _dsa_selected(self, original, module, q, k, v, indices, valid):
        output = original(module, q, k, v, indices, valid)
        positions = self.positions[self.block_offset:self.block_offset + q.shape[1]]
        self.block_offset += q.shape[1]
        if self._observe_now():
            rows = self._rows(positions, sampled=True)
            if rows.numel():
                selected = torch.zeros(rows.numel(), k.shape[1], device=q.device, dtype=torch.int32)
                selected.scatter_add_(-1, indices[0, rows].long(), valid[0, rows].int())
                selected = selected.bool()[:, None].expand(-1, k.shape[2], -1)
                key_valid = self.key_valid.to(q.device)
                record, teacher = self._record(q[0, rows], k[0], v[0], positions[rows], selected, selected, torch.zeros_like(selected), key_valid, kind="dsa")
                record["attention_output_relative_l2"] = float((output[0, rows].float() - teacher).norm() / teacher.norm().clamp_min(1e-8))
        return output

    def _nsa_selected(self, original, module, q, k, v, indices, valid):
        output = original(module, q, k, v, indices, valid)
        positions = self.positions[self.block_offset:self.block_offset + q.shape[1]]
        self.block_offset += q.shape[1]
        if self._observe_now():
            rows = self._rows(positions, sampled=True)
            if rows.numel():
                length = k.shape[1]
                selected = torch.zeros(
                    rows.numel(), k.shape[2], length, device=q.device, dtype=torch.bool,
                )
                selected.scatter_(
                    -1,
                    indices[0, rows].long().clamp(0, length - 1),
                    valid[0, rows].bool(),
                )
                key_valid = self.key_valid.to(q.device)
                local = self._local(positions[rows], length, k.shape[2], key_valid)
                record, teacher = self._record(
                    q[0, rows], k[0], v[0], positions[rows], selected, selected, local, key_valid,
                    kind="nsa",
                )
                record["attention_output_relative_l2"] = float(
                    (output[0, rows].float() - teacher).norm() / teacher.norm().clamp_min(1e-8)
                )
        return output
