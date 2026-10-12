"""Counterfactual token ranks at one fixed attention state, not a dense teacher."""
import torch


def evidence_rank_audit(scores, before, after, local, evidence, priority, *, budget=512, chunk_size=64):
    """scores: [query, KV head, token], already max-reduced over GQA heads."""
    if scores.shape != before.shape or scores.shape != after.shape or scores.shape != local.shape:
        raise ValueError("support and score shapes must agree")
    remote = torch.isfinite(scores) & ~local
    candidate = remote & before
    full_scores = scores.masked_fill(~remote, float("-inf"))
    candidate_scores = scores.masked_fill(~candidate, float("-inf"))
    k = min(budget, scores.shape[-1])
    if k <= 0:
        full_cutoff = torch.full(scores.shape[:-1], float("inf"), device=scores.device, dtype=scores.dtype)
        candidate_cutoff = full_cutoff
    else:
        full_cutoff = full_scores.topk(k, dim=-1).values[..., -1]
        candidate_cutoff = candidate_scores.topk(k, dim=-1).values[..., -1]
    positions = torch.unique(torch.cat(evidence)).to(scores.device)
    value_scores = scores.index_select(-1, positions)
    is_remote = remote.index_select(-1, positions)
    missed = is_remote & ~before.index_select(-1, positions)
    # Strict comparison avoids crediting arbitrary top-k tie breaks.
    high = missed & (value_scores > full_cutoff[..., None])
    beats_candidate = missed & (value_scores > candidate_cutoff[..., None])
    result = dict(remote_evidence_token_units=int(is_remote.sum()),
                  missed_evidence_token_units=int(missed.sum()),
                  missed_strict_full_top_budget_units=int(high.sum()),
                  missed_beats_candidate_cutoff_units=int(beats_candidate.sum()),
                  budget=budget, unit="query x KV head x unique evidence token",
                  strict_ties=True, examples=[])
    for qi, hi, ei in torch.nonzero(high, as_tuple=False)[:8].tolist():
        pos = int(positions[ei])
        item = dict(query_row=qi, kv_head=hi, token_position=pos, chunk=pos // chunk_size,
                    token_score=float(value_scores[qi, hi, ei]),
                    full_top_budget_cutoff=float(full_cutoff[qi, hi]),
                    candidate_cutoff=float(candidate_cutoff[qi, hi]))
        if priority is not None:
            row = priority[qi, hi]
            value = row[pos // chunk_size]
            item.update(chunk_rank_best=1 + int((row > value).sum()),
                        chunk_rank_worst=int((row >= value).sum()))
        result["examples"].append(item)
    if priority is not None:
        chunk_scores = priority.index_select(-1, positions // chunk_size)
        ranks = 1 + (priority[..., None, :] > chunk_scores[..., None]).sum(-1)
        for lo, hi in ((1, 16), (17, 31), (32, 64), (65, priority.shape[-1])):
            if lo <= hi:
                result[f"remote_evidence_rank_{lo}_{hi}_units"] = int((is_remote & (ranks >= lo) & (ranks <= hi)).sum())
    return result
