"""Fixed-state decomposition of route chunk-ranking signals."""
from __future__ import annotations

import math

import torch


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Membership mask with invalid chunks excluded; exact ties use torch.topk's choice."""
    k = min(k, scores.shape[-1])
    values, indices = torch.topk(scores, k, dim=-1, sorted=False)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, indices, torch.isfinite(values))
    return keep


def _component_metrics(scores: torch.Tensor, evidence_chunks: torch.Tensor, dropped: torch.Tensor) -> dict:
    """Summarize query x KV-head x remote-evidence-chunk ranking units."""
    rows, heads, chunks = scores.shape
    evidence_scores = scores.index_select(-1, evidence_chunks)
    remote = ~dropped.index_select(-1, evidence_chunks)[:, None].expand(-1, heads, -1)
    rank_best = 1 + (scores[..., None, :] > evidence_scores[..., None]).sum(-1)
    top16 = _topk_mask(scores, 16).index_select(-1, evidence_chunks)
    top31 = _topk_mask(scores, 31).index_select(-1, evidence_chunks)
    top32 = _topk_mask(scores, 32).index_select(-1, evidence_chunks)
    out = {
        "remote_evidence_chunk_units": int(remote.sum()),
        "top16_units": int((top16 & remote).sum()),
        "top31_units": int((top31 & remote).sum()),
        "top32_units": int((top32 & remote).sum()),
        "rank_sum": int(rank_best.masked_select(remote).sum()),
        "rank_le_16_units": int(((rank_best <= 16) & remote).sum()),
        "rank_17_31_units": int(((rank_best >= 17) & (rank_best <= 31) & remote).sum()),
        "rank_32_64_units": int(((rank_best >= 32) & (rank_best <= 64) & remote).sum()),
        "rank_gt_64_units": int(((rank_best > 64) & remote).sum()),
    }
    return out, top16, top31, remote


def decompose_chunk_scores(
    routing_q: torch.Tensor,
    landmark_keys: torch.Tensor,
    local_lse: torch.Tensor,
    prior_bias: torch.Tensor,
    dropped: torch.Tensor,
    key: torch.Tensor,
    text_valid: torch.Tensor,
    evidence_facts: list[torch.Tensor],
    *,
    chunk_size: int,
    token_q: torch.Tensor | None = None,
) -> dict:
    """Compare landmark, entropy-prior, and all-token upper-bound chunk rankings.

    Inputs are one batch's active query rows. The upper bound is intentionally
    dense over chunk tokens and is diagnostic-only: it cannot be an efficient
    production router. Every reported unit is one query x KV head x evidence
    chunk, so units from a single case are correlated.
    """
    if routing_q.ndim != 3 or landmark_keys.ndim != 4 or key.ndim != 3:
        raise ValueError("expected q [R,Hq,D], landmark [C,Hkv,G,D], key [N,Hkv,D]")
    rows, hq, dim = routing_q.shape
    chunks, hkv, groups, ldim = landmark_keys.shape
    if dim != ldim or hq != hkv * groups or key.shape != (chunks * chunk_size, hkv, dim):
        raise ValueError("incompatible routing score shapes")
    if local_lse.shape != (rows, hkv, groups) or prior_bias.shape != (chunks, hkv, groups):
        raise ValueError("invalid LSE or prior-bias shape")
    if dropped.shape != (rows, chunks) or text_valid.shape != (chunks * chunk_size,):
        raise ValueError("invalid chunk or token validity mask")
    if not evidence_facts:
        raise ValueError("at least one evidence fact is required")

    evidence_chunks = torch.unique(torch.cat(evidence_facts).to(routing_q.device) // chunk_size)
    evidence_chunks = evidence_chunks[(evidence_chunks >= 0) & (evidence_chunks < chunks)]
    if not evidence_chunks.numel():
        raise ValueError("no evidence chunk belongs to this sequence")
    q_grouped = routing_q.reshape(rows, hkv, groups, dim)
    raw = torch.einsum("rhgd,chgd->rhgc", q_grouped.float(), landmark_keys.float()) / math.sqrt(dim)
    bias = prior_bias.float().permute(1, 2, 0)[None]

    # These mirror the grouped max-pooling structure of routing. Per-group
    # normalizers matter because the final selector maximizes across GQA groups.
    lmk_lse = torch.logaddexp(local_lse.float(), torch.logsumexp(raw, dim=-1))
    combined_logits = raw + bias
    combined_lse = torch.logaddexp(local_lse.float(), torch.logsumexp(combined_logits, dim=-1))
    lmk_only = (raw - lmk_lse[..., None]).amax(dim=2)
    combined = (combined_logits - combined_lse[..., None]).amax(dim=2)
    prior_only = bias.amax(dim=2).expand(rows, -1, -1)

    token_q_grouped = q_grouped if token_q is None else token_q.reshape(rows, hkv, groups, dim)
    qk = torch.einsum("rhgd,nhd->rhgn", token_q_grouped.float(), key.float()) / math.sqrt(dim)
    qk = qk.masked_fill(~text_valid[None, None, None], float("-inf"))
    token_upper = qk.amax(dim=2).reshape(rows, hkv, chunks, chunk_size).amax(dim=-1)

    # Match the auxiliary route-KL teacher exactly: normalize native token QK
    # over the valid remote-token domain, then sum probability mass by chunk.
    remote_tokens = (~dropped).repeat_interleave(chunk_size, dim=-1)
    teacher_allowed = remote_tokens & text_valid[None]
    teacher_valid_rows = teacher_allowed.any(dim=-1)
    teacher_logits = qk.masked_fill(
        ~teacher_allowed[:, None, None], float("-inf")
    )
    teacher_logits = torch.where(
        teacher_valid_rows[:, None, None, None],
        teacher_logits,
        torch.zeros_like(teacher_logits),
    )
    teacher_mass = teacher_logits.softmax(dim=-1).reshape(
        rows, hkv, groups, chunks, chunk_size
    ).sum(dim=-1)
    teacher_mass = teacher_mass * (~dropped)[:, None, None]
    student_logits = combined_logits.masked_fill(
        dropped[:, None, None], torch.finfo(torch.float32).min
    )
    student_log_probs = student_logits.log_softmax(dim=-1)
    teacher_log_probs = teacher_mass.clamp_min(1e-30).log()
    teacher_student_kl = (
        teacher_mass * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    teacher_kl_valid = teacher_valid_rows[:, None, None].expand(
        -1, hkv, groups
    )
    teacher_student_kl_sum = teacher_student_kl.masked_select(
        teacher_kl_valid
    ).sum()
    teacher_student_kl_count = int(teacher_kl_valid.sum())

    components = {
        "lmk_only": lmk_only,
        "prior_only": prior_only,
        "lmk_plus_prior": combined,
        "token_qk_upper": token_upper,
    }
    metrics, masks = {}, {}
    for name, score in components.items():
        score = score.masked_fill(dropped[:, None], float("-inf"))
        metrics[name], top16, top31, remote = _component_metrics(score, evidence_chunks, dropped)
        masks[name] = (top16, top31, remote)

    teacher_gqa_max, _, _, _ = _component_metrics(
        teacher_mass.amax(dim=2), evidence_chunks, dropped
    )
    teacher_per_head, _, _, _ = _component_metrics(
        teacher_mass.reshape(rows, hkv * groups, chunks),
        evidence_chunks,
        dropped,
    )

    def transition(source: str, target: str, k: int) -> dict:
        source_mask, target_mask, remote = masks[source][0 if k == 16 else 1], masks[target][0 if k == 16 else 1], masks[source][2]
        return {
            f"{source}_to_{target}_loses_top{k}_units": int((source_mask & ~target_mask & remote).sum()),
            f"{source}_to_{target}_gains_top{k}_units": int((~source_mask & target_mask & remote).sum()),
        }

    effects = {}
    for k in (16, 31):
        effects.update(transition("lmk_only", "lmk_plus_prior", k))
        effects.update(transition("lmk_only", "token_qk_upper", k))
        effects.update(transition("lmk_plus_prior", "token_qk_upper", k))
    return {
        "unit": "query x KV head x remote evidence chunk",
        "chunk_count": chunks,
        "evidence_chunk_count": int(evidence_chunks.numel()),
        "components": metrics,
        "dense_teacher": {
            "gqa_max": teacher_gqa_max,
            "per_query_head": teacher_per_head,
            "gqa_max_unit": "query x KV head x remote evidence chunk",
            "per_query_head_unit": "query x query head x remote evidence chunk",
            "student_kl_sum": float(teacher_student_kl_sum),
            "student_kl_count": teacher_student_kl_count,
            "student_kl_mean": (
                float(teacher_student_kl_sum) / teacher_student_kl_count
                if teacher_student_kl_count
                else None
            ),
            "definition": (
                "native token-QK softmax on valid remote text tokens, summed "
                "within each chunk; this is the route-KL teacher target"
            ),
        },
        "effects": effects,
        "definitions": {
            "lmk_only": "Q dot pooled landmark, normalized per GQA group with local LSE; no entropy prior",
            "prior_only": "entropy-derived prior bias only; no Q dot landmark term",
            "lmk_plus_prior": "actual landmark routing score, including entropy prior and local normalization",
            "token_qk_upper": "maximum true QK over all valid text tokens in a remote chunk and the GQA query group; diagnostic dense upper bound",
        },
    }
