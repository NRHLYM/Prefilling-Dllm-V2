"""Small, CPU-testable helpers for controlled sparse-retrieval diagnostics."""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import torch


def diagnostic_lengths(phase: str, requested: list[int] | None) -> list[int]:
    if requested is not None:
        if not requested or any(length <= 0 or length % 64 for length in requested):
            raise ValueError("diagnostic lengths must be positive multiples of 64")
        return list(dict.fromkeys(requested))
    return [32768] if phase == "generate" else [2048, 8192, 16384, 32768]


def generation_health(records: list[dict]) -> dict:
    if not records:
        raise ValueError("generation health requires records")
    groups = {}
    for record in records:
        f1 = record["code_f1"]
        if not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError("invalid code F1")
        key = (record["identity"]["model"], record["variant"])
        group = groups.setdefault(key, {"model": key[0], "variant": key[1], "n": 0,
                                        "empty_predictions": 0, "positive_code_f1": 0})
        group["n"] += 1
        group["empty_predictions"] += not record["prediction"].strip()
        group["positive_code_f1"] += f1 > 0
    floor = all(group["positive_code_f1"] == 0 for group in groups.values())
    return {"all_groups_zero_code_f1": floor, "groups": [groups[key] for key in sorted(groups)],
            "warning": ("No generation-quality contrast is available. Execution success is not a positive-control pass; "
                        "do not infer intervention ineffectiveness from this floor.") if floor else None}


def observer_error_limit(native_relative_l2: list[float]) -> float:
    """Bound observer drift against measured repeats, with absolute guardrails."""
    if not native_relative_l2 or any(not math.isfinite(x) or x < 0 for x in native_relative_l2):
        raise ValueError("invalid native repeatability measurements")
    noise = max(native_relative_l2)
    if noise > 0.02:
        raise AssertionError(f"native forward is too unstable for this smoke: relative_l2={noise}")
    return max(1e-6, min(0.03, 2 * noise))


@dataclass
class RetrievalCase:
    prompt_ids: list[int]
    evidence_values: list[list[int]]
    evidence_facts: list[list[int]]
    question_start: int
    answers: list[str]
    task: str
    seed: int
    depth: float


def make_case(tokenizer, filler: list[int], prompt_length: int, *, task: str,
              seed: int, depth: float) -> RetrievalCase:
    if task not in {"single", "multi", "chain"} or not 0 < depth < 1:
        raise ValueError("invalid retrieval case configuration")
    rng = random.Random(seed)
    names = [f"DIAG{seed}X{i}" for i in range(8)]
    codes = [str(x) for x in rng.sample(range(10000000, 99999999), 8)]
    encode = lambda text: list(tokenizer(text, add_special_tokens=False).input_ids)
    facts = [(names[i], codes[i]) for i in range(8)]
    if task == "chain":
        facts[0] = (names[0], names[1])
        question = f"\nFollow the reference from {names[0]} to its final numerical value. Output only that number. Answer:"
        answers, required = [codes[1]], {0, 1}
    elif task == "multi":
        question = f"\nWhat are the values of {names[0]} and {names[1]}, in that order? Output only the two numbers. Answer:"
        answers, required = codes[:2], {0, 1}
    else:
        question = f"\nWhat is the value of {names[0]}? Output only the number. Answer:"
        answers, required = codes[:1], {0}
    prefix = encode("Read the records and answer using the stated values.\n")
    question_ids = encode(question)
    encoded = []
    for name, value in facts:
        left, val, right = encode(f"\nThe value of {name} is "), encode(value), encode(".\n")
        encoded.append((left + val + right, len(left), len(val)))
    budget = prompt_length - len(prefix) - len(question_ids) - sum(len(x[0]) for x in encoded)
    if budget <= 0 or len(filler) < budget:
        raise ValueError("not enough filler or prompt capacity")
    positions = []
    for i in range(len(facts)):
        position = int(budget * (depth if i == 0 else (1 - depth if i == 1 and i in required else rng.random())))
        positions.append((position, i))
    out, cursor, values, spans = list(prefix), 0, [], []
    for position, i in sorted(positions):
        out.extend(filler[cursor:position])
        tokens, offset, size = encoded[i]
        start = len(out)
        out.extend(tokens)
        if i in required:
            values.append(list(range(start + offset, start + offset + size)))
            spans.append(list(range(start, start + len(tokens))))
        cursor = position
    out.extend(filler[cursor:budget])
    question_start = len(out)
    out.extend(question_ids)
    if len(out) != prompt_length:
        raise AssertionError("case length accounting failed")
    return RetrievalCase(out, values, spans, question_start, answers, task, seed, depth)


def physical_positions(logical: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return logical + torch.div(logical, chunk_size - 1, rounding_mode="floor")


def corrupt_background(case: RetrievalCase, ratio: float, mask_id: int) -> list[int]:
    if not 0 <= ratio <= 1:
        raise ValueError("invalid corruption probability")
    protected = set(range(case.question_start, len(case.prompt_ids)))
    protected.update(p for fact in case.evidence_facts for p in fact)
    rng = random.Random(case.seed ^ 0xDAC0)
    return [mask_id if rng.random() < ratio and i not in protected else token
            for i, token in enumerate(case.prompt_ids)]


def score_codes(prediction: str, answers: list[str]) -> dict[str, float]:
    found = re.findall(r"(?<!\d)\d{8}(?!\d)", prediction)
    gold = set(answers)
    hits = len(gold.intersection(found))
    precision = hits / len(set(found)) if found else 0.0
    recall = hits / len(gold)
    return {"code_exact_match": float(found == answers), "code_precision": precision,
            "code_recall": recall, "code_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def force_chunks(indices: torch.Tensor, priority: torch.Tensor, required: list[int],
                 dropped: torch.Tensor) -> torch.Tensor:
    """Replace the lowest-priority non-evidence slots without increasing K."""
    result = indices.clone()
    for row in range(indices.shape[0]):
        needed = [c for c in required if not bool(dropped[row, c])]
        for head in range(indices.shape[1]):
            old = indices[row, head].tolist()
            valid = [c for c in old if c >= 0]
            if len(set(valid)) != len(valid):
                raise ValueError("native chunk selection contains duplicates")
            if len(needed) > len(valid):
                raise ValueError("oracle evidence exceeds existing valid chunk budget")
            keep = set(valid)
            for c in needed:
                if c not in keep:
                    removable = [x for x in keep if x not in needed]
                    victim = min(removable, key=lambda x: float(priority[row, head, x]))
                    keep.remove(victim)
                    keep.add(c)
            result[row, head] = torch.tensor(sorted(keep) + [-1] * (len(old) - len(keep)), device=indices.device, dtype=indices.dtype)
    return result


def force_tokens(keep: torch.Tensor, scores: torch.Tensor, positions: torch.Tensor,
                 evidence: torch.Tensor) -> torch.Tensor:
    """Force available evidence tokens only, preserving each row/head count."""
    shape = keep.shape
    flat = keep.reshape(-1, shape[-2] * shape[-1]).bool()
    rank = scores.reshape_as(flat).float()
    pos = positions.reshape_as(flat)
    required = torch.isin(pos, evidence) & torch.isfinite(rank)
    result = flat.clone()
    for i in range(flat.shape[0]):
        budget = int(flat[i].sum())
        if int(required[i].sum()) > budget:
            raise ValueError("oracle evidence exceeds token budget")
        missing = torch.where(required[i] & ~flat[i])[0]
        if missing.numel():
            candidates = torch.where(flat[i] & ~required[i])[0]
            removed = candidates[torch.topk(rank[i, candidates], missing.numel(), largest=False).indices]
            result[i, removed] = False
            result[i, missing] = True
    return result.reshape(shape).to(keep.dtype)


def support_mask(indices: torch.Tensor, key_valid: torch.Tensor, chunk_size: int,
                 keep: torch.Tensor | None = None) -> torch.Tensor:
    """Convert [Q,Hkv,K] chunk ids into physical-token membership."""
    q, h, _ = indices.shape
    n = key_valid.numel()
    pos = indices.long()[..., None] * chunk_size + torch.arange(chunk_size, device=indices.device)
    valid = (indices[..., None] >= 0) & key_valid[pos.clamp(0, n - 1)]
    valid &= pos.remainder(chunk_size) != chunk_size - 1
    if keep is not None:
        valid &= keep.bool()
    out = torch.zeros(q, h, n, dtype=torch.int32, device=indices.device)
    out.scatter_add_(-1, pos.clamp(0, n - 1).flatten(-2), valid.int().flatten(-2))
    return out.bool()


def dense_reference(q: torch.Tensor, k: torch.Tensor, valid: torch.Tensor):
    heads, groups = k.shape[1], q.shape[1] // k.shape[1]
    scores = torch.einsum("qhgd,nhd->qhgn", q.float().reshape(q.shape[0], heads, groups, q.shape[-1]), k.float()) / math.sqrt(q.shape[-1])
    scores.masked_fill_(~valid[None, None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return scores, probs


def coverage_metrics(probs: torch.Tensor, before: torch.Tensor, after: torch.Tensor,
                     local: torch.Tensor, evidence_values: list[torch.Tensor]) -> dict:
    def mass(mask):
        return float((probs * mask[:, :, None]).sum(-1).mean())
    def hit(mask):
        return [mask.index_select(-1, positions).all(-1).float() for positions in evidence_values]
    before_hit, after_hit = hit(before | local), hit(after | local)
    joint_before, joint_after = torch.stack(before_hit).bool().all(0), torch.stack(after_hit).bool().all(0)
    b = torch.stack(before_hit)
    a = torch.stack(after_hit)
    denom = int(b.sum())
    remote_evidence = torch.stack([(~local.index_select(-1, p)).any(-1) for p in evidence_values])
    remote_denom = int(remote_evidence.sum())
    tokens = torch.unique(torch.cat(evidence_values))
    remote_token = ~local.index_select(-1, tokens)
    token_before = before.index_select(-1, tokens) & remote_token
    token_after = after.index_select(-1, tokens) & remote_token
    # One representative evidence token per chunk; coarse support covers its body.
    representatives = torch.stack([tokens[tokens.div(64, rounding_mode="floor") == c][0]
                                   for c in tokens.div(64, rounding_mode="floor").unique()])
    remote_chunk = ~local.index_select(-1, representatives)
    routing_counts = {
        "evidence_token_remote_denominator": int(remote_token.sum()),
        "evidence_token_candidate_hits": int(token_before.sum()),
        "evidence_token_retained_hits": int(token_after.sum()),
        "evidence_token_conditional_numerator": int((token_before & token_after).sum()),
        "evidence_token_conditional_denominator": int(token_before.sum()),
        "evidence_chunk_remote_denominator": int(remote_chunk.sum()),
        "evidence_chunk_candidate_hits": int((before.index_select(-1, representatives) & remote_chunk).sum()),
        "routing_unit": "active answer query x KV head x distinct evidence token/chunk; complete values reported separately",
    }
    return {**routing_counts, "candidate_mass": mass(before | local), "retained_mass": mass(after | local),
            "local_mass": mass(local), "remote_candidate_mass": mass(before & ~local),
            "remote_retained_mass": mass(after & ~local),
            "evidence_value_recall_before": float(b.mean()), "evidence_value_recall_after": float(a.mean()),
            "all_evidence_before": float(joint_before.float().mean()), "all_evidence_after": float(joint_after.float().mean()),
            "conditional_retention_numerator": int((a.bool() & b.bool()).sum()),
            "conditional_retention_denominator": denom,
            "conditional_retention": float((a.bool() & b.bool()).sum()) / denom if denom else None,
            "remote_evidence_denominator": remote_denom,
            "remote_evidence_recall_before": float((b.bool() & remote_evidence).sum()) / remote_denom if remote_denom else None,
            "remote_evidence_recall_after": float((a.bool() & remote_evidence).sum()) / remote_denom if remote_denom else None,
            "local_tokens_mean": float(local.sum(-1).float().mean()),
            "remote_candidates_mean": float((before & ~local).sum(-1).float().mean()),
            "remote_kept_mean": float((after & ~local).sum(-1).float().mean()),
            "actual_total_tokens_mean": float((after | local).sum(-1).float().mean()),
            "kv_head_union_tokens_mean": float((after | local).any(1).sum(-1).float().mean())}
