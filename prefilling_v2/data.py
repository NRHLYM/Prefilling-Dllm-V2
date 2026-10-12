"""Answer-span denoising data utilities for Dream."""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

IGNORE_INDEX = -100
RULER_VIEW_ID_BASE = 100


def _randint_exclusive(high: int, generator: torch.Generator) -> int:
    if high <= 0:
        raise ValueError(f"randint high must be positive, got {high}")
    return int(torch.randint(0, high, (1,), generator=generator).item())


def _split_total_length(
    total: int,
    parts: int,
    *,
    min_part: int,
    generator: torch.Generator,
) -> list[int]:
    if parts < 1:
        raise ValueError("parts must be positive")
    if total < parts * min_part:
        raise ValueError(
            f"cannot split {total} into {parts} parts of at least {min_part}"
        )
    extra = total - parts * min_part
    cuts = [0]
    for _ in range(parts - 1):
        cuts.append(_randint_exclusive(extra + 1, generator))
    cuts.append(extra)
    cuts.sort()
    return [min_part + cuts[index + 1] - cuts[index] for index in range(parts)]


def _place_disjoint_spans(
    region_end: int,
    lengths: list[int],
    *,
    min_sep: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    free = [(0, int(region_end))]
    placed: list[tuple[int, int]] = []
    pad = max(int(min_sep), 0)
    for length in lengths:
        slots = [
            (index, start, end)
            for index, (start, end) in enumerate(free)
            if end - start >= length
        ]
        if not slots:
            raise ValueError("cannot place distant needles without overlap")
        weights = torch.tensor(
            [float(end - start - length + 1) for _, start, end in slots],
            dtype=torch.float,
        )
        choice = int(torch.multinomial(weights, 1, generator=generator).item())
        index, start, end = slots[choice]
        offset = _randint_exclusive(end - start - length + 1, generator)
        span_start = start + offset
        span_end = span_start + length
        placed.append((span_start, span_end))
        blocked_start = max(start, span_start - pad)
        blocked_end = min(end, span_end + pad)
        updated: list[tuple[int, int]] = []
        for free_index, (free_start, free_end) in enumerate(free):
            if free_index != index:
                updated.append((free_start, free_end))
                continue
            if blocked_start > free_start:
                updated.append((free_start, blocked_start))
            if free_end > blocked_end:
                updated.append((blocked_end, free_end))
        free = updated
    return placed


def synthesize_distant_span_copy(
    clean_ids: torch.Tensor,
    *,
    min_gap: int,
    span_min: int = 8,
    span_max: int = 16,
    generator: torch.Generator,
    num_needles: int = 1,
    min_needle_sep: int = 64,
    cue_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy prefix spans onto the suffix.

    The first ``cue_len`` tokens of each copy stay visible so the query can
    identify which remote span to complete; only the remainder is MASK'd.
    """

    if clean_ids.ndim != 1:
        raise ValueError("clean_ids must be 1D")
    total = int(clean_ids.numel())
    span_min = int(span_min)
    span_max = int(span_max)
    min_gap = int(min_gap)
    num_needles = int(num_needles)
    min_needle_sep = int(min_needle_sep)
    cue_len = int(cue_len)
    if min_gap < 1:
        raise ValueError("min_gap must be positive")
    if cue_len < 0:
        raise ValueError("cue_len must be nonnegative")
    if span_min < 1 or span_max < span_min:
        raise ValueError("invalid distant span bounds")
    if num_needles < 1:
        raise ValueError("num_needles must be positive")
    min_part = 8 if num_needles > 1 else span_min
    if cue_len:
        remainder = 8 if num_needles > 1 else 1
        min_part = max(min_part, cue_len + remainder)
    max_span = min(span_max, (total - min_gap - 1) // 2)
    if max_span < max(span_min, num_needles * min_part):
        raise ValueError(
            f"pack length {total} cannot host {num_needles} needles in "
            f"[{span_min},{span_max}] with min_gap={min_gap}"
        )
    total_span = int(
        torch.randint(
            max(span_min, num_needles * min_part),
            max_span + 1,
            (1,),
            generator=generator,
        ).item()
    )
    lengths = _split_total_length(
        total_span,
        num_needles,
        min_part=min_part,
        generator=generator,
    )
    answer_start = total - total_span
    region_end = answer_start - min_gap
    if region_end < total_span + max(0, num_needles - 1) * max(min_needle_sep, 0):
        raise ValueError("distant infill placement overflow")
    placed = _place_disjoint_spans(
        region_end,
        lengths,
        min_sep=min_needle_sep if num_needles > 1 else 0,
        generator=generator,
    )
    order = torch.randperm(num_needles, generator=generator).tolist()
    copies = []
    cue_flags = []
    for index in order:
        span = clean_ids[placed[index][0] : placed[index][1]]
        copies.append(span)
        flags = torch.ones(span.numel(), dtype=torch.bool)
        if cue_len:
            flags[: min(cue_len, span.numel())] = False
        cue_flags.append(flags)
    new_ids = clean_ids.clone()
    new_ids[answer_start:] = torch.cat(copies)
    target_mask = torch.zeros(total, dtype=torch.bool)
    target_mask[answer_start:] = torch.cat(cue_flags)
    target_mask[0] = False
    evidence_mask = torch.zeros(total, dtype=torch.bool)
    for start, end in placed:
        evidence_mask[start:end] = True
        if end + min_gap > answer_start:
            raise RuntimeError("distant evidence overlapped the MASK span")
    if bool((target_mask & evidence_mask).any()):
        raise RuntimeError("distant evidence overlapped the MASK span")
    return new_ids, target_mask, evidence_mask


def synthesize_one_token_key_value(
    clean_ids: torch.Tensor,
    *,
    min_gap: int,
    generator: torch.Generator,
    encode_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoising NIAH: far ``KEY=k VALUE=v``, near ``KEY=k VALUE=[MASK]``.

    Only the near VALUE token is MASK'd. The needle VALUE stays outside
    ``min_gap`` of that MASK. CE belongs on the MASK position, not a
    causal predecessor.
    """

    if clean_ids.ndim != 1:
        raise ValueError("clean_ids must be 1D")
    total = int(clean_ids.numel())
    min_gap = int(min_gap)
    if min_gap < 1:
        raise ValueError("min_gap must be positive")
    key_wrap = torch.as_tensor(encode_fn("KEY="), dtype=torch.long)
    val_wrap = torch.as_tensor(encode_fn(" VALUE="), dtype=torch.long)
    if key_wrap.ndim != 1 or val_wrap.ndim != 1 or key_wrap.numel() < 1 or val_wrap.numel() < 1:
        raise ValueError("KEY=/VALUE= wrappers must be nonempty 1D token ids")
    template = int(key_wrap.numel() + 1 + val_wrap.numel() + 1)
    if total < min_gap + 2 * template + 2:
        raise ValueError(
            f"pack length {total} cannot host KEY/VALUE NIAH with min_gap={min_gap}"
        )
    query_start = total - template
    region_end = query_start - min_gap
    if region_end < template + 1:
        raise ValueError("1-token KEY/VALUE placement overflow")
    pool = clean_ids[1:region_end]
    if pool.numel() < 2:
        raise ValueError("not enough tokens to sample KEY/VALUE")
    key_id = None
    value_id = None
    for _ in range(64):
        left = int(pool[_randint_exclusive(pool.numel(), generator)].item())
        right = int(pool[_randint_exclusive(pool.numel(), generator)].item())
        if left != right:
            key_id, value_id = left, right
            break
    if key_id is None or value_id is None:
        raise RuntimeError("could not sample a distinct KEY/VALUE pair")
    needle = torch.cat(
        (
            key_wrap,
            torch.tensor([key_id], dtype=torch.long),
            val_wrap,
            torch.tensor([value_id], dtype=torch.long),
        )
    )
    query = needle.clone()
    needle_start = 1 + _randint_exclusive(region_end - template, generator)
    new_ids = clean_ids.clone()
    new_ids[needle_start : needle_start + template] = needle
    new_ids[query_start:] = query
    target_mask = torch.zeros(total, dtype=torch.bool)
    target_mask[total - 1] = True
    evidence_mask = torch.zeros(total, dtype=torch.bool)
    evidence_mask[needle_start + template - 1] = True
    if int(new_ids[total - 1].item()) != value_id:
        raise RuntimeError("near VALUE was not planted")
    if int(new_ids[needle_start + template - 1].item()) != value_id:
        raise RuntimeError("far VALUE was not planted")
    if needle_start + template - 1 + min_gap > total - 1:
        raise RuntimeError("needle VALUE overlapped the local window")
    return new_ids, target_mask, evidence_mask


def _first_answer(record: Dict[str, Any]) -> str:
    answer = record.get("answer", None)
    if answer is None:
        answer = record.get("answers", "")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return str(answer)


def normalize_qa_record(record: Dict[str, Any]) -> Dict[str, str]:
    context = record.get("context") or record.get("passage") or record.get("document") or ""
    question = record.get("question") or record.get("input") or record.get("query") or ""
    return {"context": str(context), "question": str(question), "answer": _first_answer(record)}


def load_jsonl_records(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(normalize_qa_record(json.loads(line)))
            if limit is not None and len(records) >= limit:
                break
    return records


def build_synthetic_records(size: int = 128, distractors: int = 64) -> List[Dict[str, str]]:
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi"]
    cities = ["Paris", "Tokyo", "Berlin", "Nairobi", "Seoul", "Toronto", "Lima", "Zurich"]
    records = []
    for idx in range(size):
        target_name = names[idx % len(names)]
        target_city = cities[(idx * 3) % len(cities)]
        facts = []
        for j in range(distractors):
            name = names[(idx + j + 1) % len(names)]
            city = cities[(idx + j + 2) % len(cities)]
            facts.append(f"Record {j}: {name} lives in {city}.")
        insert_at = (idx * 17) % max(1, len(facts))
        facts.insert(insert_at, f"Key record: {target_name} lives in {target_city}.")
        records.append(
            {
                "context": "\n".join(facts),
                "question": f"Where does {target_name} live?",
                "answer": target_city,
            }
        )
    return records


class AnswerSpanDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict[str, str]], tokenizer, max_length: int, add_bos: bool = True):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.add_bos = add_bos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        prefix = f"Context:\n{record['context']}\n\nQuestion:\n{record['question']}\n\nAnswer:\n"
        answer = record["answer"] or "unknown"

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            answer_ids = answer_ids + [int(eos_id)]

        bos_ids: List[int] = []
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if self.add_bos and bos_id is not None:
            bos_ids = [int(bos_id)]

        min_answer = max(1, min(len(answer_ids), self.max_length - len(bos_ids)))
        answer_ids = answer_ids[:min_answer]
        prefix_budget = max(0, self.max_length - len(bos_ids) - len(answer_ids))
        if len(prefix_ids) > prefix_budget:
            prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget > 0 else []

        answer_start = len(bos_ids) + len(prefix_ids)
        clean_ids = bos_ids + prefix_ids + answer_ids
        answer_end = len(clean_ids)
        return {"clean_ids": clean_ids, "answer_start": answer_start, "answer_end": answer_end}


@dataclass
class AnswerSpanDenoisingCollator:
    mask_token_id: int
    pad_token_id: int
    lmk_token_id: int
    chunk_size: int = 64
    t_min: float = 0.2
    t_max: float = 0.8

    @property
    def real_per_chunk(self) -> int:
        return self.chunk_size - 1

    def _sample_mask(self, length: int) -> torch.Tensor:
        if length <= 0:
            return torch.zeros(0, dtype=torch.bool)
        t = random.uniform(self.t_min, self.t_max)
        mask = torch.rand(length) < t
        if not mask.any():
            mask[random.randrange(length)] = True
        if length > 1 and mask.all():
            mask[random.randrange(length)] = False
        return mask

    def _pad_orig(self, ids: List[int], padded_len: int) -> torch.Tensor:
        out = torch.full((padded_len,), int(self.pad_token_id), dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def _insert_landmarks(self, ids: torch.Tensor, orig_attention: torch.Tensor):
        if ids.numel() % self.real_per_chunk != 0:
            raise ValueError("ids must be padded to a multiple of chunk_size - 1 before inserting landmarks")
        chunks = ids.view(-1, self.real_per_chunk)
        attn_chunks = orig_attention.view(-1, self.real_per_chunk)
        num_chunks = chunks.shape[0]

        lmk = torch.full((num_chunks, 1), int(self.lmk_token_id), dtype=torch.long)
        lmk_attn = attn_chunks.any(dim=-1, keepdim=True).long()
        input_ids = torch.cat([chunks, lmk], dim=-1).reshape(-1)
        attention_mask = torch.cat([attn_chunks.long(), lmk_attn], dim=-1).reshape(-1)
        landmark_mask = torch.cat(
            [torch.zeros_like(attn_chunks, dtype=torch.bool), lmk_attn.bool()],
            dim=-1,
        ).reshape(-1)

        position_chunks = torch.arange(ids.numel(), dtype=torch.long).view(-1, self.real_per_chunk)
        lmk_pos = position_chunks[:, -1:] + 1
        position_ids = torch.cat([position_chunks, lmk_pos], dim=-1).reshape(-1)

        orig_to_lmk = torch.empty(ids.numel(), dtype=torch.long)
        for orig_idx in range(ids.numel()):
            orig_to_lmk[orig_idx] = orig_idx + orig_idx // self.real_per_chunk
        return input_ids, attention_mask, position_ids, orig_to_lmk, landmark_mask

    def _build_view(
        self,
        clean_ids: List[int],
        answer_start: int,
        answer_end: int,
        target_mask: torch.Tensor,
        padded_orig_len: int,
    ) -> Dict[str, torch.Tensor]:
        clean = self._pad_orig(clean_ids, padded_orig_len)
        corrupted = clean.clone()
        orig_attention = torch.zeros(padded_orig_len, dtype=torch.long)
        orig_attention[: len(clean_ids)] = 1

        answer_positions = torch.arange(answer_start, answer_end, dtype=torch.long)
        target_positions = answer_positions[target_mask]
        if target_positions.numel() > 0:
            corrupted[target_positions] = int(self.mask_token_id)

        input_ids, attention_mask, position_ids, orig_to_lmk, landmark_mask = self._insert_landmarks(corrupted, orig_attention)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        for orig_pos in target_positions.tolist():
            predictor_orig_pos = max(orig_pos - 1, 0)
            predictor_lmk_pos = int(orig_to_lmk[predictor_orig_pos].item())
            labels[predictor_lmk_pos] = int(clean[orig_pos].item())

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "landmark_mask": landmark_mask,
            "labels": labels,
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_orig_len = max(len(ex["clean_ids"]) for ex in examples)
        padded_orig_len = int(math.ceil(max_orig_len / self.real_per_chunk) * self.real_per_chunk)
        views: List[Dict[str, torch.Tensor]] = []

        for ex in examples:
            answer_start = int(ex["answer_start"])
            answer_end = int(ex["answer_end"])
            answer_len = max(0, answer_end - answer_start)
            mask = self._sample_mask(answer_len)
            views.append(self._build_view(ex["clean_ids"], answer_start, answer_end, mask, padded_orig_len))
            views.append(self._build_view(ex["clean_ids"], answer_start, answer_end, ~mask, padded_orig_len))

        keys = ("input_ids", "attention_mask", "position_ids", "landmark_mask", "labels")
        return {key: torch.stack([view[key] for view in views], dim=0) for key in keys}


class RulerDenoisingSynthesizer:
    """Build RULER-style retrieval prompts with explicit answer spans."""

    def __init__(
        self,
        tokenizer,
        *,
        task_ids: Sequence[int] = (0, 1, 2),
    ) -> None:
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        allowed = {0, 1, 2}
        unsupported = sorted(set(int(task_id) for task_id in task_ids) - allowed)
        if unsupported:
            raise ValueError(f"unsupported RULER task ids: {unsupported}")
        self.tokenizer = tokenizer
        self.task_ids = tuple(int(task_id) for task_id in task_ids)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("RULER denoising requires eos_token_id")
        self.eos_token_id = int(eos_token_id)

    def _encode(self, text: str) -> np.ndarray:
        return np.asarray(
            self.tokenizer(text, add_special_tokens=False).input_ids,
            dtype=np.int64,
        )

    def _rng(self, input_ids: np.ndarray, salt: int) -> np.random.RandomState:
        first = int(input_ids[0]) if input_ids.size else 0
        last = int(input_ids[-1]) if input_ids.size else 0
        seed = (
            first * 1103515245
            + last * 12345
            + int(salt) * 2654435761
        ) % (2**32 - 1)
        return np.random.RandomState(seed)

    def _random_code(
        self,
        rng: np.random.RandomState,
        *,
        length: int = 7,
    ) -> str:
        return str(int(rng.randint(10**length, 10 ** (length + 1) - 1)))

    def _random_name(
        self,
        rng: np.random.RandomState,
        *,
        length: int = 5,
    ) -> str:
        letters = rng.choice(list(string.ascii_uppercase), size=length)
        return "".join(str(letter) for letter in letters)

    def _unique_names(
        self,
        rng: np.random.RandomState,
        count: int,
        *,
        length: int = 5,
    ) -> list[str]:
        names: list[str] = []
        while len(names) < count:
            candidate = self._random_name(rng, length=length)
            if candidate not in names:
                names.append(candidate)
        return names

    def _unique_codes(
        self,
        rng: np.random.RandomState,
        count: int,
        *,
        length: int = 7,
    ) -> list[str]:
        codes: list[str] = []
        while len(codes) < count:
            candidate = self._random_code(rng, length=length)
            if candidate not in codes:
                codes.append(candidate)
        return codes

    def _compose(
        self,
        base_ids: np.ndarray,
        needles: list[np.ndarray],
        needle_evidence: list[np.ndarray],
        question_ids: np.ndarray,
        answer_ids: np.ndarray,
        rng: np.random.RandomState,
        answer_mask_ratio: float | None = None,
        haystack_mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(needles) != len(needle_evidence) or any(
            needle.shape != evidence.shape
            for needle, evidence in zip(needles, needle_evidence)
        ):
            raise ValueError("needle evidence masks must match needle tokens")
        total_len = int(base_ids.shape[0])
        required = (
            sum(int(needle.shape[0]) for needle in needles)
            + int(question_ids.shape[0])
            + int(answer_ids.shape[0])
        )
        if required >= total_len:
            raise ValueError(
                f"RULER prompt needs {required} tokens but pack has {total_len}"
            )
        filler_budget = total_len - required
        filler = base_ids[:filler_budget]
        order = list(range(len(needles)))
        rng.shuffle(order)
        positions = sorted(
            int(item)
            for item in rng.randint(0, filler_budget + 1, size=len(needles))
        )
        blocks: list[np.ndarray] = []
        evidence_blocks: list[np.ndarray] = []
        cursor = 0
        for needle_index, position in zip(order, positions):
            blocks.append(filler[cursor:position])
            evidence_blocks.append(
                np.zeros(position - cursor, dtype=np.bool_)
            )
            blocks.append(needles[needle_index])
            evidence_blocks.append(needle_evidence[needle_index])
            cursor = position
        blocks.append(filler[cursor:])
        evidence_blocks.append(
            np.zeros(filler_budget - cursor, dtype=np.bool_)
        )
        context_ids = np.concatenate(blocks) if blocks else filler
        context_evidence = (
            np.concatenate(evidence_blocks)
            if evidence_blocks
            else np.zeros_like(filler, dtype=np.bool_)
        )
        answer_start = int(context_ids.shape[0] + question_ids.shape[0])
        context_len = int(context_ids.shape[0])
        new_ids = np.concatenate((context_ids, question_ids, answer_ids))
        if new_ids.shape[0] != total_len:
            raise RuntimeError(
                f"RULER synthesis produced {new_ids.shape[0]} tokens, "
                f"expected {total_len}"
            )
        target_mask = torch.zeros(total_len, dtype=torch.bool)
        answer_n = int(answer_ids.shape[0])
        if answer_mask_ratio is None:
            target_mask[answer_start:] = True
        else:
            ratio = min(1.0, max(0.0, float(answer_mask_ratio)))
            kept = rng.rand(answer_n) < ratio
            if not bool(kept.any()):
                kept[int(rng.randint(0, answer_n))] = True
            target_mask[answer_start:] = torch.from_numpy(kept)
        if haystack_mask_ratio:
            ratio = min(1.0, max(0.0, float(haystack_mask_ratio)))
            haystack = np.ones(total_len, dtype=np.bool_)
            haystack[0] = False
            haystack[context_len:] = False
            haystack[:context_len] = ~np.asarray(context_evidence, dtype=np.bool_)
            target_mask |= torch.from_numpy(haystack & (rng.rand(total_len) < ratio))
        target_mask[0] = False
        evidence_mask = torch.zeros(total_len, dtype=torch.bool)
        evidence_mask[: context_evidence.shape[0]] = torch.from_numpy(
            context_evidence
        )
        if not bool(evidence_mask.any()):
            raise RuntimeError("RULER synthesis produced no evidence tokens")
        return (
            torch.as_tensor(new_ids, dtype=torch.long),
            target_mask,
            evidence_mask,
        )

    def synthesize(
        self,
        clean_ids: torch.Tensor,
        *,
        task_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean, target, _ = self.synthesize_with_evidence(
            clean_ids, task_id=task_id
        )
        return clean, target

    def synthesize_with_evidence(
        self,
        clean_ids: torch.Tensor,
        *,
        task_id: int,
        joint_mask: bool = False,
        t_min: float = 0.2,
        t_max: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_ids = clean_ids.detach().cpu().numpy().astype(np.int64)
        extra: dict[str, float] = {}
        if joint_mask:
            if not 0.0 < float(t_min) <= float(t_max) <= 1.0:
                raise ValueError("joint mask requires 0 < t_min <= t_max <= 1")
            mask_rng = self._rng(base_ids, salt=99)
            extra["answer_mask_ratio"] = float(mask_rng.uniform(t_min, t_max))
            extra["haystack_mask_ratio"] = float(mask_rng.uniform(t_min, t_max))
        if int(task_id) == 0:
            return self._single_needle(base_ids, **extra)
        if int(task_id) == 1:
            return self._multi_query(base_ids, **extra)
        if int(task_id) == 2:
            return self._variable_tracking(base_ids, **extra)
        raise ValueError(f"unsupported RULER task id: {task_id}")

    def sample_task_id(self, generator: torch.Generator) -> int:
        offset = int(
            torch.randint(
                0,
                len(self.task_ids),
                (),
                generator=generator,
            ).item()
        )
        return self.task_ids[offset]

    def _single_needle(
        self,
        base_ids: np.ndarray,
        *,
        length: int = 7,
        answer_mask_ratio: float | None = None,
        haystack_mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = self._rng(base_ids, salt=0)
        answer = self._random_code(rng, length=length)
        answer_value = self._encode(answer)
        answer_ids = np.concatenate(
            (answer_value, np.asarray([self.eos_token_id], dtype=np.int64))
        )
        needle_prefix = self._encode(
            " |One of the special magic numbers for long-context is: "
        )
        needle_suffix = self._encode("|")
        needle = np.concatenate((needle_prefix, answer_value, needle_suffix))
        evidence = np.concatenate(
            (
                np.zeros_like(needle_prefix, dtype=np.bool_),
                np.ones_like(answer_value, dtype=np.bool_),
                np.zeros_like(needle_suffix, dtype=np.bool_),
            )
        )
        question = self._encode(
            " What is the special magic number for long-context mentioned "
            "in the provided text? Answer: "
        )
        return self._compose(
            base_ids,
            [needle],
            [evidence],
            question,
            answer_ids,
            rng,
            answer_mask_ratio=answer_mask_ratio,
            haystack_mask_ratio=haystack_mask_ratio,
        )

    def _multi_query(
        self,
        base_ids: np.ndarray,
        *,
        total_var: int = 6,
        num_queries: int = 2,
        answer_mask_ratio: float | None = None,
        haystack_mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = self._rng(base_ids, salt=1)
        names = self._unique_names(rng, total_var)
        values = self._unique_codes(rng, total_var, length=5)
        needles: list[np.ndarray] = []
        needle_evidence: list[np.ndarray] = []
        for index, (name, value) in enumerate(zip(names, values)):
            prefix = self._encode(
                f"| One of the special magic numbers for {name} is "
            )
            value_ids = self._encode(value)
            suffix = self._encode(".|")
            needles.append(np.concatenate((prefix, value_ids, suffix)))
            needle_evidence.append(
                np.concatenate(
                    (
                        np.zeros_like(prefix, dtype=np.bool_),
                        np.full_like(
                            value_ids, index < num_queries, dtype=np.bool_
                        ),
                        np.zeros_like(suffix, dtype=np.bool_),
                    )
                )
            )
        queried = " and ".join(names[:num_queries])
        answer = " ".join(values[:num_queries])
        question = self._encode(
            f" What are all the special magic numbers for {queried} "
            "mentioned in the provided text?. Answer: "
        )
        answer_ids = np.concatenate(
            (self._encode(answer), np.asarray([self.eos_token_id], dtype=np.int64))
        )
        return self._compose(
            base_ids,
            needles,
            needle_evidence,
            question,
            answer_ids,
            rng,
            answer_mask_ratio=answer_mask_ratio,
            haystack_mask_ratio=haystack_mask_ratio,
        )

    def _variable_tracking(
        self,
        base_ids: np.ndarray,
        *,
        total_var: int = 6,
        max_hops: int = 2,
        answer_mask_ratio: float | None = None,
        haystack_mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = self._rng(base_ids, salt=2)
        names = self._unique_names(rng, total_var)
        values = self._unique_codes(
            rng,
            max(1, total_var // (max_hops + 1)),
            length=7,
        )
        needles: list[np.ndarray] = []
        needle_evidence: list[np.ndarray] = []
        for offset in range(0, total_var, max_hops + 1):
            value = values[offset // (max_hops + 1)]
            root = self._encode(f"|VAR {names[offset]} = {value}|")
            needles.append(root)
            needle_evidence.append(
                np.full_like(root, offset == 0, dtype=np.bool_)
            )
            for hop in range(1, max_hops + 1):
                relation = self._encode(
                    f"|VAR {names[offset + hop]} = "
                    f"{names[offset + hop - 1]}|"
                )
                needles.append(relation)
                needle_evidence.append(
                    np.full_like(relation, offset == 0, dtype=np.bool_)
                )
        answer = ", ".join(names[: max_hops + 1])
        question = np.concatenate(
            (
                self._encode(" Find all variables that are assigned the value "),
                self._encode(values[0]),
                self._encode(". Answer: "),
            )
        )
        answer_ids = np.concatenate(
            (self._encode(answer), np.asarray([self.eos_token_id], dtype=np.int64))
        )
        return self._compose(
            base_ids,
            needles,
            needle_evidence,
            question,
            answer_ids,
            rng,
            answer_mask_ratio=answer_mask_ratio,
            haystack_mask_ratio=haystack_mask_ratio,
        )


@dataclass
class FullTextComplementaryCollator:
    """Build deterministic complementary full-text denoising views."""

    mask_token_id: int
    pad_token_id: int
    eos_token_id: int
    lmk_token_id: int
    chunk_size: int = 64
    t_min: float = 0.2
    t_max: float = 0.8
    seed: int = 7
    epoch: int = 0
    ruler_mix_ratio: float = 0.0
    ruler_synthesizer: Optional[RulerDenoisingSynthesizer] = None
    ruler_every_step: bool = False
    ruler_all_tasks: bool = False
    ruler_joint_local_mask: bool = False
    insert_landmarks: bool = True
    pad_to: Optional[int] = None
    distant_infill: bool = False
    distant_min_gap: int = 256
    distant_span_min: int = 8
    distant_span_max: int = 16
    distant_needles_min: int = 1
    distant_needles_max: int = 1
    distant_needle_sep: int = 64
    distant_cue_len: int = 0
    distant_only: bool = False
    label_on_mask: bool = False
    tokenizer: Any = None

    def __post_init__(self) -> None:
        if self.insert_landmarks and self.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if not 0.0 <= self.t_min <= self.t_max <= 1.0:
            raise ValueError(
                f"expected 0 <= t_min <= t_max <= 1, got "
                f"{self.t_min}, {self.t_max}"
            )
        if not 0.0 <= float(self.ruler_mix_ratio) <= 1.0:
            raise ValueError(
                f"expected 0 <= ruler_mix_ratio <= 1, got {self.ruler_mix_ratio}"
            )
        if (self.ruler_mix_ratio > 0 or bool(self.ruler_every_step)) and self.ruler_synthesizer is None:
            raise ValueError(
                "ruler_synthesizer is required when ruler_mix_ratio > 0 or ruler_every_step"
            )
        if bool(self.ruler_all_tasks) and not bool(self.ruler_every_step):
            raise ValueError("ruler_all_tasks requires ruler_every_step")
        if bool(self.ruler_every_step):
            if float(self.ruler_mix_ratio) != 0.0:
                raise ValueError("ruler_every_step cannot stack with Bernoulli ruler_mix_ratio")
            if bool(self.distant_infill):
                raise ValueError("ruler_every_step cannot stack with distant infill")
        if bool(self.ruler_joint_local_mask):
            if float(self.ruler_mix_ratio) <= 0:
                raise ValueError("ruler_joint_local_mask requires ruler_mix_ratio > 0")
            if bool(self.distant_infill):
                raise ValueError("ruler_joint_local_mask cannot stack with distant infill")
        if self.pad_to is not None and int(self.pad_to) <= 0:
            raise ValueError("pad_to must be positive when set")
        if bool(self.distant_infill):
            if int(self.distant_min_gap) < 1:
                raise ValueError("distant_min_gap must be positive")
            if int(self.distant_span_min) < 1:
                raise ValueError("distant_span_min must be positive")
            if int(self.distant_span_max) < int(self.distant_span_min):
                raise ValueError("distant_span_max must be >= distant_span_min")
            if int(self.distant_needles_min) < 1:
                raise ValueError("distant_needles_min must be positive")
            if int(self.distant_needles_max) < int(self.distant_needles_min):
                raise ValueError("distant_needles_max must be >= distant_needles_min")
            if int(self.distant_cue_len) < 0:
                raise ValueError("distant_cue_len must be nonnegative")
            if float(self.ruler_mix_ratio) > 0:
                raise ValueError("distant infill cannot mix with ruler_mix_ratio")
        if bool(self.distant_only) and not bool(self.distant_infill):
            raise ValueError("distant_only requires distant_infill")

    @property
    def real_per_chunk(self) -> int:
        return self.chunk_size - 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, sample_id: int) -> torch.Generator:
        mixed_seed = (
            int(self.seed) * 6364136223846793005
            + int(self.epoch) * 1442695040888963407
            + int(sample_id) * 22695477
        ) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        return generator

    def _encode_piece(self, text: str) -> torch.Tensor:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return torch.as_tensor(ids, dtype=torch.long)
        return torch.tensor([ord(ch) for ch in text], dtype=torch.long)

    def _eligible_targets(
        self,
        clean_ids: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        eligible = torch.zeros_like(valid_tokens, dtype=torch.bool)
        eligible[1:] = (
            valid_tokens[1:]
            & valid_tokens[:-1]
            & (clean_ids[:-1] != int(self.eos_token_id))
        )
        return eligible

    def _sample_target_mask(
        self,
        eligible: torch.Tensor,
        sample_id: int,
    ) -> torch.Tensor:
        generator = self._generator(sample_id)
        unit = torch.rand((), generator=generator).item()
        mask_ratio = self.t_min + (self.t_max - self.t_min) * unit
        draws = torch.rand(eligible.numel(), generator=generator)
        return eligible & (draws < mask_ratio)

    def _use_ruler(self, sample_id: int) -> bool:
        if self.ruler_mix_ratio <= 0:
            return False
        generator = self._generator(sample_id)
        return torch.rand((), generator=generator).item() < self.ruler_mix_ratio

    def _insert_landmarks(
        self,
        token_ids: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        real_slots = token_ids.numel()
        if real_slots % self.real_per_chunk != 0:
            raise ValueError(
                f"real token slots {real_slots} must be divisible by "
                f"chunk_size - 1 = {self.real_per_chunk}"
            )
        chunks = token_ids.view(-1, self.real_per_chunk)
        valid_chunks = valid_tokens.view(-1, self.real_per_chunk)
        segment_chunks = segment_ids.view(-1, self.real_per_chunk)
        if torch.any(valid_tokens & (segment_ids <= 0)):
            raise ValueError("valid tokens must have positive segment ids")
        if torch.any((~valid_tokens) & (segment_ids != 0)):
            raise ValueError("invalid tokens must have segment id zero")
        for chunk_segments, chunk_valid in zip(segment_chunks, valid_chunks):
            if torch.unique(chunk_segments[chunk_valid]).numel() > 1:
                raise ValueError("a physical chunk cannot mix document segments")
        chunk_count = chunks.shape[0]
        landmarks = torch.full(
            (chunk_count, 1), int(self.lmk_token_id), dtype=torch.long
        )
        landmark_valid = valid_chunks.any(dim=-1, keepdim=True)
        physical_ids = torch.cat((chunks, landmarks), dim=-1).reshape(-1)
        attention_mask = torch.cat(
            (valid_chunks, landmark_valid), dim=-1
        ).reshape(-1).long()
        landmark_segments = segment_chunks.amax(dim=-1, keepdim=True)
        physical_segments = torch.cat(
            (segment_chunks, landmark_segments), dim=-1
        ).reshape(-1)

        text_positions = torch.zeros(real_slots, dtype=torch.long)
        for segment in torch.unique(segment_ids[segment_ids > 0]).tolist():
            member = segment_ids == int(segment)
            text_positions[member] = torch.arange(member.sum(), dtype=torch.long)
        text_positions = text_positions.view(-1, self.real_per_chunk)
        landmark_positions = torch.where(
            landmark_valid,
            text_positions.amax(dim=-1, keepdim=True) + 1,
            torch.zeros_like(landmark_valid, dtype=torch.long),
        )
        position_ids = torch.cat(
            (text_positions, landmark_positions), dim=-1
        ).reshape(-1)
        original_to_physical = torch.arange(real_slots, dtype=torch.long)
        original_to_physical += torch.div(
            original_to_physical,
            self.real_per_chunk,
            rounding_mode="floor",
        )
        physical_landmark_mask = torch.cat(
            (
                torch.zeros_like(valid_chunks, dtype=torch.bool),
                landmark_valid,
            ),
            dim=-1,
        ).reshape(-1)
        return (
            physical_ids,
            attention_mask,
            physical_segments,
            position_ids,
            original_to_physical,
            physical_landmark_mask,
        )

    def _plain_layout(
        self,
        token_ids: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if torch.any(valid_tokens & (segment_ids <= 0)):
            raise ValueError("valid tokens must have positive segment ids")
        if torch.any((~valid_tokens) & (segment_ids != 0)):
            raise ValueError("invalid tokens must have segment id zero")
        position_ids = torch.zeros_like(token_ids)
        for segment in torch.unique(segment_ids[segment_ids > 0]).tolist():
            member = segment_ids == int(segment)
            position_ids[member] = torch.arange(int(member.sum()), dtype=torch.long)
        mapping = torch.arange(token_ids.numel(), dtype=torch.long)
        landmark_mask = torch.zeros_like(valid_tokens)
        return (
            token_ids,
            valid_tokens.long(),
            segment_ids,
            position_ids,
            mapping,
            landmark_mask,
        )

    def _build_view(
        self,
        clean_ids: torch.Tensor,
        valid_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        target_mask: torch.Tensor,
        sample_id: int,
        view_id: int,
        evidence_mask: torch.Tensor | None = None,
        gate_live: bool = False,
    ) -> Dict[str, torch.Tensor]:
        corrupted = clean_ids.clone()
        corrupted[~valid_tokens] = int(self.pad_token_id)
        corrupted[target_mask] = int(self.mask_token_id)
        if self.insert_landmarks:
            input_ids, attention_mask, physical_segments, position_ids, mapping, landmark_mask = (
                self._insert_landmarks(corrupted, valid_tokens, segment_ids)
            )
            evidence_chunks = torch.zeros(
                input_ids.numel() // self.chunk_size, dtype=torch.bool
            )
            evidence_tokens = torch.zeros(input_ids.numel(), dtype=torch.bool)
        else:
            input_ids, attention_mask, physical_segments, position_ids, mapping, landmark_mask = (
                self._plain_layout(corrupted, valid_tokens, segment_ids)
            )
            evidence_chunks = torch.zeros(1, dtype=torch.bool)
            evidence_tokens = torch.zeros(input_ids.numel(), dtype=torch.bool)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        target_indices = torch.where(target_mask)[0]
        if target_indices.numel() > 0:
            if bool(self.label_on_mask):
                label_positions = mapping[target_indices]
            else:
                predictor_indices = target_indices - 1
                label_positions = mapping[predictor_indices]
            labels[label_positions] = clean_ids[target_indices]
        if evidence_mask is not None:
            if evidence_mask.shape != clean_ids.shape:
                raise ValueError("evidence mask must match clean tokens")
            if not self.insert_landmarks:
                raise ValueError("route evidence chunks require landmark layout")
            evidence_positions = mapping[torch.where(evidence_mask)[0]]
            evidence_tokens[evidence_positions] = True
            evidence_chunks[
                torch.div(
                    evidence_positions,
                    self.chunk_size,
                    rounding_mode="floor",
                )
            ] = True
        if self.pad_to is not None and input_ids.numel() < int(self.pad_to):
            extra = int(self.pad_to) - input_ids.numel()
            input_ids = torch.cat(
                (input_ids, torch.full((extra,), int(self.pad_token_id), dtype=torch.long))
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(extra, dtype=attention_mask.dtype))
            )
            physical_segments = torch.cat(
                (physical_segments, torch.zeros(extra, dtype=physical_segments.dtype))
            )
            position_ids = torch.cat(
                (position_ids, torch.zeros(extra, dtype=position_ids.dtype))
            )
            landmark_mask = torch.cat(
                (landmark_mask, torch.zeros(extra, dtype=torch.bool))
            )
            labels = torch.cat(
                (labels, torch.full((extra,), IGNORE_INDEX, dtype=labels.dtype))
            )
            evidence_tokens = torch.cat(
                (evidence_tokens, torch.zeros(extra, dtype=torch.bool))
            )
        elif self.pad_to is not None and input_ids.numel() > int(self.pad_to):
            raise ValueError(
                f"sequence length {input_ids.numel()} exceeds pad_to={self.pad_to}"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": physical_segments,
            "position_ids": position_ids,
            "landmark_mask": landmark_mask,
            "labels": labels,
            "target_count": target_mask.sum(dtype=torch.long),
            "sample_ids": torch.tensor(sample_id, dtype=torch.long),
            "view_ids": torch.tensor(view_id, dtype=torch.long),
            "route_evidence_chunks": evidence_chunks,
            "route_evidence_tokens": evidence_tokens,
            "gate_live": torch.tensor(bool(gate_live)),
        }

    def _append_ruler_views(
        self,
        views: List[Dict[str, torch.Tensor]],
        clean_ids: torch.Tensor,
        sample_id: int,
    ) -> None:
        assert self.ruler_synthesizer is not None
        if bool(self.ruler_all_tasks):
            task_ids = list(self.ruler_synthesizer.task_ids)
        else:
            generator = self._generator(sample_id + 17)
            task_ids = [self.ruler_synthesizer.sample_task_id(generator)]
        for task_id in task_ids:
            ruler_ids, target_mask, evidence_mask = (
                self.ruler_synthesizer.synthesize_with_evidence(
                    clean_ids,
                    task_id=int(task_id),
                    joint_mask=bool(self.ruler_joint_local_mask),
                    t_min=float(self.t_min),
                    t_max=float(self.t_max),
                )
            )
            ruler_valid = torch.ones_like(ruler_ids, dtype=torch.bool)
            ruler_segments = torch.ones_like(ruler_ids, dtype=torch.long)
            views.append(
                self._build_view(
                    ruler_ids,
                    ruler_valid,
                    ruler_segments,
                    target_mask,
                    sample_id,
                    RULER_VIEW_ID_BASE + int(task_id),
                    evidence_mask if self.insert_landmarks else None,
                )
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        views: List[Dict[str, torch.Tensor]] = []
        expected_slots: Optional[int] = None
        for example in examples:
            clean_ids = torch.as_tensor(
                example["clean_ids"], dtype=torch.long
            ).clone()
            valid_tokens = torch.as_tensor(
                example.get(
                    "valid_tokens", torch.ones_like(clean_ids, dtype=torch.bool)
                ),
                dtype=torch.bool,
            ).clone()
            segment_ids = torch.as_tensor(
                example.get(
                    "segment_ids", valid_tokens.to(dtype=torch.long)
                ),
                dtype=torch.long,
            ).clone()
            if (
                clean_ids.ndim != 1
                or valid_tokens.shape != clean_ids.shape
                or segment_ids.shape != clean_ids.shape
            ):
                raise ValueError(
                    "clean_ids, valid_tokens, and segment_ids must be equal 1D tensors"
                )
            if expected_slots is None:
                expected_slots = clean_ids.numel()
            elif clean_ids.numel() != expected_slots:
                raise ValueError("all packed examples must have equal token length")

            sample_id = int(example["sample_id"])
            if bool(self.ruler_every_step):
                eligible = self._eligible_targets(clean_ids, valid_tokens)
                target_a = self._sample_target_mask(eligible, sample_id)
                target_b = eligible & ~target_a
                views.append(
                    self._build_view(
                        clean_ids, valid_tokens, segment_ids, target_a, sample_id, 0
                    )
                )
                views.append(
                    self._build_view(
                        clean_ids, valid_tokens, segment_ids, target_b, sample_id, 1
                    )
                )
                self._append_ruler_views(views, clean_ids, sample_id)
            elif self._use_ruler(sample_id):
                generator = self._generator(sample_id + 17)
                assert self.ruler_synthesizer is not None
                task_id = self.ruler_synthesizer.sample_task_id(generator)
                ruler_ids, target_mask, evidence_mask = (
                    self.ruler_synthesizer.synthesize_with_evidence(
                        clean_ids,
                        task_id=task_id,
                        joint_mask=bool(self.ruler_joint_local_mask),
                        t_min=float(self.t_min),
                        t_max=float(self.t_max),
                    )
                )
                ruler_valid = torch.ones_like(ruler_ids, dtype=torch.bool)
                ruler_segments = torch.ones_like(ruler_ids, dtype=torch.long)
                views.append(
                    self._build_view(
                        ruler_ids,
                        ruler_valid,
                        ruler_segments,
                        target_mask,
                        sample_id,
                        RULER_VIEW_ID_BASE + task_id,
                        evidence_mask if self.insert_landmarks else None,
                    )
                )
            else:
                if not self.distant_only:
                    eligible = self._eligible_targets(clean_ids, valid_tokens)
                    target_a = self._sample_target_mask(eligible, sample_id)
                    target_b = eligible & ~target_a
                    views.append(
                        self._build_view(
                            clean_ids, valid_tokens, segment_ids, target_a, sample_id, 0
                        )
                    )
                    views.append(
                        self._build_view(
                            clean_ids, valid_tokens, segment_ids, target_b, sample_id, 1
                        )
                    )
                if self.distant_infill:
                    needle_lo = int(self.distant_needles_min)
                    needle_hi = int(self.distant_needles_max)
                    needle_gen = self._generator(sample_id + 31)
                    num_needles = needle_lo
                    if needle_hi > needle_lo:
                        num_needles = needle_lo + int(
                            torch.randint(
                                0,
                                needle_hi - needle_lo + 1,
                                (1,),
                                generator=needle_gen,
                            ).item()
                        )
                    distant_ids, distant_targets, distant_evidence = (
                        synthesize_one_token_key_value(
                            clean_ids,
                            min_gap=int(self.distant_min_gap),
                            generator=needle_gen,
                            encode_fn=self._encode_piece,
                        )
                        if bool(self.distant_only)
                        else synthesize_distant_span_copy(
                            clean_ids,
                            min_gap=int(self.distant_min_gap),
                            span_min=int(self.distant_span_min),
                            span_max=int(self.distant_span_max),
                            generator=needle_gen,
                            num_needles=num_needles,
                            min_needle_sep=int(self.distant_needle_sep),
                            cue_len=int(self.distant_cue_len),
                        )
                    )
                    views.append(
                        self._build_view(
                            distant_ids,
                            valid_tokens,
                            segment_ids,
                            distant_targets,
                            sample_id,
                            2,
                            distant_evidence if self.insert_landmarks else None,
                            gate_live=True,
                        )
                    )

        if not views:
            raise ValueError("FullTextComplementaryCollator received an empty batch")
        keys = (
            "input_ids",
            "attention_mask",
            "segment_ids",
            "position_ids",
            "landmark_mask",
            "labels",
            "target_count",
            "sample_ids",
            "view_ids",
            "route_evidence_chunks",
            "route_evidence_tokens",
            "gate_live",
        )
        return {
            key: torch.stack([view[key] for view in views], dim=0)
            for key in keys
        }
