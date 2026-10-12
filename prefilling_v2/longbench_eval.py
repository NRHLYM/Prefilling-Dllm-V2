"""Pure helpers for resumable Dream route LongBench evaluation."""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


LONGBENCH_TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

LONGBENCH_EN_TASKS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "samsum",
    "trec",
    "triviaqa",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

LONGBENCH_ALL_TASKS: tuple[str, ...] = (
    *LONGBENCH_EN_TASKS,
    "multifieldqa_zh",
    "dureader",
    "vcsum",
    "lsht",
    "passage_retrieval_zh",
)


@dataclass(frozen=True)
class GenerationLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    answer_positions: torch.Tensor
    predictor_positions: torch.Tensor
    landmark_positions: torch.Tensor


@dataclass(frozen=True)
class FastDLLMBlockLayout:
    answer_positions: torch.Tensor
    predictor_positions: torch.Tensor
    affected_chunks: torch.Tensor
    landmark_positions: torch.Tensor
    query_positions: torch.Tensor
    kv_update_positions: torch.Tensor


def build_fastdllm_block_layouts(
    layout: GenerationLayout,
    logical_block_size: int,
    chunk_size: int,
) -> tuple[FastDLLMBlockLayout, ...]:
    """Split an answer span into logical Fast-dLLM cache update blocks."""

    if logical_block_size <= 0:
        raise ValueError(
            f"logical_block_size must be positive, got {logical_block_size}"
        )
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be at least 2, got {chunk_size}")

    physical_length = int(layout.input_ids.numel())
    if physical_length <= 0 or physical_length % chunk_size:
        raise ValueError(
            f"physical length {physical_length} must be divisible by "
            f"chunk_size={chunk_size}"
        )
    if layout.answer_positions.numel() != layout.predictor_positions.numel():
        raise ValueError("answer and predictor position counts must match")
    if layout.answer_positions.numel() % logical_block_size:
        raise ValueError(
            "answer token count must be divisible by logical_block_size"
        )

    answer_positions = layout.answer_positions.to(dtype=torch.long)
    predictor_positions = layout.predictor_positions.to(dtype=torch.long)
    if torch.unique(answer_positions).numel() != answer_positions.numel():
        raise ValueError("answer positions must be unique")
    all_owned = torch.cat((answer_positions, predictor_positions))
    if bool(((all_owned < 0) | (all_owned >= physical_length)).any()):
        raise ValueError("answer and predictor positions must be in range")

    insert_landmarks = layout.landmark_positions.numel() > 0
    if insert_landmarks:
        expected_landmarks = torch.arange(
            chunk_size - 1,
            physical_length,
            chunk_size,
            dtype=torch.long,
            device=layout.landmark_positions.device,
        )
        if not torch.equal(
            layout.landmark_positions.to(dtype=torch.long),
            expected_landmarks,
        ):
            raise ValueError("layout landmark positions do not match chunk ends")

    blocks: list[FastDLLMBlockLayout] = []
    for start in range(0, answer_positions.numel(), logical_block_size):
        stop = start + logical_block_size
        answer = answer_positions[start:stop]
        predictor = predictor_positions[start:stop]
        if insert_landmarks:
            chunks = torch.unique(
                torch.div(answer, chunk_size, rounding_mode="floor"),
                sorted=True,
            )
            landmarks = chunks * chunk_size + (chunk_size - 1)
            queries = torch.unique(
                torch.cat((answer, predictor, landmarks)),
                sorted=True,
            )
            updates = torch.unique(
                torch.cat((answer, landmarks)),
                sorted=True,
            )
        else:
            chunks = answer.new_empty((0,), dtype=torch.long)
            landmarks = answer.new_empty((0,), dtype=torch.long)
            queries = torch.unique(
                torch.cat((answer, predictor)),
                sorted=True,
            )
            updates = answer.clone()
        blocks.append(
            FastDLLMBlockLayout(
                answer_positions=answer.clone(),
                predictor_positions=predictor.clone(),
                affected_chunks=chunks,
                landmark_positions=landmarks,
                query_positions=queries,
                kv_update_positions=updates,
            )
        )
    return tuple(blocks)


def physical_text_positions(real_slots: int, chunk_size: int) -> torch.Tensor:
    """Map contiguous text slots around one landmark at each chunk end."""

    if real_slots < 0:
        raise ValueError(f"real_slots must be non-negative, got {real_slots}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be at least 2, got {chunk_size}")
    real_per_chunk = chunk_size - 1
    positions = torch.arange(real_slots, dtype=torch.long)
    return positions + torch.div(
        positions,
        real_per_chunk,
        rounding_mode="floor",
    )


def build_generation_layout(
    *,
    prompt_ids: Sequence[int],
    answer_tokens: int,
    physical_length: int,
    chunk_size: int,
    mask_token_id: int,
    pad_token_id: int,
    landmark_token_id: int,
    insert_landmarks: bool = True,
) -> GenerationLayout:
    """Pack a clean prompt and masked answer span into a fixed physical length."""

    if physical_length <= 0 or physical_length % chunk_size:
        raise ValueError(
            f"physical_length={physical_length} must be positive and divisible "
            f"by chunk_size={chunk_size}"
        )
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one predictor token")
    if answer_tokens <= 0:
        raise ValueError(f"answer_tokens must be positive, got {answer_tokens}")

    if not insert_landmarks:
        answer_start = len(prompt_ids)
        answer_end = answer_start + answer_tokens
        if answer_end > physical_length:
            raise ValueError(
                f"prompt plus answer needs {answer_end} tokens, only "
                f"{physical_length} are available"
            )
        input_ids = torch.full((physical_length,), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros(physical_length, dtype=torch.bool)
        input_ids[:answer_start] = torch.as_tensor(prompt_ids, dtype=torch.long)
        input_ids[answer_start:answer_end] = int(mask_token_id)
        attention_mask[:answer_end] = True
        position_ids = torch.zeros(physical_length, dtype=torch.long)
        position_ids[:answer_end] = torch.arange(answer_end, dtype=torch.long)
        answer_positions = torch.arange(answer_start, answer_end, dtype=torch.long)
        return GenerationLayout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            answer_positions=answer_positions,
            predictor_positions=answer_positions - 1,
            landmark_positions=torch.zeros(0, dtype=torch.long),
        )

    if physical_length <= 0 or physical_length % chunk_size:
        raise ValueError(
            f"physical_length={physical_length} must be positive and divisible "
            f"by chunk_size={chunk_size}"
        )
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one predictor token")
    if answer_tokens <= 0:
        raise ValueError(f"answer_tokens must be positive, got {answer_tokens}")

    real_per_chunk = chunk_size - 1
    chunk_count = physical_length // chunk_size
    real_slots = chunk_count * real_per_chunk
    answer_start = len(prompt_ids)
    answer_end = answer_start + answer_tokens
    if answer_end > real_slots:
        raise ValueError(
            f"prompt plus answer needs {answer_end} text slots, only "
            f"{real_slots} are available"
        )

    text_ids = torch.full((real_slots,), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros(real_slots, dtype=torch.bool)
    text_ids[:answer_start] = torch.as_tensor(prompt_ids, dtype=torch.long)
    text_ids[answer_start:answer_end] = int(mask_token_id)
    valid[:answer_end] = True

    text_chunks = text_ids.view(chunk_count, real_per_chunk)
    valid_chunks = valid.view(chunk_count, real_per_chunk)
    landmarks = torch.full(
        (chunk_count, 1),
        int(landmark_token_id),
        dtype=torch.long,
    )
    landmark_valid = valid_chunks.any(dim=-1, keepdim=True)
    input_ids = torch.cat((text_chunks, landmarks), dim=-1).reshape(-1)
    attention_mask = torch.cat(
        (valid_chunks, landmark_valid), dim=-1
    ).reshape(-1)

    text_position_ids = torch.arange(real_slots, dtype=torch.long).view(
        chunk_count, real_per_chunk
    )
    landmark_position_ids = text_position_ids[:, -1:] + 1
    position_ids = torch.cat(
        (text_position_ids, landmark_position_ids), dim=-1
    ).reshape(-1)

    mapping = physical_text_positions(real_slots, chunk_size)
    answer_text_positions = torch.arange(answer_start, answer_end)
    landmark_positions = torch.arange(
        chunk_size - 1,
        physical_length,
        chunk_size,
        dtype=torch.long,
    )
    return GenerationLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        answer_positions=mapping[answer_text_positions],
        predictor_positions=mapping[answer_text_positions - 1],
        landmark_positions=landmark_positions,
    )


def build_plain_generation_layout(
    *,
    prompt_ids: Sequence[int],
    answer_tokens: int,
    physical_length: int,
    mask_token_id: int,
    pad_token_id: int,
) -> GenerationLayout:
    """Pack prompt and masked answers into a contiguous sequence with no landmarks."""

    if physical_length <= 0:
        raise ValueError(f"physical_length must be positive, got {physical_length}")
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one predictor token")
    if answer_tokens <= 0:
        raise ValueError(f"answer_tokens must be positive, got {answer_tokens}")
    answer_start = len(prompt_ids)
    answer_end = answer_start + answer_tokens
    if answer_end > physical_length:
        raise ValueError(
            f"prompt plus answer needs {answer_end} tokens, only "
            f"{physical_length} are available"
        )
    input_ids = torch.full((physical_length,), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros(physical_length, dtype=torch.bool)
    input_ids[:answer_start] = torch.as_tensor(prompt_ids, dtype=torch.long)
    input_ids[answer_start:answer_end] = int(mask_token_id)
    attention_mask[:answer_end] = True
    position_ids = torch.arange(physical_length, dtype=torch.long)
    answer_positions = torch.arange(answer_start, answer_end, dtype=torch.long)
    return GenerationLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        answer_positions=answer_positions,
        predictor_positions=answer_positions - 1,
        landmark_positions=torch.empty(0, dtype=torch.long),
    )


def build_plain_fastdllm_block_layouts(
    layout: GenerationLayout,
    logical_block_size: int,
) -> tuple[FastDLLMBlockLayout, ...]:
    """Split a contiguous answer span into Fast-dLLM blocks without route chunks."""

    if logical_block_size <= 0:
        raise ValueError(
            f"logical_block_size must be positive, got {logical_block_size}"
        )
    if layout.landmark_positions.numel():
        raise ValueError("plain Fast-dLLM layouts cannot contain landmarks")
    if layout.answer_positions.numel() != layout.predictor_positions.numel():
        raise ValueError("answer and predictor position counts must match")
    if layout.answer_positions.numel() % logical_block_size:
        raise ValueError(
            "answer token count must be divisible by logical_block_size"
        )
    answer_positions = layout.answer_positions.to(dtype=torch.long)
    predictor_positions = layout.predictor_positions.to(dtype=torch.long)
    blocks: list[FastDLLMBlockLayout] = []
    empty = torch.empty(0, dtype=torch.long)
    for start in range(0, answer_positions.numel(), logical_block_size):
        stop = start + logical_block_size
        answer = answer_positions[start:stop]
        predictor = predictor_positions[start:stop]
        queries = torch.unique(torch.cat((answer, predictor)), sorted=True)
        blocks.append(
            FastDLLMBlockLayout(
                answer_positions=answer.clone(),
                predictor_positions=predictor.clone(),
                affected_chunks=empty.clone(),
                landmark_positions=empty.clone(),
                query_positions=queries,
                kv_update_positions=answer.clone(),
            )
        )
    return tuple(blocks)


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    return list(range(rank, total, world_size))


def _normalize_answer(text: str) -> list[str]:
    punctuation = set(string.punctuation)
    unpunctuated = "".join(
        character for character in text.lower() if character not in punctuation
    )
    return [
        token
        for token in unpunctuated.split()
        if token not in {"a", "an", "the"}
    ]


def qa_f1_score(prediction: str, answers: Sequence[str]) -> float:
    prediction_tokens = _normalize_answer(prediction)
    best = 0.0
    for answer in answers:
        answer_tokens = _normalize_answer(str(answer))
        overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
        if not prediction_tokens or not answer_tokens or overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(answer_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _normalize_zh_answer(text: str) -> str:
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
        "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join(
        character
        for character in text.lower()
        if not character.isspace() and character not in punctuation
    )


def _token_f1(prediction_tokens: Sequence[str], answer_tokens: Sequence[str]) -> float:
    overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if not prediction_tokens or not answer_tokens or overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def _qa_f1_zh_score(prediction: str, answer: str) -> float:
    import jieba

    prediction_tokens = [
        token
        for piece in jieba.cut(prediction, cut_all=False)
        if (token := _normalize_zh_answer(piece))
    ]
    answer_tokens = [
        token
        for piece in jieba.cut(answer, cut_all=False)
        if (token := _normalize_zh_answer(piece))
    ]
    return _token_f1(prediction_tokens, answer_tokens)


def _rouge_l_score(prediction: str, answer: str) -> float:
    from rouge import Rouge

    try:
        return float(
            Rouge().get_scores([prediction], [answer], avg=True)["rouge-l"]["f"]
        )
    except Exception:
        return 0.0


def _rouge_l_zh_score(prediction: str, answer: str) -> float:
    import jieba

    segmented_prediction = " ".join(jieba.cut(prediction, cut_all=False))
    segmented_answer = " ".join(jieba.cut(answer, cut_all=False))
    return _rouge_l_score(segmented_prediction, segmented_answer)


def _classification_score(
    prediction: str,
    answer: str,
    all_classes: Sequence[str] | None,
) -> float:
    if not all_classes:
        raise ValueError("classification tasks require all_classes")
    matches = [class_name for class_name in all_classes if class_name in prediction]
    # Match the official LongBench handling of labels contained in another label.
    for match in list(matches):
        if match in answer and match != answer:
            matches.remove(match)
    return 1.0 / len(matches) if answer in matches else 0.0


def _number_precision_score(prediction: str, answer: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(number == str(answer) for number in numbers) / len(numbers)


def _retrieval_score(prediction: str, answer: str, *, chinese: bool) -> float:
    pattern = r"段落(\d+)" if chinese else r"Paragraph (\d+)"
    matches = re.findall(pattern, answer)
    if not matches:
        raise ValueError(f"retrieval answer has no paragraph id: {answer!r}")
    return _number_precision_score(prediction, matches[0])


def _code_similarity_score(prediction: str, answer: str) -> float:
    from fuzzywuzzy import fuzz

    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return float(fuzz.ratio(candidate, answer)) / 100.0


def longbench_score(
    task: str,
    prediction: str,
    answers: Sequence[str],
    *,
    all_classes: Sequence[str] | None = None,
) -> tuple[float, str]:
    """Score one prediction with the official LongBench v1 task metric."""

    if task not in LONGBENCH_TASK_MAX_NEW_TOKENS:
        raise ValueError(f"unsupported LongBench task: {task}")
    if not answers:
        raise ValueError(f"LongBench task {task} has no reference answers")
    scored_prediction = prediction
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        scored_prediction = prediction.lstrip("\n").split("\n", 1)[0]

    if task in {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "triviaqa",
    }:
        return qa_f1_score(scored_prediction, answers), "qa_f1"

    scorers = {
        "multifieldqa_zh": (_qa_f1_zh_score, "qa_f1_zh"),
        "dureader": (_rouge_l_zh_score, "rouge_l_zh"),
        "gov_report": (_rouge_l_score, "rouge_l"),
        "qmsum": (_rouge_l_score, "rouge_l"),
        "multi_news": (_rouge_l_score, "rouge_l"),
        "samsum": (_rouge_l_score, "rouge_l"),
        "vcsum": (_rouge_l_zh_score, "rouge_l_zh"),
        "passage_count": (_number_precision_score, "count"),
        "lcc": (_code_similarity_score, "code_similarity"),
        "repobench-p": (_code_similarity_score, "code_similarity"),
    }
    if task in {"trec", "lsht"}:
        return (
            max(
                _classification_score(scored_prediction, answer, all_classes)
                for answer in answers
            ),
            "classification",
        )
    if task in {"passage_retrieval_en", "passage_retrieval_zh"}:
        chinese = task.endswith("_zh")
        return (
            max(
                _retrieval_score(scored_prediction, answer, chinese=chinese)
                for answer in answers
            ),
            "retrieval_zh" if chinese else "retrieval",
        )
    scorer, metric = scorers[task]
    return max(scorer(scored_prediction, answer) for answer in answers), metric


def truncate_prompt_parts(
    *,
    prefix_ids: Sequence[int],
    context_ids: Sequence[int],
    query_ids: Sequence[int],
    max_prompt_tokens: int,
    bos_token_id: int | None,
) -> tuple[list[int], dict[str, int | str]]:
    """Head-tail truncate only context while preserving prompt framing."""

    prefix = list(prefix_ids)
    context = list(context_ids)
    query = list(query_ids)
    bos = [] if bos_token_id is None else [int(bos_token_id)]
    fixed_tokens = len(bos) + len(prefix) + len(query)
    if max_prompt_tokens < fixed_tokens:
        raise ValueError(
            f"prompt budget {max_prompt_tokens} cannot fit {fixed_tokens} "
            "BOS/prefix/query tokens"
        )

    context_budget = max_prompt_tokens - fixed_tokens
    raw_prompt_tokens = fixed_tokens + len(context)
    if len(context) <= context_budget:
        kept_context = context
        head_tokens = len(context)
        tail_tokens = 0
        mode = "none"
    else:
        head_tokens = (context_budget + 1) // 2
        tail_tokens = context_budget - head_tokens
        kept_context = context[:head_tokens]
        if tail_tokens:
            kept_context += context[-tail_tokens:]
        mode = "head_tail"

    query_start = len(bos) + len(prefix) + len(kept_context)
    query_end = query_start + len(query)
    prompt_ids = bos + prefix + kept_context + query
    metadata: dict[str, int | str] = {
        "raw_context_tokens": len(context),
        "raw_prompt_tokens": raw_prompt_tokens,
        "prompt_tokens_after_truncation": len(prompt_ids),
        "context_budget_tokens": context_budget,
        "context_head_tokens": head_tokens,
        "context_tail_tokens": tail_tokens,
        "context_dropped_tokens": len(context) - len(kept_context),
        "query_start_token": query_start,
        "query_end_token": query_end,
        "query_tokens": len(query),
        "truncation_mode": mode,
    }
    return prompt_ids, metadata


def load_resumable_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Read complete JSONL records and remove an interrupted final record."""

    output = Path(path)
    if not output.exists():
        return []

    records: list[dict[str, object]] = []
    last_good_offset = 0
    with output.open("rb") as stream:
        while True:
            line_start = stream.tell()
            line = stream.readline()
            if not line:
                last_good_offset = line_start
                break
            if not line.strip():
                last_good_offset = stream.tell()
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                last_good_offset = line_start
                break
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object in {output}, got {type(record)}")
            records.append(record)
            last_good_offset = stream.tell()

    if output.stat().st_size != last_good_offset:
        with output.open("rb+") as stream:
            stream.truncate(last_good_offset)
    return records


def append_jsonl_fsync(path: str | Path, record: dict[str, object]) -> None:
    """Append one complete record and force it to stable storage."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def merge_evaluation_shards(
    output_dir: str | Path,
    *,
    total_examples: int,
    expected_variant: str | None = None,
    require_zero_fallback: bool = True,
    require_nonempty_predictions: bool = True,
) -> dict[str, object]:
    """Validate, merge, and summarize complete resumable rank shards."""

    output_dir = Path(output_dir)
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    shard_paths = sorted(output_dir.glob("rank-*.jsonl"))
    if not shard_paths:
        raise ValueError(f"no rank shards found in {output_dir}")

    records: list[dict[str, object]] = []
    for shard_path in shard_paths:
        records.extend(load_resumable_jsonl(shard_path))
    indices = [int(record["index"]) for record in records]
    duplicates = sorted(
        index for index, count in Counter(indices).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate evaluation indices: {duplicates[:8]}")
    expected_indices = set(range(total_examples))
    actual_indices = set(indices)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"incomplete evaluation: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    records.sort(key=lambda record: int(record["index"]))
    if require_nonempty_predictions and any(
        not str(record.get("prediction", "")).strip()
        for record in records
    ):
        raise ValueError("evaluation contains an empty prediction")
    variants = {str(record.get("model_variant", "")) for record in records}
    cache_modes = {str(record.get("cache_mode", "")) for record in records}
    if len(variants) != 1 or len(cache_modes) != 1:
        raise ValueError("all merged records must share model_variant/cache_mode")
    variant = next(iter(variants))
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"expected model_variant={expected_variant}, got {variant}"
        )

    fallback_max = max(int(record.get("fallback_count", 0)) for record in records)
    if require_zero_fallback and fallback_max:
        raise ValueError(f"kernel fallback observed: maximum={fallback_max}")
    seconds = [float(record["seconds"]) for record in records]
    rank_seconds: dict[int, float] = {}
    for record in records:
        rank = int(record.get("rank", 0))
        rank_seconds[rank] = rank_seconds.get(rank, 0.0) + float(record["seconds"])

    truncation_modes = Counter(
        str(record.get("prompt_metadata", {}).get("truncation_mode", "unknown"))
        for record in records
    )
    metrics: dict[str, object] = {
        "examples": len(records),
        "model_variant": variant,
        "cache_mode": next(iter(cache_modes)),
        "qa_f1": 100.0 * sum(float(record["score"]) for record in records) / len(records),
        "mean_seconds": sum(seconds) / len(seconds),
        "p50_seconds": _percentile(seconds, 0.50),
        "p95_seconds": _percentile(seconds, 0.95),
        "wall_seconds_estimate": max(rank_seconds.values()),
        "mean_full_prefills": sum(int(record.get("full_prefills", 0)) for record in records) / len(records),
        "mean_cached_forwards": sum(int(record.get("cached_forwards", 0)) for record in records) / len(records),
        "mean_routing_calls": sum(int(record.get("routing_calls", 0)) for record in records) / len(records),
        "mean_recomputed_tokens": sum(int(record.get("recomputed_tokens", 0)) for record in records) / len(records),
        "peak_memory_bytes_max": max(int(record.get("peak_memory_bytes", 0)) for record in records),
        "fallback_max": fallback_max,
        "truncation_modes": dict(sorted(truncation_modes.items())),
    }
    merged_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    _write_text_atomic(output_dir / "merged.jsonl", merged_text)
    _write_text_atomic(
        output_dir / "metrics.json",
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    return metrics
