import json

import pytest
import torch

from prefilling_v2.longbench_eval import (
    LONGBENCH_ALL_TASKS,
    LONGBENCH_EN_TASKS,
    LONGBENCH_TASK_MAX_NEW_TOKENS,
    build_fastdllm_block_layouts,
    build_generation_layout,
    load_resumable_jsonl,
    longbench_score,
    physical_text_positions,
    qa_f1_score,
    shard_indices,
    truncate_prompt_parts,
)


def test_physical_text_positions_skip_landmark_slots():
    positions = physical_text_positions(real_slots=126, chunk_size=64)

    assert positions.shape == (126,)
    assert torch.equal(positions[:4], torch.tensor([0, 1, 2, 3]))
    assert int(positions[62]) == 62
    assert int(positions[63]) == 64
    assert int(positions[125]) == 126


def test_head_tail_truncation_preserves_prefix_query_and_bos():
    prompt_ids, metadata = truncate_prompt_parts(
        prefix_ids=[10, 11],
        context_ids=list(range(20, 30)),
        query_ids=[30, 31],
        max_prompt_tokens=8,
        bos_token_id=1,
    )

    assert prompt_ids == [1, 10, 11, 20, 21, 29, 30, 31]
    assert metadata == {
        "raw_context_tokens": 10,
        "raw_prompt_tokens": 15,
        "prompt_tokens_after_truncation": 8,
        "context_budget_tokens": 3,
        "context_head_tokens": 2,
        "context_tail_tokens": 1,
        "context_dropped_tokens": 7,
        "query_start_token": 6,
        "query_end_token": 8,
        "query_tokens": 2,
        "truncation_mode": "head_tail",
    }


def test_resumable_jsonl_repairs_partial_tail(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(
        b'{"index": 0, "prediction": "a"}\n'
        b'{"index": 2, "prediction": "b"}\n'
        b'{"index":'
    )

    records = load_resumable_jsonl(output)

    assert [record["index"] for record in records] == [0, 2]
    assert output.read_text() == (
        json.dumps({"index": 0, "prediction": "a"})
        + "\n"
        + json.dumps({"index": 2, "prediction": "b"})
        + "\n"
    )


def test_generation_layout_tracks_answer_and_shifted_predictors():
    layout = build_generation_layout(
        prompt_ids=[10, 11],
        answer_tokens=2,
        physical_length=8,
        chunk_size=4,
        mask_token_id=99,
        pad_token_id=0,
        landmark_token_id=98,
    )

    assert torch.equal(
        layout.input_ids,
        torch.tensor([10, 11, 99, 98, 99, 0, 0, 98]),
    )
    assert torch.equal(
        layout.attention_mask,
        torch.tensor([1, 1, 1, 1, 1, 0, 0, 1]),
    )
    assert layout.attention_mask.dtype is torch.bool
    assert torch.equal(
        layout.position_ids,
        torch.tensor([0, 1, 2, 3, 3, 4, 5, 6]),
    )
    assert torch.equal(layout.answer_positions, torch.tensor([2, 4]))
    assert torch.equal(layout.predictor_positions, torch.tensor([1, 2]))
    assert torch.equal(layout.landmark_positions, torch.tensor([3, 7]))


def test_fastdllm_blocks_track_queries_updates_and_landmarks():
    layout = build_generation_layout(
        prompt_ids=[10, 11],
        answer_tokens=4,
        physical_length=8,
        chunk_size=4,
        mask_token_id=99,
        pad_token_id=0,
        landmark_token_id=98,
    )

    blocks = build_fastdllm_block_layouts(
        layout,
        logical_block_size=2,
        chunk_size=4,
    )

    assert len(blocks) == 2
    assert torch.equal(blocks[0].answer_positions, torch.tensor([2, 4]))
    assert torch.equal(blocks[0].predictor_positions, torch.tensor([1, 2]))
    assert torch.equal(blocks[0].affected_chunks, torch.tensor([0, 1]))
    assert torch.equal(blocks[0].landmark_positions, torch.tensor([3, 7]))
    assert torch.equal(
        blocks[0].query_positions,
        torch.tensor([1, 2, 3, 4, 7]),
    )
    assert torch.equal(
        blocks[0].kv_update_positions,
        torch.tensor([2, 3, 4, 7]),
    )
    assert torch.equal(blocks[1].answer_positions, torch.tensor([5, 6]))
    assert torch.equal(blocks[1].predictor_positions, torch.tensor([4, 5]))
    assert torch.equal(blocks[1].affected_chunks, torch.tensor([1]))
    assert torch.equal(blocks[1].landmark_positions, torch.tensor([7]))
    assert torch.equal(
        blocks[1].query_positions,
        torch.tensor([4, 5, 6, 7]),
    )
    assert torch.equal(
        blocks[1].kv_update_positions,
        torch.tensor([5, 6, 7]),
    )


def test_two_rank_shards_cover_dataset_once():
    rank_zero = shard_indices(total=7, rank=0, world_size=2)
    rank_one = shard_indices(total=7, rank=1, world_size=2)

    assert rank_zero == [0, 2, 4, 6]
    assert rank_one == [1, 3, 5]
    assert sorted(rank_zero + rank_one) == list(range(7))


def test_qa_f1_uses_best_reference_answer():
    score = qa_f1_score(
        "The South West Ultras fan club",
        ["unrelated", "South West Ultras fan club."],
    )

    assert score == pytest.approx(1.0, abs=1e-8)


def test_longbench_task_registry_covers_all_21_and_english_16():
    assert len(LONGBENCH_ALL_TASKS) == 21
    assert len(LONGBENCH_EN_TASKS) == 16
    assert set(LONGBENCH_ALL_TASKS) == set(LONGBENCH_TASK_MAX_NEW_TOKENS)
    assert LONGBENCH_TASK_MAX_NEW_TOKENS["hotpotqa"] == 32
    assert LONGBENCH_TASK_MAX_NEW_TOKENS["gov_report"] == 512


def test_longbench_retrieval_and_count_use_number_precision():
    retrieval, retrieval_metric = longbench_score(
        "passage_retrieval_en",
        "Paragraphs 3 and 4",
        ["Paragraph 3"],
    )
    count, count_metric = longbench_score(
        "passage_count",
        "There are 7, not 8.",
        ["7"],
    )

    assert retrieval == 0.5
    assert retrieval_metric == "retrieval"
    assert count == 0.5
    assert count_metric == "count"


def test_longbench_classification_uses_official_class_penalty():
    score, metric = longbench_score(
        "trec",
        "The label is DESC, not ENTY.",
        ["DESC"],
        all_classes=["DESC", "ENTY", "HUM"],
    )

    assert score == 0.5
    assert metric == "classification"


def test_longbench_rouge_uses_best_reference():
    score, metric = longbench_score(
        "gov_report",
        "the exact summary",
        ["unrelated", "the exact summary"],
    )

    assert score == pytest.approx(1.0, abs=1e-8)
    assert metric == "rouge_l"
