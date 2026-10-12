import torch

from prefilling_v2.data import FullTextComplementaryCollator
from prefilling_v2.longbench_eval import (
    build_plain_fastdllm_block_layouts,
    build_plain_generation_layout,
)


def test_plain_collator_keeps_contiguous_tokens_and_predictor_labels():
    collator = FullTextComplementaryCollator(
        mask_token_id=1,
        pad_token_id=0,
        eos_token_id=2,
        lmk_token_id=3,
        chunk_size=64,
        t_min=1.0,
        t_max=1.0,
        seed=7,
        insert_landmarks=False,
        pad_to=8,
        ruler_mix_ratio=0.0,
    )
    clean = torch.arange(5, 11)  # 6 tokens
    valid = torch.ones(6, dtype=torch.bool)
    segments = torch.ones(6, dtype=torch.long)
    batch = collator(
        [
            {
                "clean_ids": clean,
                "valid_tokens": valid,
                "segment_ids": segments,
                "sample_id": 0,
            }
        ]
    )
    # complementary views, first view
    ids = batch["input_ids"][0]
    assert ids.numel() == 8
    assert not bool(ids.eq(3).any()), "must not insert landmark tokens"
    labeled = batch["labels"][0].ne(-100)
    assert int(labeled.sum()) >= 1
    # labeled query is immediately before a MASK
    for pos in torch.where(labeled)[0].tolist():
        assert ids[pos + 1].item() == 1


def test_plain_generation_layout_has_no_landmark_holes():
    layout = build_plain_generation_layout(
        prompt_ids=[10, 11, 12, 13],
        answer_tokens=4,
        physical_length=16,
        mask_token_id=1,
        pad_token_id=0,
    )
    assert layout.landmark_positions.numel() == 0
    assert layout.answer_positions.tolist() == [4, 5, 6, 7]
    assert layout.predictor_positions.tolist() == [3, 4, 5, 6]
    blocks = build_plain_fastdllm_block_layouts(layout, 2)
    assert len(blocks) == 2
    assert blocks[0].affected_chunks.numel() == 0
    assert blocks[0].kv_update_positions.tolist() == [4, 5]
