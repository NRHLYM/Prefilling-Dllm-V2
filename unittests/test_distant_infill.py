import torch

from prefilling_v2.data import (
    FullTextComplementaryCollator,
    synthesize_distant_span_copy,
)
from prefilling_v2.train_fulltext import DEFAULTS, validate_training_config


def _asymmetric_cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(
        {
            "attention_mode": "route",
            "route_backend": "kernel_bidir",
            "no_kernel_fallback": True,
            "route_qcal_rank": 64,
            "route_trainable_scope": "lora_qcal_lmk",
            "lmk_token_mode": "mask_type",
            "route_chunk_aux_loss_weight": 0.0,
            "route_detach_fusion_weights": False,
            "route_dense_teacher_weight": 0.0,
            "route_asymmetric_gate_ce": True,
            "ruler_mix_ratio": 0.0,
            "local_window": 256,
            "route_distant_min_gap": 256,
            "initialize_from": "/ckpt",
            "max_length": 2048,
            "chunk_size": 64,
            "gradient_accumulation_steps": 1,
        }
    )
    cfg.update(overrides)
    return cfg


def test_distant_span_stays_outside_local_window():
    clean = torch.arange(800, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(7)
    ids, targets, evidence = synthesize_distant_span_copy(
        clean, min_gap=256, span_min=8, span_max=16, generator=generator
    )
    target_pos = torch.where(targets)[0]
    evidence_pos = torch.where(evidence)[0]
    assert target_pos.numel() >= 8
    assert int(evidence_pos.max()) + 256 <= int(target_pos.min())
    assert torch.equal(ids[target_pos], ids[evidence_pos])
    assert not bool((targets & evidence).any())


def test_collator_emits_local_and_distant_views():
    collator = FullTextComplementaryCollator(
        mask_token_id=999,
        pad_token_id=0,
        eos_token_id=3,
        lmk_token_id=998,
        chunk_size=64,
        distant_infill=True,
        distant_min_gap=64,
        distant_span_min=8,
        distant_span_max=8,
    )
    batch = collator(
        [{"clean_ids": torch.arange(63 * 8, dtype=torch.long), "sample_id": 4}]
    )
    assert batch["input_ids"].shape[0] == 3
    assert int(batch["gate_live"].sum().item()) == 1
    assert int((~batch["gate_live"].bool()).sum().item()) == 2
    distant = batch["gate_live"].bool()
    assert int(batch["target_count"][distant].min().item()) >= 7


def test_asymmetric_config_rejects_ruler_mix_and_teacher():
    validate_training_config(_asymmetric_cfg())
    try:
        validate_training_config(_asymmetric_cfg(ruler_mix_ratio=0.2))
    except ValueError as exc:
        assert "ruler_mix_ratio" in str(exc)
    else:
        raise AssertionError("expected ruler_mix rejection")
    try:
        validate_training_config(
            _asymmetric_cfg(
                route_dense_teacher_weight=0.01,
                route_dense_teacher_source="dense",
            )
        )
    except ValueError as exc:
        assert "dense chunk KL" in str(exc)
    else:
        raise AssertionError("expected teacher rejection")
    try:
        validate_training_config(_asymmetric_cfg(route_detach_fusion_weights=True))
    except ValueError as exc:
        assert "live fusion" in str(exc)
    else:
        raise AssertionError("expected detach rejection")


def test_balanced_view_ce_rejects_mix_and_asymmetric_stack():
    cfg = _asymmetric_cfg(
        route_asymmetric_gate_ce=False,
        route_balanced_view_ce=True,
        route_distant_span_min=256,
        route_distant_span_max=512,
    )
    validate_training_config(cfg)
    try:
        validate_training_config({**cfg, "ruler_mix_ratio": 0.05})
    except ValueError as exc:
        assert "ruler_mix_ratio" in str(exc)
    else:
        raise AssertionError("expected mix rejection")
    try:
        validate_training_config(
            _asymmetric_cfg(route_balanced_view_ce=True)
        )
    except ValueError as exc:
        assert "stack" in str(exc)
    else:
        raise AssertionError("expected stack rejection")


def test_long_distant_span_stays_outside_window():
    clean = torch.arange(4000, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(3)
    ids, targets, evidence = synthesize_distant_span_copy(
        clean, min_gap=256, span_min=256, span_max=512, generator=generator
    )
    target_pos = torch.where(targets)[0]
    evidence_pos = torch.where(evidence)[0]
    assert int(target_pos.numel()) >= 256
    assert int(evidence_pos.max()) + 256 <= int(target_pos.min())
    assert torch.equal(ids[target_pos], ids[evidence_pos])


def test_multi_needle_copies_are_disjoint_and_distant():
    clean = torch.arange(4000, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(11)
    ids, targets, evidence = synthesize_distant_span_copy(
        clean,
        min_gap=256,
        span_min=256,
        span_max=512,
        generator=generator,
        num_needles=3,
        min_needle_sep=64,
    )
    evidence_pos = torch.where(evidence)[0]
    target_pos = torch.where(targets)[0]
    assert int(evidence_pos.max()) + 256 <= int(target_pos.min())
    runs = []
    start = int(evidence_pos[0])
    prev = start
    for pos in evidence_pos[1:].tolist():
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev + 1))
        start = pos
        prev = pos
    runs.append((start, prev + 1))
    assert len(runs) == 3
    for left, right in zip(runs, runs[1:]):
        assert right[0] - left[1] >= 64
    assert torch.equal(
        ids[target_pos].sort().values,
        ids[evidence_pos].sort().values,
    )


def test_visible_cue_is_not_masked():
    clean = torch.arange(4000, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(5)
    ids, targets, evidence = synthesize_distant_span_copy(
        clean,
        min_gap=256,
        span_min=256,
        span_max=256,
        generator=generator,
        num_needles=1,
        cue_len=16,
    )
    suffix = ids[-256:]
    suffix_mask = targets[-256:]
    assert not bool(suffix_mask[:16].any())
    assert bool(suffix_mask[16:].all())
    evidence_ids = ids[evidence]
    assert torch.equal(suffix, evidence_ids)
    assert torch.equal(suffix[:16], evidence_ids[:16])
    assert int(targets.sum().item()) == 240


def test_short_copy_masks_only_the_remainder():
    clean = torch.arange(4000, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(9)
    ids, targets, evidence = synthesize_distant_span_copy(
        clean,
        min_gap=256,
        span_min=32,
        span_max=64,
        generator=generator,
        num_needles=1,
        cue_len=16,
    )
    masked = int(targets.sum().item())
    copied = int(evidence.sum().item())
    assert 32 <= copied <= 64
    assert masked == copied - 16
    suffix = ids[-copied:]
    assert not bool(targets[-copied:][:16].any())
    assert bool(targets[-copied:][16:].all())
    assert torch.equal(suffix, ids[evidence])


def test_balanced_shortcopy_config_keeps_a_mask_remainder():
    cfg = _asymmetric_cfg(
        route_asymmetric_gate_ce=False,
        route_balanced_view_ce=True,
        route_distant_span_min=32,
        route_distant_span_max=64,
        route_distant_needles_max=1,
        route_distant_cue_len=16,
    )
    validate_training_config(cfg)
    try:
        validate_training_config({**cfg, "route_distant_cue_len": 32})
    except ValueError as exc:
        assert "remainder" in str(exc)
    else:
        raise AssertionError("expected cue remainder rejection")
