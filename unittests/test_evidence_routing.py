import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from prefilling_v2.data import (
    FullTextComplementaryCollator,
    RULER_VIEW_ID_BASE,
    RulerDenoisingSynthesizer,
)
from prefilling_v2.distributed import scale_local_loss_for_ddp
from prefilling_v2.evidence_routing import evidence_route_loss
from prefilling_v2.train_fulltext import (
    _split_dolma_ruler_views,
    _sync_ruler_window_target_counts,
    scale_sync_task_loss,
    validate_training_config,
)


class CharacterTokenizer:
    eos_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_ruler_evidence_labels_survive_landmark_insertion():
    tokenizer = CharacterTokenizer()
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
    collator = FullTextComplementaryCollator(
        mask_token_id=999,
        pad_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        lmk_token_id=998,
        chunk_size=64,
        ruler_mix_ratio=1.0,
        ruler_synthesizer=synthesizer,
    )
    batch = collator(
        [{"clean_ids": torch.full((63 * 8,), ord("x")), "sample_id": 7}]
    )
    assert batch["input_ids"].shape == (1, 512)
    assert batch["route_evidence_chunks"].shape == (1, 8)
    assert batch["route_evidence_chunks"].sum() >= 1
    assert batch["route_evidence_tokens"].shape == batch["input_ids"].shape
    assert batch["route_evidence_tokens"].sum() >= 1
    assert batch["labels"].ne(-100).sum() > 0


def test_sync_ruler_collator_emits_dolma_and_ruler_views():
    tokenizer = CharacterTokenizer()
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0, 1, 2))
    collator = FullTextComplementaryCollator(
        mask_token_id=999,
        pad_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        lmk_token_id=998,
        chunk_size=64,
        ruler_mix_ratio=0.0,
        ruler_every_step=True,
        ruler_synthesizer=synthesizer,
    )
    batch = collator(
        [{"clean_ids": torch.full((63 * 8,), ord("x")), "sample_id": 11}]
    )
    assert batch["input_ids"].shape[0] == 3
    ruler_rows = batch["view_ids"] >= RULER_VIEW_ID_BASE
    assert int(ruler_rows.sum().item()) == 1
    assert int((~ruler_rows).sum().item()) == 2
    dolma, ruler = _split_dolma_ruler_views(batch)
    assert dolma["input_ids"].shape[0] == 2
    assert ruler["input_ids"].shape[0] == 1
    assert int(ruler["target_count"].item()) >= 1
    assert int(dolma["target_count"].sum().item()) >= 1


def test_sync_all_tasks_collator_emits_dolma_and_three_ruler_views():
    tokenizer = CharacterTokenizer()
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0, 1, 2))
    collator = FullTextComplementaryCollator(
        mask_token_id=999,
        pad_token_id=0,
        eos_token_id=tokenizer.eos_token_id,
        lmk_token_id=998,
        chunk_size=64,
        ruler_mix_ratio=0.0,
        ruler_every_step=True,
        ruler_all_tasks=True,
        ruler_synthesizer=synthesizer,
    )
    batch = collator(
        [{"clean_ids": torch.full((63 * 8,), ord("x")), "sample_id": 11}]
    )
    assert batch["input_ids"].shape[0] == 5
    ruler_rows = batch["view_ids"] >= RULER_VIEW_ID_BASE
    assert int(ruler_rows.sum().item()) == 3
    assert int((~ruler_rows).sum().item()) == 2
    tasks = sorted(
        int(value) - RULER_VIEW_ID_BASE
        for value in batch["view_ids"][ruler_rows].tolist()
    )
    assert tasks == [0, 1, 2]


def test_sync_ruler_config_rejects_mix_and_copy_stack():
    cfg = {
        "attention_mode": "route",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_qcal_rank": 64,
        "route_trainable_scope": "lora_qcal_lmk",
        "lmk_token_mode": "mask_type",
        "route_chunk_aux_loss_weight": 0.0,
        "route_detach_fusion_weights": False,
        "route_dense_teacher_weight": 0.0,
        "route_sync_ruler_ce": True,
        "ruler_mix_ratio": 0.0,
        "local_window": 256,
        "initialize_from": "/ckpt",
        "max_length": 16384,
        "chunk_size": 64,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "num_workers": 0,
    }
    validate_training_config(cfg)
    try:
        validate_training_config({**cfg, "ruler_mix_ratio": 0.05})
    except ValueError as exc:
        assert "ruler_mix_ratio" in str(exc)
    else:
        raise AssertionError("expected mix rejection")
    try:
        validate_training_config({**cfg, "route_balanced_view_ce": True})
    except ValueError as exc:
        assert "stack" in str(exc)
    else:
        raise AssertionError("expected stack rejection")
    try:
        validate_training_config({**cfg, "route_detach_fusion_weights": True})
    except ValueError as exc:
        assert "live fusion" in str(exc)
    else:
        raise AssertionError("expected detach rejection")
    validate_training_config({**cfg, "route_allchunk_st_queries": 16})
    try:
        validate_training_config(
            {
                **cfg,
                "route_allchunk_st_queries": 16,
                "route_relaxation": "gumbel_softmax_topk",
            }
        )
    except ValueError as exc:
        assert "all-chunk ST" in str(exc)
    else:
        raise AssertionError("expected allchunk vs gumbel stack rejection")


def test_dense_sync_ruler_configs_are_valid():
    cfg = {
        "attention_mode": "dense",
        "non_route_attention": "dense",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_trainable_scope": "full",
        "lmk_token_mode": "mask",
        "route_sync_ruler_ce": True,
        "ruler_mix_ratio": 0.0,
        "max_length": 16384,
        "chunk_size": 64,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "num_workers": 0,
    }
    validate_training_config(cfg)
    validate_training_config(
        {
            **cfg,
            "route_sync_ruler_ce": False,
            "sync_ruler_all_tasks": True,
            "ruler_task_ids": [0, 1, 2],
        }
    )
    try:
        validate_training_config(
            {**cfg, "sync_ruler_all_tasks": True}
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected exclusive flag rejection")


def test_joint_ruler_mask_keeps_some_answer_tokens_visible():
    tokenizer = CharacterTokenizer()
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
    ids = torch.full((63 * 16,), ord("x"))
    full_ids, full_target, evidence = synthesizer.synthesize_with_evidence(
        ids, task_id=0
    )
    joint_ids, joint_target, joint_evidence = synthesizer.synthesize_with_evidence(
        ids, task_id=0, joint_mask=True, t_min=0.2, t_max=0.8
    )
    assert torch.equal(full_ids, joint_ids)
    assert torch.equal(evidence, joint_evidence)
    answer_start = int(torch.where(full_target)[0].min())
    full_answer = int(full_target[answer_start:].sum().item())
    joint_answer = int(joint_target[answer_start:].sum().item())
    assert 1 <= joint_answer < full_answer
    assert not bool((joint_target & evidence).any())
    assert int(joint_target[:answer_start].sum().item()) >= 1


def test_evidence_route_loss_only_backpropagates_through_route_query():
    query = torch.tensor(
        [[[[0.0, 1.0]], [[1.0, 0.0]]]], requires_grad=True
    )
    landmarks = torch.tensor(
        [[[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[-1.0, 0.0]]]]],
        requires_grad=True,
    )
    prior = torch.zeros(1, 3, 1, 1, requires_grad=True)
    local_lse = torch.zeros(1, 2, 1, 1, requires_grad=True)
    dropped = torch.zeros(1, 2, 3, dtype=torch.bool)
    query_mask = torch.tensor([[False, True]])
    evidence = torch.tensor([[True, False, False]])

    loss = evidence_route_loss(
        query, landmarks, prior, local_lse, dropped, query_mask, evidence
    )
    loss.backward()

    assert loss > 0
    assert query.grad is not None and torch.count_nonzero(query.grad)
    assert landmarks.grad is None
    assert prior.grad is None
    assert local_lse.grad is None


def test_evidence_route_loss_matches_max_pooled_router_priority():
    query = torch.tensor([[[[1.0], [1.0]]]], requires_grad=True)
    landmarks = torch.tensor(
        [[[[[2.0], [0.0]]], [[[0.0], [3.0]]], [[[-1.0], [-1.0]]]]]
    )
    prior = torch.zeros(1, 3, 1, 2)
    local_lse = torch.tensor([[[[10.0, 0.0]]]])
    dropped = torch.zeros(1, 1, 3, dtype=torch.bool)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    evidence = torch.tensor([[True, False, False]])

    actual = evidence_route_loss(
        query,
        landmarks,
        prior,
        local_lse,
        dropped,
        query_mask,
        evidence,
    )
    logits = landmarks[0, :, 0, :, 0].transpose(0, 1)
    total_lse = torch.logaddexp(local_lse[0, 0, 0], torch.logsumexp(logits, -1))
    priorities = (logits - total_lse[:, None]).amax(0)
    expected = -priorities.log_softmax(-1)[0]

    assert torch.allclose(actual, expected)


def test_qcal_only_evidence_config_contract():
    valid = {
        "attention_mode": "route",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_qcal_rank": 64,
        "route_trainable_scope": "qcal_only",
        "route_evidence_route_loss_weight": 0.01,
        "route_chunk_aux_loss_weight": 0.0,
        "initialize_from": "/checkpoint/source",
        "ruler_mix_ratio": 1.0,
        "max_length": 32768,
        "chunk_size": 64,
        "route_topk": 32,
        "route_token_budget": 512,
        "route_token_policy": "global_qk",
        "route_min_tokens_per_chunk": 1,
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "num_workers": 0,
    }
    validate_training_config(valid)

    invalid = dict(valid)
    invalid["ruler_mix_ratio"] = 0.0
    try:
        validate_training_config(invalid)
    except ValueError as exc:
        assert "RULER" in str(exc)
    else:
        raise AssertionError("evidence supervision accepted data without labels")


class _TinyCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, batch):
        gold = batch["labels"][batch["labels"].ne(-100)]
        positive = self.theta.expand(gold.numel())
        logits = torch.stack((positive, torch.zeros_like(positive)), dim=-1)
        return F.cross_entropy(logits, gold, reduction="sum"), gold.numel()


def _sync_batch(rank: int, unequal: bool) -> dict[str, torch.Tensor]:
    ruler_n = (1 if rank % 2 == 0 else 3) if unequal else 3
    labels = torch.full((3, 3), -100, dtype=torch.long)
    labels[0, 0] = 0
    labels[1, 0] = 0
    labels[2, :ruler_n] = rank % 2
    return {
        "labels": labels,
        "target_count": labels.ne(-100).sum(-1),
        "view_ids": torch.tensor([0, 1, RULER_VIEW_ID_BASE]),
    }


def _reference_sync_grad(world_size: int, accumulation: int, unequal: bool) -> float:
    ref = _TinyCE()
    sums = [0, 0]
    counts = [0, 0]
    for _ in range(accumulation):
        for rank in range(world_size):
            for task, batch in enumerate(_split_dolma_ruler_views(_sync_batch(rank, unequal))):
                loss, n = ref(batch)
                sums[task] = sums[task] + loss
                counts[task] += n
    objective = 0.5 * sums[0] / counts[0] + 0.5 * sums[1] / counts[1]
    objective.backward()
    return float(ref.theta.grad.item())


def _simulated_ddp_sync_grad(world_size: int, accumulation: int, unequal: bool) -> float:
    windows = [
        [_sync_batch(rank, unequal) for rank in range(world_size)]
        for _ in range(accumulation)
    ]
    rank_counts = [
        _sync_ruler_window_target_counts(
            [windows[micro][rank] for micro in range(accumulation)],
            device=torch.device("cpu"),
        )
        for rank in range(world_size)
    ]
    global_dolma = torch.stack([counts[0] for counts in rank_counts]).sum()
    global_ruler = torch.stack([counts[1] for counts in rank_counts]).sum()
    grads = []
    for rank in range(world_size):
        model = _TinyCE()
        for micro in range(accumulation):
            dolma, ruler = _split_dolma_ruler_views(windows[micro][rank])
            dolma_sum, _ = model(dolma)
            scale_sync_task_loss(
                dolma_sum, global_count=global_dolma, world_size=world_size
            ).backward()
            ruler_sum, _ = model(ruler)
            scale_sync_task_loss(
                ruler_sum, global_count=global_ruler, world_size=world_size
            ).backward()
        grads.append(model.theta.grad.detach().clone())
    return float(torch.stack(grads).mean().item())


def test_scale_sync_task_loss_is_half_ddp_token_mean():
    local_sum = torch.tensor(14.0)
    scaled = scale_sync_task_loss(
        local_sum, global_count=torch.tensor(55.0), world_size=4
    )
    assert abs(float(scaled.item()) - (0.5 * 4 * 14.0 / 55.0)) < 1e-6
    assert abs(
        float(scaled.item())
        - 0.5
        * float(
            scale_local_loss_for_ddp(
                local_sum, global_count=torch.tensor(55.0), world_size=4
            ).item()
        )
    ) < 1e-12
    quarter = scale_sync_task_loss(
        local_sum, global_count=torch.tensor(55.0), world_size=4, task_weight=0.25
    )
    assert abs(float(quarter.item()) - (0.25 * 4 * 14.0 / 55.0)) < 1e-6


def test_sync_window_counts_cover_accumulation_and_all_ranks():
    window = [_sync_batch(0, True), _sync_batch(0, True)]
    dolma_n, ruler_n = _sync_ruler_window_target_counts(
        window,
        device=torch.device("cpu"),
        all_reduce_sum=lambda value: value * 4,
    )
    assert float(dolma_n.item()) == 16.0
    assert float(ruler_n.item()) == 8.0


def _simulated_old_rank_mean_grad(world_size: int, accumulation: int, unequal: bool) -> float:
    windows = [
        [_sync_batch(rank, unequal) for rank in range(world_size)]
        for _ in range(accumulation)
    ]
    grads = []
    for rank in range(world_size):
        model = _TinyCE()
        for micro in range(accumulation):
            dolma, ruler = _split_dolma_ruler_views(windows[micro][rank])
            dolma_sum, dolma_n = model(dolma)
            (0.5 * dolma_sum / dolma_n).backward()
            ruler_sum, ruler_n = model(ruler)
            (0.5 * ruler_sum / ruler_n).backward()
        grads.append(model.theta.grad.detach().clone())
    return float(torch.stack(grads).mean().item())


def test_sync_backward_matches_global_task_token_mean_when_counts_unequal():
    actual = _simulated_ddp_sync_grad(4, 1, True)
    expected = _reference_sync_grad(4, 1, True)
    assert abs(actual - expected) < 1e-12
    old = _simulated_old_rank_mean_grad(4, 1, True)
    assert abs(old - expected) > 1e-6


def test_sync_backward_accumulation_does_not_double_grad():
    accum1 = _simulated_ddp_sync_grad(4, 1, False)
    accum2 = _simulated_ddp_sync_grad(4, 2, False)
    expected = _reference_sync_grad(4, 2, False)
    assert abs(accum2 - expected) < 1e-12
    assert abs(accum1 - accum2) < 1e-12
    old_double = _simulated_old_rank_mean_grad(4, 2, False)
    assert abs(old_double - 2.0 * expected) < 1e-12


def test_resume_skips_incomplete_and_refuses_silent_reinit(tmp_path: Path):
    helper = (
        Path(__file__).resolve().parents[2]
        / "nsa-dream-20260917/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
    )
    if not helper.is_file():
        helper = Path(
            "src/nsa-dream-20260917/scripts/from_dense16k/"
            "select_latest_complete_checkpoint.sh"
        )
    assert helper.is_file()
    files = (
        "checkpoint_manifest.json",
        "trainable_state.pt",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.pt",
    )

    def complete(step_dir: Path) -> None:
        step_dir.mkdir()
        for name in files:
            (step_dir / name).write_text("ok")

    complete(tmp_path / "step-50")
    (tmp_path / "step-100.incomplete").mkdir()
    for name in files[:2]:
        ((tmp_path / "step-100.incomplete") / name).write_text("partial")
    selected = subprocess.check_output(
        ["bash", "-lc", f"source '{helper}'; select_latest_complete_checkpoint '{tmp_path}'"],
        text=True,
    ).strip()
    assert selected.endswith("step-50")

    empty = tmp_path / "empty"
    empty.mkdir()
    none = subprocess.check_output(
        ["bash", "-lc", f"source '{helper}'; select_latest_complete_checkpoint '{empty}'"],
        text=True,
    ).strip()
    assert none == ""

    leftover = tmp_path / "leftover"
    leftover.mkdir()
    (leftover / "step-25.incomplete").mkdir()
    result = subprocess.run(
        ["bash", "-lc", f"source '{helper}'; select_latest_complete_checkpoint '{leftover}'"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "refusing to reinitialize" in result.stderr
