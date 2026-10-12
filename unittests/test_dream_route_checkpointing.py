import json
import random
from pathlib import Path

import pytest
import torch

from prefilling_v2.checkpointing import (
    load_trainable_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.entropy_bias_scale = torch.nn.Parameter(torch.tensor([0.5]))
        self.frozen_weight = torch.nn.Parameter(
            torch.tensor([9.0]), requires_grad=False
        )

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.marker").write_text("saved")


class TinyTokenizer:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.marker").write_text("saved")


def _optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
    )
    loss = (model.lora_weight.square().sum() + model.entropy_bias_scale.square())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return optimizer, scheduler


def test_only_rank_zero_writes_and_all_ranks_cross_checkpoint_barriers(tmp_path):
    model = TinyAdapterModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    barrier_calls = []

    result = save_training_checkpoint(
        model=model,
        tokenizer=TinyTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path / "rank-one-output",
        step=3,
        sampler_epoch=0,
        batches_in_epoch=12,
        config={"seed": 7},
        dataset_manifest_hash="dataset-sha",
        rank=1,
        world_size=2,
        barrier_fn=lambda: barrier_calls.append("barrier"),
        gather_object_fn=lambda state: [state, state],
        git_commit="test-commit",
    )

    assert result is None
    assert barrier_calls == ["barrier", "barrier"]
    assert not (tmp_path / "rank-one-output").exists()


def test_checkpoint_contains_every_resumable_state_and_restores_trainables(tmp_path):
    random.seed(123)
    torch.manual_seed(456)
    model = TinyAdapterModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    expected_trainables = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    output_dir = tmp_path / "output"

    path = save_training_checkpoint(
        model=model,
        tokenizer=TinyTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        step=7,
        sampler_epoch=2,
        batches_in_epoch=19,
        config={"seed": 7, "route_backend": "kernel_bidir"},
        dataset_manifest_hash="dataset-sha",
        rank=0,
        world_size=1,
        barrier_fn=lambda: None,
        gather_object_fn=lambda state: [state],
        git_commit="test-commit",
        resume_contract={
            "attention_mode": "dsa",
            "dsa_backend": "tilelang",
            "dsa_selected_attention_semantics": "fp32-v1",
        },
    )

    assert path == output_dir / "step-7"
    assert (path / "adapter" / "adapter.marker").is_file()
    assert (path / "tokenizer" / "tokenizer.marker").is_file()
    for filename in (
        "trainable_state.pt",
        "route_extra_trainable.pt",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.pt",
        "checkpoint_manifest.json",
    ):
        assert (path / filename).is_file()

    manifest = json.loads((path / "checkpoint_manifest.json").read_text())
    assert manifest["step"] == 7
    assert manifest["git_commit"] == "test-commit"
    assert manifest["dataset_manifest_hash"] == "dataset-sha"
    assert manifest["format_version"] == 2
    assert manifest["resume_contract"] == {
        "attention_mode": "dsa",
        "dsa_backend": "tilelang",
        "dsa_selected_attention_semantics": "fp32-v1",
    }
    trainer_state = torch.load(path / "trainer_state.pt", weights_only=False)
    assert trainer_state["sampler_epoch"] == 2
    assert trainer_state["batches_in_epoch"] == 19
    assert len(trainer_state["rng_states"]) == 1
    assert "python" in trainer_state["rng_states"][0]
    assert "torch" in trainer_state["rng_states"][0]

    with torch.no_grad():
        model.lora_weight.fill_(-10)
        model.entropy_bias_scale.fill_(-20)
        model.frozen_weight.fill_(-30)
    state = load_training_checkpoint(
        checkpoint_dir=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=0,
        restore_rng=False,
        expected_resume_contract={
            "attention_mode": "dsa",
            "dsa_backend": "tilelang",
            "dsa_selected_attention_semantics": "fp32-v1",
        },
    )

    for name, expected in expected_trainables.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected)
    # Frozen base weights are deliberately not part of the adapter checkpoint.
    torch.testing.assert_close(model.frozen_weight, torch.tensor([-30.0]))
    assert state["step"] == 7
    assert state["sampler_epoch"] == 2
    assert state["batches_in_epoch"] == 19
    assert state["resume_contract"]["dsa_backend"] == "tilelang"


def test_exact_resume_rejects_legacy_or_changed_dsa_numerics(tmp_path):
    model = TinyAdapterModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    path = save_training_checkpoint(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path / "output",
        step=1,
        sampler_epoch=0,
        batches_in_epoch=1,
        config={},
        dataset_manifest_hash="sha",
        rank=0,
        world_size=1,
        barrier_fn=lambda: None,
        gather_object_fn=lambda state: [state],
        git_commit="commit",
    )
    expected = {
        "attention_mode": "dsa",
        "dsa_backend": "tilelang",
        "dsa_selected_attention_semantics": "fp32-v1",
    }

    with pytest.raises(ValueError, match="resume_contract"):
        load_training_checkpoint(
            checkpoint_dir=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=0,
            restore_rng=False,
            expected_resume_contract=expected,
        )

    manifest_path = path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resume_contract"] = {
        **expected,
        "dsa_backend": "torch",
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        load_training_checkpoint(
            checkpoint_dir=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=0,
            restore_rng=False,
            expected_resume_contract=expected,
        )


def test_load_rejects_a_checkpoint_missing_a_current_trainable(tmp_path):
    model = TinyAdapterModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    path = save_training_checkpoint(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path / "output",
        step=1,
        sampler_epoch=0,
        batches_in_epoch=1,
        config={},
        dataset_manifest_hash="sha",
        rank=0,
        world_size=1,
        barrier_fn=lambda: None,
        gather_object_fn=lambda state: [state],
        git_commit="commit",
    )
    trainables = torch.load(path / "trainable_state.pt", weights_only=False)
    trainables.pop("lora_weight")
    torch.save(trainables, path / "trainable_state.pt")

    try:
        load_training_checkpoint(
            checkpoint_dir=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=0,
            restore_rng=False,
        )
    except ValueError as error:
        assert "trainable" in str(error)
    else:
        raise AssertionError("incomplete trainable state must fail")


def test_initialize_loads_only_trainables_and_leaves_optimizer_untouched(tmp_path):
    source = TinyAdapterModel()
    source_optimizer, source_scheduler = _optimizer_and_scheduler(source)
    path = save_training_checkpoint(
        model=source,
        tokenizer=None,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        output_dir=tmp_path / "source",
        step=4,
        sampler_epoch=1,
        batches_in_epoch=9,
        config={},
        dataset_manifest_hash="sha",
        rank=0,
        world_size=1,
        barrier_fn=lambda: None,
        gather_object_fn=lambda state: [state],
        git_commit="commit",
    )

    target = TinyAdapterModel()
    target_optimizer, _ = _optimizer_and_scheduler(target)
    optimizer_before = target_optimizer.state_dict()
    with torch.no_grad():
        target.lora_weight.fill_(-10)
        target.entropy_bias_scale.fill_(-20)
        target.frozen_weight.fill_(-30)

    loaded = load_trainable_checkpoint(checkpoint_dir=path, model=target)

    assert loaded == ["entropy_bias_scale", "lora_weight"]
    torch.testing.assert_close(target.lora_weight, source.lora_weight)
    torch.testing.assert_close(
        target.entropy_bias_scale, source.entropy_bias_scale
    )
    torch.testing.assert_close(target.frozen_weight, torch.tensor([-30.0]))
    assert target_optimizer.state_dict() == optimizer_before
