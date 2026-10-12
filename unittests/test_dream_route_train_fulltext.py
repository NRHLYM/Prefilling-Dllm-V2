import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from prefilling_v2.train_fulltext import (
    ResumableDataIterator,
    DenseDreamAttentionAdapter,
    _attention_mask_for_model,
    _gradient_audit,
    _gradient_group_norms,
    _landmark_inputs_embeds,
    _preserve_rng_state,
    configure_training_attention,
    audit_frozen_gradients,
    denoising_loss_sum_for_batch,
    install_external_lmk_embedding,
    install_mask_lmk_type_embedding,
    parse_args,
    resolve_landmark_token_id,
    ruler_sn_span_counts,
    split_packed_indices,
    weighted_supervised_count,
    validate_training_config,
)


class SampleDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"sample_id": index}


class EpochCollator:
    def __init__(self):
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, examples):
        return {
            "sample_ids": torch.tensor([item["sample_id"] for item in examples]),
            "epoch": torch.tensor(self.epoch),
        }


def test_preserve_rng_state_keeps_resumed_training_stream_stable():
    torch.manual_seed(1234)
    expected = torch.rand(8)

    torch.manual_seed(1234)
    with _preserve_rng_state(enabled=True):
        torch.randperm(32768)
    actual = torch.rand(8)

    torch.testing.assert_close(actual, expected)


def test_config_values_are_loaded_and_explicit_cli_values_win(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_path": "/model",
                "corpus_bin": "/data/train.bin",
                "corpus_meta": "/data/train.meta.json",
                "output_dir": "/output/from-config",
                "max_steps": 500,
                "route_backend": "kernel_bidir",
                "no_kernel_fallback": True,
            }
        )
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--max_steps",
            "3",
            "--output_dir",
            "/output/from-cli",
        ]
    )

    assert args.model_path == "/model"
    assert args.max_steps == 3
    assert args.output_dir == "/output/from-cli"
    assert args.no_kernel_fallback is True


def test_config_accepts_mixed_ruler_and_external_lmk(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_path": "/model",
                "corpus_bin": "/data/train.bin",
                "corpus_meta": "/data/train.meta.json",
                "output_dir": "/output",
                "max_length": 2048,
                "chunk_size": 64,
                "route_backend": "kernel_bidir",
                "no_kernel_fallback": True,
                "lmk_token_mode": "mask_type",
                "ruler_mix_ratio": 0.05,
                "ruler_task_ids": [0, 1, 2],
                "route_relaxation": "gumbel_softmax_topk",
                "route_temperature": 0.7,
                "route_gumbel_scale": 0.5,
                "route_token_relaxation": "gumbel_topk",
                "route_token_gumbel_scale": 0.25,
            }
        )
    )

    args = parse_args(["--config", str(config)])

    assert args.lmk_token_mode == "mask_type"
    assert args.ruler_mix_ratio == 0.05
    assert args.ruler_task_ids == [0, 1, 2]
    assert args.ruler_val_batches == 0
    assert args.route_relaxation == "gumbel_softmax_topk"
    assert args.route_temperature == 0.7
    assert args.route_gumbel_scale == 0.5
    assert args.route_token_relaxation == "gumbel_topk"
    assert args.route_token_gumbel_scale == 0.25


def test_config_accepts_hope_rope_scaling():
    validate_training_config(
        {
            "attention_mode": "route",
            "route_backend": "kernel_bidir",
            "no_kernel_fallback": True,
            "max_length": 16384,
            "model_max_position_embeddings": 16384,
            "chunk_size": 64,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            "model_rope_scaling": {
                "rope_type": "hope",
                "original_max_position_embeddings": 2048,
            },
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "num_workers": 0,
        }
    )


def test_hope_inrange_zeros_long_period_freqs():
    from prefilling_v2.hope import apply_hope_inrange

    class _Rope(nn.Module):
        def __init__(self):
            super().__init__()
            # High freq (short period) then low freq (period >> 2048).
            self.register_buffer(
                "inv_freq",
                torch.tensor([1.0, 1e-4], dtype=torch.float32),
            )

    model = nn.Module()
    model.rotary_emb = _Rope()
    patched = apply_hope_inrange(model, context_length=2048, period_multiplier=1.0)
    assert patched == 1
    assert float(model.rotary_emb.inv_freq[0]) == 1.0
    assert float(model.rotary_emb.inv_freq[1]) == 0.0


def test_config_accepts_yarn_model_override(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_path": "/model",
                "corpus_bin": "/data/train.bin",
                "corpus_meta": "/data/train.meta.json",
                "output_dir": "/output",
                "max_length": 32768,
                "chunk_size": 64,
                "route_backend": "kernel_bidir",
                "no_kernel_fallback": True,
                "model_max_position_embeddings": 32768,
                "model_rope_theta": 1000000.0,
                "model_rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 8192,
                },
            }
        )
    )

    args = parse_args(["--config", str(config)])

    assert args.model_max_position_embeddings == 32768
    assert args.model_rope_theta == 1000000.0
    assert args.model_rope_scaling["rope_type"] == "yarn"
    assert args.model_rope_scaling["factor"] == 4.0


def test_unknown_config_key_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"not_a_training_option": 1}))
    with pytest.raises(ValueError, match="Unknown config key"):
        parse_args(["--config", str(config)])


def test_dsa_warmup_config_accepts_indexer_only_initialization():
    config = {
        "attention_mode": "dsa",
        "max_length": 2048,
        "chunk_size": 64,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "dsa_topk": 2048,
        "dsa_aux_queries": 64,
        "dsa_aux_loss_weight": 1.0,
        "dsa_aux_loss_scope": "full",
        "dsa_lm_loss_weight": 0.0,
        "initialize_from": "/checkpoints/indexer-warmup",
    }

    validate_training_config(config)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_training_config(
            {**config, "resume_from": "/checkpoints/resume"}
        )
    with pytest.raises(ValueError, match="non-zero LM or auxiliary"):
        validate_training_config(
            {
                **config,
                "initialize_from": None,
                "dsa_aux_loss_weight": 0.0,
            }
        )


def test_validation_split_is_fixed_disjoint_and_spans_the_corpus():
    train_a, validation_a = split_packed_indices(1000, 100, seed=7)
    train_b, validation_b = split_packed_indices(1000, 100, seed=7)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 900
    assert len(validation_a) == 100
    assert set(train_a).isdisjoint(validation_a)
    assert min(validation_a) < 100
    assert max(validation_a) > 900


def test_resumable_iterator_replays_the_exact_next_batch_per_rank():
    dataset = SampleDataset()
    expected_next = []
    resumed_next = []
    rank_samples = []

    for rank in (0, 1):
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=7,
            drop_last=False,
        )
        collator = EpochCollator()
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collator,
        )
        iterator = ResumableDataIterator(loader, sampler, collator)
        seen = [int(next(iterator)["sample_ids"].item()) for _ in range(4)]
        expected = int(next(iterator)["sample_ids"].item())
        expected_next.append(expected)
        rank_samples.append(set(seen + [expected]))

        resumed = ResumableDataIterator(
            loader,
            sampler,
            collator,
            epoch=0,
            batches_in_epoch=4,
        )
        resumed_next.append(int(next(resumed)["sample_ids"].item()))
        assert int(next(resumed)["epoch"].item()) == 0

    assert resumed_next == expected_next
    assert rank_samples[0].isdisjoint(rank_samples[1])


def test_training_config_enforces_the_no_fallback_8k_layout():
    valid = {
        "max_length": 8192,
        "chunk_size": 64,
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 1,
    }
    validate_training_config(valid)

    with pytest.raises(ValueError, match="divisible"):
        validate_training_config({**valid, "max_length": 8191})
    with pytest.raises(ValueError, match="fallback"):
        validate_training_config({**valid, "no_kernel_fallback": False})
    with pytest.raises(ValueError, match="route_token_budget"):
        validate_training_config({**valid, "route_token_budget": 2048})
    with pytest.raises(ValueError, match="lora_target_modules"):
        validate_training_config(
            {**valid, "lora_target_modules": ["q_proj", "q_proj"]}
        )
    with pytest.raises(ValueError, match="model_max_position_embeddings"):
        validate_training_config(
            {**valid, "max_length": 32768, "model_max_position_embeddings": 8192}
        )
    with pytest.raises(ValueError, match="rope_type=yarn"):
        validate_training_config(
            {
                **valid,
                "model_rope_scaling": {"rope_type": "linear", "factor": 4.0},
            }
        )
    with pytest.raises(ValueError, match="route_temperature"):
        validate_training_config({**valid, "route_temperature": 0.0})
    with pytest.raises(ValueError, match="route_gumbel_scale"):
        validate_training_config({**valid, "route_gumbel_scale": -0.1})
    with pytest.raises(ValueError, match="route_token_relaxation"):
        validate_training_config({**valid, "route_token_relaxation": "gumbel_softmax_topk"})
    with pytest.raises(ValueError, match="route_token_gumbel_scale"):
        validate_training_config({**valid, "route_token_gumbel_scale": -0.1})


def test_gradient_group_norms_separate_lora_and_entropy_parameters():
    model = nn.Module()
    model.register_parameter(
        "lora_weight", nn.Parameter(torch.tensor([3.0, 4.0]))
    )
    model.register_parameter(
        "entropy_bias_scale", nn.Parameter(torch.tensor([1.0]))
    )
    model.register_parameter(
        "dream_route_lmk_embed", nn.Parameter(torch.tensor([2.0]))
    )
    model.register_parameter(
        "dream_route_lmk_type_embed", nn.Parameter(torch.tensor([5.0, 12.0]))
    )
    model.lora_weight.grad = torch.tensor([3.0, 4.0])
    model.entropy_bias_scale.grad = torch.tensor([12.0])
    model.dream_route_lmk_embed.grad = torch.tensor([8.0])
    model.dream_route_lmk_type_embed.grad = torch.tensor([5.0, 12.0])

    norms = _gradient_group_norms(model)

    assert norms["lora"] == 5.0
    assert norms["lora_other"] == 5.0
    assert norms["entropy"] == 12.0
    assert norms["external_lmk"] == 8.0
    assert norms["lmk_type"] == 13.0
    assert norms["other"] == 0.0


def test_qcal_detach_requires_a_router_loss():
    valid = {
        "attention_mode": "route",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_qcal_rank": 64,
        "route_trainable_scope": "lora_qcal_lmk",
        "lmk_token_mode": "mask_type",
        "route_chunk_aux_loss_weight": 0.0,
        "route_detach_fusion_weights": False,
        "route_dense_teacher_weight": 0.0,
        "initialize_from": "/ckpt",
    }
    validate_training_config(valid)
    validate_training_config(
        {
            **valid,
            "route_detach_fusion_weights": True,
            "route_dense_teacher_weight": 0.01,
            "route_dense_teacher_source": "dense",
        }
    )
    with pytest.raises(ValueError, match="detached fusion"):
        validate_training_config(
            {**valid, "route_detach_fusion_weights": True}
        )


def _route_ablation_base():
    return {
        "attention_mode": "route",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_qcal_rank": 64,
        "route_trainable_scope": "lora_qcal_lmk",
        "lmk_token_mode": "mask_type",
        "route_chunk_aux_loss_weight": 0.0,
        "route_relaxation": "none",
        "route_residual_weight": 0.0,
        "route_token_budget": 0,
        "route_lmk_kl_ste": False,
        "route_lmk_ce_ste": False,
        "route_allchunk_st_queries": 0,
        "route_detach_fusion_weights": False,
        "route_dense_teacher_weight": 0.01,
        "route_dense_teacher_source": "dense",
        "initialize_from": "/ckpt",
    }


def test_route_ablation_s1_to_s4_contracts():
    base = _route_ablation_base()
    validate_training_config(base)
    validate_training_config({**base, "route_dense_teacher_weight": 0.0})
    validate_training_config(
        {
            **base,
            "route_dense_teacher_weight": 0.0,
            "route_lmk_ce_ste": True,
        }
    )
    validate_training_config(
        {
            **base,
            "route_dense_teacher_weight": 0.0,
            "route_allchunk_st_queries": 16,
        }
    )
    with pytest.raises(ValueError, match="do not stack"):
        validate_training_config(
            {
                **base,
                "route_dense_teacher_weight": 0.0,
                "route_lmk_ce_ste": True,
                "route_allchunk_st_queries": 16,
            }
        )
    with pytest.raises(ValueError, match="all-chunk ST requires live fusion"):
        validate_training_config(
            {
                **base,
                "route_allchunk_st_queries": 16,
            }
        )


def test_allchunk_st_temperature_is_independent_of_fusion_tau():
    valid = {
        "max_length": 8192,
        "chunk_size": 64,
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 1,
        "attention_mode": "route",
        "route_qcal_rank": 64,
        "route_trainable_scope": "lora_qcal_lmk",
        "lmk_token_mode": "mask_type",
        "route_chunk_aux_loss_weight": 0.0,
        "route_relaxation": "none",
        "route_residual_weight": 0.0,
        "route_token_budget": 0,
        "route_lmk_kl_ste": False,
        "route_lmk_ce_ste": False,
        "route_detach_fusion_weights": False,
        "route_dense_teacher_weight": 0.0,
        "route_dense_teacher_source": "dense",
        "initialize_from": "/ckpt",
        "route_allchunk_st_queries": 16,
        "route_temperature": 1.0,
        "route_allchunk_st_temperature": 1.0,
        "route_allchunk_st_temperature_end": 0.3,
        "route_allchunk_st_anneal": "cosine",
    }
    validate_training_config(valid)
    with pytest.raises(ValueError, match="route_allchunk_st_temperature"):
        validate_training_config({**valid, "route_allchunk_st_temperature": 0.0})
    with pytest.raises(ValueError, match="route_allchunk_st_anneal"):
        validate_training_config({**valid, "route_allchunk_st_anneal": "gumbel"})


def test_dense_training_config_keeps_the_same_8k_data_contract():
    validate_training_config(
        {
            "attention_mode": "dense",
            "max_length": 8192,
            "chunk_size": 64,
            "route_backend": "kernel_bidir",
            "no_kernel_fallback": True,
            "gradient_accumulation_steps": 4,
            "micro_batch_size": 1,
        }
    )


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _OriginalAttention()


class _OriginalAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        del (
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
        )
        return hidden_states + 1, None, None


class _DenseDream(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([_Layer() for _ in range(layers)]))
        self.config = SimpleNamespace()


def test_dense_attention_mode_wraps_but_preserves_original_attention_math():
    model = _DenseDream()
    before = [layer.self_attn for layer in model.model.layers]
    args = SimpleNamespace(
        attention_mode="dense",
        route_interleave=4,
        local_window=128,
        chunk_size=64,
        route_topk=16,
        route_backend="kernel_bidir",
        no_kernel_fallback=True,
    )

    plan = configure_training_attention(model, args)

    assert all(
        isinstance(layer.self_attn, DenseDreamAttentionAdapter)
        for layer in model.model.layers
    )
    assert [layer.self_attn.source_attn for layer in model.model.layers] == before
    output = model.model.layers[0].self_attn(
        torch.zeros(1, 2, 3),
        sparse_keep_indices=torch.tensor([0]),
        sparse_prompt_len=2,
    )[0]
    assert torch.equal(output, torch.ones(1, 2, 3))
    assert plan.route_layers == []
    assert plan.sliding_window_layers == []
    assert plan.dense_layers == [0, 1, 2]


class _TinyEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, initializer_range=0.02)
        self.embed = nn.Embedding(5, 4)
        self.proj = nn.Linear(4, 5)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        return SimpleNamespace(logits=self.proj(inputs_embeds))


class _TinyDreamBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 4)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        del kwargs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _TinyDreamLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            dream_route_lmk_token_mode="mask",
            hidden_size=4,
            initializer_range=0.02,
        )
        self.model = _TinyDreamBackbone()
        self.lm_head = nn.Linear(4, 7, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_denoising_loss_only_projects_supervised_positions():
    model = _TinyDreamLM()
    args = SimpleNamespace(
        attention_mode="route",
        non_route_attention="dense",
        loss_chunk_size=1,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "segment_ids": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.tensor([[-100, 4, -100, 5]]),
    }

    loss_sum, target_count = denoising_loss_sum_for_batch(model, args, batch)

    hidden = model.model.embed_tokens(batch["input_ids"])
    expected = F.cross_entropy(
        model.lm_head(hidden[batch["labels"].ne(-100)]),
        torch.tensor([4, 5]),
        reduction="sum",
    )
    assert target_count.item() == 2
    assert torch.allclose(loss_sum, expected)


def test_dsa_indexer_only_loss_keeps_backbone_graph_with_zero_gradient(monkeypatch):
    model = _TinyDreamLM()
    model.dsa_indexer = nn.Linear(1, 1, bias=False)
    args = SimpleNamespace(
        attention_mode="dsa",
        non_route_attention="dense",
        loss_chunk_size=1,
        dsa_aux_loss_weight=1.0,
        dsa_lm_loss_weight=0.0,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "segment_ids": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.tensor([[-100, 4, -100, 5]]),
    }
    monkeypatch.setattr(
        "prefilling_v2.train_fulltext.prepare_dsa_index_losses",
        lambda _model: 1,
    )
    monkeypatch.setattr(
        "prefilling_v2.train_fulltext.collect_dsa_index_loss",
        lambda _model: model.dsa_indexer.weight.square().sum(),
    )

    loss_sum, target_count = denoising_loss_sum_for_batch(model, args, batch)
    loss_sum.backward()

    assert target_count.item() == 2
    assert torch.allclose(loss_sum, 2 * model.dsa_indexer.weight.square().sum())
    assert model.model.embed_tokens.weight.grad is not None
    assert torch.count_nonzero(model.model.embed_tokens.weight.grad) == 0
    assert torch.count_nonzero(model.dsa_indexer.weight.grad) > 0


def test_gradient_audit_can_allow_zero_lora_during_indexer_warmup():
    model = nn.Module()
    model.register_parameter("lora_weight", nn.Parameter(torch.tensor([1.0])))
    model.lora_weight.grad = torch.zeros_like(model.lora_weight)

    stats = _gradient_audit(model, require_lora_nonzero=False)

    assert stats["lora_nonzero"] == 0
    with pytest.raises(RuntimeError, match="all LoRA gradients"):
        _gradient_audit(model)


def test_external_lmk_embedding_replaces_out_of_vocab_input_and_gets_grad():
    model = _TinyEmbeddingModel()
    args = SimpleNamespace(lmk_token_mode="external")
    tokenizer = SimpleNamespace(vocab_size=5, mask_token_id=3)

    lmk_token_id = install_external_lmk_embedding(model, args, tokenizer)
    assert lmk_token_id == 5
    assert resolve_landmark_token_id(args, tokenizer, model) == 5

    outputs = model(input_ids=torch.tensor([[0, 5, 1]]))
    loss = outputs.logits[0, 1].sum()
    loss.backward()

    assert model.dream_route_lmk_embed.grad is not None
    assert torch.count_nonzero(model.dream_route_lmk_embed.grad) > 0


def test_mask_lmk_type_embedding_offsets_only_landmark_masks_and_gets_grad():
    model = _TinyDreamLM()
    args = SimpleNamespace(lmk_token_mode="mask_type")
    tokenizer = SimpleNamespace(vocab_size=7, mask_token_id=3)

    lmk_token_id = install_mask_lmk_type_embedding(model, args, tokenizer)
    assert lmk_token_id == 3
    assert resolve_landmark_token_id(args, tokenizer, model) == 3

    input_ids = torch.tensor([[0, 3, 3, 1]])
    landmark_mask = torch.tensor([[False, False, True, False]])
    model_input_ids, inputs_embeds = _landmark_inputs_embeds(
        model,
        input_ids,
        landmark_mask,
    )

    assert model_input_ids is None
    baseline = model.get_input_embeddings()(input_ids)
    assert torch.allclose(inputs_embeds[0, 1], baseline[0, 1])
    assert torch.allclose(
        inputs_embeds[0, 2],
        baseline[0, 2] + model.dream_route_lmk_type_embed.to(baseline.dtype),
    )

    outputs = model.model(inputs_embeds=inputs_embeds)
    loss = outputs.last_hidden_state[0, 2].sum()
    loss.backward()

    assert model.dream_route_lmk_type_embed.grad is not None
    assert torch.count_nonzero(model.dream_route_lmk_type_embed.grad) > 0


def test_route_with_dense_non_route_layers_reports_the_7_0_21_plan(monkeypatch):
    model = _DenseDream(layers=28)
    original = [layer.self_attn for layer in model.model.layers]
    args = SimpleNamespace(
        attention_mode="route",
        non_route_attention="dense",
        route_interleave=4,
        local_window=128,
        chunk_size=64,
        route_topk=16,
        route_backend="kernel_bidir",
        no_kernel_fallback=True,
    )

    class _Plan:
        route_layers = [3, 7, 11, 15, 19, 23, 27]
        sliding_window_layers = []
        dense_layers = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        16, 17, 18, 20, 21, 22, 24, 25, 26]

    def _fake_install(actual_model, **kwargs):
        assert actual_model is model
        assert kwargs["non_route_attention"] == "dense"
        return _Plan()

    monkeypatch.setattr(
        "prefilling_v2.train_fulltext.install_dream_sparse_attention",
        _fake_install,
    )
    plan = configure_training_attention(model, args)

    assert plan.route_layers == _Plan.route_layers
    assert plan.sliding_window_layers == []
    assert plan.dense_layers == _Plan.dense_layers
    assert all(
        isinstance(model.model.layers[index].self_attn, DenseDreamAttentionAdapter)
        for index in plan.dense_layers
    )
    assert all(
        model.model.layers[index].self_attn.source_attn is original[index]
        for index in plan.dense_layers
    )


def test_packed_documents_use_a_bidirectional_block_diagonal_mask():
    args = SimpleNamespace(
        attention_mode="route",
        non_route_attention="dense",
    )

    actual = _attention_mask_for_model(
        args,
        torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
        torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long),
    )

    expected = torch.tensor(
        [[
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]],
        dtype=torch.bool,
    ).unsqueeze(1)
    assert torch.equal(actual, expected)


def test_packed_document_mask_rejects_invalid_segment_metadata():
    args = SimpleNamespace(attention_mode="route")
    with pytest.raises(ValueError, match="positive segment"):
        _attention_mask_for_model(
            args,
            torch.tensor([[1, 1]]),
            torch.tensor([[1, 0]]),
        )


def test_dense_attention_uses_implicit_full_mask_and_rejects_padding():
    args = SimpleNamespace(attention_mode="dense")

    assert _attention_mask_for_model(
        args,
        torch.ones(2, 8, dtype=torch.long),
    ) == "full"
    with pytest.raises(ValueError, match="fully valid"):
        _attention_mask_for_model(
            args,
            torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )


def test_frozen_gradient_audit_detects_accidental_base_updates():
    model = torch.nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(RuntimeError, match="frozen"):
        audit_frozen_gradients(model)


def test_weighted_supervised_count_uses_ruler_answer_alpha():
    batch = {
        "labels": torch.tensor([[-100, 4, -100, 5], [-100, 6, -100, -100]]),
        "view_ids": torch.tensor([0, 100]),
    }
    assert weighted_supervised_count(batch, 1.0) == 3.0
    assert weighted_supervised_count(batch, 2.0) == 4.0


def test_ruler_answer_ce_weight_scales_numerator_and_denominator():
    model = _TinyDreamLM()
    args = SimpleNamespace(
        attention_mode="route",
        non_route_attention="dense",
        loss_chunk_size=1,
        ruler_answer_ce_weight=1.0,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "segment_ids": torch.ones(2, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0).expand(2, -1),
        "labels": torch.tensor([[-100, 4, -100, -100], [-100, 5, -100, -100]]),
        "view_ids": torch.tensor([0, 100]),
    }
    loss1, count1 = denoising_loss_sum_for_batch(model, args, batch)
    args.ruler_answer_ce_weight = 2.0
    loss2, count2 = denoising_loss_sum_for_batch(model, args, batch)
    hidden = model.model.embed_tokens(batch["input_ids"])
    dolma = F.cross_entropy(model.lm_head(hidden[0, 1:2]), torch.tensor([4]), reduction="sum")
    ruler = F.cross_entropy(model.lm_head(hidden[1, 1:2]), torch.tensor([5]), reduction="sum")
    assert count1.item() == 2
    assert count2.item() == 3
    assert torch.allclose(loss1, dolma + ruler)
    assert torch.allclose(loss2, dolma + 2 * ruler)
    assert model._last_dolma_targets == 1
    assert model._last_ruler_targets == 1


def test_ruler_sn_span_counts_splits_digits_from_eos():
    eos_id = 99
    gold = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, eos_id])
    # 2/8 digits + EOS: token acc 3/9, digit acc 2/8, not digit-EM.
    pred = torch.tensor([1, 0, 3, 0, 0, 0, 0, 0, eos_id])
    counts = ruler_sn_span_counts(pred, gold, eos_id=eos_id)
    assert counts["token_ok"].item() == 3
    assert counts["token_n"].item() == 9
    assert counts["digit_ok"].item() == 2
    assert counts["digit_n"].item() == 8
    assert counts["eos_ok"].item() == 1
    assert counts["eos_n"].item() == 1
    assert counts["digit_em"].item() == 0
    assert counts["seq_em"].item() == 0

    digit_pred = gold.clone()
    digit_pred[-1] = 0
    perfect_digits = ruler_sn_span_counts(digit_pred, gold, eos_id=eos_id)
    assert perfect_digits["digit_em"].item() == 1
    assert perfect_digits["seq_em"].item() == 0
    assert perfect_digits["eos_ok"].item() == 0
    assert perfect_digits["token_ok"].item() == 8
