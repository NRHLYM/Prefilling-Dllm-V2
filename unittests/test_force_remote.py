import torch

from prefilling_v2.data import (
    FullTextComplementaryCollator,
    synthesize_one_token_key_value,
)
from prefilling_v2.force_remote import apply_forced_remote_gate, splice_oracle_indices
from prefilling_v2.train_fulltext import DEFAULTS, validate_training_config


def _force_cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(
        {
            "attention_mode": "route",
            "route_backend": "kernel_bidir",
            "no_kernel_fallback": True,
            "route_qcal_rank": 64,
            "route_trainable_scope": "qcal_lmk",
            "lmk_token_mode": "mask_type",
            "route_chunk_aux_loss_weight": 0.0,
            "route_detach_fusion_weights": False,
            "route_dense_teacher_weight": 0.0,
            "route_force_remote_unit": True,
            "ruler_mix_ratio": 0.0,
            "local_window": 256,
            "route_distant_min_gap": 256,
            "route_distant_span_min": 2,
            "route_distant_span_max": 2,
            "route_distant_cue_len": 1,
            "route_distant_needles_min": 1,
            "route_distant_needles_max": 1,
            "initialize_from": "/ckpt",
            "max_length": 2048,
            "chunk_size": 64,
            "gradient_accumulation_steps": 1,
        }
    )
    cfg.update(overrides)
    return cfg


def _ascii_encode(text: str):
    return torch.tensor([ord(ch) for ch in text], dtype=torch.long)


def test_one_token_key_value_is_distinct_and_outside_window():
    clean = torch.arange(800, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(7)
    ids, targets, evidence = synthesize_one_token_key_value(
        clean, min_gap=256, generator=generator, encode_fn=_ascii_encode
    )
    target_pos = int(torch.where(targets)[0].item())
    evidence_pos = int(torch.where(evidence)[0].item())
    assert int(targets.sum().item()) == 1
    assert target_pos == 799
    assert evidence_pos + 256 < target_pos
    assert int(ids[target_pos].item()) != int(ids[target_pos - 1].item())
    assert int(ids[target_pos].item()) == int(ids[evidence_pos].item())
    val_wrap = _ascii_encode(" VALUE=")
    key_wrap = _ascii_encode("KEY=")
    assert torch.equal(ids[target_pos - val_wrap.numel() : target_pos], val_wrap)
    key_pos = target_pos - val_wrap.numel() - 1
    assert torch.equal(ids[key_pos - key_wrap.numel() : key_pos], key_wrap)
    assert int(ids[key_pos].item()) != int(ids[target_pos].item())
    assert not bool((targets & evidence).any())


def test_collator_distant_only_emits_one_value_mask():
    collator = FullTextComplementaryCollator(
        mask_token_id=999,
        pad_token_id=0,
        eos_token_id=3,
        lmk_token_id=998,
        chunk_size=64,
        distant_infill=True,
        distant_only=True,
        distant_min_gap=64,
        distant_span_min=2,
        distant_span_max=2,
        distant_cue_len=1,
    )
    batch = collator(
        [{"clean_ids": torch.arange(63 * 8, dtype=torch.long), "sample_id": 4}]
    )
    assert batch["input_ids"].shape[0] == 1
    assert int(batch["target_count"].item()) == 1
    assert bool(batch["gate_live"].item())
    assert int((batch["labels"] != -100).sum().item()) == 1
    label_pos = int((batch["labels"][0] != -100).nonzero(as_tuple=False).flatten()[0].item())
    mask_id = 999
    mask_pos = int((batch["input_ids"][0] == mask_id).nonzero(as_tuple=False).flatten()[0].item())
    assert int((batch["input_ids"][0] == mask_id).sum().item()) == 1
    assert label_pos != mask_pos
    assert int(batch["input_ids"][0, mask_pos].item()) == mask_id
    assert int(batch["input_ids"][0, label_pos].item()) != mask_id
    assert int(batch["route_evidence_chunks"].sum().item()) >= 1


def test_force_remote_config_freezes_lora_and_rejects_stacks():
    validate_training_config(_force_cfg())
    try:
        validate_training_config(_force_cfg(route_trainable_scope="lora_qcal_lmk"))
    except ValueError as exc:
        assert "qcal_lmk" in str(exc)
    else:
        raise AssertionError("expected frozen-LoRA rejection")
    try:
        validate_training_config(_force_cfg(route_balanced_view_ce=True))
    except ValueError as exc:
        assert "stack" in str(exc)
    else:
        raise AssertionError("expected stack rejection")
    try:
        validate_training_config(_force_cfg(route_detach_fusion_weights=True))
    except ValueError as exc:
        assert "live remote" in str(exc)
    else:
        raise AssertionError("expected detach rejection")


def test_forced_remote_renormalizes_and_routes_gradient():
    remote = torch.tensor([[[[0.2, 0.1]]]], dtype=torch.float32, requires_grad=True)
    local = torch.tensor([[[0.7]]], dtype=torch.float32, requires_grad=True)
    indices = torch.tensor([[[[3, 8]]]], dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    new_remote, new_local, new_idx = apply_forced_remote_gate(
        remote, local, indices, query_mask=mask, ablation="none"
    )
    torch.testing.assert_close(new_local, torch.zeros_like(local))
    torch.testing.assert_close(new_remote.sum(-1), torch.ones(1, 1, 1))
    torch.testing.assert_close(new_idx, indices)
    new_remote.sum().backward()
    assert remote.grad is not None and float(remote.grad.abs().sum()) > 0
    assert local.grad is None


def test_gate_bce_pushes_remote_and_requires_oracle():
    from prefilling_v2.force_remote import fusion_gate_bce

    logit = torch.tensor([[[-2.0, -1.0]]], dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([[True]])
    loss = fusion_gate_bce(logit, mask, target_remote=True)
    loss.backward()
    assert logit.grad is not None
    assert float(logit.grad.sum()) < 0
    validate_training_config(
        _force_cfg(
            route_force_remote_oracle_route=True,
            route_gate_bce_weight=1.0,
            route_distant_min_gap=256,
        )
    )
    try:
        validate_training_config(_force_cfg(route_gate_bce_weight=1.0))
    except ValueError as exc:
        assert "oracle" in str(exc)
    else:
        raise AssertionError("expected oracle-route requirement")
    validate_training_config(
        _force_cfg(
            route_force_remote_oracle_route=True,
            route_gate_bce_weight=1.0,
            route_gate_ce_force=True,
            route_qcal_max_grad_norm=1.0,
            route_distant_min_gap=256,
        )
    )
    try:
        validate_training_config(
            _force_cfg(
                route_force_remote_oracle_route=True,
                route_gate_ce_force=True,
                route_distant_min_gap=256,
            )
        )
    except ValueError as exc:
        assert "route_gate_ce_force" in str(exc)
    else:
        raise AssertionError("expected gate_ce_force to require BCE")
    validate_training_config(
        _force_cfg(
            route_force_remote_oracle_route=True,
            route_gate_bce_weight=1.0,
            route_gate_ce_force=True,
            route_trainable_scope="lora_qcal_lmk",
            route_lora_q_only=True,
            route_lora_q_lr=5e-5,
            route_qcal_lr=2e-4,
            route_lora_q_max_grad_norm=1.0,
            route_qcal_max_grad_norm=1.0,
            lr_schedule="constant",
            route_distant_min_gap=256,
        )
    )


def test_remote_off_and_shuffle_ablations():
    remote = torch.tensor([[[[0.4, 0.1]]]], dtype=torch.float32)
    local = torch.tensor([[[0.5]]], dtype=torch.float32)
    indices = torch.tensor([[[[1, 2]]]], dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    off_remote, off_local, off_idx = apply_forced_remote_gate(
        remote, local, indices, query_mask=mask, ablation="off"
    )
    torch.testing.assert_close(off_remote, torch.zeros_like(remote))
    torch.testing.assert_close(off_local, torch.ones_like(local))
    torch.testing.assert_close(off_idx, indices)
    found = False
    for seed in range(32):
        shuffled_remote, shuffled_local, shuffled_idx = apply_forced_remote_gate(
            remote, local, indices, query_mask=mask, ablation="shuffle", seed=seed
        )
        if not torch.equal(shuffled_idx, indices):
            found = True
            torch.testing.assert_close(shuffled_local, torch.zeros_like(local))
            torch.testing.assert_close(shuffled_remote.sum(-1), torch.ones(1, 1, 1))
            break
    assert found


def test_force_remote_rejects_gap_inside_swa():
    try:
        validate_training_config(
            _force_cfg(swa_local_window=512, route_distant_min_gap=256, max_length=2048)
        )
    except ValueError as exc:
        assert "SWA" in str(exc)
    else:
        raise AssertionError("expected SWA-gap rejection")


def test_gate_affine_only_config():
    validate_training_config(
        _force_cfg(
            route_force_remote_oracle_route=True,
            route_gate_bce_weight=1.0,
            route_gate_ce_force=True,
            route_gate_affine_only=True,
            route_gate_offset_lr=0.02,
            route_gate_scale_lr=0.01,
            route_gate_cal_clip_each=True,
            route_distant_min_gap=256,
        )
    )
    try:
        validate_training_config(
            _force_cfg(
                route_force_remote_oracle_route=True,
                route_gate_bce_weight=1.0,
                route_gate_offset_only=True,
                route_gate_affine_only=True,
                route_distant_min_gap=256,
            )
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected affine/offset exclusivity")


def test_gate_layerwise_affine_config():
    from prefilling_v2.train_fulltext import _as_float_map, _as_int_list

    validate_training_config(
        _force_cfg(
            route_force_remote_oracle_route=True,
            route_gate_bce_weight=1.0,
            route_gate_ce_force=True,
            route_gate_affine_only=True,
            route_gate_train_offset_layers=[0],
            route_gate_train_scale_layers=[6],
            route_gate_offset_layer_lr={"0": 0.08},
            route_gate_scale_layer_lr={"6": 0.003},
            route_gate_offset_layer_clip={"0": 8.0},
            route_gate_scale_layer_clip={"6": 8.0},
            route_gate_cal_init_from="/ckpt/affine",
            route_distant_min_gap=256,
        )
    )
    assert _as_int_list([0]) == [0]
    assert _as_float_map({"6": 0.003})[6] == 0.003
    try:
        validate_training_config(
            _force_cfg(
                route_force_remote_oracle_route=True,
                route_gate_bce_weight=1.0,
                route_gate_train_offset_layers=[0],
                route_distant_min_gap=256,
            )
        )
    except ValueError as exc:
        assert "route_gate_affine_only" in str(exc)
    else:
        raise AssertionError("expected layer freeze to require affine-only")


def test_gate_affine_matches_offset_and_scales_logit():
    from prefilling_v2.attention import _route_weights_torch

    scores = torch.tensor([[[[2.0, 0.5]], [[-1.0, 0.25]]]], dtype=torch.float32)
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.long)
    prior = torch.zeros(1, 4, 1, 1)
    local = torch.tensor([[[0.0], [1.5]]], dtype=torch.float32)
    kwargs = dict(
        scores=scores,
        indices=indices,
        prior_bias=prior,
        local_lse=local,
        temperature=1.0,
        return_gate_logit=True,
    )
    _, _, native = _route_weights_torch(**kwargs)
    affine_remote, affine_local, affine_logit = _route_weights_torch(
        **kwargs, gate_scale=torch.tensor(1.0), gate_offset=torch.tensor(0.7)
    )
    offset_remote, offset_local, offset_logit = _route_weights_torch(
        **kwargs, gate_offset=torch.tensor(0.7)
    )
    torch.testing.assert_close(affine_logit, offset_logit)
    torch.testing.assert_close(affine_remote, offset_remote)
    torch.testing.assert_close(affine_local, offset_local)
    _, _, scaled = _route_weights_torch(
        **kwargs, gate_scale=torch.tensor(0.5), gate_offset=torch.tensor(0.0)
    )
    torch.testing.assert_close(scaled, native * 0.5)


def test_l0_local_control_and_two_class_metrics():
    from prefilling_v2.force_remote import (
        local_control_query_mask,
        two_class_logit_metrics,
    )

    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 7] = True
    control = local_control_query_mask(mask, shift=2)
    assert bool(control[0, 5])
    assert not bool(control[0, 7])
    pos = torch.tensor([2.0, 1.0, 0.5])
    neg = torch.tensor([-1.0, -0.5, 0.0])
    stats = two_class_logit_metrics(pos, neg)
    assert stats["n_pos"] == 3
    assert stats["pos_frac"] == 0.5
    assert stats["delta_pos_minus_neg"] > 0
    assert stats["auc"] == 1.0
    assert stats["bce_grad_y1"] < 0
    assert stats["sign_ok"] == 1.0
    flipped = two_class_logit_metrics(neg, pos)
    assert flipped["auc"] == 0.0
    assert flipped["sign_ok"] == 0.0


def test_gate_vector_stats_and_high_mass_fraction():

    logit = torch.tensor([[[-2.0, 2.0]]], dtype=torch.float32)
    remote = torch.tensor([[[[0.1, 0.0], [0.9, 0.0]]]], dtype=torch.float32)
    mask = torch.tensor([[True]])
    flat_logit, flat_mass = labeled_gate_vectors(logit, remote, mask)
    stats = summarize_gate_vectors(flat_logit, flat_mass)
    assert stats["live_frac_w_gt_0.8"] == 0.5
    assert stats["gate_logit_p90"] >= stats["gate_logit_p50"]
    assert stats["live_gate_bce"] > 0
    from prefilling_v2.force_remote import layer_gate_metrics

    layered = layer_gate_metrics([{"logit": flat_logit, "mass": flat_mass}])
    assert "l0_live_w_remote" in layered
    assert "l0_gate_logit_mean" in layered
    indices = torch.tensor([[[[7, 8]]]], dtype=torch.long)
    scores = torch.tensor([[[[0.1, 0.9]]]], dtype=torch.float32)
    evidence = torch.zeros(1, 12, dtype=torch.bool)
    evidence[0, 3] = True
    mask = torch.tensor([[False, True]])
    indices = indices.expand(1, 2, 1, 2).clone()
    scores = scores.expand(1, 2, 1, 2).clone()
    new_idx, new_scores = splice_oracle_indices(indices, scores, evidence, mask)
    assert int(new_idx[0, 0, 0, 0].item()) == 7
    assert int(new_idx[0, 1, 0, 0].item()) == 3
    torch.testing.assert_close(new_scores, scores)


def test_gumbel_softmax_topk_st_matches_hard_forward_and_soft_grad():
    from prefilling_v2.attention import (
        _route_weights_torch,
        _straight_through,
    )

    scores = torch.tensor([[[[2.0, 0.5]]]], dtype=torch.float32, requires_grad=True)
    indices = torch.tensor([[[[0, 1]]]], dtype=torch.long)
    prior = torch.zeros(1, 4, 1, 1)
    local = torch.tensor([[[0.25]]], dtype=torch.float32)
    kwargs = dict(
        scores=scores,
        indices=indices,
        prior_bias=prior,
        local_lse=local,
        temperature=1.0,
        return_gate_logit=True,
    )
    hard_r, hard_l, _ = _route_weights_torch(**kwargs, selected_gumbel=None)
    noise = torch.ones_like(scores)
    soft_r, soft_l, _ = _route_weights_torch(**kwargs, selected_gumbel=noise)
    st_r = _straight_through(hard_r, soft_r)
    st_l = _straight_through(hard_l, soft_l)
    torch.testing.assert_close(st_r, hard_r)
    torch.testing.assert_close(st_l, hard_l)
    (st_r.sum() + st_l.sum()).backward()
    g_st = scores.grad.detach().clone()
    scores.grad = None
    soft_r, soft_l, _ = _route_weights_torch(**kwargs, selected_gumbel=noise)
    (soft_r.sum() + soft_l.sum()).backward()
    torch.testing.assert_close(g_st, scores.grad)

