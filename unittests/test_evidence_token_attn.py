import torch

from prefilling_v2.evidence_token_attn import evidence_token_attn_loss
from prefilling_v2.train_fulltext import validate_training_config


def test_evidence_token_attn_ce_hits_needle_and_trains_qk():
    q = torch.zeros(1, 8, 2, 2, requires_grad=True)
    k = torch.zeros(1, 8, 1, 2, requires_grad=True)
    with torch.no_grad():
        q[0, 7, :, 0] = 1.0
        k[0, 0, 0, 0] = 4.0
        k[0, 1, 0, 1] = 4.0
    indices = torch.zeros(1, 8, 1, 1, dtype=torch.long)
    query_mask = torch.zeros(1, 8, dtype=torch.bool)
    query_mask[0, 7] = True
    evidence = torch.zeros(1, 8, dtype=torch.bool)
    evidence[0, 0] = True
    key_valid = torch.ones(1, 8, dtype=torch.bool)
    local_weight = torch.full((1, 8, 2), 0.8)

    loss, stats = evidence_token_attn_loss(
        q,
        k,
        indices,
        query_mask,
        evidence,
        key_valid,
        chunk_size=8,
        local_weight=local_weight,
    )
    loss.backward()

    assert loss.item() < 0.5
    assert float(stats["remote_needle_qk_mass"]) > 0.7
    assert abs(float(stats["local_weight"]) - 0.8) < 1e-5
    assert q.grad is not None and torch.count_nonzero(q.grad)
    assert k.grad is not None and torch.count_nonzero(k.grad)


def test_evidence_token_attn_bf16_autocast_mask_is_finite():
    q = torch.zeros(1, 8, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 8, 1, 2, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        q[0, 7, :, 0] = 1.0
        k[0, 0, 0, 0] = 4.0
    indices = torch.zeros(1, 8, 1, 1, dtype=torch.long)
    query_mask = torch.zeros(1, 8, dtype=torch.bool)
    query_mask[0, 7] = True
    evidence = torch.zeros(1, 8, dtype=torch.bool)
    evidence[0, 0] = True
    key_valid = torch.ones(1, 8, dtype=torch.bool)
    device_type = "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        loss, stats = evidence_token_attn_loss(
            q, k, indices, query_mask, evidence, key_valid, chunk_size=8
        )
    assert torch.isfinite(loss)
    assert float(stats["remote_needle_qk_mass"]) > 0.0
    loss.backward()


def test_evidence_token_attn_multiple_queries_not_broadcast_against_kv_heads():
    # R=9 queries, Hkv=4: denom without keepdim used to collide these axes.
    seq_len, h_kv, groups, dim, n_q = 16, 4, 1, 2, 9
    h_q = h_kv * groups
    q = torch.zeros(1, seq_len, h_q, dim, requires_grad=True)
    k = torch.zeros(1, seq_len, h_kv, dim, requires_grad=True)
    with torch.no_grad():
        q[0, 7:16, :, 0] = 1.0
        k[0, 0, :, 0] = 4.0
    indices = torch.zeros(1, seq_len, h_kv, 1, dtype=torch.long)
    query_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    query_mask[0, 7:16] = True
    evidence = torch.zeros(1, seq_len, dtype=torch.bool)
    evidence[0, 0] = True
    key_valid = torch.ones(1, seq_len, dtype=torch.bool)
    loss, stats = evidence_token_attn_loss(
        q, k, indices, query_mask, evidence, key_valid, chunk_size=16
    )
    assert torch.isfinite(loss)
    assert float(stats["n_supervised"]) > 0
    loss.backward()


def test_evidence_token_attn_skips_when_needle_not_selected():
    q = torch.zeros(1, 8, 2, 2, requires_grad=True)
    k = torch.zeros(1, 8, 1, 2, requires_grad=True)
    indices = torch.ones(1, 8, 1, 1, dtype=torch.long)
    query_mask = torch.zeros(1, 8, dtype=torch.bool)
    query_mask[0, 7] = True
    evidence = torch.zeros(1, 8, dtype=torch.bool)
    evidence[0, 0] = True
    key_valid = torch.ones(1, 8, dtype=torch.bool)

    loss, stats = evidence_token_attn_loss(
        q, k, indices, query_mask, evidence, key_valid, chunk_size=4
    )
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert float(stats["remote_needle_qk_mass"]) == 0.0
    assert q.grad is not None
    assert torch.count_nonzero(q.grad) == 0


def test_s2_token_attn_config_contract():
    valid = {
        "attention_mode": "route",
        "route_backend": "kernel_bidir",
        "no_kernel_fallback": True,
        "route_qcal_rank": 64,
        "route_trainable_scope": "lora_qcal_lmk",
        "route_detach_fusion_weights": False,
        "route_dense_teacher_weight": 0.0,
        "route_evidence_token_attn_weight": 0.1,
        "route_chunk_aux_loss_weight": 0.0,
        "initialize_from": "/checkpoint/dense-500",
        "ruler_mix_ratio": 0.05,
        "max_length": 16384,
        "chunk_size": 64,
        "route_topk": 32,
        "route_token_budget": 0,
        "route_token_policy": "global_qk",
        "route_min_tokens_per_chunk": 1,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "num_workers": 0,
        "lmk_token_mode": "mask_type",
    }
    validate_training_config(valid)
    try:
        validate_training_config({**valid, "route_detach_fusion_weights": True})
    except ValueError as exc:
        assert "live fusion" in str(exc)
    else:
        raise AssertionError("detached fusion was accepted")
    try:
        validate_training_config({**valid, "route_allchunk_st_queries": 16})
    except ValueError as exc:
        assert "all-chunk" in str(exc)
    else:
        raise AssertionError("S4 stack was accepted")
