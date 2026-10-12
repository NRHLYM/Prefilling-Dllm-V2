import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@pytest.mark.parametrize("topk", [3, 5, 16, 31])
@pytest.mark.parametrize("training", [False, True])
def test_sort_preserves_logical_budget_and_invalid_slots(topk, training):
    from ops.topk_head_softmax import sort_topk_indices_kernel

    torch.manual_seed(71)
    indices = torch.randint(0, 128, (1, 35, 4, topk), dtype=torch.int32, device="cuda")
    indices[torch.rand(indices.shape, device="cuda") < 0.3] = -1
    indices[:, 0] = -1
    original = indices.clone()
    kernel = sort_topk_indices_kernel(1, h_kv=4, topk=topk,
        seq_len=35 if training else None, is_training=training)
    actual = kernel(indices)
    sentinel = torch.iinfo(torch.int32).max
    expected = indices.masked_fill(indices < 0, sentinel).sort(-1).values
    expected.masked_fill_(expected == sentinel, -1)
    assert actual.shape == indices.shape
    assert torch.equal(actual, expected)
    assert torch.equal(indices, original)
    assert torch.equal((actual >= 0).sum(-1), (indices >= 0).sum(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@torch.inference_mode()
def test_g7_topk31_returns_31_chunks_and_aligned_raw_scores():
    from prefilling_v2.routing import route_topk_g7

    torch.manual_seed(73)
    q = torch.randn(1, 35, 28, 128, dtype=torch.bfloat16, device="cuda")
    landmarks = torch.randn(1, 64, 4, 7, 128, dtype=torch.bfloat16, device="cuda")
    local_lse = torch.zeros(1, 35, 4, 7, device="cuda")
    bias = torch.zeros(1, 64, 4, 7, device="cuda")
    dropped = torch.zeros(1, 35, 64, dtype=torch.int32, device="cuda")
    indices, scores = route_topk_g7(q, landmarks, local_lse, bias, dropped, 31, 64, 512, training=False)
    assert indices.shape == (1, 35, 4, 31)
    assert scores.shape == (1, 35, 28, 31)
    assert bool((indices >= 0).all()) and bool((indices < 64).all())
    assert bool((indices[..., 1:] > indices[..., :-1]).all())
    raw = torch.einsum("bqhgd,bchgd->bqhgc", q.float().reshape(1, 35, 4, 7, 128), landmarks.float()) / math.sqrt(128)
    expected = torch.gather(raw.flatten(2, 3), -1, indices.long().repeat_interleave(7, dim=2))
    torch.testing.assert_close(scores.float(), expected, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_g7_inference_route_uses_deterministic_reference(monkeypatch):
    from prefilling_v2 import routing
    from ops.topk_head_softmax import ref_softmax_topk_max_pooling

    torch.manual_seed(74)
    batch, length, chunks, h_kv, groups, dim, topk = 1, 1031, 40, 4, 7, 128, 32
    q = torch.randn(batch, length, h_kv * groups, dim, dtype=torch.bfloat16, device="cuda")
    landmarks = torch.randn(batch, chunks, h_kv, groups, dim, dtype=torch.bfloat16, device="cuda")
    local_lse = torch.randn(batch, length, h_kv, groups, device="cuda")
    bias = torch.randn(batch, chunks, h_kv, groups, device="cuda")
    dropped = torch.zeros(batch, length, chunks, dtype=torch.int32, device="cuda")
    dropped[:, ::11, 3] = 1

    def fail_tilelang(*args, **kwargs):
        raise AssertionError("inference routing called the TileLang kernel")

    monkeypatch.setattr(routing, "online_softmax_topk_head", fail_tilelang)
    first = routing.route_topk_g7(
        q, landmarks, local_lse, bias, dropped, topk, 64, 512, training=False
    )
    second = routing.route_topk_g7(
        q, landmarks, local_lse, bias, dropped, topk, 64, 512, training=False
    )
    expected_indices, expected_scores = ref_softmax_topk_max_pooling(
        q.reshape(batch, length, h_kv, groups, dim),
        landmarks.reshape(batch, chunks, h_kv * groups, dim),
        local_lse,
        topk,
        64,
        512,
        is_causal=False,
        drop_mask=dropped,
        bias=bias,
    )
    expected_scores = expected_scores.reshape(batch, length, h_kv * groups, topk)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[0], expected_indices)
    assert torch.equal(first[1], second[1])
    torch.testing.assert_close(first[1].float(), expected_scores, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
@torch.inference_mode()
def test_selected_k31_matches_weighted_chunk_reference(cached, filtered):
    from prefilling_v2.routing import selected_attention_g7, selected_attention_g7_cached

    torch.manual_seed(79)
    query_len, kv_len, h_kv, g, dim, topk, size = 35 if cached else 2048, 2048, 4, 7, 128, 31, 64
    q = torch.randn(1, query_len, h_kv * g, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, h_kv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.arange(topk, dtype=torch.int32, device="cuda").expand(1, query_len, h_kv, -1).clone()
    indices[:, 0] = -1
    weights = torch.randn(1, query_len, h_kv * g, topk, device="cuda").softmax(-1).bfloat16()
    weights[:, 0] = 0
    valid = torch.ones(1, kv_len, dtype=torch.bool, device="cuda")
    valid[:, ::97] = False
    keep = (torch.rand(1, query_len, h_kv, topk, size, device="cuda") > 0.4) if filtered else None
    originals = [x.clone() for x in (indices, weights, valid)]
    original_keep = None if keep is None else keep.clone()
    args = (q, k, v, weights, indices, valid, size)
    actual = (selected_attention_g7_cached(*args, token_keep=keep) if cached else
              selected_attention_g7(*args, training=False, token_keep=keep))
    assert actual.shape == q.shape and bool(torch.isfinite(actual).all())
    assert torch.equal(actual[:, 0], torch.zeros_like(actual[:, 0]))

    rows = torch.tensor([1, query_len // 2, query_len - 1], device="cuda")
    qs = q[0, rows].float().reshape(-1, h_kv, g, dim)
    expected = torch.zeros_like(qs)
    ws = weights[0, rows].float().reshape(-1, h_kv, g, topk)
    for chunk in range(topk):
        start = chunk * size
        ks, vs = k[0, start:start + size].float(), v[0, start:start + size].float()
        logits = torch.einsum("rhgd,shd->rhgs", qs, ks) / math.sqrt(dim)
        allowed = valid[0, start:start + size].expand(rows.numel(), h_kv, -1).clone()
        allowed[..., -1] = False
        if keep is not None:
            allowed &= keep[0, rows, :, chunk]
        probs = logits.masked_fill(~allowed[:, :, None], float("-inf")).softmax(-1)
        expected += torch.einsum("rhgs,shd->rhgd", probs, vs) * ws[..., chunk, None]
    observed = actual[0, rows].float().reshape_as(expected)
    assert float((observed - expected).norm() / expected.norm()) < 0.02
    torch.testing.assert_close(observed, expected, atol=0.01, rtol=0.025)
    for value, original in zip((indices, weights, valid), originals):
        assert torch.equal(value, original)
    if keep is not None:
        assert torch.equal(keep, original_keep)
