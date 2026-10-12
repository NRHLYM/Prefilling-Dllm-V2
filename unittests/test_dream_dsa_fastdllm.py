from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prefilling_v2.dsa_attention import DreamDsaAttention, selected_token_attention_reference
from prefilling_v2.fastdllm_attention import cached_attention_forward


class Source(nn.Module):
    def __init__(self, device="cpu", dim=8):
        super().__init__()
        self.config = SimpleNamespace(rms_norm_eps=1e-6, dream_dsa_chunk_size=8)
        self.layer_idx = 0
        self.num_heads, self.num_key_value_heads = 28, 4
        self.num_key_value_groups, self.head_dim = 7, dim
        self.hidden_size, self.attention_dropout = 28 * dim, 0.0
        self.rotary_emb = None
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        for name, heads in (("q_proj", 28), ("k_proj", 4), ("v_proj", 4), ("o_proj", 28)):
            setattr(self, name, nn.Linear(self.hidden_size, heads * dim, bias=False, device=device, dtype=dtype))


def make_attention(device="cpu", dim=8, backend="torch"):
    return DreamDsaAttention(
        Source(device, dim), topk=12, num_index_heads=2, index_head_dim=dim,
        query_block_size=7, attention_query_block_size=7, aux_queries=0,
        backend=backend,
    ).eval()


def rope(length, dim, device="cpu", dtype=torch.float32):
    phases = torch.arange(length, device=device)[:, None] * torch.linspace(0.001, 0.1, dim // 2, device=device)[None]
    phases = torch.cat((phases, phases), dim=-1)[None]
    return phases.cos().to(dtype), phases.sin().to(dtype)


@torch.inference_mode()
def test_partial_queries_match_frozen_context_reference_and_preserve_cache():
    torch.manual_seed(17)
    attn = make_attention()
    hidden = torch.randn(1, 32, attn.hidden_size)
    valid = torch.ones(1, 32, dtype=torch.bool)
    valid[:, 27:] = False
    positions = torch.arange(32)[None]
    embeddings = rope(32, 8)
    attn.begin_prefill_capture()
    attn(hidden, attention_mask=valid, position_ids=positions, position_embeddings=embeddings)
    cache = attn.end_prefill_capture()
    original = [x.clone() for x in (cache.key, cache.value, cache.dsa_index_keys)]
    queries = torch.tensor([9, 10, 11, 12, 13, 14])
    updates = queries[1:]
    changed = hidden.clone()
    changed[:, updates] += 0.3 * torch.randn_like(changed[:, updates])
    actual, stats = cached_attention_forward(
        attn, changed[:, queries], position_ids=positions[:, queries],
        position_embeddings=tuple(x[:, queries] for x in embeddings),
        query_positions=queries, kv_update_positions=updates,
        affected_chunks=torch.empty(0, dtype=torch.long), cache=cache,
    )
    expected = attn(changed, attention_mask=valid, position_ids=positions, position_embeddings=embeddings)[0][:, queries]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    unchanged = torch.ones(32, dtype=torch.bool)
    unchanged[updates] = False
    for before, after in zip(original, (cache.key, cache.value, cache.dsa_index_keys)):
        assert torch.equal(before[:, unchanged], after[:, unchanged])
        assert not torch.equal(before[:, updates], after[:, updates])
    assert not cache.key_valid[:, 7::8].any()
    assert not cache.key_valid[:, 27:].any()
    assert stats.routing_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@pytest.mark.parametrize("query_len,kv_len,topk", [(1, 128, 64), (35, 2048, 1024), (67, 32768, 1024)])
@torch.inference_mode()
def test_rectangular_tilelang_matches_reference(query_len, kv_len, topk):
    from ops.dsa_selected_attention_tilelang import selected_token_attention_tilelang

    torch.manual_seed(23)
    q = torch.randn(1, query_len, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.randint(kv_len, (1, query_len, topk), device="cuda")
    valid = torch.rand_like(indices, dtype=torch.float32) > 0.2
    valid[:, 0] = False
    actual = selected_token_attention_tilelang(q, k, v, indices, valid)
    expected = selected_token_attention_reference(q, k, v, indices, valid)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)
    assert relative_l2 < 0.01
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, 0]) == 0


def test_block_single_token_cache_matches_uncached_schedule():
    from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM, PrefillOutput
    from prefilling_v2.longbench_eval import build_generation_layout, build_fastdllm_block_layouts

    class Decoder(DreamRouteFastDLLM):
        def prefill(self, input_ids, attention_mask, position_ids, landmark_positions=None):
            logits = torch.zeros(1, input_ids.shape[1], 16)
            logits[:, :, int(input_ids.sum()) % 14 + 1] = 12
            return PrefillOutput(logits=logits, cache=None)

    layout = build_generation_layout(prompt_ids=[2, 3], answer_tokens=4,
        physical_length=8, chunk_size=4, mask_token_id=0, pad_token_id=15, landmark_token_id=0)
    args = dict(input_ids=layout.input_ids[None], attention_mask=layout.attention_mask[None],
        position_ids=layout.position_ids[None], blocks=build_fastdllm_block_layouts(layout, 1, 4))
    cached, cs = Decoder(model=None, mask_token_id=0).generate(**args)
    exact, es = Decoder(model=None, mask_token_id=0, use_cache=False).generate(**args)
    assert torch.equal(cached, exact)
    assert cs.full_prefills == es.full_prefills == 4
    assert cs.cached_forwards == es.cached_forwards == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@torch.inference_mode()
def test_cached_attention_multiple_query_blocks_have_canonical_strides():
    torch.manual_seed(31)
    attn = make_attention("cuda", dim=128, backend="tilelang")
    hidden = torch.randn(1, 32, attn.hidden_size, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(32, device="cuda")[None]
    embeddings = rope(32, 128, "cuda", torch.bfloat16)
    attn.begin_prefill_capture()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = attn(hidden, position_ids=pos, position_embeddings=embeddings)[0]
        cache = attn.end_prefill_capture()
        actual, _ = cached_attention_forward(
            attn, hidden, position_ids=pos, position_embeddings=embeddings,
            query_positions=pos[0], kv_update_positions=pos[0],
            affected_chunks=pos.new_empty(0), cache=cache,
        )
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
