import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from prefilling_v2.dsa_attention import (
    DreamDsaAttention,
    DreamDsaIndexer,
    collect_dsa_index_loss,
    install_dream_dsa_attention,
    prepare_dsa_index_losses,
    selected_token_attention_reference,
)


class SourceAttention(nn.Module):
    def __init__(
        self,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=None,
    ):
        super().__init__()
        self.config = SimpleNamespace(rms_norm_eps=1e-6)
        self.layer_idx = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(
            hidden_size, num_heads * head_dim, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim, hidden_size, bias=False, dtype=dtype
        )
        self.rotary_emb = None


def identity_rope(batch, length, dim):
    return torch.ones(batch, length, dim), torch.zeros(batch, length, dim)


def test_selected_token_attention_matches_dense_masked_reference():
    torch.manual_seed(11)
    batch, length, h_q, h_kv, dim, topk = 2, 7, 4, 2, 8, 3
    q = torch.randn(batch, length, h_q, dim, requires_grad=True)
    k = torch.randn(batch, length, h_kv, dim, requires_grad=True)
    v = torch.randn(batch, length, h_kv, dim, requires_grad=True)
    indices = torch.stack(
        [
            torch.stack([torch.randperm(length)[:topk] for _ in range(length)])
            for _ in range(batch)
        ]
    )
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[0, 0, -1] = False

    actual = selected_token_attention_reference(q, k, v, indices, valid)

    groups = h_q // h_kv
    k_expanded = k.repeat_interleave(groups, dim=2)
    v_expanded = v.repeat_interleave(groups, dim=2)
    dense_scores = torch.einsum("bqhd,bkhd->bqhk", q, k_expanded) / math.sqrt(dim)
    mask = torch.zeros(batch, length, length, dtype=torch.bool)
    mask.scatter_(2, indices, valid)
    dense_scores = dense_scores.masked_fill(~mask[:, :, None], float("-inf"))
    dense_probs = torch.softmax(dense_scores, dim=-1)
    expected = torch.einsum("bqhk,bkhd->bqhd", dense_probs, v_expanded)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_selected_token_attention_keeps_fp32_scores_under_outer_autocast():
    torch.manual_seed(13)
    batch, length, h_q, h_kv, dim, topk = 1, 11, 8, 2, 32, 9
    q = torch.randn(batch, length, h_q, dim)
    k = torch.randn(batch, length, h_kv, dim)
    v = torch.randn(batch, length, h_kv, dim)
    indices = torch.stack(
        [torch.randperm(length)[:topk] for _ in range(length)]
    ).unsqueeze(0)
    valid = torch.ones_like(indices, dtype=torch.bool)

    expected = selected_token_attention_reference(q, k, v, indices, valid)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = selected_token_attention_reference(q, k, v, indices, valid)

    # A missing inner autocast guard changes both the dtype and values here.
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_indexer_selection_respects_bidirectional_allowed_mask():
    torch.manual_seed(17)
    source = SourceAttention()
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=3,
        query_block_size=2,
    )
    hidden = torch.randn(1, 8, 32)
    allowed = torch.ones(1, 8, 8, dtype=torch.bool)
    allowed[:, :, 1::2] = False
    allowed[:, 0, 4:] = False

    indices, valid = indexer.select(hidden, identity_rope(1, 8, 8), allowed)

    assert indices.shape == (1, 8, 3)
    assert valid.shape == indices.shape
    selected_allowed = torch.gather(allowed, 2, indices)
    assert torch.all(selected_allowed[valid])
    assert torch.all(indices[valid] % 2 == 0)
    # Query zero has only positions 0 and 2 available, so one slot is invalid.
    assert valid[0, 0].sum().item() == 2


def test_indexer_score_matches_dsa_signed_weight_equation():
    source = SourceAttention(hidden_size=4, num_heads=2, num_kv_heads=1, head_dim=2)
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=2,
        topk=1,
        query_block_size=1,
    )
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[1.0, 1.0], [2.0, -3.0]]])
    weights = torch.tensor([[[2.0, -1.0]]])

    scores = indexer._score(query, key, weights)

    expected = torch.tensor([[[1.0, 4.0]]]) / math.sqrt(2.0)
    torch.testing.assert_close(scores, expected)


def test_indexer_head_weights_are_raw_fp32_projections():
    torch.manual_seed(19)
    source = SourceAttention()
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=3,
        query_block_size=2,
    )
    hidden = torch.randn(1, 3, 32)

    _, _, weights = indexer._project(hidden, identity_rope(1, 3, 8))

    expected = torch.nn.functional.linear(
        hidden.float(), indexer.index_w.weight
    ) / math.sqrt(2.0)
    assert indexer.index_w.weight.dtype is torch.float32
    assert torch.count_nonzero(indexer.index_w.weight).item() > 0
    torch.testing.assert_close(weights, expected)
    assert bool((weights < 0).any() or (weights > 0).any())


def test_indexer_norm_scales_stay_fp32_with_bfloat16_attention():
    source = SourceAttention(dtype=torch.bfloat16)
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=3,
        query_block_size=2,
    )

    assert indexer.index_q.weight.dtype is torch.bfloat16
    assert indexer.index_k.weight.dtype is torch.bfloat16
    assert indexer.index_w.weight.dtype is torch.float32
    assert indexer.q_norm.weight.dtype is torch.float32
    assert indexer.k_norm.weight.dtype is torch.float32


def test_dsa_attention_auxiliary_loss_reaches_indexer_parameters():
    torch.manual_seed(23)
    source = SourceAttention()
    source.config.dream_dsa_chunk_size = 4
    attention = DreamDsaAttention(
        source,
        topk=3,
        num_index_heads=2,
        index_head_dim=8,
        query_block_size=4,
        attention_query_block_size=4,
        aux_queries=3,
    )
    attention.train()
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    attention.prepare_index_loss()

    output = attention(
        hidden,
        attention_mask=allowed,
        position_embeddings=identity_rope(1, 8, 8),
    )[0]
    loss = output.square().mean() + attention.index_loss
    loss.backward()

    assert attention.index_loss is not None
    assert torch.isfinite(attention.index_loss)
    gradients = [
        parameter.grad
        for parameter in attention.dsa_indexer.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert sum(torch.count_nonzero(gradient).item() for gradient in gradients) > 0


@pytest.mark.parametrize("loss_scope", ["full", "selected"])
def test_indexer_distillation_scopes_are_finite_and_trainable(loss_scope):
    torch.manual_seed(27)
    source = SourceAttention()
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=3,
        query_block_size=4,
    )
    hidden = torch.randn(1, 8, 32)
    teacher_q = torch.randn(1, 8, 4, 8)
    teacher_k = torch.randn(1, 8, 2, 8)
    allowed = torch.ones(1, 8, 8, dtype=torch.bool)
    allowed[:, :, -1] = False

    loss = indexer.distillation_loss(
        hidden,
        identity_rope(1, 8, 8),
        teacher_q,
        teacher_k,
        allowed,
        sample_count=4,
        loss_scope=loss_scope,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert indexer.index_w.weight.grad is not None
    assert torch.count_nonzero(indexer.index_w.weight.grad) > 0


def test_selected_distillation_matches_full_when_topk_covers_context():
    torch.manual_seed(31)
    source = SourceAttention()
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=8,
        query_block_size=4,
    )
    hidden = torch.randn(1, 8, 32)
    teacher_q = torch.randn(1, 8, 4, 8)
    teacher_k = torch.randn(1, 8, 2, 8)
    allowed = torch.ones(1, 8, 8, dtype=torch.bool)
    allowed[:, :, -1] = False

    torch.manual_seed(37)
    full = indexer.distillation_loss(
        hidden,
        identity_rope(1, 8, 8),
        teacher_q,
        teacher_k,
        allowed,
        sample_count=4,
        loss_scope="full",
    )
    torch.manual_seed(37)
    selected = indexer.distillation_loss(
        hidden,
        identity_rope(1, 8, 8),
        teacher_q,
        teacher_k,
        allowed,
        sample_count=4,
        loss_scope="selected",
    )

    torch.testing.assert_close(selected, full, rtol=1e-5, atol=1e-6)


def test_dsa_auxiliary_loss_is_checkpoint_recomputation_safe():
    torch.manual_seed(29)
    source = SourceAttention()
    source.config.dream_dsa_chunk_size = 4
    attention = DreamDsaAttention(
        source,
        topk=3,
        num_index_heads=2,
        index_head_dim=8,
        query_block_size=4,
        attention_query_block_size=4,
        aux_queries=3,
    )
    attention.train()
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    rope = identity_rope(1, 8, 8)
    attention.prepare_index_loss()

    def checkpointed_layer(states):
        return attention(
            states,
            attention_mask=allowed,
            position_embeddings=rope,
        )[0]

    output = checkpoint(checkpointed_layer, hidden, use_reentrant=False)
    index_loss = attention.index_loss
    assert index_loss is not None
    (output.square().mean() + index_loss).backward()

    assert torch.isfinite(index_loss)
    assert hidden.grad is not None
    gradients = [
        parameter.grad
        for parameter in attention.dsa_indexer.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert sum(torch.count_nonzero(gradient).item() for gradient in gradients) > 0


class TinyDream(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([])
        for layer_idx in range(layers):
            layer = nn.Module()
            layer.self_attn = SourceAttention()
            layer.self_attn.layer_idx = layer_idx
            self.model.layers.append(layer)
        self.config = SimpleNamespace()


def test_install_dsa_uses_same_interleave_and_collects_losses():
    model = TinyDream()
    plan = install_dream_dsa_attention(
        model,
        interleave=2,
        local_window=2,
        chunk_size=4,
        topk=3,
        num_index_heads=2,
        index_head_dim=8,
        query_block_size=2,
        attention_query_block_size=2,
        aux_queries=2,
        non_dsa_attention="dense",
    )
    assert plan.dsa_layers == [1, 3]
    assert plan.dense_layers == [0, 2]
    assert prepare_dsa_index_losses(model) == 2
    for layer_idx in plan.dsa_layers:
        model.model.layers[layer_idx].self_attn.index_loss = torch.tensor(
            float(layer_idx), requires_grad=True
        )
    assert collect_dsa_index_loss(model).item() == 2.0


def test_all_query_distillation_is_finite():
    torch.manual_seed(41)
    source = SourceAttention()
    indexer = DreamDsaIndexer(
        source,
        num_index_heads=2,
        index_head_dim=8,
        topk=3,
        query_block_size=3,
    )
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    teacher_q = torch.randn(1, 8, 4, 8)
    teacher_k = torch.randn(1, 8, 2, 8)
    allowed = torch.ones(1, 8, dtype=torch.bool)
    loss = indexer.distillation_loss(
        hidden,
        identity_rope(1, 8, 8),
        teacher_q,
        teacher_k,
        allowed,
        sample_count=0,
        loss_scope="full",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert indexer.index_w.weight.grad is not None


def test_warmup_dense_forward_trains_indexer_without_topk():
    torch.manual_seed(43)
    source = SourceAttention()
    source.config.dream_dsa_chunk_size = 4
    attention = DreamDsaAttention(
        source,
        topk=3,
        num_index_heads=2,
        index_head_dim=8,
        query_block_size=4,
        attention_query_block_size=4,
        aux_queries=0,
        aux_loss_scope="full",
    )
    attention.warmup_dense = True
    attention.train()
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    attention.prepare_index_loss()
    output = attention(
        hidden,
        attention_mask=allowed,
        position_embeddings=identity_rope(1, 8, 8),
    )[0]
    recorded = []
    original = attention.dsa_indexer._teacher_key_mass

    def wrapped(*args, **kwargs):
        recorded.append(kwargs.get("selected_indices"))
        return original(*args, **kwargs)

    attention.dsa_indexer._teacher_key_mass = wrapped
    attention.materialize_index_loss()
    (output.square().mean() * 0.0 + attention.index_loss).backward()
    assert attention.index_loss is not None
    assert torch.isfinite(attention.index_loss)
    assert recorded and all(indices is not None for indices in recorded)
    assert attention.dsa_indexer.index_w.weight.grad is not None
    assert torch.count_nonzero(attention.dsa_indexer.index_w.weight.grad) > 0


def test_materialize_index_loss_is_selected_even_if_aux_scope_is_full():
    torch.manual_seed(44)
    source = SourceAttention()
    source.config.dream_dsa_chunk_size = 4
    attention = DreamDsaAttention(
        source,
        topk=3,
        num_index_heads=2,
        index_head_dim=8,
        query_block_size=4,
        attention_query_block_size=4,
        aux_queries=0,
        aux_loss_scope="full",
    )
    attention.warmup_dense = False
    attention.train()
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    attention.prepare_index_loss()
    attention(
        hidden,
        attention_mask=allowed,
        position_embeddings=identity_rope(1, 8, 8),
    )
    recorded = []
    original = attention.dsa_indexer._teacher_key_mass

    def wrapped(*args, **kwargs):
        recorded.append(kwargs.get("selected_indices"))
        return original(*args, **kwargs)

    attention.dsa_indexer._teacher_key_mass = wrapped
    attention.materialize_index_loss()
    assert recorded and all(indices is not None for indices in recorded)
