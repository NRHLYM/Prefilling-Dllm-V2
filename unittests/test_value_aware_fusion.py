from __future__ import annotations

import torch
from torch import nn

from prefilling_v2.attention import _route_weights_torch
from prefilling_v2.value_aware_fusion import (
    selected_chunk_outputs,
    value_fusion_bonus,
)


def test_value_bonus_zero_matches_s2_fusion():
    torch.manual_seed(0)
    b, l, hq, hkv, k, chunks = 1, 8, 4, 2, 2, 4
    scores = torch.randn(b, l, hq, k)
    indices = torch.zeros(b, l, hkv, k, dtype=torch.long)
    indices[..., 1] = 1
    prior = torch.zeros(b, chunks, hkv, hq // hkv)
    local = torch.randn(b, l, hq)
    bonus = torch.zeros_like(scores)
    hard = _route_weights_torch(scores, indices, prior, local, temperature=1.0)
    with_bonus = _route_weights_torch(
        scores, indices, prior, local, temperature=1.0, value_bonus=bonus
    )
    assert torch.allclose(hard[0], with_bonus[0])
    assert torch.allclose(hard[1], with_bonus[1])
    mass = with_bonus[0].float().sum(-1) + with_bonus[1].float()
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)


def test_selected_chunk_outputs_and_bonus_shapes():
    torch.manual_seed(0)
    b, l, hq, hkv, d, cs, k = 1, 16, 4, 2, 8, 4, 2
    q = torch.randn(b, l, hq, d)
    kv = torch.randn(b, l, hkv, d)
    indices = torch.zeros(b, l, hkv, k, dtype=torch.long)
    indices[..., 0] = 0
    indices[..., 1] = 1
    valid = torch.ones(b, l, dtype=torch.bool)
    valid[:, cs - 1 :: cs] = False
    out = selected_chunk_outputs(q, kv, kv, indices, valid, cs, query_block=8)
    assert out.shape == (b, l, hq, k, d)
    query_proj = nn.Linear(d, 4, bias=False)
    value_proj = nn.Linear(d, 4, bias=False)
    nn.init.zeros_(value_proj.weight)
    bonus = value_fusion_bonus(q, out, query_proj, value_proj)
    assert bonus.shape == (b, l, hq, k)
    assert torch.equal(bonus, torch.zeros_like(bonus))
