from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prefilling_v2.nsa_attention import DreamNsaAttention


class Source(nn.Module):
    def __init__(self, dim=8, device="cpu"):
        super().__init__()
        self.config = SimpleNamespace(rms_norm_eps=1e-6)
        self.layer_idx = 0
        self.num_heads, self.num_key_value_heads = 28, 4
        self.num_key_value_groups, self.head_dim = 7, dim
        self.hidden_size, self.attention_dropout = 28 * dim, 0.0
        self.rotary_emb = None
        for name, heads in (
            ("q_proj", 28), ("k_proj", 4), ("v_proj", 4), ("o_proj", 28)
        ):
            setattr(
                self,
                name,
                nn.Linear(self.hidden_size, heads * dim, bias=False, device=device),
            )


def make_attention(dim=8, device="cpu"):
    return DreamNsaAttention(
        Source(dim=dim, device=device),
        local_window=8,
        chunk_size=8,
        block_count=1,
        compress_block=4,
        compress_stride=2,
        select_block=8,
    )


def test_phi_sees_every_token_in_the_block():
    attn = make_attention()
    key = torch.zeros(1, 16, 4, 8)
    value = torch.zeros_like(key)
    key[:, 3] = 10_000.0
    valid = torch.ones(1, 16, dtype=torch.bool)
    with_last, _, windows = attn._compress_kv(key, value, valid)
    key[:, 3] = 0
    without_last, _, _ = attn._compress_kv(key, value, valid)
    assert not torch.allclose(with_last[:, 0], without_last[:, 0])
    assert windows.shape[-1] == 7
    assert windows[0].tolist() == [True] * 7


def test_gqa_groups_select_independently():
    attn = make_attention()
    probs = torch.zeros(1, 16, 28, 7)
    probs[:, :, :7, 0] = 1.0
    probs[:, :, 7:14, 6] = 1.0
    valid = torch.ones(1, 16, dtype=torch.bool)
    indices, _ = attn._selected_indices(probs, valid)
    assert indices.shape[:3] == (1, 16, 4)
    assert indices[0, 0, 0].tolist() == list(range(0, 8))
    assert indices[0, 0, 1].tolist() == list(range(8, 16))


def test_eq9_adds_overlapping_compress_mass():
    attn = make_attention()
    probs = torch.zeros(1, 8, 28, 7)
    probs[..., 0] = 0.25
    probs[..., 1] = 0.75
    valid = torch.ones(1, 16, dtype=torch.bool)
    indices, _ = attn._selected_indices(probs, valid)
    # Windows 0 and 1 both overlap selection block 0 only, so top-1 is block 0
    # even if a later unique window is unused.
    assert indices[0, 0, 0].tolist() == list(range(0, 8))


def test_selection_keeps_local_blocks():
    attn = make_attention()
    probs = torch.zeros(1, 16, 28, 7)
    probs[..., 0] = 1.0
    valid = torch.ones(1, 16, dtype=torch.bool)
    indices, _ = attn._selected_indices(probs, valid)
    assert indices[0, 0, 0, 0].item() == 0


def test_nsa_parameters_receive_gradient(monkeypatch):
    attn = make_attention()

    def selected(query, key, value, indices, valid, **kwargs):
        return query.new_zeros(query.shape)

    def local_attention(query, key, value, *args, **kwargs):
        return query.new_zeros(query.shape), None

    monkeypatch.setattr(attn, "_selected_attention", selected)
    monkeypatch.setattr(
        "prefilling_v2.nsa_attention.bidir_local_attention",
        local_attention,
    )
    hidden = torch.randn(2, 16, attn.hidden_size, requires_grad=True)
    query_rope = torch.randn(2, 16, 28, 8, requires_grad=True)
    query_raw = torch.randn(2, 16, 28, 8, requires_grad=True)
    key_rope = torch.randn(2, 16, 4, 8, requires_grad=True)
    key_raw = torch.randn(2, 16, 4, 8, requires_grad=True)
    value = torch.randn_like(key_rope)
    valid = torch.ones(2, 16, dtype=torch.bool)
    attn._nsa_output(
        hidden, query_rope, query_raw, key_rope, key_raw, value, valid, None
    ).sum().backward()
    assert attn.nsa_gate.weight.grad is not None
    assert torch.count_nonzero(attn.nsa_gate.weight.grad)
    assert attn.nsa_phi_k_down.weight.grad is not None
    assert attn.nsa_cmp_k.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@torch.inference_mode()
def test_tilelang_selected_branch_is_invoked(monkeypatch):
    attn = make_attention(dim=128, device="cuda").eval()
    calls = 0

    def selected(query, key, value, indices, valid, **kwargs):
        nonlocal calls
        calls += 1
        assert query.is_cuda and key.is_cuda and value.is_cuda
        assert query.shape[2] == 7
        assert key.shape[2] == 1
        return torch.zeros_like(query)

    import ops.dsa_selected_attention_tilelang as selected_ops

    monkeypatch.setattr(selected_ops, "selected_token_attention_tilelang", selected)
    q = torch.randn(1, 3, 28, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    index = torch.arange(0, 7, device="cuda")[None, None, None].expand(1, 3, 4, -1)
    valid = torch.ones_like(index, dtype=torch.bool)
    output = attn._selected_attention(q, k, v, index, valid)
    assert calls == 4
    assert output.shape == q.shape
