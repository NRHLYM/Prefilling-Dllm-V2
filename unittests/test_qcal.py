from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prefilling_v2.attention import (
    KernelDreamFullRouteAttention,
    apply_rotary_pos_emb,
    invert_rotary_pos_emb,
)
from prefilling_v2.qcal import install_qcal, set_qcal_scale
from prefilling_v2.checkpointing import load_trainable_checkpoint


class Source(nn.Module):
    def __init__(self, device="cpu", dim=8):
        super().__init__()
        self.config = SimpleNamespace()
        self.layer_idx = 0
        self.num_heads, self.num_key_value_heads = 28, 4
        self.num_key_value_groups, self.head_dim = 7, dim
        self.hidden_size, self.attention_dropout = 28 * dim, 0.0
        self.rotary_emb = None
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        for name, heads in (("q_proj", 28), ("k_proj", 4), ("v_proj", 4), ("o_proj", 28)):
            setattr(self, name, nn.Linear(self.hidden_size, heads * dim, bias=False, device=device, dtype=dtype))


def rope(length, dim, device="cpu", dtype=torch.float32):
    phases = torch.arange(length, device=device)[:, None] * torch.linspace(0.001, 0.1, dim // 2, device=device)[None]
    phases = torch.cat((phases, phases), dim=-1)[None]
    return phases.cos().to(dtype), phases.sin().to(dtype)


def expected_route_q(attn, hidden, q, pe, scale=1.0):
    delta = scale * attn.qcal(hidden).view_as(q)
    q_unrot = invert_rotary_pos_emb(q.transpose(1, 2), *pe).transpose(1, 2)
    combined = attn.qcal_norm(q_unrot + delta)
    rotated, _ = apply_rotary_pos_emb(
        combined.transpose(1, 2), combined.transpose(1, 2), *pe
    )
    return rotated.transpose(1, 2)


def make_model(device="cpu", dim=8):
    model = nn.Module()
    model.config = SimpleNamespace()
    model.attn = KernelDreamFullRouteAttention(
        Source(device, dim), local_window=512, chunk_size=64, topk=16,
        token_budget=512, allow_fallback=False,
    )
    return model


def test_identity_rng_gradients_rope_and_checkpoint(tmp_path):
    torch.manual_seed(37)
    model = make_model()
    rng = torch.get_rng_state().clone()
    install_qcal(model, 4)
    assert torch.equal(rng, torch.get_rng_state())
    a = model.attn
    assert float(a.qcal[1].weight.detach().float().abs().sum()) > 0
    hidden = torch.randn(1, 16, a.hidden_size)
    pe = rope(16, 8)
    q, k, v = a._project_qkv_blhd(hidden, None, pe)
    torch.testing.assert_close(
        a._calibrated_query(hidden, q, None, pe),
        expected_route_q(a, hidden, q, pe, scale=1.0),
    )
    assert not torch.equal(a._calibrated_query(hidden, q, None, pe), q)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in a.qcal.parameters():
        p.requires_grad_(True)
    a.qcal_norm.weight.requires_grad_(True)
    out = a._calibrated_query(hidden, q.detach(), None, pe)
    out.square().mean().backward()
    assert a.qcal[1].weight.grad.norm() > 0
    assert a.qcal_norm.weight.grad.norm() > 0
    assert a.q_proj.weight.grad is None
    with torch.no_grad():
        a.qcal[1].weight.normal_(std=0.02)
    actual = a._calibrated_query(hidden, q, None, pe)
    torch.testing.assert_close(actual, expected_route_q(a, hidden, q, pe, scale=1.0))
    assert set_qcal_scale(model, 0.0) == 1
    torch.testing.assert_close(
        a._calibrated_query(hidden, q, None, pe),
        expected_route_q(a, hidden, q, pe, scale=0.0),
    )
    set_qcal_scale(model, 0.5)
    torch.testing.assert_close(
        a._calibrated_query(hidden, q, None, pe),
        expected_route_q(a, hidden, q, pe, scale=0.5),
    )
    set_qcal_scale(model, 1.0)
    rows = torch.tensor([1, 7, 15])
    partial = a._calibrated_query(hidden[:, rows], q[:, rows], None, tuple(x[:, rows] for x in pe))
    torch.testing.assert_close(partial, actual[:, rows])
    after = a._project_qkv_blhd(hidden, None, pe)
    for before, value in zip((q, k, v), after):
        assert torch.equal(before, value)
    model.zero_grad()
    actual.square().mean().backward()
    assert all(p.grad.norm() > 0 for p in a.qcal.parameters())
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(saved, tmp_path / "trainable_state.pt")
    with torch.no_grad():
        a.qcal[1].weight.zero_()
    load_trainable_checkpoint(model=model, checkpoint_dir=tmp_path)
    assert all(torch.equal(dict(model.named_parameters())[n], p) for n, p in saved.items())


def test_qcal_scale_validation():
    model = make_model()
    assert set_qcal_scale(model, 1.0) == 0
    with pytest.raises(ValueError, match="installed Q-Cal"):
        set_qcal_scale(model, 0.0)
    install_qcal(model, 4)
    with pytest.raises(ValueError, match="finite"):
        set_qcal_scale(model, float("nan"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang")
@torch.inference_mode()
def test_nonzero_qcal_full_and_cached_refresh_match():
    from prefilling_v2.fastdllm_attention import cached_attention_forward
    torch.manual_seed(39)
    model = make_model("cuda", 128).eval()
    install_qcal(model, 64)
    a = model.attn
    a.qcal[1].weight.normal_(std=0.01)
    hidden = torch.randn(1, 2048, a.hidden_size, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 2048, device="cuda", dtype=torch.bool)
    pos = torch.arange(2048, device="cuda")[None]
    pe = rope(2048, 128, "cuda", torch.bfloat16)
    a.begin_prefill_capture()
    a(hidden, attention_mask=valid, position_ids=pos, position_embeddings=pe)
    cache = a.end_prefill_capture()
    original_lmk = cache.landmark_keys.clone()
    rows = torch.arange(1984, 2048, device="cuda")
    changed = hidden.clone()
    changed[:, rows] += torch.randn_like(changed[:, rows]) * 0.1
    actual, _ = cached_attention_forward(
        a, changed[:, rows], position_ids=pos[:, rows],
        position_embeddings=tuple(x[:, rows] for x in pe), query_positions=rows,
        kv_update_positions=rows, affected_chunks=torch.tensor([31], device="cuda"), cache=cache,
    )
    torch.cuda.synchronize()
    a.begin_prefill_capture()
    expected = a(changed, attention_mask=valid, position_ids=pos, position_embeddings=pe)[0][:, rows]
    reference = a.end_prefill_capture()
    torch.cuda.synchronize()
    assert torch.equal(cache.landmark_keys[:, :31], original_lmk[:, :31])
    torch.testing.assert_close(cache.landmark_keys, reference.landmark_keys, atol=0.01, rtol=0.03)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 0.03, float(error)
