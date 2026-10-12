from __future__ import annotations

import torch

from prefilling_v2.support_attn_distill import (
    branch_masks,
    fused_support_kl,
    last_k_mask,
    span_first_mask,
)


def test_last_k_mask_keeps_suffix_spans():
    flags = torch.tensor([[1, 0, 1, 0, 1, 1]], dtype=torch.bool)
    got = last_k_mask(flags, 2)
    assert got.tolist() == [[False, False, False, False, True, True]]


def test_span_first_keeps_gold_start_not_whole_run():
    flags = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 1]], dtype=torch.bool)
    got = span_first_mask(flags)
    assert got.tolist() == [[False, True, False, False, True, False, True, False, False]]


def test_branch_masks_detach_indices_and_keep_local_remote_disjoint():
    positions = torch.tensor([[10]])
    indices = torch.full((1, 16, 1, 2), -1, dtype=torch.long)
    indices[0, 10, 0, 0] = 1
    key_valid = torch.ones(1, 16, dtype=torch.bool)
    local, remote = branch_masks(
        positions, indices, key_valid, chunk_size=4, window=2
    )
    assert not (local & remote).any()
    assert bool(local[0, 0, 8:11].all())
    assert bool(local[0, 0, 12])
    assert not bool(local[0, 0, 11])
    assert bool(remote[0, 0, 4:7].all())
    assert not bool(remote[0, 0, 7])


def test_fused_kl_trains_content_qk_not_teacher_or_route_q():
    torch.manual_seed(0)
    b, n, hq, hkv, d, qn = 1, 8, 2, 1, 4, 2
    student_q = torch.randn(b, n, hq, d, requires_grad=True)
    student_k = torch.randn(b, n, hkv, d, requires_grad=True)
    route_q = torch.randn(b, n, hq, d, requires_grad=True)
    teacher_q = torch.randn(b, n, hq, d)
    teacher_k = torch.randn(b, n, hkv, d)
    positions = torch.tensor([[1, 4]])
    keep = torch.ones(b, qn, dtype=torch.bool)
    local = torch.zeros(b, qn, n, dtype=torch.bool)
    remote = torch.zeros(b, qn, n, dtype=torch.bool)
    local[0, 0, :3] = True
    remote[0, 0, 4:] = True
    local[0, 1, 2:5] = True
    remote[0, 1, 5:] = True
    local_weight = torch.full((b, qn, hq), 0.7, requires_grad=True)
    loss = fused_support_kl(
        student_q,
        student_k,
        teacher_q,
        teacher_k,
        positions,
        keep,
        local,
        remote,
        local_weight,
        1.5,
        query_chunk=1,
    )
    loss.backward()
    assert float(student_q.grad.abs().sum()) > 0
    assert float(student_k.grad.abs().sum()) > 0
    assert float(local_weight.grad.abs().sum()) > 0
    assert route_q.grad is None
    assert teacher_q.grad is None
    assert teacher_k.grad is None


def test_detached_gate_does_not_train_fusion_weight():
    torch.manual_seed(0)
    b, n, hq, hkv, d, qn = 1, 8, 2, 1, 4, 2
    student_q = torch.randn(b, n, hq, d, requires_grad=True)
    student_k = torch.randn(b, n, hkv, d, requires_grad=True)
    teacher_q = torch.randn(b, n, hq, d)
    teacher_k = torch.randn(b, n, hkv, d)
    positions = torch.tensor([[1, 4]])
    keep = torch.ones(b, qn, dtype=torch.bool)
    local = torch.zeros(b, qn, n, dtype=torch.bool)
    remote = torch.zeros(b, qn, n, dtype=torch.bool)
    local[0, 0, :3] = True
    remote[0, 0, 4:] = True
    local[0, 1, 2:5] = True
    remote[0, 1, 5:] = True
    local_weight = torch.full((b, qn, hq), 0.7, requires_grad=True)
    loss = fused_support_kl(
        student_q,
        student_k,
        teacher_q,
        teacher_k,
        positions,
        keep,
        local,
        remote,
        local_weight.detach(),
        1.5,
    )
    loss.backward()
    assert float(student_q.grad.abs().sum()) > 0
    assert local_weight.grad is None

