import torch

import math

from prefilling_v2.allchunk_gumbel import (
    allchunk_st_temperature, attach_allchunk_st, detach_sampled_gate,
    exact_chunk_outputs,
)


def test_allchunk_st_temperature_cosine_hits_endpoints():
    assert allchunk_st_temperature(1, 500, 1.0, 0.3, "cosine") == 1.0
    assert allchunk_st_temperature(500, 500, 1.0, 0.3, "cosine") == 0.3
    mid = allchunk_st_temperature(251, 500, 1.0, 0.3, "cosine")
    progress = 250 / 499
    expected = 0.3 + 0.5 * 0.7 * (1.0 + math.cos(math.pi * progress))
    assert abs(mid - expected) < 1e-12
    assert 0.3 < mid < 1.0
    assert allchunk_st_temperature(100, 500, 1.0, 0.3, "none") == 1.0
    assert abs(allchunk_st_temperature(251, 500, 1.0, 0.3, "linear") - (1.0 - 0.7 * progress)) < 1e-12


def test_exact_outputs_match_individual_token_attention():
    torch.manual_seed(17)
    q = torch.randn(2, 24, 4, 8, requires_grad=True)
    k, v = torch.randn(2, 2, 24, 2, 8, requires_grad=True)
    valid = torch.ones(2, 24, dtype=torch.bool)
    valid[1, 20:] = False
    positions = torch.tensor([[0, 9, 23], [1, 11, 22]])
    got = exact_chunk_outputs(q, k, v, valid, positions, 4)
    assert not got.requires_grad
    for b in range(2):
        for r, pos in enumerate(positions[b]):
            for head in range(4):
                for chunk in range(6):
                    rows = torch.arange(chunk * 4, chunk * 4 + 3)
                    rows = rows[valid[b, rows]]
                    expected = torch.zeros(8)
                    if len(rows):
                        scores = k[b, rows, head // 2] @ q[b, pos, head] / 8**.5
                        expected = scores.softmax(-1) @ v[b, rows, head // 2]
                    torch.testing.assert_close(got[b, r, head, chunk], expected)


def test_detach_replaces_only_sampled_gate_gradient():
    weights = torch.randn(2, 12, 4, 3, requires_grad=True)
    positions = torch.tensor([[1, 7], [2, 9]])
    result = detach_sampled_gate(weights, positions)
    torch.testing.assert_close(result, weights, rtol=0, atol=0)
    result.sum().backward()
    expected = torch.ones_like(weights)
    expected[torch.arange(2)[:, None], positions] = 0
    torch.testing.assert_close(weights.grad, expected)


def test_st_preserves_forward_and_has_unselected_content_dependent_gradient():
    torch.manual_seed(7)
    q = torch.randn(1, 24, 4, 8, requires_grad=True)
    k = torch.randn(1, 24, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    route = torch.randn_like(q, requires_grad=True)
    landmarks = torch.randn(1, 6, 2, 2, 8, requires_grad=True)
    prior = torch.randn(1, 6, 2, 2, requires_grad=True)
    local_lse = torch.randn(1, 24, 4, requires_grad=True)
    local_output = torch.randn_like(q, requires_grad=True)
    hard = torch.randn_like(q, requires_grad=True)
    valid = torch.ones(1, 24, dtype=torch.bool)
    dropped = torch.zeros(1, 24, 6, dtype=torch.bool)
    dropped[..., 0] = True
    positions = torch.tensor([[1, 7]])
    capture = {}
    def observe(logits, soft, responses):
        logits.retain_grad()
        capture.update(logits=logits, responses=responses)
    out = attach_allchunk_st(hard, q, k, v, route, landmarks, prior, local_lse,
        dropped, valid, local_output, positions, torch.zeros(1, 1, 4, 6), 4, 1., observe)
    torch.testing.assert_close(out, hard, atol=0, rtol=0)
    direction = torch.randn_like(out)
    (out * direction).sum().backward()
    torch.testing.assert_close(hard.grad, direction)
    assert (capture["logits"].grad[..., 1:] != 0).all()
    assert (capture["logits"].grad[..., 0] == 0).all()
    assert q.grad is None and k.grad is None and v.grad is None
    assert local_output.grad is None
    assert route.grad.abs().sum() > 0 and landmarks.grad.abs().sum() > 0
    assert prior.grad.abs().sum() > 0 and local_lse.grad.abs().sum() > 0
    assert not capture["responses"].requires_grad


def test_changing_unselected_value_changes_router_gradient_without_changing_forward():
    gradients = []
    for value in (1., 5.):
        q = torch.zeros(1, 8, 1, 1)
        k, v = torch.zeros_like(q), torch.ones_like(q)
        v[:, 4:7] = value
        landmarks = torch.zeros(1, 2, 1, 1, 1)
        prior = torch.zeros(1, 2, 1, 1, requires_grad=True)
        out = attach_allchunk_st(q, q, k, v, q, landmarks, prior,
            torch.zeros(1, 8, 1), torch.zeros(1, 8, 2, dtype=torch.bool),
            torch.ones(1, 8, dtype=torch.bool), q, torch.tensor([[0]]), None, 4, 1.)
        assert torch.equal(out, q)
        out.sum().backward()
        gradients.append(prior.grad.clone())
    assert not torch.allclose(*gradients)
