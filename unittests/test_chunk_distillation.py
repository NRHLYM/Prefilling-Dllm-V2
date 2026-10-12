import torch
from torch.utils.checkpoint import checkpoint

from prefilling_v2.chunk_distillation import chunk_distillation_loss


def inputs():
    torch.manual_seed(11)
    q = torch.randn(1, 12, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 1, 4, requires_grad=True)
    lmk = torch.randn(1, 3, 1, 2, 4, requires_grad=True)
    prior = torch.randn(1, 3, 1, 2, requires_grad=True)
    valid = torch.ones(1, 12, dtype=torch.bool)
    drop = torch.zeros(1, 12, 3, dtype=torch.int32)
    drop[:, :, 0] = 1
    return q, k, lmk, prior, valid, drop


def test_matches_dense_reference_and_unselected_gradients():
    q, k, lmk, prior, valid, drop = inputs()
    pos = torch.tensor([[0, 4]])
    loss = chunk_distillation_loss(q, k, lmk, prior, valid, drop, 4, 2, positions=pos)
    keys = torch.tensor([4, 5, 6, 8, 9, 10])
    scores = torch.einsum("rhd,nd->rhn", q[0, pos[0]].detach(), k[0, keys, 0].detach()) / 2
    target = scores.softmax(-1).reshape(2, 2, 2, 3).sum(-1)
    student = torch.einsum("rhd,chd->rhc", q[0, pos[0]], lmk[0, 1:, 0]) / 2 + prior[0, 1:, 0].T
    reference = (target * (target.log() - student.log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert k.grad is None  # teacher is detached
    assert lmk.grad[:, 1:].abs().sum(dim=(2, 3, 4)).gt(0).all()
    assert lmk.grad[:, 0].count_nonzero() == 0
    assert prior.grad[:, 1:].abs().sum() > 0


def test_rng_unchanged_and_checkpoint_gradients():
    q, k, lmk, prior, valid, drop = inputs()
    rng = torch.get_rng_state().clone()
    loss = chunk_distillation_loss(q, k, lmk, prior, valid, drop, 4, 2)
    assert torch.equal(rng, torch.get_rng_state())
    loss.backward()
    expected = lmk.grad.clone()
    lmk.grad = None
    torch.set_rng_state(rng)
    checkpoint(lambda x: chunk_distillation_loss(q, k, x, prior, valid, drop, 4, 2), lmk, use_reentrant=False).backward()
    torch.testing.assert_close(lmk.grad, expected)


def test_empty_remote_rows_finite_zero():
    q, k, lmk, prior, valid, drop = inputs()
    drop.fill_(1)
    loss = chunk_distillation_loss(q, k, lmk, prior, valid, drop, 4, 2)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(lmk.grad).all()


def test_excluded_tokens_do_not_change_teacher():
    q, k, lmk, prior, valid, drop = inputs()
    pos = torch.tensor([[0, 4]])
    first = chunk_distillation_loss(q, k, lmk, prior, valid, drop, 4, 2, positions=pos)
    changed = k.detach().clone()
    changed[:, [0, 1, 2, 3, 7, 11]] = 10000
    second = chunk_distillation_loss(q, changed, lmk, prior, valid, drop, 4, 2, positions=pos)
    torch.testing.assert_close(first, second)
