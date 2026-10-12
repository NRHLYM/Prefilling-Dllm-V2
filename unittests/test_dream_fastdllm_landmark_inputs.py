from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prefilling_v2.fastdllm_v1 import _embed_partial_inputs, _prepare_landmark_inputs


class LandmarkModel(nn.Module):
    def __init__(self, mode, device):
        super().__init__()
        self.embedding = nn.Embedding(8, 4, device=device)
        self.config = SimpleNamespace(
            dream_route_lmk_token_mode=mode,
            dream_route_lmk_token_id=8 if mode == "external" else 7,
        )
        self.dream_route_lmk_embed = nn.Parameter(torch.arange(4., device=device) + 20)
        self.dream_route_lmk_type_embed = nn.Parameter(torch.arange(4., device=device) + 10)

    def get_input_embeddings(self):
        return self.embedding


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["external", "mask", "mask_type"])
@pytest.mark.parametrize("wrapped", [False, True])
@torch.inference_mode()
def test_cached_embedding_matches_full_prefill(mode, device, wrapped):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    base = LandmarkModel(mode, device)
    model = SimpleNamespace(module=base) if wrapped else base
    core = SimpleNamespace(model=SimpleNamespace(embed_tokens=base.embedding))
    lmk = base.config.dream_route_lmk_token_id
    ids = torch.tensor([[1, 7, 2, lmk, 3, lmk]], device=device)
    original = ids.clone()
    landmarks = torch.tensor([3, 5], device=device)
    queries = torch.tensor([1, 3, 4, 5], device=device)
    lookups = []

    def assert_safe_lookup(module, args):
        looked_up = args[0]
        # Catch an unsafe lookup before it poisons the CUDA test process.
        assert bool(((looked_up >= 0) & (looked_up < module.num_embeddings)).all())
        lookups.append(looked_up.clone())

    handle = base.embedding.register_forward_pre_hook(assert_safe_lookup)
    try:
        full_ids, full_embeds = _prepare_landmark_inputs(model, ids, landmarks)
        expected = full_embeds if full_embeds is not None else base.embedding(full_ids)
        actual = _embed_partial_inputs(model, core, ids, queries, landmarks)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected[:, queries], rtol=0, atol=0)
    assert torch.equal(ids, original)
    assert actual.shape == (1, 4, 4)
    assert torch.equal(actual[:, 0], base.embedding(ids[:, 1]))
    if mode == "external":
        assert lookups[-1].tolist() == [[7, 0, 3, 0]]
        torch.testing.assert_close(actual[0, [1, 3]], base.dream_route_lmk_embed.expand(2, -1))


def test_external_partial_without_landmarks_is_plain_embedding():
    base = LandmarkModel("external", "cpu")
    core = SimpleNamespace(model=SimpleNamespace(embed_tokens=base.embedding))
    ids = torch.tensor([[1, 8, 2, 8]])
    queries = torch.tensor([0, 2])
    actual = _embed_partial_inputs(base, core, ids, queries, None)
    torch.testing.assert_close(actual, base.embedding(ids[:, queries]), rtol=0, atol=0)


def test_external_missing_embedding_raises_clear_error_not_index_error():
    base = LandmarkModel("external", "cpu")
    del base.dream_route_lmk_embed
    core = SimpleNamespace(model=SimpleNamespace(embed_tokens=base.embedding))
    with pytest.raises(RuntimeError, match="missing dream_route_lmk_embed"):
        _embed_partial_inputs(base, core, torch.tensor([[1, 8]]), torch.tensor([1]), None)
