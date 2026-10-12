import torch

from prefilling_v2.routing_score_decomposition import decompose_chunk_scores


def test_decomposition_separates_landmark_prior_and_token_upper_bound():
    # Two chunks of two text tokens; chunk 1 contains the evidence at token 2.
    # Its pooled landmark is weak, its entropy prior can either rescue or suppress
    # it, while the raw evidence-token QK remains the diagnostic upper bound.
    q = torch.tensor([[[1.0]]])
    lmks = torch.tensor([[[[3.0]]], [[[1.0]]]])
    local_lse = torch.zeros(1, 1, 1)
    key = torch.tensor([[[0.0]], [[0.0]], [[5.0]], [[0.0]]])
    common = dict(
        routing_q=q,
        landmark_keys=lmks,
        local_lse=local_lse,
        dropped=torch.zeros(1, 2, dtype=torch.bool),
        key=key,
        text_valid=torch.tensor([True, True, True, True]),
        evidence_facts=[torch.tensor([2])],
        chunk_size=2,
    )
    suppressed = decompose_chunk_scores(prior_bias=torch.tensor([[[-4.0]], [[-8.0]]]), **common)
    rescued = decompose_chunk_scores(prior_bias=torch.tensor([[[-4.0]], [[4.0]]]), **common)

    assert suppressed["components"]["lmk_only"]["top16_units"] == 1
    assert suppressed["components"]["lmk_only"]["rank_sum"] == 2
    assert suppressed["components"]["lmk_plus_prior"]["rank_sum"] == 2
    assert rescued["components"]["lmk_plus_prior"]["rank_sum"] == 1
    assert rescued["effects"]["lmk_only_to_lmk_plus_prior_gains_top16_units"] == 0
    assert rescued["components"]["token_qk_upper"]["rank_sum"] == 1


def test_decomposition_excludes_dropped_chunks_and_landmark_tokens():
    q = torch.tensor([[[1.0]]])
    lmks = torch.ones(2, 1, 1, 1)
    # Token 3 represents the landmark slot and has the highest score, but is invalid.
    key = torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[100.0]]])
    result = decompose_chunk_scores(
        q, lmks, torch.zeros(1, 1, 1), torch.zeros(2, 1, 1),
        torch.tensor([[False, True]]), key,
        torch.tensor([True, True, True, False]), [torch.tensor([2])], chunk_size=2,
    )
    for component in result["components"].values():
        assert component["remote_evidence_chunk_units"] == 0
        assert component["top16_units"] == 0


def test_qcal_does_not_replace_main_query_in_token_upper_bound():
    args = (torch.ones(1, 1, 1), torch.tensor([[[[3.0]]], [[[1.0]]]]),
            torch.zeros(1, 1, 1), torch.zeros(2, 1, 1), torch.zeros(1, 2, dtype=torch.bool),
            torch.tensor([[[2.0]], [[2.0]], [[5.0]], [[5.0]]]),
            torch.ones(4, dtype=torch.bool), [torch.tensor([2])])
    shared = decompose_chunk_scores(*args, chunk_size=2)
    split = decompose_chunk_scores(*args, chunk_size=2, token_q=-torch.ones(1, 1, 1))
    assert shared['components']['lmk_only'] == split['components']['lmk_only']
    assert shared['components']['token_qk_upper']['rank_sum'] == 1
    assert split['components']['token_qk_upper']['rank_sum'] == 2


def test_dense_teacher_reports_chunk_mass_not_max_token_qk():
    # The evidence chunk has the best individual token, while the other chunk
    # wins after dense attention probability is summed over its two tokens.
    result = decompose_chunk_scores(
        routing_q=torch.ones(1, 1, 1),
        landmark_keys=torch.ones(2, 1, 1, 1),
        local_lse=torch.zeros(1, 1, 1),
        prior_bias=torch.zeros(2, 1, 1),
        dropped=torch.zeros(1, 2, dtype=torch.bool),
        key=torch.tensor([[[3.8]], [[3.8]], [[4.0]], [[-100.0]]]),
        text_valid=torch.ones(4, dtype=torch.bool),
        evidence_facts=[torch.tensor([2])],
        chunk_size=2,
    )

    assert result["components"]["token_qk_upper"]["rank_sum"] == 1
    assert result["dense_teacher"]["gqa_max"]["rank_sum"] == 2
    assert result["dense_teacher"]["per_query_head"]["rank_sum"] == 2
    assert result["dense_teacher"]["student_kl_count"] == 1
    assert result["dense_teacher"]["student_kl_mean"] > 0


def test_dense_teacher_reports_chunk_mass_not_max_token_qk():
    # The evidence chunk has the best individual token, while the other chunk
    # wins after dense attention probability is summed over its two tokens.
    result = decompose_chunk_scores(
        routing_q=torch.ones(1, 1, 1),
        landmark_keys=torch.ones(2, 1, 1, 1),
        local_lse=torch.zeros(1, 1, 1),
        prior_bias=torch.zeros(2, 1, 1),
        dropped=torch.zeros(1, 2, dtype=torch.bool),
        key=torch.tensor([[[3.8]], [[3.8]], [[4.0]], [[-100.0]]]),
        text_valid=torch.ones(4, dtype=torch.bool),
        evidence_facts=[torch.tensor([2])],
        chunk_size=2,
    )

    assert result["components"]["token_qk_upper"]["rank_sum"] == 1
    assert result["dense_teacher"]["gqa_max"]["rank_sum"] == 2
    assert result["dense_teacher"]["per_query_head"]["rank_sum"] == 2
