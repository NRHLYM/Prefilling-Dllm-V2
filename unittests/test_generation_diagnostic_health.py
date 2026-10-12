import pytest

from prefilling_v2.diagnostic_helpers import diagnostic_lengths, generation_health


def record(model, variant="baseline", prediction="wrong", f1=0.0):
    return {"identity": {"model": model}, "variant": variant, "prediction": prediction, "code_f1": f1}


def test_length_override_is_not_silently_restricted_to_32k():
    assert diagnostic_lengths("generate", [2048, 8192]) == [2048, 8192]
    assert diagnostic_lengths("generate", None) == [32768]
    assert diagnostic_lengths("probe", None) == [2048, 8192, 16384, 32768]
    assert diagnostic_lengths("generate", [2048, 2048]) == [2048]


@pytest.mark.parametrize("lengths", [[], [0], [-64], [2047]])
def test_invalid_lengths_fail(lengths):
    with pytest.raises(ValueError):
        diagnostic_lengths("generate", lengths)


def test_completed_but_all_zero_matrix_is_flagged():
    result = generation_health([record("dsa"), record("route", prediction="  "), record("route", "oracle_both")])
    assert result["all_groups_zero_code_f1"] and result["warning"]
    assert sum(group["empty_predictions"] for group in result["groups"]) == 1


def test_positive_f1_is_not_confused_with_all_zero_exact_match():
    result = generation_health([record("dsa", f1=0.5), record("route")])
    assert not result["all_groups_zero_code_f1"] and result["warning"] is None
    assert result["groups"][0]["positive_code_f1"] == 1


@pytest.mark.parametrize("f1", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_fail(f1):
    with pytest.raises(ValueError):
        generation_health([record("dsa", f1=f1)])


def test_empty_results_do_not_pass_health_check():
    with pytest.raises(ValueError):
        generation_health([])
