"""Tests for eval/stats.py's bootstrap CI and significance testing."""

import stats


def test_bootstrap_ci_narrower_for_tight_cluster_than_noisy_data():
    tight = [0.8, 0.81, 0.79, 0.8, 0.82, 0.78, 0.8, 0.81, 0.79, 0.8, 0.8, 0.79]
    noisy = [0.1, 0.9, 0.2, 0.8, 0.3, 0.95, 0.05, 0.85, 0.15, 0.9, 0.4, 0.6]

    tight_ci = stats.bootstrap_ci(tight)
    noisy_ci = stats.bootstrap_ci(noisy)

    assert (tight_ci["ci_high"] - tight_ci["ci_low"]) < (noisy_ci["ci_high"] - noisy_ci["ci_low"])


def test_bootstrap_ci_empty_and_single_point():
    assert stats.bootstrap_ci([]) == {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    single = stats.bootstrap_ci([0.75])
    assert single == {"mean": 0.75, "ci_low": 0.75, "ci_high": 0.75, "n": 1}


def test_bootstrap_ci_is_deterministic_with_default_seed():
    scores = [0.3, 0.5, 0.9, 0.2, 0.7, 0.4, 0.6, 0.55, 0.65, 0.45]
    assert stats.bootstrap_ci(scores) == stats.bootstrap_ci(scores)


def test_reliability_caveat_threshold():
    assert stats.reliability_caveat(stats.MIN_RELIABLE_N - 1) is not None
    assert stats.reliability_caveat(stats.MIN_RELIABLE_N) is None
    assert stats.reliability_caveat(stats.MIN_RELIABLE_N + 5) is None


def test_diff_test_identical_distributions_not_significant():
    a = [0.8] * 15
    b = [0.8] * 15
    result = stats.bootstrap_mean_diff_test(a, b)
    assert result["significant"] is False
    assert result["mean_diff"] == 0.0


def test_diff_test_clearly_different_unpaired_is_significant():
    high = [0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.9, 0.91, 0.89, 0.9, 0.88]
    low = [0.3, 0.32, 0.28, 0.31, 0.29, 0.33, 0.27, 0.3, 0.31, 0.29, 0.3, 0.28]
    result = stats.bootstrap_mean_diff_test(high, low)
    assert result["significant"] is True
    assert result["p_value"] < 0.05
    assert result["paired"] is False


def test_diff_test_pairs_by_shared_question_ids():
    ids = [f"q{i}" for i in range(12)]
    a_scores = [0.9] * 12
    b_scores = [0.7] * 12
    result = stats.bootstrap_mean_diff_test(a_scores, b_scores, ids_a=ids, ids_b=ids)
    assert result["paired"] is True
    assert result["n"] == 12
    assert result["mean_diff"] == 0.2
    assert result["significant"] is True


def test_diff_test_falls_back_to_unpaired_when_ids_dont_overlap():
    result = stats.bootstrap_mean_diff_test(
        [0.9] * 10, [0.7] * 10,
        ids_a=[f"a{i}" for i in range(10)], ids_b=[f"b{i}" for i in range(10)],
    )
    assert result["paired"] is False
    assert "n_a" in result and "n_b" in result


def test_small_n_gates_significance_even_with_huge_gap():
    """A huge, consistent gap on only 3 points must not be labeled
    significant -- the whole point of MIN_RELIABLE_N."""
    tiny_a = [0.95, 0.94, 0.96]
    tiny_b = [0.1, 0.12, 0.09]
    result = stats.bootstrap_mean_diff_test(tiny_a, tiny_b)
    assert result["significant"] is False
    assert result["p_value"] < 0.05  # p-value itself is low; "significant" flag still gated off


def test_empty_inputs_return_empty_result():
    assert stats.bootstrap_mean_diff_test([], [0.5]) == stats._empty_diff_result()
    assert stats.bootstrap_mean_diff_test([0.5], []) == stats._empty_diff_result()
