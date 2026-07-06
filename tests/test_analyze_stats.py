"""
Tests for the statistical-rigor additions to eval/analyze.py: CI fields on
aggregate() and the new pairwise_significance() function.
"""

import analyze


def _record(question_id, model, score, language="english", category="math", difficulty="easy"):
    return {
        "question_id": question_id, "language": language, "category": category,
        "difficulty": difficulty, "model": model,
        "scores": {"overall": score, "judge_parse_failed": False},
    }


def test_aggregate_includes_ci_and_caveat_fields():
    records = [_record(f"q{i}", "model-a", 0.8) for i in range(12)]
    result = analyze.aggregate(records, ["model"])
    group = result[("model-a",)]
    assert "ci_low" in group and "ci_high" in group
    assert group["ci_low"] <= group["mean"] <= group["ci_high"]
    assert group["reliability_caveat"] is None  # n=12 >= MIN_RELIABLE_N


def test_aggregate_flags_small_n_with_reliability_caveat():
    records = [_record(f"q{i}", "model-a", 0.8) for i in range(3)]
    result = analyze.aggregate(records, ["model"])
    group = result[("model-a",)]
    assert group["reliability_caveat"] is not None


def test_pairwise_significance_detects_real_gap():
    records = (
        [_record(f"q{i}", "strong-model", 0.9) for i in range(15)]
        + [_record(f"q{i}", "weak-model", 0.5) for i in range(15)]
    )
    comparisons = analyze.pairwise_significance(records, "model")
    assert len(comparisons) == 1
    c = comparisons[0]
    assert {c["model_a"], c["model_b"]} == {"strong-model", "weak-model"}
    assert c["paired"] is True  # same question_ids for both
    assert c["significant"] is True
    assert abs(abs(c["mean_diff"]) - 0.4) < 1e-6


def test_pairwise_significance_no_comparisons_for_single_model():
    records = [_record(f"q{i}", "only-model", 0.7) for i in range(10)]
    assert analyze.pairwise_significance(records, "model") == []


def test_pairwise_significance_three_models_gives_three_pairs():
    records = (
        [_record(f"q{i}", "a", 0.9) for i in range(10)]
        + [_record(f"q{i}", "b", 0.6) for i in range(10)]
        + [_record(f"q{i}", "c", 0.3) for i in range(10)]
    )
    comparisons = analyze.pairwise_significance(records, "model")
    pairs = {frozenset((c["model_a"], c["model_b"])) for c in comparisons}
    assert pairs == {frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}


def test_dashboard_json_includes_pairwise_comparisons():
    records = (
        [_record(f"q{i}", "a", 0.9) for i in range(10)]
        + [_record(f"q{i}", "b", 0.5) for i in range(10)]
    )
    metadata = {"run_id": "test", "models": ["a", "b"], "languages": ["english"], "total_questions": 10}
    gaps = analyze.compute_language_gap(records)
    dashboard = analyze.generate_dashboard_json(metadata, records, gaps)
    assert "pairwise_model_comparisons" in dashboard
    assert len(dashboard["pairwise_model_comparisons"]) == 1
