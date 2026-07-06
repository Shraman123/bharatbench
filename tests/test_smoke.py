"""
Smoke test for the evaluation harness. Uses providers.set_provider_override()
to stub out every provider -- no real API key or network access is needed
or used, and no SDK client is constructed.
"""

import json
import pytest

import runner
from providers import CompletionResult, Provider, set_provider_override, clear_provider_overrides


class FakeProvider(Provider):
    """Returns a valid judge JSON when asked to score, a canned answer otherwise."""

    def __init__(self, name="fake", judge_response='{"correctness":0.8,"completeness":0.7,'
                                                    '"language_quality":0.9,"clarity":0.6}'):
        super().__init__(name)
        self.judge_response = judge_response
        self.calls = []

    async def complete(self, *, model_id, messages, max_tokens, temperature, timeout=30.0):
        self.calls.append(messages)
        if len(messages) == 1:  # judge calls are a single user message
            return CompletionResult(self.judge_response, 10.0, 42)
        return CompletionResult("This is a stubbed model answer; no real API was called.", 10.0, 42)


@pytest.fixture(autouse=True)
def stub_all_providers():
    for name in ("groq", "openai", "sarvam"):
        set_provider_override(name, FakeProvider(name))
    yield
    clear_provider_overrides()


@pytest.mark.asyncio
async def test_quick_run_produces_well_formed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

    output_path = await runner.run_evaluation(
        models=["llama3-8b"],
        languages=["english"],
        limit=2,
        output_tag="ci_smoke",
    )

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["total_questions"] == 2
    assert "judge" in data["metadata"]
    assert len(data["results"]) == 2
    for record in data["results"]:
        assert record["provider"] == "groq"
        assert record["scores"]["judge_parse_failed"] is False
        assert 0.0 <= record["scores"]["overall"] <= 1.0


@pytest.mark.asyncio
async def test_sarvam_alias_routes_through_sarvam_provider(tmp_path, monkeypatch):
    """Multi-provider wiring: a sarvam-alias model must route through the
    sarvam provider override, not silently fall back to groq."""
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

    output_path = await runner.run_evaluation(
        models=["sarvam-105b"],
        languages=["english"],
        limit=1,
        output_tag="ci_smoke_sarvam",
    )

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["results"][0]["provider"] == "sarvam"
    assert data["results"][0]["model_id"] == "sarvam-105b"


@pytest.mark.asyncio
async def test_truncated_judge_json_is_flagged_not_silently_scored():
    """Regression test: a judge response cut off mid-JSON must not be
    silently averaged in as a neutral 0.5 -- it must be flagged."""
    set_provider_override(
        runner.JUDGE.provider,
        FakeProvider(runner.JUDGE.provider, judge_response='{"correctness":0.8,"completeness":0.7,"language_qual'),
    )

    scores = await runner.judge_response("2+2?", "4", "The answer is 4.", "english", "math")

    assert scores["judge_parse_failed"] is True
    assert scores["overall"] is None


@pytest.mark.asyncio
async def test_judge_overlap_with_subject_is_detected(caplog):
    """If a run's subject list includes the exact model the judge is
    configured to, that must be surfaced as a warning, not silent."""
    overlapping_alias = next(
        alias for alias, spec in runner.MODELS.items()
        if spec.provider == runner.JUDGE.provider and spec.model_id == runner.JUDGE.model_id
    )

    with caplog.at_level("WARNING"):
        runner._warn_if_judge_overlaps_subjects([overlapping_alias])

    assert any("self-grading bias risk" in message for message in caplog.messages)
