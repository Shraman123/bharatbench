"""
Smoke test for the evaluation harness. Stubs the Groq client entirely --
no real GROQ_API_KEY or network access is needed or used.
"""

import json
import pytest

import runner


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMsg(content)


class FakeUsage:
    total_tokens = 42


class FakeResp:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    """Returns a valid judge JSON for judge calls, a canned answer otherwise."""

    async def create(self, model, messages, max_tokens, temperature, **kwargs):
        if len(messages) == 1:  # judge calls are a single user message
            return FakeResp(
                '{"correctness":0.8,"completeness":0.7,'
                '"language_quality":0.9,"clarity":0.6}'
            )
        return FakeResp("This is a stubbed model answer; no real API was called.")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeAsyncGroq:
    def __init__(self, api_key=""):
        self.chat = FakeChat()


@pytest.fixture(autouse=True)
def stub_groq_client(monkeypatch):
    monkeypatch.setattr(runner, "AsyncGroq", FakeAsyncGroq)


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
    assert len(data["results"]) == 2
    for record in data["results"]:
        assert record["scores"]["judge_parse_failed"] is False
        assert 0.0 <= record["scores"]["overall"] <= 1.0


@pytest.mark.asyncio
async def test_truncated_judge_json_is_flagged_not_silently_scored():
    """Regression test: a judge response cut off mid-JSON must not be
    silently averaged in as a neutral 0.5 -- it must be flagged."""

    class TruncatingCompletions:
        async def create(self, model, messages, max_tokens, temperature, **kwargs):
            return FakeResp('{"correctness":0.8,"completeness":0.7,"language_qual')

    class TruncatingAsyncGroq:
        def __init__(self, api_key=""):
            self.chat = type("Chat", (), {"completions": TruncatingCompletions()})()

    scores = await runner.judge_response(
        TruncatingAsyncGroq(), "2+2?", "4", "The answer is 4.", "english", "math",
    )

    assert scores["judge_parse_failed"] is True
    assert scores["overall"] is None
