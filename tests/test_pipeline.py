"""
Tests for the LangGraph generate->judge->verify->aggregate pipeline.
Stubs providers the same way tests/test_smoke.py does -- no real API keys.
"""

import pytest

import pipeline
from providers import CompletionResult, Provider, set_provider_override, clear_provider_overrides


class FakeProvider(Provider):
    def __init__(self, name="fake", judge_response='{"correctness":0.8,"completeness":0.7,'
                                                    '"language_quality":0.9,"clarity":0.6}'):
        super().__init__(name)
        self.judge_response = judge_response

    async def complete(self, *, model_id, messages, max_tokens, temperature, timeout=30.0):
        if len(messages) == 1:  # judge call
            return CompletionResult(self.judge_response, 10.0, 42)
        return CompletionResult("Stubbed model answer.", 10.0, 42)


@pytest.fixture(autouse=True)
def stub_providers():
    for name in ("groq", "openai", "sarvam"):
        set_provider_override(name, FakeProvider(name))
    yield
    clear_provider_overrides()


@pytest.mark.asyncio
async def test_pipeline_runs_all_four_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)

    state = await pipeline.run_pipeline(
        models=["llama3-8b"], languages=["english"], limit=2, output_tag="pipeline_test",
    )

    assert len(state["generated"]) == 2
    assert len(state["judged"]) == 2
    assert len(state["verified"]) == 2
    assert state["verification_issues"] == []
    for record in state["verified"]:
        assert record["verification_ok"] is True
        assert record["scores"]["judge_parse_failed"] is False

    agg = state["aggregate"]
    assert agg["total_evaluations"] == 2
    assert agg["usable_evaluations"] == 2
    assert agg["degraded_evaluations"] == 0
    assert agg["verification_issue_count"] == 0
    assert "NOT parallel" in agg["language_gap_caveat"]
    assert agg["pairwise_model_comparisons"] == []  # only one model in this run
    assert "ci_low" in agg["by_model"]["('llama3-8b',)"]


@pytest.mark.asyncio
async def test_degraded_judge_output_flows_through_without_verification_issue(tmp_path, monkeypatch):
    """A judge_parse_failed record (None scores) is the *expected* shape for
    a degraded record -- verify must not flag it as inconsistent."""
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    for name in ("groq", "openai", "sarvam"):
        set_provider_override(name, FakeProvider(
            name, judge_response='{"correctness":0.8,"completeness":0.7,"language_qual'  # truncated
        ))

    state = await pipeline.run_pipeline(
        models=["llama3-8b"], languages=["english"], limit=1, output_tag="pipeline_degraded_test",
    )

    record = state["verified"][0]
    assert record["scores"]["judge_parse_failed"] is True
    assert record["scores"]["overall"] is None
    assert record["verification_ok"] is True  # None-scores IS the correct shape for a degraded record
    assert state["verification_issues"] == []
    assert state["aggregate"]["degraded_evaluations"] == 1
    assert state["aggregate"]["usable_evaluations"] == 0


def test_verify_flags_inconsistent_record():
    """Unit test for the consistency checker itself: a failed model call
    that somehow got a non-zero score must be flagged."""
    bad_record = {
        "response": "[ERROR: something broke]",
        "scores": {"overall": 0.7, "judge_parse_failed": False},
    }
    issues = pipeline._check_consistency(bad_record)
    assert len(issues) == 1
    assert "model call failed" in issues[0]


@pytest.mark.asyncio
async def test_graph_structure_has_four_nodes():
    graph = pipeline.build_graph()
    node_names = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert node_names == {"generate", "judge", "verify", "aggregate"}
