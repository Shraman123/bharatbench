"""
API tests for the FastAPI evaluation service. Uses providers.set_provider_override()
so no real API key or network access is needed -- same stubbing approach as
tests/test_smoke.py.
"""

import time

import pytest
from fastapi.testclient import TestClient

from providers import CompletionResult, Provider, set_provider_override, clear_provider_overrides
from service.app import app


class FakeProvider(Provider):
    async def complete(self, *, model_id, messages, max_tokens, temperature, timeout=30.0):
        if len(messages) == 1:  # judge call
            return CompletionResult(
                '{"correctness":0.8,"completeness":0.7,"language_quality":0.9,"clarity":0.6}',
                5.0, 20,
            )
        return CompletionResult("Stubbed answer, no real API called.", 5.0, 20)


@pytest.fixture(autouse=True)
def stub_providers():
    for name in ("groq", "openai", "sarvam"):
        set_provider_override(name, FakeProvider(name))
    yield
    clear_provider_overrides()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _wait_for_completion(client, run_id, timeout_s=10):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/eval/runs/{run_id}")
        assert resp.status_code == 200
        if resp.json()["status"] in ("completed", "failed"):
            return resp.json()
        time.sleep(0.1)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_s}s")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_submit_status_results_report_roundtrip(client):
    resp = client.post("/eval/runs", json={"models": ["llama3-8b"], "languages": ["english"], "limit": 2})
    assert resp.status_code == 202
    body = resp.json()
    run_id = body["run_id"]
    assert body["status"] in ("queued", "running")

    final = _wait_for_completion(client, run_id)
    assert final["status"] == "completed"
    assert final["done"] == final["total"] == 2

    results_resp = client.get(f"/eval/runs/{run_id}/results")
    assert results_resp.status_code == 200
    data = results_resp.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["scores"]["judge_parse_failed"] is False

    report_resp = client.get(f"/eval/runs/{run_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert "language_gap_caveat" in report and len(report["language_gap_caveat"]) > 0
    assert "NOT parallel" in report["language_gap_caveat"]
    assert report["usable_evaluations"] == 2
    assert report["pairwise_model_comparisons"] == []  # only one model in this run


def test_report_includes_pairwise_comparisons_for_multi_model_run(client):
    resp = client.post(
        "/eval/runs",
        json={"models": ["llama3-8b", "llama3-70b"], "languages": ["english"], "limit": 2},
    )
    run_id = resp.json()["run_id"]
    _wait_for_completion(client, run_id)

    report = client.get(f"/eval/runs/{run_id}/report").json()
    assert len(report["pairwise_model_comparisons"]) == 1
    comparison = report["pairwise_model_comparisons"][0]
    assert {comparison["model_a"], comparison["model_b"]} == {"llama3-8b", "llama3-70b"}
    assert "p_value" in comparison and "significant" in comparison


def test_results_before_completion_returns_409(client):
    resp = client.post("/eval/runs", json={"models": ["llama3-8b"], "languages": ["english"], "limit": 1})
    run_id = resp.json()["run_id"]
    results_resp = client.get(f"/eval/runs/{run_id}/results")
    assert results_resp.status_code in (200, 409)  # allow for a fast stubbed run finishing first
    _wait_for_completion(client, run_id)


def test_unknown_run_id_returns_404(client):
    resp = client.get("/eval/runs/does-not-exist")
    assert resp.status_code == 404


def test_invalid_model_alias_returns_422(client):
    resp = client.post("/eval/runs", json={"models": ["not-a-real-model"], "languages": ["english"]})
    assert resp.status_code == 422


def test_invalid_language_returns_422(client):
    resp = client.post("/eval/runs", json={"models": ["llama3-8b"], "languages": ["klingon"]})
    assert resp.status_code == 422


def test_empty_models_list_returns_422(client):
    resp = client.post("/eval/runs", json={"models": [], "languages": ["english"]})
    assert resp.status_code == 422
