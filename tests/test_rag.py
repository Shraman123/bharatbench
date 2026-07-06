"""
Tests for the RAG evaluation infrastructure (rag/). Uses a tiny deterministic
hashing embedding function instead of Chroma's real default -- the real one
downloads a ~79MB ONNX model on first use, which would make CI slow/flaky and
network-dependent. Production code (rag/vector_store.py) still defaults to
Chroma's real embedder; tests just inject a fake one via the same
embedding_function parameter, same stubbing philosophy as the rest of this
project's tests.
"""

import hashlib

import pytest
from chromadb.api.types import EmbeddingFunction

from rag import vector_store
from rag.reranker import rerank
from rag.pipeline import evaluate_rag_question, load_kb_documents, load_rag_questions, run_rag_evaluation
from providers import CompletionResult, Provider, set_provider_override, clear_provider_overrides


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words hashing vectorizer -- not semantically
    meaningful, just fast and stable enough to make retrieval tests
    reproducible without downloading a real model."""

    DIM = 32

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "test-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @classmethod
    def build_from_config(cls, config: dict) -> "HashEmbedding":
        return cls()

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = [0.0] * self.DIM
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.DIM] += 1.0
            vectors.append(vec)
        return vectors


class FakeProvider(Provider):
    def __init__(self, name="fake", judge_response='{"correctness":0.8,"completeness":0.7,'
                                                    '"language_quality":0.9,"clarity":0.6}',
                 rerank_response=None):
        super().__init__(name)
        self.judge_response = judge_response
        self.rerank_response = rerank_response

    async def complete(self, *, model_id, messages, max_tokens, temperature, timeout=30.0):
        content = messages[0]["content"] if messages else ""
        if "Return ONLY a JSON array" in content:  # rerank call
            return CompletionResult(self.rerank_response or "[1, 2]", 5.0, 10)
        if len(messages) == 1:  # judge call
            return CompletionResult(self.judge_response, 10.0, 42)
        return CompletionResult("Stubbed model answer.", 10.0, 42)


@pytest.fixture(autouse=True)
def stub_providers():
    for name in ("groq", "openai", "sarvam"):
        set_provider_override(name, FakeProvider(name))
    yield
    clear_provider_overrides()


PLACEHOLDER_DOCS = [
    {"id": "doc_a", "language": "english", "title": "Cats", "text": "Cats are small domesticated mammals.", "source": "test"},
    {"id": "doc_b", "language": "english", "title": "Dogs", "text": "Dogs are loyal domesticated mammals.", "source": "test"},
]


def test_vector_store_build_and_query_returns_relevant_doc(tmp_path):
    client = vector_store.get_client(persist_dir=tmp_path)
    collection = vector_store.build_index(
        PLACEHOLDER_DOCS, collection_name="test_kb", client=client, embedding_function=HashEmbedding()
    )
    results = vector_store.query(collection, "tell me about cats", k=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc_a"  # closer match to "cats" should rank first
    assert results[0]["distance"] <= results[1]["distance"]


def test_vector_store_empty_collection_returns_no_results(tmp_path):
    client = vector_store.get_client(persist_dir=tmp_path)
    collection = vector_store.build_index([], collection_name="empty_kb", client=client, embedding_function=HashEmbedding())
    assert vector_store.query(collection, "anything", k=5) == []


@pytest.mark.asyncio
async def test_rerank_reorders_candidates():
    candidates = [
        {"id": "c1", "text": "irrelevant passage"},
        {"id": "c2", "text": "the actually relevant passage"},
    ]
    provider = FakeProvider(rerank_response="[2, 1]")
    result = await rerank(provider, "some-model", "query", candidates, top_n=2)
    assert [c["id"] for c in result] == ["c2", "c1"]


@pytest.mark.asyncio
async def test_rerank_falls_back_on_malformed_output():
    candidates = [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}]
    provider = FakeProvider(rerank_response="not valid json at all")
    result = await rerank(provider, "some-model", "query", candidates, top_n=2)
    assert [c["id"] for c in result] == ["c1", "c2"]  # falls back to original (distance) order


@pytest.mark.asyncio
async def test_rerank_single_candidate_is_passthrough():
    candidates = [{"id": "only", "text": "x"}]
    result = await rerank(FakeProvider(), "some-model", "query", candidates, top_n=3)
    assert result == candidates


def test_placeholder_content_loads_and_is_flagged():
    docs = load_kb_documents()
    questions = load_rag_questions()
    assert len(docs) >= 1
    assert len(questions) >= 1
    assert all(d.get("placeholder") for d in docs)
    assert all(q.get("placeholder") for q in questions)


@pytest.mark.asyncio
async def test_evaluate_rag_question_end_to_end(tmp_path):
    client = vector_store.get_client(persist_dir=tmp_path)
    collection = vector_store.build_index(
        PLACEHOLDER_DOCS, collection_name="e2e_kb", client=client, embedding_function=HashEmbedding()
    )

    from config import MODELS
    spec = MODELS["llama3-8b"]
    q = {
        "id": "test_rag_001", "language": "english", "category": "rag", "difficulty": "easy",
        "question": "What are cats?", "reference": "Small domesticated mammals.",
        "relevant_doc_ids": ["doc_a"],
    }

    result = await evaluate_rag_question("llama3-8b", spec, q, collection, retrieval_k=2, rerank_top_n=2)

    assert result["question_id"] == "test_rag_001"
    assert "doc_a" in result["retrieved_doc_ids"]
    assert result["retrieval_hit"] is True
    assert result["scores"]["judge_parse_failed"] is False
    assert 0.0 <= result["scores"]["overall"] <= 1.0


@pytest.mark.asyncio
async def test_run_rag_evaluation_with_placeholder_content_warns_and_completes(tmp_path, monkeypatch):
    from rag import pipeline
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)

    output_path = await run_rag_evaluation(models=["llama3-8b"], retrieval_k=2, rerank_top_n=2, output_tag="rag_test")

    import json
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["used_placeholder_content"] is True
    assert len(data["results"]) >= 1
