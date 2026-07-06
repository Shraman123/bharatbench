"""
RAG evaluation pipeline: retrieve -> rerank -> generate -> judge.

Reuses eval/runner.py's call_model()/judge_response() (same provider
abstraction, same scoring semantics as the non-RAG harness) and
eval/config.py's MODELS/JUDGE registry -- a RAG question is graded the same
way a regular question is (LLM-as-judge against a reference answer). The
only difference is the subject model receives retrieved+reranked context
before answering.

Ships with PLACEHOLDER-only knowledge-base/question content (see
rag/knowledge_base/README.md) so this is runnable and testable end-to-end
without any real content existing yet. To use real content, drop
rag/knowledge_base/documents.json and rag/knowledge_base/questions.json in
place (see rag/kb_schema.json / rag/schema.json) -- this module prefers
those over the PLACEHOLDER_ files automatically, with a loud warning if it
falls back to placeholders.

Usage (run from the repo root -- rag/ is a package, unlike eval/, so this
must be invoked with -m, not as a bare script):
    python -m rag.pipeline --models llama3-8b --k 5 --top-n 3
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import _eval_path  # noqa: F401  (must run before importing config/providers/runner)
from . import vector_store
from .reranker import rerank

from config import JUDGE, MODELS  # noqa: E402
from providers import get_provider  # noqa: E402
from runner import RESULTS_DIR, call_model, judge_response  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent / "knowledge_base"

RAG_PROMPT_TEMPLATE = """\
Use the following retrieved context to answer the question. If the context \
doesn't contain the answer, say so rather than guessing.

CONTEXT:
{context}

QUESTION: {question}"""


def _load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_kb_documents(path: Optional[Path] = None) -> list:
    if path:
        return _load_json(path)
    real = KB_DIR / "documents.json"
    if real.exists():
        return _load_json(real)
    logger.warning(
        "No rag/knowledge_base/documents.json found -- using "
        "PLACEHOLDER_documents.json (not real content). See "
        "rag/knowledge_base/README.md."
    )
    return _load_json(KB_DIR / "PLACEHOLDER_documents.json")


def load_rag_questions(path: Optional[Path] = None) -> list:
    if path:
        return _load_json(path)
    real = KB_DIR / "questions.json"
    if real.exists():
        return _load_json(real)
    logger.warning(
        "No rag/knowledge_base/questions.json found -- using "
        "PLACEHOLDER_questions.json (not real content). See "
        "rag/knowledge_base/README.md."
    )
    return _load_json(KB_DIR / "PLACEHOLDER_questions.json")


async def evaluate_rag_question(
    model_alias: str,
    spec,
    q: dict,
    collection,
    retrieval_k: int = 5,
    rerank_top_n: int = 3,
) -> dict:
    """Retrieve -> rerank -> generate -> judge for one (model, question) pair."""
    candidates = vector_store.query(collection, q["question"], k=retrieval_k)
    reranked = await rerank(
        get_provider(spec.provider), spec.model_id, q["question"], candidates, top_n=rerank_top_n
    )

    context = "\n\n".join(f"[{c['title']}] {c['text']}" for c in reranked) or "(no context retrieved)"
    augmented_question = RAG_PROMPT_TEMPLATE.format(context=context, question=q["question"])

    response, latency_ms, tokens = await call_model(spec, augmented_question, q["language"])
    await asyncio.sleep(0.5)

    scores = await judge_response(q["question"], q["reference"], response, q["language"], q["category"])

    relevant_ids = set(q.get("relevant_doc_ids", []))
    retrieved_ids = {c["id"] for c in reranked}
    retrieval_hit = bool(retrieved_ids & relevant_ids) if relevant_ids else None

    return {
        "question_id":       q["id"],
        "language":          q["language"],
        "category":          q["category"],
        "difficulty":        q["difficulty"],
        "model":             model_alias,
        "provider":          spec.provider,
        "model_id":          spec.model_id,
        "question":          q["question"],
        "reference":         q["reference"],
        "response":          response,
        "latency_ms":        latency_ms,
        "token_count":       tokens,
        "retrieved_doc_ids": [c["id"] for c in candidates],
        "reranked_doc_ids":  [c["id"] for c in reranked],
        "relevant_doc_ids":  sorted(relevant_ids),
        "retrieval_hit":     retrieval_hit,
        "scores":            scores,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


async def run_rag_evaluation(
    models: list,
    documents: Optional[list] = None,
    questions: Optional[list] = None,
    retrieval_k: int = 5,
    rerank_top_n: int = 3,
    output_tag: str = "",
) -> str:
    documents = documents if documents is not None else load_kb_documents()
    questions = questions if questions is not None else load_rag_questions()

    using_placeholder = (
        any(d.get("placeholder") for d in documents) or any(q.get("placeholder") for q in questions)
    )
    if using_placeholder:
        logger.warning(
            "Running with PLACEHOLDER knowledge-base/question content -- this "
            "exercises the pipeline only, it is not a real evaluation. See "
            "rag/knowledge_base/README.md."
        )

    if not questions:
        raise ValueError("No RAG questions loaded.")

    collection = vector_store.build_index(documents)

    results = []
    for alias in models:
        spec = MODELS.get(alias)
        if not spec:
            logger.warning(f"Unknown model: {alias}. Skipping.")
            continue

        logger.info(f"Evaluating RAG task: {alias} ({spec.provider}/{spec.model_id})")
        for q in questions:
            result = await evaluate_rag_question(alias, spec, q, collection, retrieval_k, rerank_top_n)
            results.append(result)
            logger.info(
                f"  [{alias}] {q['id']}: retrieval_hit={result['retrieval_hit']} "
                f"overall={result['scores']['overall']}"
            )

    tag = output_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"rag_eval_{tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "run_id": tag,
                    "models": models,
                    "judge": {"provider": JUDGE.provider, "model_id": JUDGE.model_id},
                    "retrieval_k": retrieval_k,
                    "rerank_top_n": rerank_top_n,
                    "num_documents": len(documents),
                    "num_questions": len(questions),
                    "used_placeholder_content": using_placeholder,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )

    logger.info(f"Results saved to: {output_path}")
    return str(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BharatBench RAG Evaluation Pipeline")
    parser.add_argument("--models", nargs="+", default=["llama3-8b"],
                        choices=list(MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--k", type=int, default=5, help="Candidates to retrieve before reranking")
    parser.add_argument("--top-n", type=int, default=3, help="Candidates to keep after reranking")
    parser.add_argument("--tag", type=str, default="", help="Tag for output file name")
    args = parser.parse_args()

    asyncio.run(run_rag_evaluation(
        models=args.models,
        retrieval_k=args.k,
        rerank_top_n=args.top_n,
        output_tag=args.tag,
    ))
