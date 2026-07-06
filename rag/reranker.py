"""
LLM-based reranking step for the RAG pipeline.

Vector similarity search (rag/vector_store.py) is a cheap bi-encoder
first pass over the whole knowledge base; reranking is a second, more
expensive pass over just the shortlist it returns, using a real model to
judge relevance rather than embedding distance. This reuses the existing
Provider abstraction (eval/providers/) via a prompt rather than adding a
second heavy ML dependency (e.g. a cross-encoder model) -- any configured
provider/model can act as the reranker.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

RERANK_PROMPT = """\
You are ranking retrieved passages by how relevant they are to a question.

QUESTION: {query}

PASSAGES:
{passages}

Return ONLY a JSON array of the passage numbers (1-indexed), ordered from
most to least relevant to the question. Example: [3, 1, 2]"""


def _format_passages(candidates: list) -> str:
    return "\n".join(f"{i + 1}. {c['text'][:300]}" for i, c in enumerate(candidates))


def _fallback_order(candidates: list) -> list:
    """If the LLM rerank call/parse fails, fall back to the vector store's
    original distance-ranked order rather than raising."""
    return candidates


async def rerank(provider, model_id: str, query: str, candidates: list, top_n: int = 3) -> list:
    """Rerank retrieval candidates using an LLM. Returns up to top_n
    candidates in reranked order (most relevant first). Falls back to the
    original (distance-ranked) order if the rerank call or parse fails --
    reranking failure should degrade gracefully, not break retrieval."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    prompt = RERANK_PROMPT.format(query=query, passages=_format_passages(candidates))
    result = await provider.complete(
        model_id=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0,
        timeout=30,
    )

    if result.text.startswith("[ERROR") or result.text.startswith("[TIMEOUT"):
        logger.warning(f"Rerank call failed ({result.text}); falling back to distance order")
        return _fallback_order(candidates)[:top_n]

    match = re.search(r"\[[\d,\s]+\]", result.text)
    if not match:
        logger.warning(f"Rerank output had no parsable order ({result.text[:200]!r}); falling back to distance order")
        return _fallback_order(candidates)[:top_n]

    try:
        order = json.loads(match.group())
        reranked = [candidates[i - 1] for i in order if 1 <= i <= len(candidates)]
        if not reranked:
            raise ValueError("empty or out-of-range order")
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"Rerank order unusable ({e}); falling back to distance order")
        return _fallback_order(candidates)[:top_n]

    return reranked[:top_n]
