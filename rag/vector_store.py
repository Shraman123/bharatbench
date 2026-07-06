"""
Thin wrapper around a local Chroma vector store for the RAG eval task.

Uses Chroma's bundled default embedding function (a small ONNX MiniLM model,
downloaded once on first use, then runs fully offline/locally -- no API key
needed for retrieval itself). That default is English-centric and is a
placeholder choice for infrastructure purposes: for real Indic-language
retrieval quality, swap in a multilingual embedding function (e.g. a
sentence-transformers multilingual model, or a provider's embeddings API if
one is added later) via the `embedding_function` parameter below.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction

DEFAULT_PERSIST_DIR = Path(__file__).parent / "chroma_store"


def get_client(persist_dir: Optional[Path] = None) -> chromadb.ClientAPI:
    """A persistent local Chroma client. Pass persist_dir=None (default) for
    the repo-local store, or a tmp_path in tests to avoid writing to disk."""
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(
    documents: list,
    collection_name: str = "bharatbench_kb",
    client: Optional[chromadb.ClientAPI] = None,
    embedding_function: Optional[EmbeddingFunction] = None,
) -> chromadb.api.models.Collection.Collection:
    """Embed and index a list of KB documents (see rag/kb_schema.json for
    shape). Recreates the collection from scratch each call -- this is meant
    for (re)building an index from a document set, not incremental upserts."""
    if client is None:
        client = get_client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist yet

    kwargs = {}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    collection = client.get_or_create_collection(collection_name, **kwargs)

    if documents:
        collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[
                {"language": doc["language"], "title": doc["title"], "source": doc["source"]}
                for doc in documents
            ],
        )
    return collection


def query(collection, query_text: str, k: int = 5) -> list:
    """Return up to k candidate chunks as
    [{"id", "text", "title", "source", "language", "distance"}, ...],
    ordered by relevance (ascending distance = more relevant first)."""
    if collection.count() == 0:
        return []

    n_results = min(k, collection.count())
    res = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for i, doc_id in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        candidates.append({
            "id": doc_id,
            "text": res["documents"][0][i],
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "language": meta.get("language", ""),
            "distance": res["distances"][0][i],
        })
    return candidates
