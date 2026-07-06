# Knowledge base -- PLACEHOLDER CONTENT ONLY

`PLACEHOLDER_documents.json` and `PLACEHOLDER_questions.json` in this
directory are **not real content**. They exist only so the RAG
infrastructure (`rag/vector_store.py`, `rag/reranker.py`, `rag/pipeline.py`)
has something to embed, index, retrieve, rerank, and score end-to-end in
tests, without depending on any real knowledge base existing yet.

Every entry in both files has `"placeholder": true` and text that says
"PLACEHOLDER" in it, deliberately, so it can't be mistaken for real
benchmark content if it leaks into a real run.

**Before using this pipeline for actual evaluation:**

1. Replace `PLACEHOLDER_documents.json` with real Indic-language knowledge-
   base documents, each with a real `source` field (see `rag/kb_schema.json`
   and `CONTRIBUTING.md` -- same provenance requirement as the main dataset).
2. Replace `PLACEHOLDER_questions.json` with real RAG questions, each with a
   real `source` and `relevant_doc_ids` pointing at real document IDs (see
   `rag/schema.json`).
3. Run `scripts/validate_rag_content.py` to confirm both pass schema
   validation with no `placeholder: true` entries remaining.

This mirrors the main dataset's rule: question/knowledge-base *content* is
not something this codebase generates or invents -- see
`README.md#known-limitations` and `CONTRIBUTING.md`.
