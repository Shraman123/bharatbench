"""
Validate rag/knowledge_base/*.json against rag/schema.json (questions) and
rag/kb_schema.json (documents). Mirrors scripts/validate_dataset.py.

Usage:
    python scripts/validate_rag_content.py                # report, exit 0
    python scripts/validate_rag_content.py --strict        # exit 1 on any violation
    python scripts/validate_rag_content.py --reject-placeholders
        # also fail if any entry has "placeholder": true -- use this once you've
        # replaced the placeholder files with real content, to confirm none of
        # it slipped through.

This does not modify rag/ content. If you haven't replaced the placeholder
files yet, this will report 0 schema violations (the placeholder entries are
schema-valid) but --reject-placeholders will still fail, which is correct:
schema-valid placeholder content is still not real content.
"""

import json
import sys
import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parent.parent
RAG_DIR = ROOT / "rag"
KB_DIR = RAG_DIR / "knowledge_base"


def _validate_file(path: Path, schema: dict, reject_placeholders: bool) -> list:
    if not path.exists():
        return []

    validator = Draft202012Validator(schema)
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    violations = []
    for entry in entries:
        entry_id = entry.get("id", "<no id>")
        for e in validator.iter_errors(entry):
            violations.append(f"{path.name}/{entry_id}: {e.message}")
        if reject_placeholders and entry.get("placeholder"):
            violations.append(f"{path.name}/{entry_id}: placeholder=true (not real content)")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RAG knowledge-base/question content")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any schema violation is found")
    parser.add_argument("--reject-placeholders", action="store_true",
                         help="Also fail if any entry has placeholder: true")
    args = parser.parse_args()

    kb_schema = json.loads((RAG_DIR / "kb_schema.json").read_text(encoding="utf-8"))
    q_schema = json.loads((RAG_DIR / "schema.json").read_text(encoding="utf-8"))

    violations = []
    # Prefer real content files; fall back to the PLACEHOLDER_ ones, same as rag/pipeline.py
    doc_file = KB_DIR / "documents.json" if (KB_DIR / "documents.json").exists() else KB_DIR / "PLACEHOLDER_documents.json"
    q_file = KB_DIR / "questions.json" if (KB_DIR / "questions.json").exists() else KB_DIR / "PLACEHOLDER_questions.json"

    violations += _validate_file(doc_file, kb_schema, args.reject_placeholders)
    violations += _validate_file(q_file, q_schema, args.reject_placeholders)

    print(f"Checked: {doc_file.name}, {q_file.name}\n")

    if not violations:
        print("No violations found.")
        return 0

    print(f"Found {len(violations)} violation(s):\n")
    for v in violations:
        print(f"  {v}")

    if args.strict or args.reject_placeholders:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
