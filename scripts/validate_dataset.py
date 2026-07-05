"""
Validate dataset/{language}/questions.json against dataset/schema.json.

Usage:
    python scripts/validate_dataset.py                # report violations, exit 0
    python scripts/validate_dataset.py --strict        # exit 1 if any violations found

This does not modify the dataset. As of this script's introduction, the
existing 67 questions are expected to fail on the missing "source" field --
that's the point: it makes the provenance gap visible and machine-checkable
instead of silent. New questions submitted via CONTRIBUTING.md's process must
pass with --strict.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"
SCHEMA_PATH = DATASET_DIR / "schema.json"
LANGUAGES = ["bengali", "hindi", "english"]


def load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_language(validator: Draft202012Validator, language: str) -> list[dict]:
    path = DATASET_DIR / language / "questions.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    violations = []
    seen_ids = set()
    for q in questions:
        qid = q.get("id", "<no id>")
        errors = sorted(validator.iter_errors(q), key=lambda e: e.path)
        for e in errors:
            violations.append({
                "language": language,
                "id": qid,
                "error": e.message,
            })
        if qid in seen_ids:
            violations.append({
                "language": language,
                "id": qid,
                "error": "duplicate id within this file",
            })
        seen_ids.add(qid)

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BharatBench question files against the schema")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any violations are found")
    args = parser.parse_args()

    schema = load_schema()
    validator = Draft202012Validator(schema)

    all_violations = []
    for language in LANGUAGES:
        all_violations.extend(validate_language(validator, language))

    if not all_violations:
        print("All questions pass schema validation.")
        return 0

    by_field = defaultdict(int)
    print(f"Found {len(all_violations)} schema violation(s):\n")
    for v in all_violations:
        print(f"  [{v['language']}] {v['id']}: {v['error']}")
        by_field[v["error"]] += 1

    print(f"\n{len(all_violations)} total violation(s) across "
          f"{len({(v['language'], v['id']) for v in all_violations})} question(s).")

    if args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
