"""
Package dataset/ into a HuggingFace-datasets-ready local directory.

Loads all questions, validates them against dataset/schema.json, and writes
one JSONL file per language plus a combined file and a minimal dataset card
into build/hf_dataset/. Does NOT upload anywhere -- there is no
huggingface_hub call in this script, deliberately.

By default, refuses to package if any question fails schema validation
(currently: all 67 existing questions, because they lack a "source" field).
Use --allow-invalid to package anyway for local inspection.

Usage:
    python scripts/package_for_hf.py
    python scripts/package_for_hf.py --allow-invalid
    python scripts/package_for_hf.py --output build/hf_dataset
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"
SCHEMA_PATH = DATASET_DIR / "schema.json"
LANGUAGES = ["bengali", "hindi", "english"]

DATASET_CARD_TEMPLATE = """---
language:
- bn
- hi
- en
license: mit
task_categories:
- question-answering
pretty_name: BharatBench
---

# BharatBench

An evaluation benchmark for LLMs on Bengali, Hindi, and English tasks across
five categories (math, reasoning, knowledge, instruction, code).

Packaged from the BharatBench GitHub repository on {timestamp}.
See the repository README for methodology, known limitations, and licensing
before relying on these numbers -- in particular the language-gap
methodology and judge-model overlap caveats.

## Splits

{split_table}
"""


def load_all() -> dict:
    data = {}
    for lang in LANGUAGES:
        path = DATASET_DIR / lang / "questions.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[lang] = json.load(f)
        else:
            data[lang] = []
    return data


def validate(data: dict) -> list:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    problems = []
    for lang, questions in data.items():
        for q in questions:
            for e in validator.iter_errors(q):
                problems.append(f"[{lang}] {q.get('id', '<no id>')}: {e.message}")
    return problems


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package BharatBench for HuggingFace datasets (local only, no upload)"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "hf_dataset")
    parser.add_argument(
        "--allow-invalid", action="store_true",
        help="Package even if questions fail schema validation (e.g. missing provenance)",
    )
    args = parser.parse_args()

    data = load_all()
    total = sum(len(v) for v in data.values())
    if total == 0:
        print("No questions found in dataset/. Nothing to package.", file=sys.stderr)
        return 2

    problems = validate(data)
    if problems and not args.allow_invalid:
        print(f"Refusing to package: {len(problems)} schema violation(s) found.\n")
        for p in problems[:10]:
            print(f"  {p}")
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
        print(
            "\nRun scripts/validate_dataset.py for the full list, or pass "
            "--allow-invalid to package anyway (not recommended for public "
            "release while questions lack a 'source' field)."
        )
        return 1

    if problems:
        print(f"WARNING: packaging despite {len(problems)} schema violation(s) (--allow-invalid set).\n")

    args.output.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for lang, questions in data.items():
        write_jsonl(args.output / f"{lang}.jsonl", questions)
        split_rows.append(f"| {lang} | {len(questions)} |")

    all_rows = [q for questions in data.values() for q in questions]
    write_jsonl(args.output / "all.jsonl", all_rows)

    card = DATASET_CARD_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        split_table="| Language | Questions |\n|---|---|\n" + "\n".join(split_rows),
    )
    (args.output / "README.md").write_text(card, encoding="utf-8")

    print(f"Packaged {total} questions across {len(data)} languages to {args.output}")
    print("This is a LOCAL package only -- nothing has been uploaded to HuggingFace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
