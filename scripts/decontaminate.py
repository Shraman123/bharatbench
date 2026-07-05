"""
Flag n-gram overlap between BharatBench questions and a reference corpus.

Checks whether benchmark questions (and their reference answers) appear to
have leaked into a training/reference corpus, using the same n-gram overlap
approach used by GPT-3/PaLM-style decontamination reports.

Two modes:
  - char (default): character n-grams. Language-agnostic -- works uniformly
    across English, Bengali, and Hindi without needing per-language word
    tokenization (Indic scripts don't always segment cleanly on whitespace).
  - word: whitespace-delimited word n-grams. More standard for English-style
    corpora; use --mode word if your reference corpus is English-only.

This does not modify the dataset -- it only reports.

Usage:
    python scripts/decontaminate.py --corpus path/to/corpus_dir_or_file.txt
    python scripts/decontaminate.py --corpus corpus.txt --mode word --n 13
    python scripts/decontaminate.py --corpus corpus.txt --threshold 0.1
"""

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"
LANGUAGES = ["bengali", "hindi", "english"]

DEFAULT_N = {"char": 50, "word": 13}


def load_questions() -> list[dict]:
    questions = []
    for language in LANGUAGES:
        path = DATASET_DIR / language / "questions.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            questions.extend(json.load(f))
    return questions


def read_corpus_text(corpus_path: Path) -> str:
    if corpus_path.is_dir():
        parts = []
        for file in sorted(corpus_path.rglob("*")):
            if file.is_file():
                try:
                    parts.append(file.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    continue
        return "\n".join(parts)
    return corpus_path.read_text(encoding="utf-8", errors="ignore")


def char_ngrams(text: str, n: int) -> set[str]:
    text = " ".join(text.split())  # normalize whitespace
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngrams(text: str, n: int, mode: str) -> set[str]:
    return char_ngrams(text, n) if mode == "char" else word_ngrams(text, n)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check BharatBench questions for overlap with a reference corpus")
    parser.add_argument("--corpus", required=True, type=Path, help="Path to a text file or directory of text files")
    parser.add_argument("--mode", choices=["char", "word"], default="char")
    parser.add_argument("--n", type=int, default=None, help="n-gram size (default: 50 for char, 13 for word)")
    parser.add_argument("--threshold", type=float, default=0.05,
                         help="Fraction of a question's n-grams found in the corpus to flag it (default 0.05)")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Corpus path does not exist: {args.corpus}", file=sys.stderr)
        return 2

    n = args.n or DEFAULT_N[args.mode]

    print(f"Reading corpus from {args.corpus} ...")
    corpus_text = read_corpus_text(args.corpus)
    print(f"Corpus size: {len(corpus_text):,} characters. Building {n}-gram index ({args.mode} mode) ...")
    corpus_ngrams = ngrams(corpus_text, n, args.mode)
    print(f"Corpus n-gram index: {len(corpus_ngrams):,} unique {n}-grams.\n")

    questions = load_questions()
    flagged = []

    for q in questions:
        combined = f"{q.get('question', '')} {q.get('reference', '')}"
        q_ngrams = ngrams(combined, n, args.mode)
        if not q_ngrams:
            continue
        overlap = q_ngrams & corpus_ngrams
        overlap_ratio = len(overlap) / len(q_ngrams)
        if overlap_ratio >= args.threshold:
            flagged.append({
                "id": q.get("id"),
                "language": q.get("language"),
                "overlap_ratio": round(overlap_ratio, 4),
                "overlapping_ngrams": len(overlap),
                "total_ngrams": len(q_ngrams),
                "sample_match": next(iter(overlap), ""),
            })

    if not flagged:
        print(f"No questions exceeded the overlap threshold ({args.threshold}). "
              f"Checked {len(questions)} questions.")
        return 0

    flagged.sort(key=lambda f: f["overlap_ratio"], reverse=True)
    print(f"Flagged {len(flagged)}/{len(questions)} question(s) at or above "
          f"{args.threshold:.0%} {n}-gram overlap:\n")
    for f in flagged:
        print(f"  [{f['language']}] {f['id']}: {f['overlap_ratio']:.1%} overlap "
              f"({f['overlapping_ngrams']}/{f['total_ngrams']} {n}-grams)")
        print(f"    sample match: {f['sample_match']!r}")

    print(
        "\nNote: overlap does not automatically mean contamination -- short "
        "common phrases (numbers, stock question wording) can overlap by "
        "chance, especially in char mode with mid-size n. Review flagged "
        "questions manually before concluding leakage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
