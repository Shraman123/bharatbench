# Contributing to BharatBench

## Code contributions

1. Fork and clone the repo.
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Lint before committing: `ruff check .`
4. Run the smoke test (no API key required, uses a stubbed Groq client): `pytest tests/`
5. Keep PRs focused — one logical change per PR, with a clear description of what and why.

## Contributing benchmark questions

The credibility of this benchmark depends entirely on its questions being
human-sourced and human-verified. To keep that true as the dataset grows:

- **Every new question must include a `source` field** describing its
  provenance (e.g. `"author-written"`, `"NCERT Class 10 Math Textbook, Ch. 4"`,
  `"contributor: <name/handle>, reviewed by: <name/handle>"`). Questions
  without a `source` field will fail schema validation
  (`scripts/validate_dataset.py`) and will not be merged.
- **Do not submit LLM-generated questions.** If a question was drafted with
  LLM assistance, it must be disclosed in the `source` field and independently
  verified by a human fluent in the target language before submission.
- **Verify the reference answer yourself.** Run through the math/logic by
  hand; don't copy an answer you haven't checked. (We found and flagged, but
  did not touch, at least one existing question with an unverifiable
  reference during an engineering audit — see `README.md#known-limitations`.)
- Before submitting a new question set, run it through
  `scripts/validate_dataset.py` (schema/provenance check) and, if you have a
  reference corpus, `scripts/decontaminate.py` (checks for accidental overlap
  with common training corpora) locally.
- Maintainers reviewing dataset PRs should spot-check reference answers, not
  just schema validity — a well-formed JSON object can still contain a wrong
  answer.

## Reporting issues

Open a GitHub issue with a minimal reproduction. For evaluation-harness bugs,
include the model/language/category combination and, if possible, the raw
judge output that triggered the problem.
