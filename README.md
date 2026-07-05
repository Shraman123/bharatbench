# BharatBench 🇮🇳

An evaluation benchmark for LLMs on Bengali, Hindi, and English across 5 task categories,
with automated LLM-as-judge scoring and language gap analysis.

This is one benchmark among several covering Indic languages (see prior work
like IndicGLUE, IndicXTREME, and MILU) — its contribution is a small,
hand-written question set focused specifically on comparing math/reasoning/
knowledge/instruction/code performance across Bengali, Hindi, and English.

**Status:** Work in progress. No paper has been published yet; the citation
below is a placeholder for when/if one is.

---

## Dataset

| Language | Questions | Categories |
|---|---|---|
| Bengali | 23 | math, reasoning, knowledge, instruction, code |
| Hindi   | 22 | math, reasoning, knowledge, instruction, code |
| English | 22 | math, reasoning, knowledge, instruction, code |
| **Total** | **67** | 5 categories, uneven distribution per language (see `scripts/validate_dataset.py`) |

## Models Evaluated

| Model ID | Alias |
|---|---|
| llama-3.3-70b-versatile | llama3-70b |
| llama-3.1-8b-instant | llama3-8b |
| openai/gpt-oss-20b | gpt-oss-20b |
| openai/gpt-oss-120b | gpt-oss-120b |

`gemma2-9b-it` and `mixtral-8x7b-32768` were deprecated by Groq (2025-10-08
and 2025-03-20 respectively) and are no longer callable; replaced above with
Groq's current production models as of 2026-07-05. Note also that
`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` — the latter is also the
judge model, see limitations — have a deprecation announced for 2026-08-16 on
free/developer tiers; revisit before then.

## Scoring Dimensions

Each response is scored by `llama-3.1-8b-instant` (LLM-as-judge) on:

- **Correctness** — Factual/mathematical accuracy vs reference answer
- **Completeness** — All parts of the question addressed
- **Language Quality** — Response in correct language, fluent
- **Clarity** — Clear, well-structured explanation

**Overall** = mean of all 4 dimensions

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY
```

## Run Evaluation

```bash
# Quick test (5 questions per language, 2 models) — ~5 min
python eval/runner.py --quick

# Bengali + Hindi only, 2 models — ~15 min
python eval/runner.py --models llama3-70b llama3-8b --langs bengali hindi

# Full benchmark (all models, all languages) — ~45 min
python eval/runner.py --all
```

## Analyze Results

```bash
# Print report
python eval/analyze.py results/eval_TIMESTAMP.json

# Generate LaTeX table for paper
python eval/analyze.py results/eval_TIMESTAMP.json --latex

# Generate dashboard JSON
python eval/analyze.py results/eval_TIMESTAMP.json --dashboard
```

---

## Key Research Questions

1. **Language Gap**: How much do models underperform on Bengali/Hindi vs English?
2. **Category Sensitivity**: Which task types (math, reasoning, code) fail most in Indic languages?
3. **Scale Effect**: Do larger models close the Indic language gap?
4. **Instruction Following**: Can models respond in Bengali when asked in Bengali?

---

## Project Structure

```
bharatbench/
├── dataset/
│   ├── schema.json           # JSON Schema for question entries (requires provenance)
│   ├── bengali/questions.json    (23 questions)
│   ├── hindi/questions.json      (22 questions)
│   └── english/questions.json    (22 questions)
├── eval/
│   ├── runner.py      # Multi-model evaluation runner
│   └── analyze.py     # Results analysis + LaTeX + dashboard
├── scripts/
│   ├── validate_dataset.py   # Schema/provenance validation, report-only
│   ├── decontaminate.py      # n-gram overlap check against a reference corpus
│   └── package_for_hf.py     # Local HF-datasets packaging (does not upload)
├── tests/             # Smoke tests, stubbed Groq client, no API key needed
└── results/            # JSON output files (gitignored)
```

There is no `paper/` folder yet — a prior version of this README referenced
one that was never created.

---

## Citation

No paper has been published for this project. If you use this benchmark,
please cite the repository directly rather than the placeholder below, which
is left here only in case a paper is published later:

```bibtex
@misc{hazra2025bharatbench,
  title   = {BharatBench: Benchmarking Large Language Models on Indic Language Tasks},
  author  = {Hazra, Shraman},
  year    = {2025},
  note    = {Unpublished. https://github.com/Shraman123/bharatbench}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — in particular, all new questions
must include a `source`/provenance field and be human-verified.

## License

[MIT](LICENSE) for the code. The dataset content (questions/reference
answers) is included under the same license for now; if you plan to reuse
the dataset specifically, check back here as this may be revisited with a
data-specific license.
