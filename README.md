# BharatBench 🇮🇳

**The First Evaluation Benchmark for AI Models on Indic Language Tasks**

A rigorous benchmark evaluating LLMs on Bengali, Hindi, and English across 5 task categories,
with automated LLM-as-judge scoring and language gap analysis.

**Paper:** Shraman Hazra, 2025. *BharatBench: Benchmarking Large Language Models on Indic Language Reasoning Tasks*

---

## Dataset

| Language | Questions | Categories |
|---|---|---|
| Bengali | 40 | math, reasoning, knowledge, instruction, code |
| Hindi   | 40 | math, reasoning, knowledge, instruction, code |
| English | 40 | math, reasoning, knowledge, instruction, code |
| **Total** | **120** | 5 categories × 3 difficulties |

## Models Evaluated

| Model ID | Alias |
|---|---|
| llama-3.3-70b-versatile | llama3-70b |
| llama-3.1-8b-instant | llama3-8b |
| gemma2-9b-it | gemma2-9b |
| mixtral-8x7b-32768 | mixtral-8x7b |

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
│   ├── bengali/questions.json    (40 questions)
│   ├── hindi/questions.json      (40 questions)
│   └── english/questions.json   (40 questions)
├── eval/
│   ├── runner.py      # Multi-model evaluation runner
│   └── analyze.py     # Results analysis + LaTeX + dashboard
├── results/           # JSON output files (gitignored)
└── paper/             # LaTeX paper (Week 3)
```

---

## Citation

```bibtex
@misc{hazra2025bharatbench,
  title   = {BharatBench: Benchmarking Large Language Models on Indic Language Tasks},
  author  = {Hazra, Shraman},
  year    = {2025},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
