"""
BharatBench Evaluation Runner
==============================
Runs all benchmark questions through multiple Groq models
and scores each response using LLM-as-judge.

Usage:
    python eval/runner.py --models llama3-70b gemma2-9b --langs bengali hindi
    python eval/runner.py --all          # Run everything (takes ~30 min)
    python eval/runner.py --quick        # Run 10 questions per language (testing)
"""

import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from groq import AsyncGroq
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Models available on Groq free tier
MODELS = {
    "llama3-70b":   "llama-3.3-70b-versatile",
    "llama3-8b":    "llama-3.1-8b-instant",
    "gemma2-9b":    "gemma2-9b-it",
    "mixtral-8x7b": "mixtral-8x7b-32768",
}

LANGUAGES = ["bengali", "hindi", "english"]

JUDGE_MODEL = "llama-3.1-8b-instant"   # Fast model for scoring

# ── Judge Prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert multilingual evaluator. Score this AI response objectively.

QUESTION: {question}
REFERENCE ANSWER: {reference}
MODEL RESPONSE: {response}
LANGUAGE: {language}
CATEGORY: {category}

Score on these dimensions (0.0 to 1.0):
- correctness   : Is the answer factually/mathematically correct vs the reference?
- completeness  : Does it cover all parts of the question?
- language_quality : Is the response in the correct language ({language})? Is it fluent?
- clarity       : Is the explanation clear and well-structured?

Respond ONLY with valid JSON, nothing else:
{{"correctness": 0.X, "completeness": 0.X, "language_quality": 0.X, "clarity": 0.X}}"""


# ── Core Functions ────────────────────────────────────────────────────────────

async def call_model(
    client: AsyncGroq,
    model_id: str,
    question: str,
    language: str,
    timeout: int = 30,
) -> tuple[str, float, int]:
    """Call a model and return (response, latency_ms, tokens)."""

    system = f"""You are a helpful AI assistant. 
Answer the question accurately. 
If the question is in {language}, respond in {language}.
Be concise but complete."""

    start = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": question},
                ],
                max_tokens=600,
                temperature=0.1,
            ),
            timeout=timeout,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return response, latency_ms, tokens
    except asyncio.TimeoutError:
        return "[TIMEOUT]", timeout * 1000, 0
    except Exception as e:
        return f"[ERROR: {str(e)[:100]}]", 0, 0


async def judge_response(
    client: AsyncGroq,
    question: str,
    reference: str,
    response: str,
    language: str,
    category: str,
) -> dict:
    """Score a model response using LLM-as-judge."""
    import re

    if response.startswith("[ERROR") or response.startswith("[TIMEOUT"):
        return {
            "correctness": 0.0,
            "completeness": 0.0,
            "language_quality": 0.0,
            "clarity": 0.0,
            "overall": 0.0,
        }

    prompt = JUDGE_PROMPT.format(
        question=question[:400],
        reference=reference[:400],
        response=response[:500],
        language=language,
        category=category,
    )

    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\{[^}]+\}", raw)
        if match:
            scores = json.loads(match.group())
            c = round(float(scores.get("correctness",     0.5)), 3)
            co = round(float(scores.get("completeness",    0.5)), 3)
            lq = round(float(scores.get("language_quality",0.5)), 3)
            cl = round(float(scores.get("clarity",         0.5)), 3)
            overall = round((c + co + lq + cl) / 4, 3)
            return {
                "correctness":      max(0.0, min(1.0, c)),
                "completeness":     max(0.0, min(1.0, co)),
                "language_quality": max(0.0, min(1.0, lq)),
                "clarity":          max(0.0, min(1.0, cl)),
                "overall":          max(0.0, min(1.0, overall)),
            }
    except Exception as e:
        logger.warning(f"Judge failed: {e}")

    return {"correctness": 0.5, "completeness": 0.5, "language_quality": 0.5, "clarity": 0.5, "overall": 0.5}


async def evaluate_question(
    client: AsyncGroq,
    model_name: str,
    model_id: str,
    q: dict,
) -> dict:
    """Run one question through one model and return a result record."""
    logger.info(f"  [{model_name}] {q['id']}")

    response, latency_ms, tokens = await call_model(
        client, model_id, q["question"], q["language"]
    )

    # Small delay to avoid rate limiting
    await asyncio.sleep(0.5)

    scores = await judge_response(
        client,
        q["question"],
        q["reference"],
        response,
        q["language"],
        q["category"],
    )

    return {
        "question_id":  q["id"],
        "language":     q["language"],
        "category":     q["category"],
        "difficulty":   q["difficulty"],
        "model":        model_name,
        "model_id":     model_id,
        "question":     q["question"],
        "reference":    q["reference"],
        "response":     response,
        "latency_ms":   latency_ms,
        "token_count":  tokens,
        "scores":       scores,
        "timestamp":    datetime.utcnow().isoformat(),
    }


def load_questions(languages: list, limit: Optional[int] = None) -> list:
    """Load questions from dataset JSON files."""
    questions = []
    for lang in languages:
        path = DATASET_DIR / lang / "questions.json"
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            qs = json.load(f)
        if limit:
            qs = qs[:limit]
        questions.extend(qs)
        logger.info(f"Loaded {len(qs)} {lang} questions")
    return questions


async def run_evaluation(
    models: list,
    languages: list,
    limit: Optional[int] = None,
    output_tag: str = "",
) -> str:
    """Main evaluation loop. Returns path to results file."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))
    questions = load_questions(languages, limit)

    if not questions:
        raise ValueError("No questions loaded. Check dataset paths.")

    results = []
    total = len(questions) * len(models)
    done = 0

    for model_name in models:
        model_id = MODELS.get(model_name)
        if not model_id:
            logger.warning(f"Unknown model: {model_name}. Skipping.")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {model_name} ({model_id})")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"{'='*50}")

        for q in questions:
            result = await evaluate_question(client, model_name, model_id, q)
            results.append(result)
            done += 1
            avg = sum(r["scores"]["overall"] for r in results[-10:]) / min(10, len(results))
            logger.info(f"  Progress: {done}/{total} | Running avg: {avg:.3f}")
            await asyncio.sleep(0.3)   # Rate limit buffer

    # Save results
    tag = output_tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "run_id":       tag,
                    "models":       models,
                    "languages":    languages,
                    "total_questions": len(questions),
                    "timestamp":    datetime.utcnow().isoformat(),
                },
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )

    logger.info(f"\nResults saved to: {output_path}")
    return str(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BharatBench Evaluation Runner")
    parser.add_argument("--models",  nargs="+", default=["llama3-70b", "llama3-8b"],
                        choices=list(MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--langs",   nargs="+", default=LANGUAGES,
                        choices=LANGUAGES, help="Languages to test")
    parser.add_argument("--quick",   action="store_true",
                        help="Quick run: 5 questions per language")
    parser.add_argument("--all",     action="store_true",
                        help="Full run: all questions, all models")
    parser.add_argument("--tag",     type=str, default="",
                        help="Tag for output file name")
    args = parser.parse_args()

    if args.all:
        args.models = list(MODELS.keys())
        limit = None
    else:
        limit = 5 if args.quick else None

    asyncio.run(run_evaluation(
        models=args.models,
        languages=args.langs,
        limit=limit,
        output_tag=args.tag,
    ))
