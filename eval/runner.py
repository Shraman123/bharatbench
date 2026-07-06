"""
BharatBench Evaluation Runner
==============================
Runs all benchmark questions through multiple models (any mix of Groq,
Sarvam, and OpenAI-compatible providers -- see eval/config.py) and scores
each response using LLM-as-judge.

Usage:
    python eval/runner.py --models llama3-70b sarvam-105b --langs bengali hindi
    python eval/runner.py --all          # Run everything (takes ~30 min)
    python eval/runner.py --quick        # Run 5 questions per language (testing)
"""

import json
import re
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from config import MODELS, JUDGE, ModelSpec
from providers import get_provider

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LANGUAGES = ["bengali", "hindi", "english"]

JUDGE_MAX_TOKENS = 200  # headroom so the JSON score object isn't truncated

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
    spec: ModelSpec,
    question: str,
    language: str,
    timeout: int = 30,
) -> tuple[str, float, int]:
    """Call a subject model (via its configured provider) and return
    (response, latency_ms, tokens)."""

    system = f"""You are a helpful AI assistant.
Answer the question accurately.
If the question is in {language}, respond in {language}.
Be concise but complete."""

    provider = get_provider(spec.provider)
    result = await provider.complete(
        model_id=spec.model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        max_tokens=600,
        temperature=0.1,
        timeout=timeout,
    )
    return result.text, result.latency_ms, result.tokens


REQUIRED_SCORE_KEYS = {"correctness", "completeness", "language_quality", "clarity"}


def _degraded_scores(reason: str, raw: str) -> dict:
    """A judge record that could not be scored. Scores are None (not 0.5) so
    they can't silently blend into an aggregate mean — callers must explicitly
    handle judge_parse_failed records."""
    logger.warning(f"Judge output rejected ({reason}): {raw[:200]!r}")
    return {
        "correctness": None,
        "completeness": None,
        "language_quality": None,
        "clarity": None,
        "overall": None,
        "judge_parse_failed": True,
        "judge_raw_output": raw[:500],
    }


async def judge_response(
    question: str,
    reference: str,
    response: str,
    language: str,
    category: str,
) -> dict:
    """Score a model response using LLM-as-judge (provider/model from config.JUDGE)."""
    if response.startswith("[ERROR") or response.startswith("[TIMEOUT"):
        return {
            "correctness": 0.0,
            "completeness": 0.0,
            "language_quality": 0.0,
            "clarity": 0.0,
            "overall": 0.0,
            "judge_parse_failed": False,
        }

    prompt = JUDGE_PROMPT.format(
        question=question[:400],
        reference=reference[:400],
        response=response[:500],
        language=language,
        category=category,
    )

    provider = get_provider(JUDGE.provider)
    result = await provider.complete(
        model_id=JUDGE.model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0.0,
        timeout=30,
    )
    raw = result.text
    if raw.startswith("[ERROR") or raw.startswith("[TIMEOUT"):
        return _degraded_scores(f"judge call failed: {raw}", raw)

    match = re.search(r"\{[^}]+\}", raw)
    if not match:
        return _degraded_scores("no JSON object found (likely truncated)", raw)

    try:
        scores = json.loads(match.group())
    except json.JSONDecodeError as e:
        return _degraded_scores(f"invalid JSON ({e})", raw)

    if not REQUIRED_SCORE_KEYS.issubset(scores.keys()):
        missing = REQUIRED_SCORE_KEYS - scores.keys()
        return _degraded_scores(f"missing keys {missing}", raw)

    try:
        c  = max(0.0, min(1.0, round(float(scores["correctness"]),      3)))
        co = max(0.0, min(1.0, round(float(scores["completeness"]),     3)))
        lq = max(0.0, min(1.0, round(float(scores["language_quality"]), 3)))
        cl = max(0.0, min(1.0, round(float(scores["clarity"]),          3)))
    except (TypeError, ValueError) as e:
        return _degraded_scores(f"non-numeric score value ({e})", raw)

    return {
        "correctness":      c,
        "completeness":     co,
        "language_quality": lq,
        "clarity":          cl,
        "overall":          round((c + co + lq + cl) / 4, 3),
        "judge_parse_failed": False,
    }


async def evaluate_question(
    model_alias: str,
    spec: ModelSpec,
    q: dict,
) -> dict:
    """Run one question through one model and return a result record."""
    logger.info(f"  [{model_alias}] {q['id']}")

    # q["requires_tool"] is intentionally not read here -- it's informational
    # only. Every question is answered via plain chat completion regardless;
    # see README.md#known-limitations.
    response, latency_ms, tokens = await call_model(
        spec, q["question"], q["language"]
    )

    # Small delay to avoid rate limiting
    await asyncio.sleep(0.5)

    scores = await judge_response(
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
        "model":        model_alias,
        "provider":     spec.provider,
        "model_id":     spec.model_id,
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


def _warn_if_judge_overlaps_subjects(model_aliases: list) -> None:
    """Runtime safety net for the judge-is-also-a-subject problem: since the
    judge is now independently configurable (JUDGE_PROVIDER/JUDGE_MODEL_ID),
    this makes the overlap visible instead of silent whenever it does happen."""
    for alias in model_aliases:
        spec = MODELS.get(alias)
        if spec and spec.provider == JUDGE.provider and spec.model_id == JUDGE.model_id:
            logger.warning(
                f"Judge model ({JUDGE.provider}/{JUDGE.model_id}) is also being evaluated "
                f"as subject '{alias}' in this run -- self-grading bias risk. Set "
                f"JUDGE_PROVIDER/JUDGE_MODEL_ID env vars to an independent model to avoid "
                f"this. See README.md#known-limitations."
            )


async def run_evaluation(
    models: list,
    languages: list,
    limit: Optional[int] = None,
    output_tag: str = "",
    on_progress: Optional[callable] = None,
) -> str:
    """Main evaluation loop. Returns path to results file.

    on_progress(done, total), if given, is called after every question --
    used by service/jobs.py to report progress without polling logs."""
    questions = load_questions(languages, limit)

    if not questions:
        raise ValueError("No questions loaded. Check dataset paths.")

    _warn_if_judge_overlaps_subjects(models)

    results = []
    total = len(questions) * len(models)
    done = 0

    for model_alias in models:
        spec = MODELS.get(model_alias)
        if not spec:
            logger.warning(f"Unknown model: {model_alias}. Skipping.")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {model_alias} ({spec.provider}/{spec.model_id})")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"{'='*50}")

        for q in questions:
            result = await evaluate_question(model_alias, spec, q)
            results.append(result)
            done += 1
            recent = [r["scores"]["overall"] for r in results[-10:] if not r["scores"]["judge_parse_failed"]]
            avg_str = f"{sum(recent) / len(recent):.3f}" if recent else "N/A (all degraded)"
            logger.info(f"  Progress: {done}/{total} | Running avg: {avg_str}")
            if on_progress:
                on_progress(done, total)
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
                    "judge":        {"provider": JUDGE.provider, "model_id": JUDGE.model_id},
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
