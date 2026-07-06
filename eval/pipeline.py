"""
BharatBench Evaluation Pipeline (LangGraph)
=============================================
Reimplements the same eval flow as runner.py -- generate a subject response,
score it with the judge, then aggregate -- as an explicit, inspectable
LangGraph graph, with an added verify stage:

    generate -> judge -> verify -> aggregate

This does not change what a score means: generate/judge call the exact same
eval.runner.call_model()/judge_response() functions runner.py uses, so
scoring semantics are identical whether you run runner.py or this pipeline.
What changes is structure -- each stage is a named, independently-inspectable
node with typed state passed between them, instead of one function doing
everything inline.

verify is new: it does not re-score anything. It's a consistency/invariant
check over what generate+judge already produced (e.g. "a failed model call
must score 0.0, not something else", "a judge_parse_failed record must have
None scores, not partial ones"), flagging anomalies for visibility rather
than silently trusting upstream output.

Usage:
    python eval/pipeline.py --models llama3-70b llama3-8b --langs bengali hindi
    python eval/pipeline.py --quick
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import analyze
from config import MODELS
from runner import (
    LANGUAGES,
    RESULTS_DIR,
    _warn_if_judge_overlaps_subjects,
    call_model,
    judge_response,
    load_questions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Mirrors service/models.py's LANGUAGE_GAP_CAVEAT -- duplicated rather than
# imported so this module has no dependency on service/, and vice versa.
LANGUAGE_GAP_CAVEAT = (
    "The Bengali/Hindi/English question sets are NOT parallel/translated -- "
    "they cover different topics per language. Any cross-language gap "
    "reported here reflects topic/question-difficulty differences as well "
    "as language capability, not a controlled comparison. "
    "See README.md#known-limitations."
)


class PipelineState(TypedDict):
    questions: list
    model_aliases: list
    generated: list
    judged: list
    verified: list
    verification_issues: list
    aggregate: dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def generate_node(state: PipelineState) -> dict:
    """Run every (model, question) pair through the subject model. No scoring yet."""
    generated = []
    for alias in state["model_aliases"]:
        spec = MODELS.get(alias)
        if not spec:
            logger.warning(f"Unknown model: {alias}. Skipping.")
            continue

        logger.info(f"[generate] {alias} ({spec.provider}/{spec.model_id})")
        for q in state["questions"]:
            response, latency_ms, tokens = await call_model(spec, q["question"], q["language"])
            generated.append({
                "question_id":  q["id"],
                "language":     q["language"],
                "category":     q["category"],
                "difficulty":   q["difficulty"],
                "model":        alias,
                "provider":     spec.provider,
                "model_id":     spec.model_id,
                "question":     q["question"],
                "reference":    q["reference"],
                "response":     response,
                "latency_ms":   latency_ms,
                "token_count":  tokens,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            })
            await asyncio.sleep(0.3)  # rate-limit buffer, same as runner.py

    return {"generated": generated}


async def judge_node(state: PipelineState) -> dict:
    """Score every generated response with the configured judge."""
    judged = []
    for record in state["generated"]:
        scores = await judge_response(
            record["question"], record["reference"], record["response"],
            record["language"], record["category"],
        )
        judged.append({**record, "scores": scores})
        await asyncio.sleep(0.3)

    return {"judged": judged}


def _check_consistency(record: dict) -> list:
    """Invariants that must hold between a record's response and its scores.
    Returns a list of human-readable issue descriptions (empty if none)."""
    issues = []
    scores = record["scores"]
    failed_call = record["response"].startswith("[ERROR") or record["response"].startswith("[TIMEOUT")

    if failed_call and scores.get("overall") != 0.0:
        issues.append("model call failed but scores.overall is not 0.0")

    if scores.get("judge_parse_failed"):
        if scores.get("overall") is not None:
            issues.append("judge_parse_failed=True but overall is not None")
    else:
        overall = scores.get("overall")
        if overall is not None and not (0.0 <= overall <= 1.0):
            issues.append(f"overall score {overall} out of [0,1] bounds")

    return issues


async def verify_node(state: PipelineState) -> dict:
    """Consistency-check generate+judge output. Does not alter any score --
    only flags anomalies so they're visible rather than silently aggregated."""
    verified = []
    all_issues = []
    for record in state["judged"]:
        record_issues = _check_consistency(record)
        verified.append({
            **record,
            "verification_ok": len(record_issues) == 0,
            "verification_issues": record_issues,
        })
        all_issues.extend(f"{record['question_id']}/{record['model']}: {i}" for i in record_issues)

    if all_issues:
        logger.warning(f"[verify] {len(all_issues)} consistency issue(s) found: {all_issues}")
    else:
        logger.info(f"[verify] {len(verified)} record(s), no consistency issues")

    return {"verified": verified, "verification_issues": all_issues}


async def aggregate_node(state: PipelineState) -> dict:
    """Summarize the run -- same aggregation functions analyze.py uses on
    runner.py's output, so both paths are comparable."""
    records = state["verified"]
    good = analyze.usable(records)
    bad = analyze.degraded(records)

    summary = {
        "total_evaluations": len(records),
        "usable_evaluations": len(good),
        "degraded_evaluations": len(bad),
        "verification_issue_count": len(state["verification_issues"]),
        "by_model": {str(k): v for k, v in analyze.aggregate(good, ["model"]).items()},
        "by_language": {str(k): v for k, v in analyze.aggregate(good, ["language"]).items()},
        "by_category": {str(k): v for k, v in analyze.aggregate(good, ["category"]).items()},
        "language_gaps": analyze.compute_language_gap(good),
        "language_gap_caveat": LANGUAGE_GAP_CAVEAT,
    }
    logger.info(f"[aggregate] {summary['usable_evaluations']} usable, "
                f"{summary['degraded_evaluations']} degraded")
    return {"aggregate": summary}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate", generate_node)
    graph.add_node("judge", judge_node)
    graph.add_node("verify", verify_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "judge")
    graph.add_edge("judge", "verify")
    graph.add_edge("verify", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


async def run_pipeline(
    models: list,
    languages: list,
    limit: Optional[int] = None,
    output_tag: str = "",
) -> dict:
    """Run the full generate->judge->verify->aggregate graph and persist
    results in the same JSON shape runner.py produces, so eval/analyze.py and
    the FastAPI service can consume either path's output interchangeably."""
    questions = load_questions(languages, limit)
    if not questions:
        raise ValueError("No questions loaded. Check dataset paths.")

    _warn_if_judge_overlaps_subjects(models)

    graph = build_graph()
    final_state = await graph.ainvoke({
        "questions": questions,
        "model_aliases": models,
        "generated": [],
        "judged": [],
        "verified": [],
        "verification_issues": [],
        "aggregate": {},
    })

    tag = output_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"pipeline_{tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "run_id": tag,
                    "models": models,
                    "languages": languages,
                    "pipeline": "langgraph:generate->judge->verify->aggregate",
                    "total_questions": len(questions),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "results": final_state["verified"],
                "aggregate": final_state["aggregate"],
            },
            f, ensure_ascii=False, indent=2,
        )

    logger.info(f"Results saved to: {output_path}")
    return {**final_state, "output_path": str(output_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BharatBench LangGraph Evaluation Pipeline")
    parser.add_argument("--models", nargs="+", default=["llama3-70b", "llama3-8b"],
                        choices=list(MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--langs", nargs="+", default=LANGUAGES,
                        choices=LANGUAGES, help="Languages to test")
    parser.add_argument("--quick", action="store_true", help="Quick run: 5 questions per language")
    parser.add_argument("--all", action="store_true", help="Full run: all questions, all models")
    parser.add_argument("--tag", type=str, default="", help="Tag for output file name")
    args = parser.parse_args()

    if args.all:
        args.models = list(MODELS.keys())
        run_limit = None
    else:
        run_limit = 5 if args.quick else None

    state = asyncio.run(run_pipeline(
        models=args.models,
        languages=args.langs,
        limit=run_limit,
        output_tag=args.tag,
    ))

    print(json.dumps(state["aggregate"], indent=2, ensure_ascii=False))
