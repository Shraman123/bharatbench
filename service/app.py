"""
FastAPI wrapper around the BharatBench evaluation harness.

Endpoints:
    GET  /health                       liveness check
    POST /eval/runs                    submit a new evaluation run (async)
    GET  /eval/runs                    list all runs (in-memory, this process only)
    GET  /eval/runs/{run_id}           status + progress
    GET  /eval/runs/{run_id}/results   raw per-question results (once completed)
    GET  /eval/runs/{run_id}/report    aggregated report, incl. language-gap caveat

Run locally: `uvicorn service.app:app --reload` from the repo root.
See README.md's "FastAPI Evaluation Service" section for details and the
in-memory-job-store caveat.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import _eval_path  # noqa: F401  (must run before importing analyze)
from .jobs import job_manager
from .models import (
    LANGUAGE_GAP_CAVEAT,
    EvalRunAccepted,
    EvalRunRequest,
    EvalRunStatusResponse,
    ReportResponse,
    RunStatus,
)

import analyze  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BharatBench Evaluation Service",
    description="Submit, track, and fetch results for Indic-language LLM evaluation runs.",
    version="0.1.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"error": str(exc)})


def _status_response(job) -> EvalRunStatusResponse:
    return EvalRunStatusResponse(
        run_id=job.run_id,
        status=job.status,
        done=job.done,
        total=job.total,
        models=job.models,
        languages=job.languages,
        error=job.error,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/eval/runs", response_model=EvalRunAccepted, status_code=202)
async def submit_run(req: EvalRunRequest) -> EvalRunAccepted:
    # Must be async: asyncio.create_task() (inside job_manager.create) needs a
    # running event loop, but FastAPI runs sync `def` endpoints in a worker
    # thread pool with no loop attached.
    job = job_manager.create(req.models, req.languages, req.limit)
    return EvalRunAccepted(run_id=job.run_id, status=job.status)


@app.get("/eval/runs", response_model=list[EvalRunStatusResponse])
def list_runs() -> list:
    return [_status_response(job) for job in job_manager.list()]


@app.get("/eval/runs/{run_id}", response_model=EvalRunStatusResponse)
def get_run_status(run_id: str) -> EvalRunStatusResponse:
    job = job_manager.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id!r}")
    return _status_response(job)


@app.get("/eval/runs/{run_id}/results")
def get_run_results(run_id: str) -> dict:
    job = job_manager.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id!r}")
    if job.status in (RunStatus.queued, RunStatus.running):
        raise HTTPException(status_code=409, detail=f"Run {run_id!r} is not finished yet (status={job.status.value})")
    if job.status == RunStatus.failed:
        raise HTTPException(status_code=500, detail=f"Run {run_id!r} failed: {job.error}")
    return job_manager.load_results(run_id)


@app.get("/eval/runs/{run_id}/report", response_model=ReportResponse)
def get_run_report(run_id: str) -> ReportResponse:
    job = job_manager.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id!r}")
    if job.status in (RunStatus.queued, RunStatus.running):
        raise HTTPException(status_code=409, detail=f"Run {run_id!r} is not finished yet (status={job.status.value})")
    if job.status == RunStatus.failed:
        raise HTTPException(status_code=500, detail=f"Run {run_id!r} failed: {job.error}")

    data = job_manager.load_results(run_id)
    results = data["results"]
    good = analyze.usable(results)
    bad = analyze.degraded(results)

    return ReportResponse(
        run_id=run_id,
        language_gap_caveat=LANGUAGE_GAP_CAVEAT,
        usable_evaluations=len(good),
        degraded_evaluations=len(bad),
        summary={
            "by_model": {str(k): v for k, v in analyze.aggregate(good, ["model"]).items()},
            "by_language": {str(k): v for k, v in analyze.aggregate(good, ["language"]).items()},
            "by_category": {str(k): v for k, v in analyze.aggregate(good, ["category"]).items()},
        },
        language_gaps=analyze.compute_language_gap(good),
    )
