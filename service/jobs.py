"""
In-memory async job manager for evaluation runs.

This is a single-process, in-memory store: jobs are lost on restart and not
shared across worker processes. That's fine for one instance (e.g. `uvicorn
service.app:app`), but a real multi-worker production deployment would want
a shared job store (DB/Redis) and a task queue (Celery/RQ/arq) instead of
asyncio.create_task -- see README's FastAPI service section.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import _eval_path  # noqa: F401  (must run before importing runner)
from .models import RunStatus

import runner  # noqa: E402


@dataclass
class JobRecord:
    run_id: str
    models: list
    languages: list
    status: RunStatus = RunStatus.queued
    done: int = 0
    total: int = 0
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobManager:
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, models: list, languages: list, limit: Optional[int]) -> JobRecord:
        run_id = uuid.uuid4().hex[:12]
        job = JobRecord(run_id=run_id, models=models, languages=languages)
        self._jobs[run_id] = job
        self._tasks[run_id] = asyncio.create_task(self._run(job, models, languages, limit))
        return job

    async def _run(self, job: JobRecord, models: list, languages: list, limit: Optional[int]) -> None:
        job.status = RunStatus.running

        def on_progress(done: int, total: int) -> None:
            job.done = done
            job.total = total

        try:
            path = await runner.run_evaluation(
                models=models,
                languages=languages,
                limit=limit,
                output_tag=job.run_id,
                on_progress=on_progress,
            )
            job.result_path = path
            job.status = RunStatus.completed
        except Exception as e:
            job.error = str(e)
            job.status = RunStatus.failed

    def get(self, run_id: str) -> Optional[JobRecord]:
        return self._jobs.get(run_id)

    def list(self) -> list:
        return list(self._jobs.values())

    def load_results(self, run_id: str) -> dict:
        job = self._jobs[run_id]
        with open(job.result_path, encoding="utf-8") as f:
            return json.load(f)


job_manager = JobManager()
