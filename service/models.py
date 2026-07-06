from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator

from . import _eval_path  # noqa: F401  (must run before importing config)
from config import MODELS  # noqa: E402


VALID_LANGUAGES = {"bengali", "hindi", "english"}

LANGUAGE_GAP_CAVEAT = (
    "The Bengali/Hindi/English question sets are NOT parallel/translated -- "
    "they cover different topics per language. Any cross-language gap "
    "reported here reflects topic/question-difficulty differences as well "
    "as language capability, not a controlled comparison. "
    "See README.md#known-limitations."
)


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class EvalRunRequest(BaseModel):
    models: list[str]
    languages: list[str]
    limit: Optional[int] = None

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("models must be a non-empty list")
        unknown = [m for m in v if m not in MODELS]
        if unknown:
            raise ValueError(f"Unknown model alias(es): {unknown}. Valid: {sorted(MODELS)}")
        return v

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("languages must be a non-empty list")
        unknown = [lang for lang in v if lang not in VALID_LANGUAGES]
        if unknown:
            raise ValueError(f"Unknown language(s): {unknown}. Valid: {sorted(VALID_LANGUAGES)}")
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("limit must be a positive integer if given")
        return v


class EvalRunAccepted(BaseModel):
    run_id: str
    status: RunStatus


class EvalRunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    done: int
    total: int
    models: list[str]
    languages: list[str]
    error: Optional[str] = None


class ReportResponse(BaseModel):
    run_id: str
    language_gap_caveat: str
    usable_evaluations: int
    degraded_evaluations: int
    summary: dict
    language_gaps: dict
    pairwise_model_comparisons: list
