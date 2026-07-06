"""
Model + judge registry. Subjects and the judge each resolve to a
(provider, model_id) pair -- neither is hardcoded to one vendor SDK, and the
judge can be pointed at a different provider/model than every subject via
JUDGE_PROVIDER / JUDGE_MODEL_ID env vars. See README.md for how to configure
an independent judge, and eval/runner.py's _warn_if_judge_overlaps_subjects
for the runtime safety net if you don't.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "groq" | "openai" | "sarvam"
    model_id: str


# NOTE: gemma2-9b-it and mixtral-8x7b-32768 were deprecated by Groq on
# 2025-10-08 and 2025-03-20 respectively and are no longer callable.
# llama-3.3-70b-versatile and llama-3.1-8b-instant are active today but Groq
# has announced their deprecation for 2026-08-16 on free/developer tiers --
# revisit before that date.
MODELS: dict[str, ModelSpec] = {
    "llama3-70b":   ModelSpec("groq", "llama-3.3-70b-versatile"),
    "llama3-8b":    ModelSpec("groq", "llama-3.1-8b-instant"),
    "gpt-oss-20b":  ModelSpec("groq", "openai/gpt-oss-20b"),
    "gpt-oss-120b": ModelSpec("groq", "openai/gpt-oss-120b"),
    "sarvam-105b":  ModelSpec("sarvam", "sarvam-105b"),
    "sarvam-30b":   ModelSpec("sarvam", "sarvam-30b"),
    "sarvam-m":     ModelSpec("sarvam", "sarvam-m"),
    "gpt-4o-mini":  ModelSpec("openai", "gpt-4o-mini"),
}

# Judge defaults to the same provider/model as before this refactor
# (groq/llama-3.1-8b-instant) to preserve existing behavior -- that model is
# also subject alias "llama3-8b", the known self-grading overlap flagged in
# README.md#known-limitations. Override both env vars together to point the
# judge at an independent model, e.g.:
#   JUDGE_PROVIDER=openai JUDGE_MODEL_ID=gpt-4o-mini
JUDGE = ModelSpec(
    provider=os.getenv("JUDGE_PROVIDER", "groq"),
    model_id=os.getenv("JUDGE_MODEL_ID", "llama-3.1-8b-instant"),
)
