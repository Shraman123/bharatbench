"""
Provider abstraction so subject models and the judge model aren't hardcoded
to one vendor SDK. A Provider knows how to run one chat completion; callers
(runner.py) assemble messages and read back a CompletionResult.

On failure, complete() returns a CompletionResult whose text is the sentinel
"[ERROR: ...]" or "[TIMEOUT]" rather than raising -- this matches the
pre-existing behavior in eval/runner.py, where judge_response() keys off
those prefixes to short-circuit scoring. Every provider implementation must
preserve that contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    text: str
    latency_ms: float
    tokens: int


class Provider(ABC):
    """A model provider capable of running chat completions for both subject
    models and the judge -- the two are not distinguished at this layer."""

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        timeout: float = 30.0,
    ) -> CompletionResult:
        """Run one chat completion. Must not raise -- failures come back as
        a CompletionResult with an "[ERROR: ...]" / "[TIMEOUT]" text sentinel."""
        raise NotImplementedError
