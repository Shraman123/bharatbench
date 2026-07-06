import os
import time
import asyncio

from groq import (
    AsyncGroq,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .base import Provider, CompletionResult
from .retry import retry_async

RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class GroqProvider(Provider):
    def __init__(self, api_key: str | None = None):
        super().__init__("groq")
        self._client = AsyncGroq(api_key=api_key or os.getenv("GROQ_API_KEY", ""))

    async def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        timeout: float = 30.0,
    ) -> CompletionResult:
        start = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                retry_async(
                    self._client.chat.completions.create,
                    RETRYABLE,
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            text = resp.choices[0].message.content.strip()
            tokens = resp.usage.total_tokens if resp.usage else 0
            return CompletionResult(text, latency_ms, tokens)
        except asyncio.TimeoutError:
            return CompletionResult("[TIMEOUT]", timeout * 1000, 0)
        except Exception as e:
            return CompletionResult(f"[ERROR: {str(e)[:100]}]", 0, 0)
