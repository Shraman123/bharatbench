"""
Sarvam AI's official SDK (`sarvamai`). Verified against the installed
sarvamai==0.1.28 package and docs.sarvam.ai: the chat completions response is
OpenAI-shaped (choices[0].message.content, usage.total_tokens), but the call
itself is `client.chat.completions(...)` -- directly callable, not
`.create(...)` like Groq/OpenAI.
"""

import os
import time
import asyncio

from sarvamai import AsyncSarvamAI
from sarvamai import TooManyRequestsError, ServiceUnavailableError, InternalServerError

from .base import Provider, CompletionResult
from .retry import retry_async

RETRYABLE = (TooManyRequestsError, ServiceUnavailableError, InternalServerError)


class SarvamProvider(Provider):
    def __init__(self, api_key: str | None = None):
        super().__init__("sarvam")
        self._client = AsyncSarvamAI(
            api_subscription_key=api_key or os.getenv("SARVAM_API_KEY", "")
        )

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
                    self._client.chat.completions,
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
