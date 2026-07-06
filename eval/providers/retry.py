"""Shared exponential-backoff retry helper used by every provider."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_async(coro_fn, retryable_errors: tuple, *args, retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Call an async function, retrying transient errors with exponential backoff."""
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except retryable_errors as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Transient API error ({type(e).__name__}): {e}. "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{retries})"
            )
            await asyncio.sleep(delay)
