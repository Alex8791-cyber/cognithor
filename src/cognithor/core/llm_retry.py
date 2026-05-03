"""LLM call retry logic with stream-to-sync fallback.

Provides exponential-backoff retry for transient LLM errors (network
timeouts, 429/5xx status codes) and an optional fallback from streaming
to synchronous mode when streaming fails repeatedly.

Usage (future integration)::

    from cognithor.core.llm_retry import retry_llm_call

    result = await retry_llm_call(
        lambda: backend.chat(messages, model),
        stream_fallback_fn=lambda: backend.chat(messages, model, stream=False),
    )
"""

from __future__ import annotations

import asyncio
import re

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSIENT_ERROR_RE = re.compile(
    r"fetch failed|network|socket|timeout|timed out"
    r"|ECONNRESET|ECONNREFUSED|EAI_AGAIN",
    re.IGNORECASE,
)

MAX_RETRIES = 3
BASE_DELAY_MS = 200
MAX_DELAY_MS = 2000

# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def is_retryable_error(error: Exception) -> bool:
    """Return *True* if *error* looks like a transient LLM/network failure."""
    msg = str(error)
    if hasattr(error, "status_code"):
        status = error.status_code
        if status is not None and (status == 429 or 500 <= status <= 504):
            return True
    return bool(TRANSIENT_ERROR_RE.search(msg))


def should_fallback_stream_to_sync(error: Exception) -> bool:
    """Return *True* if the error warrants a stream-to-sync fallback.

    Rate-limit errors (429 / "rate") are excluded because switching to
    sync mode would not help — the server is throttling the caller.
    """
    msg = str(error)
    if "429" in msg or "rate" in msg.lower():
        return False
    if hasattr(error, "status_code") and error.status_code == 429:
        return False
    return is_retryable_error(error)


# ---------------------------------------------------------------------------
# Core retry wrapper
# ---------------------------------------------------------------------------


async def retry_llm_call(
    call_fn,
    *,
    max_retries: int = MAX_RETRIES,
    stream_fallback_fn=None,
):
    """Retry an async LLM call with exponential backoff.

    Args:
        call_fn: Async callable that performs the LLM request.
        max_retries: Maximum number of attempts (including the first).
        stream_fallback_fn: Optional async callable used as a last-resort
            non-streaming fallback when all retries are exhausted and the
            error is eligible for stream-to-sync fallback.

    Returns:
        The result of *call_fn* (or *stream_fallback_fn*).

    Raises:
        The last caught exception when all retries and fallback fail.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await call_fn()
        except Exception as exc:
            last_error = exc
            if not is_retryable_error(exc):
                raise

            if attempt < max_retries:
                delay = (
                    min(
                        BASE_DELAY_MS * (2 ** (attempt - 1)),
                        MAX_DELAY_MS,
                    )
                    / 1000
                )
                log.warning(
                    "llm_retry",
                    attempt=attempt,
                    delay_s=delay,
                    error=str(exc)[:200],
                )
                await asyncio.sleep(delay)
            elif stream_fallback_fn and should_fallback_stream_to_sync(exc):
                log.info("llm_stream_fallback_to_sync")
                try:
                    return await stream_fallback_fn()
                except Exception as fb_exc:
                    last_error = fb_exc

    # All retries exhausted.
    raise last_error  # type: ignore[misc]
