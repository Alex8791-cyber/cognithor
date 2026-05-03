"""CAPTCHA Solver — orchestrator: detect, classify, solve, retry, learn."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from cognithor.browser.captcha.classifier import select_vision_model
from cognithor.browser.captcha.detector import detect_captcha
from cognithor.browser.captcha.models import CaptchaChallenge, CaptchaType, SolveResult
from cognithor.browser.captcha.strategies import get_strategy
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)

__all__ = ["CaptchaConfig", "CaptchaSolver"]


@dataclass
class CaptchaConfig:
    """Configuration for the CAPTCHA solver."""

    enabled: bool = False
    max_retries: int = 3
    stealth_enabled: bool = True
    auto_solve: bool = True
    preferred_simple_model: str = "openbmb/minicpm-v4.5:latest"
    preferred_complex_model: str = "qwen3-vl:32b"
    solve_timeout_seconds: int = 30


class CaptchaSolver:
    """Orchestrates CAPTCHA detection and solving."""

    def __init__(
        self,
        vision_fn: Callable | None = None,  # type: ignore[type-arg]
        config: CaptchaConfig | None = None,
        tactical_memory: Any = None,
    ) -> None:
        self._vision_fn = vision_fn
        self._config = config or CaptchaConfig()
        self._tactical = tactical_memory

    async def solve(
        self,
        page: Any,
        challenge: CaptchaChallenge | None = None,
    ) -> SolveResult:
        """Detect and solve a CAPTCHA on the given page.

        If no challenge is provided, runs detection first.
        Retries up to config.max_retries times.
        Records result to TacticalMemory if available.
        """
        t0 = time.monotonic()

        # Step 1: Detect if no challenge provided
        if challenge is None:
            challenges = await detect_captcha(page)
            if not challenges:
                return SolveResult(
                    success=True,
                    captcha_type=CaptchaType.UNKNOWN,
                    model_used="",
                    attempts=0,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    answer="No CAPTCHA detected",
                )
            challenge = challenges[0]

        # Step 2: Select strategy
        strategy = get_strategy(challenge.captcha_type)
        if strategy is None:
            return SolveResult(
                success=False,
                captcha_type=challenge.captcha_type,
                model_used="",
                attempts=0,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error=f"No strategy for {challenge.captcha_type}",
            )

        model = select_vision_model(challenge.captcha_type)

        # Step 3: Retry loop
        last_result: SolveResult | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                result = await strategy(page, challenge, self._vision_fn)
                result.attempts = attempt
                result.model_used = model or result.model_used
                result.duration_ms = int((time.monotonic() - t0) * 1000)

                if result.success:
                    log.info(
                        "captcha_solved",
                        type=challenge.captcha_type.value,
                        model=model,
                        attempts=attempt,
                        duration_ms=result.duration_ms,
                    )
                    self._record_tactical(result, challenge)
                    return cast("SolveResult", result)

                last_result = result
                log.debug(
                    "captcha_solve_attempt_failed",
                    type=challenge.captcha_type.value,
                    attempt=attempt,
                    error=result.error[:80],
                )
            except Exception as exc:
                last_result = SolveResult(
                    success=False,
                    captcha_type=challenge.captcha_type,
                    model_used=model,
                    attempts=attempt,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    error=str(exc)[:200],
                )
                log.debug("captcha_solve_exception", attempt=attempt, error=str(exc)[:80])

        # All retries exhausted
        final = last_result or SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used=model,
            attempts=self._config.max_retries,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error="Max retries exhausted",
        )
        log.warning(
            "captcha_solve_failed",
            type=challenge.captcha_type.value,
            attempts=self._config.max_retries,
        )
        self._record_tactical(final, challenge)
        return final

    def _record_tactical(self, result: SolveResult, challenge: CaptchaChallenge) -> None:
        """Record solve result to TacticalMemory for learning."""
        if not self._tactical:
            return
        try:
            self._tactical.record_outcome(
                tool_name="browser_solve_captcha",
                params={
                    "type": challenge.captcha_type.value,
                    "model": result.model_used,
                    "domain": challenge.page_url[:50],
                },
                success=result.success,
                duration_ms=result.duration_ms,
            )
        except Exception:
            log.debug("captcha_tactical_record_failed", exc_info=True)
