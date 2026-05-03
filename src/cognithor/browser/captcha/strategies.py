"""CAPTCHA solve strategies — one async function per captcha type."""

from __future__ import annotations

import base64
import re
import time
from typing import TYPE_CHECKING, Any, cast

from cognithor.browser.captcha.models import CaptchaChallenge, CaptchaType, SolveResult
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "build_image_grid_prompt",
    "build_text_captcha_prompt",
    "get_strategy",
    "parse_grid_coordinates",
    "parse_text_answer",
]

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_text_captcha_prompt() -> str:
    return (
        "Lies den verzerrten Text in diesem CAPTCHA-Bild. "
        "Antworte NUR mit dem Text, keine Erklaerung, keine Anfuehrungszeichen."
    )


def build_image_grid_prompt(challenge_text: str) -> str:
    return (
        f"Dieses Bild zeigt ein CAPTCHA-Grid. Die Aufgabe ist: '{challenge_text}'. "
        "Antworte mit den Positionen der richtigen Bilder als Liste: [[row,col]] "
        "mit 0-basiertem Index. Beispiel: [[0,1],[1,2],[2,0]]. "
        "NUR die Liste, keine Erklaerung."
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_EXPLANATION_PREFIXES = re.compile(
    r"^(?:The text reads:|Der Text lautet:|The text is:|Der Text ist:)\s*",
    re.IGNORECASE,
)

_COORD_RE = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")


def parse_text_answer(raw: str) -> str:
    """Extract the actual CAPTCHA text from a potentially verbose LLM reply."""
    text = raw.strip()
    # Strip surrounding quotes
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1].strip()
    # Strip known explanation prefixes
    text = _EXPLANATION_PREFIXES.sub("", text).strip()
    # If there are still multiple words and it looks like an explanation,
    # take the last word (likely the actual captcha text).
    if " " in text and len(text.split()) > 3:
        text = text.split()[-1]
    return text


def parse_grid_coordinates(raw: str) -> list[tuple[int, int]]:
    """Parse ``[[row,col], ...]`` coordinate pairs from *raw* LLM output."""
    matches = _COORD_RE.findall(raw)
    return [(int(r), int(c)) for r, c in matches]


# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------


async def text_strategy(
    page: Any,
    challenge: CaptchaChallenge,
    vision_fn: Callable[..., Any],
) -> SolveResult:
    """Solve a simple text-based CAPTCHA via OCR."""
    t0 = time.monotonic()
    try:
        el = await page.query_selector(challenge.selector)
        if not el:
            return SolveResult(
                success=False,
                captcha_type=challenge.captcha_type,
                model_used="",
                attempts=1,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error="CAPTCHA element not found",
            )
        screenshot = await el.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode()
        raw_answer = await vision_fn(screenshot_b64, build_text_captcha_prompt())
        answer = parse_text_answer(raw_answer)

        # Find the nearby text input
        input_el = await page.query_selector(
            f"{challenge.selector} ~ input, {challenge.selector} + input"
        )
        if not input_el:
            input_el = await page.query_selector('input[type="text"]')
        if input_el:
            await input_el.fill(answer)

        return SolveResult(
            success=True,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            answer=answer,
        )
    except Exception as exc:
        _log.warning("text_strategy failed: %s", exc)
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )


async def checkbox_strategy(
    page: Any,
    challenge: CaptchaChallenge,
    vision_fn: Callable[..., Any],
) -> SolveResult:
    """Click the reCAPTCHA v2 checkbox and hope stealth is enough."""
    t0 = time.monotonic()
    try:
        try:
            await page.click(challenge.selector + " iframe")
        except Exception:
            await (
                page.frame_locator(challenge.selector + " iframe")
                .locator(".recaptcha-checkbox-border")
                .click()
            )
        await page.wait_for_timeout(3000)

        # Check if an image challenge appeared (not solved yet)
        image_challenge = await page.query_selector('iframe[src*="recaptcha/api2/bframe"]')
        solved = image_challenge is None

        return SolveResult(
            success=solved,
            captcha_type=challenge.captcha_type,
            model_used="",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            answer="checkbox_clicked",
            error="" if solved else "image_challenge_appeared",
        )
    except Exception as exc:
        _log.warning("checkbox_strategy failed: %s", exc)
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )


async def image_grid_strategy(
    page: Any,
    challenge: CaptchaChallenge,
    vision_fn: Callable[..., Any],
) -> SolveResult:
    """Solve an image-grid CAPTCHA (reCAPTCHA v2 image / hCaptcha)."""
    t0 = time.monotonic()
    try:
        # Screenshot the challenge iframe area
        iframe_el = await page.query_selector(
            'iframe[src*="recaptcha/api2/bframe"], iframe[src*="hcaptcha.com/captcha"]'
        )
        if not iframe_el:
            iframe_el = await page.query_selector(challenge.selector)
        if not iframe_el:
            return SolveResult(
                success=False,
                captcha_type=challenge.captcha_type,
                model_used="",
                attempts=1,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error="Challenge iframe not found",
            )

        screenshot = await iframe_el.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode()

        # Extract challenge text
        challenge_text = challenge.challenge_text or "Select the matching images"

        raw = await vision_fn(screenshot_b64, build_image_grid_prompt(challenge_text))
        coords = parse_grid_coordinates(raw)

        if not coords:
            return SolveResult(
                success=False,
                captcha_type=challenge.captcha_type,
                model_used="vision",
                attempts=1,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error="No grid coordinates parsed",
            )

        # Click each identified cell (approximate click positions)
        box = await iframe_el.bounding_box()
        if box:
            cell_w = box["width"] / 3
            cell_h = box["height"] / 3
            for row, col in coords:
                x = box["x"] + col * cell_w + cell_w / 2
                y = box["y"] + row * cell_h + cell_h / 2
                await page.mouse.click(x, y)

        return SolveResult(
            success=True,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            answer=str(coords),
        )
    except Exception as exc:
        _log.warning("image_grid_strategy failed: %s", exc)
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )


async def stealth_strategy(
    page: Any,
    challenge: CaptchaChallenge,
    vision_fn: Callable[..., Any],
) -> SolveResult:
    """Wait for Turnstile / reCAPTCHA v3 to auto-solve via stealth."""
    t0 = time.monotonic()
    try:
        await page.wait_for_timeout(5000)

        # Check for a token in hidden inputs
        token_el = await page.query_selector(
            'input[name="cf-turnstile-response"], '
            'input[name="g-recaptcha-response"], '
            'textarea[name="g-recaptcha-response"]'
        )
        token_value = ""
        if token_el:
            token_value = await token_el.get_attribute("value") or ""
            if not token_value:
                token_value = await token_el.inner_text() if hasattr(token_el, "inner_text") else ""

        solved = bool(token_value)

        return SolveResult(
            success=solved,
            captcha_type=challenge.captcha_type,
            model_used="",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            answer=token_value[:32] + "..." if len(token_value) > 32 else token_value,
            error="" if solved else "no_token_found",
        )
    except Exception as exc:
        _log.warning("stealth_strategy failed: %s", exc)
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )


async def generic_strategy(
    page: Any,
    challenge: CaptchaChallenge,
    vision_fn: Callable[..., Any],
) -> SolveResult:
    """Fallback: screenshot and ask the vision model what to do."""
    t0 = time.monotonic()
    try:
        screenshot = await page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode()
        description = await vision_fn(
            screenshot_b64,
            "Describe the CAPTCHA on this page and how to solve it. Be concise.",
        )
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            answer=description,
            error="generic_strategy_needs_human",
        )
    except Exception as exc:
        _log.warning("generic_strategy failed: %s", exc)
        return SolveResult(
            success=False,
            captcha_type=challenge.captcha_type,
            model_used="vision",
            attempts=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[CaptchaType, Callable[..., Any]] = {
    CaptchaType.TEXT: text_strategy,
    CaptchaType.RECAPTCHA_V2_CHECKBOX: checkbox_strategy,
    CaptchaType.RECAPTCHA_V2_IMAGE: image_grid_strategy,
    CaptchaType.RECAPTCHA_V3: stealth_strategy,
    CaptchaType.HCAPTCHA: image_grid_strategy,
    CaptchaType.TURNSTILE: stealth_strategy,
    CaptchaType.FUNCAPTCHA: image_grid_strategy,
    CaptchaType.UNKNOWN: generic_strategy,
}


def get_strategy(captcha_type: CaptchaType) -> Callable | None:
    """Return the async strategy function for *captcha_type*."""
    return cast("Callable[..., Any] | None", _STRATEGY_MAP.get(captcha_type, generic_strategy))
