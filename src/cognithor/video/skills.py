"""Sprint-27 HF-5 — video-composition skills layer.

Domain-aware preset functions that wrap the raw
``compose_html`` / ``video_compose`` surface (HF-3) with sensible
defaults for common video shapes. Each skill is:

* Pure: turns a high-level intent (title + bullet points, hook +
  beats + outro, base composition + caption track) into a
  structured spec the existing ``compose_html`` accepts.
* Stateless: no filesystem, no subprocess. The caller pipes the
  spec into ``video_compose`` (GREEN-risk MCP tool) to get HTML
  back, then optionally hands the HTML to ``video_render``.
* Defensively-typed: malformed inputs (wrong types, empty strings,
  oversize lists) are clamped or dropped silently — the skill
  should never raise.

Three skills land in HF-5:

* :func:`compose_explainer` — title scene + N body scenes + CTA
  scene. 16:9 by default. Captions auto-laid-out.
* :func:`compose_social_cut` — vertical 9:16 short with hook, fast
  beats, outro. Defaults to 15s total / 1080x1920.
* :func:`caption_overlay` — given a base spec + a parallel caption
  list, returns a new spec with the captions glued onto each
  scene by track-index.

All three are registered as MCP tools in :mod:`cognithor.mcp.video_tools`
under GREEN risk-level. Apache-2.0, no domain coupling.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_EXPLAINER_FPS",
    "DEFAULT_EXPLAINER_HEIGHT",
    "DEFAULT_EXPLAINER_WIDTH",
    "DEFAULT_SOCIAL_HEIGHT",
    "DEFAULT_SOCIAL_WIDTH",
    "MAX_BEATS",
    "MAX_SECTIONS",
    "caption_overlay",
    "compose_explainer",
    "compose_social_cut",
]


DEFAULT_EXPLAINER_WIDTH = 1920
DEFAULT_EXPLAINER_HEIGHT = 1080
DEFAULT_EXPLAINER_FPS = 30

DEFAULT_SOCIAL_WIDTH = 1080
DEFAULT_SOCIAL_HEIGHT = 1920

MAX_SECTIONS = 12  # explainer body cap — anything more belongs in chapters
MAX_BEATS = 8  # social-cut beat cap — TikTok / Shorts work best with ≤8 cuts

_DEFAULT_TITLE_SECONDS = 3.0
_DEFAULT_SECTION_SECONDS = 5.0
_DEFAULT_CTA_SECONDS = 3.0
_DEFAULT_HOOK_SECONDS = 2.0
_DEFAULT_BEAT_SECONDS = 1.5
_DEFAULT_OUTRO_SECONDS = 2.0


def _str_or_empty(value: Any) -> str:
    """Coerce a value to ``str`` — non-strings (incl. None) become ''."""

    if isinstance(value, str):
        return value
    return ""


def _positive_float(value: Any, fallback: float) -> float:
    """Parse a positive float; falls back when value is missing / invalid."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if out > 0 else fallback


def compose_explainer(
    *,
    title: str,
    sections: list[Any] | None = None,
    cta: str | None = None,
    width: int = DEFAULT_EXPLAINER_WIDTH,
    height: int = DEFAULT_EXPLAINER_HEIGHT,
    composition_id: str = "explainer",
    title_seconds: float = _DEFAULT_TITLE_SECONDS,
    section_seconds: float = _DEFAULT_SECTION_SECONDS,
    cta_seconds: float = _DEFAULT_CTA_SECONDS,
) -> dict[str, Any]:
    """Build an explainer-style composition spec.

    Layout:

    * Scene 0 — title card (centered headline).
    * Scenes 1..N — body sections (each with optional caption + image_url).
    * Scene N+1 — CTA card (only emitted if ``cta`` is set).

    ``sections`` is a list of either:

    * ``str`` — used as the caption.
    * ``dict`` with keys ``{"caption": str, "image_url": str}``.

    The returned spec is consumable by :func:`cognithor.mcp.video_tools.compose_html`.
    """

    title_text = _str_or_empty(title).strip()
    cta_text = _str_or_empty(cta).strip()
    safe_sections: list[dict[str, Any]] = []
    if isinstance(sections, list):
        for raw in sections[:MAX_SECTIONS]:
            if isinstance(raw, str):
                caption = raw.strip()
                if caption:
                    safe_sections.append({"caption": caption})
            elif isinstance(raw, dict):
                caption = _str_or_empty(raw.get("caption")).strip()
                image_url = _str_or_empty(raw.get("image_url")).strip()
                entry: dict[str, Any] = {}
                if caption:
                    entry["caption"] = caption
                if image_url:
                    entry["image_url"] = image_url
                if entry:
                    safe_sections.append(entry)

    title_dur = _positive_float(title_seconds, _DEFAULT_TITLE_SECONDS)
    section_dur = _positive_float(section_seconds, _DEFAULT_SECTION_SECONDS)
    cta_dur = _positive_float(cta_seconds, _DEFAULT_CTA_SECONDS)

    scenes: list[dict[str, Any]] = []
    cursor = 0.0

    if title_text:
        scenes.append(
            {
                "start": cursor,
                "duration": title_dur,
                "title": title_text,
            },
        )
        cursor += title_dur

    for sec in safe_sections:
        scene: dict[str, Any] = {
            "start": cursor,
            "duration": section_dur,
            **sec,
        }
        scenes.append(scene)
        cursor += section_dur

    if cta_text:
        scenes.append(
            {
                "start": cursor,
                "duration": cta_dur,
                "title": cta_text,
            },
        )
        cursor += cta_dur

    return {
        "title": title_text or "Cognithor Explainer",
        "composition_id": composition_id,
        "width": width,
        "height": height,
        "scenes": scenes,
    }


def compose_social_cut(
    *,
    hook: str,
    beats: list[Any] | None = None,
    outro: str | None = None,
    width: int = DEFAULT_SOCIAL_WIDTH,
    height: int = DEFAULT_SOCIAL_HEIGHT,
    composition_id: str = "social-cut",
    hook_seconds: float = _DEFAULT_HOOK_SECONDS,
    beat_seconds: float = _DEFAULT_BEAT_SECONDS,
    outro_seconds: float = _DEFAULT_OUTRO_SECONDS,
) -> dict[str, Any]:
    """Build a vertical social-cut composition spec (9:16 by default).

    Layout:

    * Scene 0 — hook (1-2 second teaser, big caption).
    * Scenes 1..N — beats (fast cuts, 1-2 seconds each).
    * Scene N+1 — outro (CTA / sign-off).

    ``beats`` accepts the same shape as ``compose_explainer.sections``.
    """

    hook_text = _str_or_empty(hook).strip()
    outro_text = _str_or_empty(outro).strip()
    safe_beats: list[dict[str, Any]] = []
    if isinstance(beats, list):
        for raw in beats[:MAX_BEATS]:
            if isinstance(raw, str):
                caption = raw.strip()
                if caption:
                    safe_beats.append({"caption": caption})
            elif isinstance(raw, dict):
                caption = _str_or_empty(raw.get("caption")).strip()
                image_url = _str_or_empty(raw.get("image_url")).strip()
                entry: dict[str, Any] = {}
                if caption:
                    entry["caption"] = caption
                if image_url:
                    entry["image_url"] = image_url
                if entry:
                    safe_beats.append(entry)

    hook_dur = _positive_float(hook_seconds, _DEFAULT_HOOK_SECONDS)
    beat_dur = _positive_float(beat_seconds, _DEFAULT_BEAT_SECONDS)
    outro_dur = _positive_float(outro_seconds, _DEFAULT_OUTRO_SECONDS)

    scenes: list[dict[str, Any]] = []
    cursor = 0.0

    if hook_text:
        scenes.append(
            {
                "start": cursor,
                "duration": hook_dur,
                "title": hook_text,
            },
        )
        cursor += hook_dur

    for beat in safe_beats:
        scene: dict[str, Any] = {
            "start": cursor,
            "duration": beat_dur,
            **beat,
        }
        scenes.append(scene)
        cursor += beat_dur

    if outro_text:
        scenes.append(
            {
                "start": cursor,
                "duration": outro_dur,
                "title": outro_text,
            },
        )
        cursor += outro_dur

    return {
        "title": hook_text or "Cognithor Social Cut",
        "composition_id": composition_id,
        "width": width,
        "height": height,
        "scenes": scenes,
    }


def caption_overlay(
    *,
    base_spec: dict[str, Any],
    captions: list[Any],
) -> dict[str, Any]:
    """Glue caption strings onto an existing composition spec.

    ``captions`` is a parallel list to ``base_spec["scenes"]`` —
    index *i* of ``captions`` is applied to scene *i*. A non-string
    or empty entry leaves that scene's caption untouched. Extra
    captions past the scene count are dropped silently.

    Returns a NEW spec (does not mutate the input).
    """

    if not isinstance(base_spec, dict):
        return {"scenes": []}
    if not isinstance(captions, list):
        return dict(base_spec)

    scenes_raw = base_spec.get("scenes")
    if not isinstance(scenes_raw, list):
        new_spec = dict(base_spec)
        new_spec["scenes"] = []
        return new_spec

    new_scenes: list[dict[str, Any]] = []
    for idx, scene in enumerate(scenes_raw):
        if not isinstance(scene, dict):
            new_scenes.append({})
            continue
        merged = dict(scene)
        if idx < len(captions):
            cap = captions[idx]
            if isinstance(cap, str) and cap.strip():
                merged["caption"] = cap.strip()
        new_scenes.append(merged)

    new_spec = dict(base_spec)
    new_spec["scenes"] = new_scenes
    return new_spec
