"""Sprint-27 HF-3 — `video_compose` + `video_render` MCP tools.

Exposes the `cognithor.video` renderer-abstraction (HF-2) to the
agent loop via two MCP tools:

* ``video_compose(spec, run_id)`` — pure HTML-emission. Takes a
  structured composition spec (scenes, durations, asset
  references) and returns a self-contained HTML composition.
  No subprocess, no filesystem write, no network. **GREEN risk
  level** by default.

* ``video_render(html_text|html_path, run_id, ...)`` — actual
  render via the registered backend (default: HyperFrames).
  Spawns a subprocess + writes MP4 to
  ``~/.cognithor/render/<run_id>/``. **ORANGE risk level**
  (requires user approval per agent-context risk-ceiling).

The Gatekeeper-side wiring (RED for raw user-supplied HTML
without an allowlist match) lives in HF-3's gatekeeper patch
which lands alongside the tool registration; this module ships
the tools + their input schemas + the Python-side handlers.

Apache-2.0 — no insurance / domain coupling, per Sprint-27 D4
"extension stays free, free packs ship the same surface".
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger
from cognithor.video import (
    DEFAULT_ALLOWED_ADAPTERS,
    FrameAdapter,
    OutputFormat,
    RenderError,
    RenderRequest,
    renderer_registry,
)
from cognithor.video.skills import (
    caption_overlay,
    compose_explainer,
    compose_social_cut,
)

if TYPE_CHECKING:
    from cognithor.mcp.server import JarvisMCPServer
    from cognithor.video.renderer_base import RenderResult

log = get_logger(__name__)

__all__ = ["register_video_tools"]


# Producer-side cap on the inline composition spec — protects the
# subprocess + Gatekeeper from absurd payloads. ~64 KB ≈ 5 long
# scenes with images / captions; anything bigger should land via
# html_path rather than inline.
_MAX_HTML_TEXT_BYTES = 64 * 1024


_HTML_SHELL = """\
<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 0; background: #000; color: #fff;
          font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  #stage {{ position: relative; overflow: hidden;
           width: {width}px; height: {height}px; }}
  .scene {{ position: absolute; inset: 0; opacity: 0; }}
</style>
</head>
<body>
<div id="stage" data-composition-id="{composition_id}"
     data-width="{width}" data-height="{height}">
{scenes_html}
</div>
</body>
</html>
"""


_SCENE_TEMPLATE = (
    '  <div class="scene" data-track-index="{idx}" '
    'data-start="{start:g}" data-duration="{duration:g}">\n'
    "{body}\n"
    "  </div>"
)


def _sanitize_text(s: str) -> str:
    """Conservative HTML-escape for caption / title strings.

    The composition is rendered in a sandboxed Puppeteer with a
    strict CSP, but defense-in-depth: never embed raw user text
    into the DOM without escaping. Allows letters, numbers,
    spaces, and a small set of punctuation; everything else is
    stripped.
    """

    return re.sub(r"[^\w\s.,:;!?\-—()À-ÿ]", "", s)[:512]


def _scene_body(scene: dict[str, Any]) -> str:
    """Build the inner HTML for a single scene from its spec."""

    parts: list[str] = []
    title = scene.get("title")
    if isinstance(title, str) and title:
        parts.append(
            f'    <h1 style="position:absolute;top:5%;left:5%;font-size:5vw;'
            f'margin:0">{_sanitize_text(title)}</h1>',
        )
    caption = scene.get("caption")
    if isinstance(caption, str) and caption:
        parts.append(
            f'    <p style="position:absolute;bottom:8%;left:5%;right:5%;'
            f'font-size:2.5vw;margin:0">{_sanitize_text(caption)}</p>',
        )
    image_url = scene.get("image_url")
    if isinstance(image_url, str) and image_url.startswith(
        ("file://", "/", "./", "../"),
    ):
        # Local-asset references only (no http(s)) — composition
        # must self-contain its assets per HF-1 threat model.
        parts.append(
            f'    <img src="{image_url}" '
            'style="position:absolute;inset:0;width:100%;height:100%;'
            'object-fit:cover" />',
        )
    return "\n".join(parts) if parts else "    <!-- empty scene -->"


def compose_html(spec: dict[str, Any]) -> str:
    """Pure function: turn a structured spec into a HyperFrames HTML.

    The accepted spec shape (intentionally tight):

    .. code-block:: json

        {
          "title": "My Composition",
          "lang": "en",
          "width": 1920,
          "height": 1080,
          "composition_id": "demo-1",
          "scenes": [
            {"start": 0, "duration": 3, "title": "Hello",
             "caption": "world", "image_url": "./assets/hero.png"}
          ]
        }

    Anything outside the accepted shape is dropped (defensive
    parsing). Returns the rendered HTML as a string. Does NOT
    write to disk; the caller (typically the ``video_render`` MCP
    tool) handles that.
    """

    title = _sanitize_text(str(spec.get("title", "Cognithor Composition")))
    lang = re.sub(r"[^a-zA-Z\-]", "", str(spec.get("lang", "en")))[:8] or "en"
    composition_id = (
        re.sub(
            r"[^a-zA-Z0-9_\-]",
            "",
            str(spec.get("composition_id", "comp")),
        )[:64]
        or "comp"
    )

    width = int(spec.get("width", 1920))
    height = int(spec.get("height", 1080))
    width = max(16, min(7680, width))
    height = max(16, min(4320, height))

    scenes_raw = spec.get("scenes")
    if not isinstance(scenes_raw, list):
        scenes_raw = []

    scene_html_parts: list[str] = []
    for idx, scene in enumerate(scenes_raw):
        if not isinstance(scene, dict):
            continue
        try:
            start = float(scene.get("start", 0.0))
            duration = float(scene.get("duration", 3.0))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        scene_html_parts.append(
            _SCENE_TEMPLATE.format(
                idx=idx,
                start=max(0.0, start),
                duration=duration,
                body=_scene_body(scene),
            ),
        )

    return _HTML_SHELL.format(
        lang=lang,
        title=title,
        width=width,
        height=height,
        composition_id=composition_id,
        scenes_html="\n".join(scene_html_parts) or "  <!-- no scenes -->",
    )


def _coerce_adapters(value: Any) -> frozenset[FrameAdapter]:
    """Parse caller-supplied adapter list into a typed frozenset.

    Falls back to :data:`DEFAULT_ALLOWED_ADAPTERS` when the value
    is missing or invalid.
    """

    if value is None:
        return DEFAULT_ALLOWED_ADAPTERS
    if not isinstance(value, list):
        return DEFAULT_ALLOWED_ADAPTERS
    out: set[FrameAdapter] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        try:
            out.add(FrameAdapter(item.lower()))
        except ValueError:
            continue
    return frozenset(out) if out else DEFAULT_ALLOWED_ADAPTERS


def _coerce_format(value: Any) -> OutputFormat:
    if isinstance(value, str):
        try:
            return OutputFormat(value.lower())
        except ValueError:
            pass
    return OutputFormat.MP4


# ---------------------------------------------------------------------------
# MCP tool handlers
# ---------------------------------------------------------------------------


async def _video_compose_handler(
    spec: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """``video_compose`` MCP-tool handler.

    Returns the rendered HTML (string) inline. The caller decides
    whether to pipe it directly into ``video_render`` or stash it
    in the vault for later editing.
    """

    if not isinstance(spec, dict):
        return {"ok": False, "error": "spec must be a JSON object"}
    html = compose_html(spec)
    if len(html.encode("utf-8")) > _MAX_HTML_TEXT_BYTES:
        return {
            "ok": False,
            "error": (
                f"composed HTML exceeds {_MAX_HTML_TEXT_BYTES // 1024} KB cap "
                "— split into multiple scenes or pre-render assets"
            ),
        }
    return {
        "ok": True,
        "html": html,
        "byte_size": len(html.encode("utf-8")),
        "run_id": run_id or "",
    }


async def _video_render_handler(
    run_id: str,
    html_text: str | None = None,
    html_path: str | None = None,
    output_format: str = "mp4",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    duration_seconds: float | None = None,
    allowed_adapters: list[str] | None = None,
    timeout_seconds: float = 300.0,
    renderer: str = "hyperframes",
) -> dict[str, Any]:
    """``video_render`` MCP-tool handler — dispatches to the registered renderer."""

    if not run_id or not isinstance(run_id, str):
        return {"ok": False, "error": "run_id is required"}

    factory = renderer_registry.get(renderer)
    if factory is None:
        return {
            "ok": False,
            "error": f"unknown renderer {renderer!r}",
            "available": sorted(renderer_registry),
        }

    try:
        request = RenderRequest(
            run_id=run_id,
            html_path=__import__("pathlib").Path(html_path) if html_path else None,
            html_text=html_text,
            output_format=_coerce_format(output_format),
            width=int(width),
            height=int(height),
            fps=int(fps),
            duration_seconds=duration_seconds,
            allowed_adapters=_coerce_adapters(allowed_adapters),
            timeout_seconds=float(timeout_seconds),
        )
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"bad render request: {exc}"}

    try:
        result = await factory().render(request)
    except RenderError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "run_id": exc.run_id,
            "stderr_excerpt": exc.stderr_excerpt,
        }

    # HF-4: TRUST wiring. Best-effort emission to canonical
    # ledgers — never raise into the caller. Each tier is wrapped
    # in suppress() per the established TRUST-stack pattern.
    _emit_trust_entries(request=request, result=result, html_text=html_text)

    return {"ok": True, **result.to_dict()}


def _hash_html_source(html_text: str | None, html_path: str | None) -> str:
    """Stable identifier for the rendered composition.

    Used as the ``ProvenanceTag.source_id`` for the rendered MP4
    so downstream tools can correlate the binary with the exact
    HTML that produced it.
    """

    if html_text is not None:
        digest = hashlib.sha256(html_text.encode("utf-8")).hexdigest()
        return f"html-text:{digest[:16]}"
    if html_path is not None:
        return f"html-path:{html_path}"
    return "html:unknown"


def _emit_trust_entries(
    *,
    request: RenderRequest,
    result: RenderResult,
    html_text: str | None,
) -> None:
    """HF-4 — emit ProvenanceTag + CostEntry for a successful video render.

    All three sub-emissions are individually wrapped in
    :func:`contextlib.suppress` so a TRUST-tier fault cannot
    surface into the MCP-tool caller. The pattern matches the
    Sprint-26 TRUST-stack rollout: best-effort capture, never
    block the calling tier.
    """

    html_path_str = str(request.html_path) if request.html_path is not None else None
    source_id = _hash_html_source(html_text, html_path_str)
    output_path_str = str(result.output_path)

    # TRUST-9 ProvenanceTag for the rendered MP4 file.
    with contextlib.suppress(Exception):
        from cognithor.memory.provenance import (
            PROVENANCE_LEDGER,
            ExpiryPolicy,
            ProvenanceTag,
            SourceType,
        )

        PROVENANCE_LEDGER.tag(
            output_path_str,
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id=source_id,
                expiry_policy=ExpiryPolicy.PERMANENT,
                notes=(
                    f"video_render run_id={request.run_id} "
                    f"format={result.output_format.value} "
                    f"{result.width}x{result.height}@{result.fps}fps"
                ),
            ),
        )

    # TRUST-9 CostEntry. Local renders carry zero monetary cost
    # but the entry exists so the operator's per-run accounting
    # surfaces "this run rendered N MP4s" in the cost-summary
    # tile alongside text-token costs. cost_usd_micro=0 — the
    # subprocess runtime in milliseconds rides ``unit_count`` so
    # the Trace-UI can render a duration histogram.
    with contextlib.suppress(Exception):
        from cognithor.security.cost_ledger import COST_LEDGER, CostEntry, CostKind

        COST_LEDGER.record(
            CostEntry(
                kind=CostKind.OTHER,
                tool="video_render",
                cost_usd_micro=0,
                backend=(request.allowed_adapters and "hyperframes") or "",
                run_id=request.run_id,
                unit_count=result.duration_ms,
                notes=(f"render={result.output_format.value} bytes={result.bytes_written}"),
            ),
        )


# ---------------------------------------------------------------------------
# Skill-layer handlers (HF-5) — pure-function presets
# ---------------------------------------------------------------------------


def _compose_spec_to_html(spec: dict[str, Any]) -> dict[str, Any]:
    """Shared tail for all HF-5 skills: HTML-emit + size-cap check."""

    html = compose_html(spec)
    if len(html.encode("utf-8")) > _MAX_HTML_TEXT_BYTES:
        return {
            "ok": False,
            "error": (
                f"composed HTML exceeds {_MAX_HTML_TEXT_BYTES // 1024} KB cap "
                "— split into multiple scenes or pre-render assets"
            ),
        }
    return {
        "ok": True,
        "html": html,
        "byte_size": len(html.encode("utf-8")),
        "spec": spec,
    }


async def _video_compose_explainer_handler(
    title: str,
    sections: list[Any] | None = None,
    cta: str | None = None,
    width: int = 1920,
    height: int = 1080,
    composition_id: str = "explainer",
    run_id: str | None = None,
) -> dict[str, Any]:
    """``video_compose_explainer`` — title + body sections + optional CTA."""

    if not isinstance(title, str) or not title.strip():
        return {"ok": False, "error": "title is required"}
    spec = compose_explainer(
        title=title,
        sections=sections,
        cta=cta,
        width=int(width),
        height=int(height),
        composition_id=composition_id,
    )
    out = _compose_spec_to_html(spec)
    if run_id is not None:
        out["run_id"] = run_id
    return out


async def _video_compose_social_cut_handler(
    hook: str,
    beats: list[Any] | None = None,
    outro: str | None = None,
    width: int = 1080,
    height: int = 1920,
    composition_id: str = "social-cut",
    run_id: str | None = None,
) -> dict[str, Any]:
    """``video_compose_social_cut`` — vertical 9:16 short with fast cuts."""

    if not isinstance(hook, str) or not hook.strip():
        return {"ok": False, "error": "hook is required"}
    spec = compose_social_cut(
        hook=hook,
        beats=beats,
        outro=outro,
        width=int(width),
        height=int(height),
        composition_id=composition_id,
    )
    out = _compose_spec_to_html(spec)
    if run_id is not None:
        out["run_id"] = run_id
    return out


async def _video_caption_overlay_handler(
    base_spec: dict[str, Any],
    captions: list[Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """``video_caption_overlay`` — glue captions onto an existing spec."""

    if not isinstance(base_spec, dict):
        return {"ok": False, "error": "base_spec must be a JSON object"}
    if not isinstance(captions, list):
        return {"ok": False, "error": "captions must be a list of strings"}
    spec = caption_overlay(base_spec=base_spec, captions=captions)
    out = _compose_spec_to_html(spec)
    if run_id is not None:
        out["run_id"] = run_id
    return out


# ---------------------------------------------------------------------------
# Registration — called by the MCP server bootstrapping
# ---------------------------------------------------------------------------


_VIDEO_COMPOSE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec": {
            "type": "object",
            "description": (
                "Structured composition spec — title, lang, width, "
                "height, composition_id, scenes (list of {start, "
                "duration, title, caption, image_url})."
            ),
        },
        "run_id": {
            "type": "string",
            "description": (
                "Stable identifier for the composition run; reused as trace_id by the receipt API."
            ),
        },
    },
    "required": ["spec"],
}

_VIDEO_RENDER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "run_id": {
            "type": "string",
            "description": "Same run_id as the originating agent run.",
        },
        "html_text": {
            "type": "string",
            "description": (
                "Inline composition HTML. Mutually exclusive with html_path. "
                "Caller should typically use the output of video_compose."
            ),
        },
        "html_path": {
            "type": "string",
            "description": "Path to a self-contained composition HTML file.",
        },
        "output_format": {
            "type": "string",
            "enum": ["mp4", "mov", "webm"],
            "default": "mp4",
        },
        "width": {"type": "integer", "minimum": 16, "maximum": 7680, "default": 1920},
        "height": {"type": "integer", "minimum": 16, "maximum": 4320, "default": 1080},
        "fps": {"type": "integer", "minimum": 1, "maximum": 240, "default": 30},
        "duration_seconds": {"type": "number", "minimum": 0.0},
        "allowed_adapters": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["css", "waapi", "anime", "lottie", "three", "gsap"],
            },
            "description": (
                "Override the default adapter allowlist. GSAP is opt-in "
                "only — see the HyperFrames license note in the spike "
                "doc."
            ),
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 1800.0,
            "default": 300.0,
        },
        "renderer": {
            "type": "string",
            "default": "hyperframes",
            "description": "Renderer name from cognithor.video.renderer_registry.",
        },
    },
    "required": ["run_id"],
}


_VIDEO_COMPOSE_EXPLAINER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Headline rendered on the title card (scene 0).",
        },
        "sections": {
            "type": "array",
            "description": (
                "List of body sections. Each item may be a string "
                "(used as caption) or an object with keys "
                "'caption' and/or 'image_url' (local-asset path)."
            ),
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "caption": {"type": "string"},
                            "image_url": {"type": "string"},
                        },
                    },
                ],
            },
        },
        "cta": {
            "type": "string",
            "description": "Optional call-to-action shown as the final scene.",
        },
        "width": {"type": "integer", "minimum": 16, "maximum": 7680, "default": 1920},
        "height": {"type": "integer", "minimum": 16, "maximum": 4320, "default": 1080},
        "composition_id": {"type": "string", "default": "explainer"},
        "run_id": {"type": "string"},
    },
    "required": ["title"],
}

_VIDEO_COMPOSE_SOCIAL_CUT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook": {
            "type": "string",
            "description": "Opening hook line (1-2 seconds, big caption).",
        },
        "beats": {
            "type": "array",
            "description": (
                "List of fast-cut beats. Each item may be a string "
                "(caption) or an object with keys 'caption' and/or "
                "'image_url'. Capped at 8 beats for short-form pacing."
            ),
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "caption": {"type": "string"},
                            "image_url": {"type": "string"},
                        },
                    },
                ],
            },
        },
        "outro": {
            "type": "string",
            "description": "Optional closing line (CTA / sign-off).",
        },
        "width": {"type": "integer", "minimum": 16, "maximum": 7680, "default": 1080},
        "height": {"type": "integer", "minimum": 16, "maximum": 4320, "default": 1920},
        "composition_id": {"type": "string", "default": "social-cut"},
        "run_id": {"type": "string"},
    },
    "required": ["hook"],
}

_VIDEO_CAPTION_OVERLAY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_spec": {
            "type": "object",
            "description": ("Base composition spec (same shape as video_compose's spec)."),
        },
        "captions": {
            "type": "array",
            "description": (
                "Parallel caption track — captions[i] applies to scene i. "
                "Empty / non-string entries leave the existing caption untouched."
            ),
            "items": {"type": "string"},
        },
        "run_id": {"type": "string"},
    },
    "required": ["base_spec", "captions"],
}


def register_video_tools(server: JarvisMCPServer) -> None:
    """Register video composition + render MCP tools on a running server."""

    from cognithor.mcp.server import MCPToolDef  # local import to avoid cycles

    server.register_tool(
        MCPToolDef(
            name="video_compose",
            description=(
                "Build a self-contained HTML composition from a structured spec. "
                "GREEN risk level — pure function, no subprocess, no filesystem write."
            ),
            input_schema=_VIDEO_COMPOSE_INPUT_SCHEMA,
            handler=_video_compose_handler,
            annotations={
                "risk_level": "green",
                "category": "video",
                "supports_streaming": False,
            },
        ),
    )
    server.register_tool(
        MCPToolDef(
            name="video_render",
            description=(
                "Render a composition HTML to MP4 / MOV / WebM via the configured "
                "renderer (default: HyperFrames). Writes under "
                "~/.cognithor/render/<run_id>/. ORANGE risk level — requires user "
                "approval. Raw user-supplied HTML without adapter allowlist match "
                "should be rejected RED upstream by the Gatekeeper."
            ),
            input_schema=_VIDEO_RENDER_INPUT_SCHEMA,
            handler=_video_render_handler,
            annotations={
                "risk_level": "orange",
                "category": "video",
                "supports_streaming": False,
            },
        ),
    )
    server.register_tool(
        MCPToolDef(
            name="video_compose_explainer",
            description=(
                "Title card + body sections + optional CTA in 16:9. "
                "GREEN risk level — pure-function preset over video_compose."
            ),
            input_schema=_VIDEO_COMPOSE_EXPLAINER_INPUT_SCHEMA,
            handler=_video_compose_explainer_handler,
            annotations={
                "risk_level": "green",
                "category": "video",
                "supports_streaming": False,
            },
        ),
    )
    server.register_tool(
        MCPToolDef(
            name="video_compose_social_cut",
            description=(
                "Vertical 9:16 short with hook + fast-cut beats + outro. "
                "GREEN risk level — pure-function preset over video_compose."
            ),
            input_schema=_VIDEO_COMPOSE_SOCIAL_CUT_INPUT_SCHEMA,
            handler=_video_compose_social_cut_handler,
            annotations={
                "risk_level": "green",
                "category": "video",
                "supports_streaming": False,
            },
        ),
    )
    server.register_tool(
        MCPToolDef(
            name="video_caption_overlay",
            description=(
                "Glue a parallel caption track onto an existing composition spec. "
                "GREEN risk level — pure-function transform; emits new HTML."
            ),
            input_schema=_VIDEO_CAPTION_OVERLAY_INPUT_SCHEMA,
            handler=_video_caption_overlay_handler,
            annotations={
                "risk_level": "green",
                "category": "video",
                "supports_streaming": False,
            },
        ),
    )
    log.info(
        "video_tools_registered",
        tools=[
            "video_compose",
            "video_render",
            "video_compose_explainer",
            "video_compose_social_cut",
            "video_caption_overlay",
        ],
    )
