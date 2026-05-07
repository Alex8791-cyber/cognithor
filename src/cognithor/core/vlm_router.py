"""VLM (Vision-Language-Model) router with quality tiers + heuristic classifier.

Sprint-VLM-Router (2026-05-07). Mirrors the existing ConciergeProfile /
ContextProfile pattern from :mod:`cognithor.core.model_router`: a small
set of named profiles, a heuristic classifier, ContextVar isolation,
a scope context manager, and a TRUST-2 explanation surface so the
Gatekeeper / Receipt sidebar can show *why* a particular VLM was chosen.

The router solves a real production problem: video/image understanding
on a 32 GB RTX 5090 forces a hard trade-off — Qwen3.6-27B-NVFP4 + vision
encoder needs ``--cpu-offload-gb 4`` (PCIe-bound, ~2.7 tok/s), whereas
Qwen2.5-VL-7B-Instruct fits with cuda graphs and runs ~80–150 tok/s but
is ~10–20 % weaker on Video-MME and ~20 % weaker on multi-step
reasoning. The right pick depends on what the user is actually asking
about the video — a "describe in one sentence" prompt routes ``fast``,
a "compare frame 3 to frame 12" prompt routes ``premium``.

Public surface:

* :class:`VlmProfile` — frozen dataclass; the operational truth about
  one VLM deployment (model_id, vllm flags, throughput, footprint).
* :data:`VLM_PROFILES` — three built-in profiles: ``fast``, ``balanced``,
  ``premium``.
* :class:`VlmTaskClass` — enum of routing decisions.
* :func:`classify_vlm_task` — pure-function heuristic (deterministic,
  unit-testable).
* :class:`VlmRouter` — selection API + TRUST-2 explanation + ContextVar
  scope helper.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import re
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VlmProfile:
    """One VLM deployment as the router sees it.

    The ``vllm_flags`` tuple is the *exact* list of flags an operator
    would pass to ``vllm serve <model_id>`` — it is the single source
    of truth for both the launch wizard (Flutter Settings → vLLM) and
    the smoke tests under ``scripts/smoke_vllm_backend.py``. Keeping
    them embedded here means a config drift between docs and runtime
    cannot happen silently — the router refuses to pick a profile
    whose flags it did not ship.

    Throughput numbers are *measured* in our own bench (RTX 5090, no
    other GPU consumers), not vendor-published peaks. The ``fast`` tier
    figures come from a 2026-05 measurement of Qwen2.5-VL-7B with
    cuda graphs; the ``premium`` figures from the spike-2026-04-23
    iteration #3 (Qwen3.6-27B-NVFP4 + cpu-offload=4 + enforce-eager).
    """

    #: Routing-friendly name (``"fast"``, ``"balanced"``, ``"premium"``).
    name: str

    #: HuggingFace identifier (``"Qwen/Qwen2.5-VL-7B-Instruct"`` etc.).
    #: This is what gets passed to ``--model`` and what shows up in
    #: ``backend.list_models()`` once the server is up.
    model_id: str

    #: Exact CLI flags for ``vllm serve <model_id>``. The launch wizard
    #: + smoke tests read this verbatim; do not mutate.
    vllm_flags: tuple[str, ...]

    #: Hard ceiling: clips longer than this should not route here. The
    #: classifier penalises this profile when video duration exceeds
    #: this value (premium is happy with longer; fast is not).
    max_video_seconds: int

    #: Maximum frame count the profile is configured to ingest. With
    #: vLLM's ``--media-io-kwargs '{"video": {"num_frames": N}}'`` this
    #: caps the encoder cost per request.
    max_frames: int

    #: Measured decode throughput in tokens/second for our own setup.
    #: NOT a vendor benchmark — re-measured per release.
    expected_throughput_tok_s: int

    #: ``"fast"`` (sub-second/clip), ``"balanced"`` (a-few-seconds), or
    #: ``"premium"`` (wait-and-think). Used by the Flutter UI to pick
    #: a progress-indicator style.
    quality_tier: str

    #: GPU memory footprint in GiB *including* vision encoder + buffers
    #: but *excluding* KV cache. Used by the launch wizard to refuse
    #: a profile that does not fit the detected GPU.
    memory_footprint_gib: float

    #: One-line human description, shown in Settings → vLLM dropdown.
    description: str

    #: Quality estimate as a percentage of the strongest profile we ship,
    #: based on Video-MME / MVBench / MathVista averages. Used by the
    #: explainability surface (TRUST-2) to communicate the trade-off.
    relative_quality_pct: int

    def vllm_serve_command(self) -> list[str]:
        """Return the full ``vllm serve …`` argv list for ops.

        Ops scripts and the Flutter launch wizard build their actual
        process invocation from this list — no string-formatting, no
        shell quoting drift. Spike-quality discipline.
        """
        return ["vllm", "serve", self.model_id, *self.vllm_flags]


# ---------------------------------------------------------------------------
# Built-in profiles — measured, not promised
# ---------------------------------------------------------------------------


_FAST_FLAGS: tuple[str, ...] = (
    "--max-model-len",
    "32768",
    "--max-num-seqs",
    "4",
    "--gpu-memory-utilization",
    "0.90",
    "--kv-cache-dtype",
    "fp8",
    "--enable-prefix-caching",
    "--trust-remote-code",
    "--media-io-kwargs",
    '{"video": {"num_frames": 32}}',
)


_BALANCED_FLAGS: tuple[str, ...] = (
    "--max-model-len",
    "16384",
    "--max-num-seqs",
    "2",
    "--gpu-memory-utilization",
    "0.92",
    "--kv-cache-dtype",
    "fp8",
    "--enforce-eager",
    "--trust-remote-code",
    "--media-io-kwargs",
    '{"video": {"num_frames": 32}}',
)


_PREMIUM_FLAGS: tuple[str, ...] = (
    # Spike 2026-04-23 iteration #3 — the only known stable Qwen3.6-27B
    # + Vision config on RTX 5090 32 GB with active Windows compositor.
    # CPU-offload is hardware-forced, not a config choice; see the
    # spike-findings doc for the three iterations that proved it.
    "--max-model-len",
    "16384",
    "--max-num-seqs",
    "2",
    "--max-num-batched-tokens",
    "2048",
    "--gpu-memory-utilization",
    "0.94",
    "--cpu-offload-gb",
    "4",
    "--enforce-eager",
    "--reasoning-parser",
    "qwen3",
    "--trust-remote-code",
    "--media-io-kwargs",
    '{"video": {"num_frames": -1}}',
)


VLM_PROFILES: dict[str, VlmProfile] = {
    "fast": VlmProfile(
        name="fast",
        # Replaces the prior Qwen2.5-VL-7B pick. Research 2026-05-07
        # confirmed Qwen3-VL-8B-Instruct ties on MVBench (the
        # short-clip benchmark closest to our use case) and is +10–20
        # pts ahead on long-clip understanding (LVBench +10.5),
        # temporal grounding (CharadesSTA +16.3), and hallucination
        # resistance (+12.5) — all at ~10 % more parameters
        # (negligible speed delta on RTX 5090). Native 256 K context
        # vs 128 K for 2.5-VL. Only tradeoff: ~5 pts lower OCRBench;
        # the OCR-Dominant task class in classify_vlm_task is set up
        # to keep routing here regardless because the absolute number
        # is still strong (~82).
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        vllm_flags=_FAST_FLAGS,
        max_video_seconds=120,  # 8B handles longer clips than 7B
        max_frames=32,
        expected_throughput_tok_s=95,
        quality_tier="fast",
        memory_footprint_gib=17.0,
        description=(
            "Qwen3-VL 8B Instruct (FP16) — fits cleanly on 32 GB GPU "
            "with cuda graphs. ~95 tok/s decode, ~3 s per typical "
            "caption. Best for short-form description, captioning, "
            "OCR, scene-classification, social-media cuts. Native "
            "256 K context. Beats Qwen2.5-VL-7B on long-video "
            "(LVBench +10.5) and reasoning (Math/MMMU +20–37 pts) "
            "without a meaningful speed cost."
        ),
        relative_quality_pct=85,
    ),
    "balanced": VlmProfile(
        name="balanced",
        # Same model as fast, but Thinking mode emits a private
        # reasoning trace before answering. Costs ~3× tokens per
        # response → ~30 tok/s effective, but quality jump is large
        # on multi-step tasks (MathVista 68 → 81, MMMU-Pro 38 → 60).
        # Single-model balance: no extra GPU memory, no extra Docker
        # image, just a flag toggle at request time.
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        vllm_flags=_BALANCED_FLAGS,
        max_video_seconds=180,
        max_frames=32,
        expected_throughput_tok_s=30,
        quality_tier="balanced",
        memory_footprint_gib=17.0,
        description=(
            "Qwen3-VL 8B Thinking — same architecture as fast, with "
            "explicit chain-of-thought reasoning. ~30 tok/s effective "
            "(emits ~3× tokens including reasoning trace). Closes the "
            "reasoning gap to premium for math, multi-step inference, "
            "and complex video Q&A — without the 27B memory hit."
        ),
        relative_quality_pct=93,
    ),
    "premium": VlmProfile(
        name="premium",
        model_id="mmangkad/Qwen3.6-27B-NVFP4",
        vllm_flags=_PREMIUM_FLAGS,
        max_video_seconds=600,
        max_frames=64,
        expected_throughput_tok_s=3,
        quality_tier="premium",
        memory_footprint_gib=28.5,
        description=(
            "Qwen3.6-27B NVFP4 with CPU offload — premium reasoning + "
            "video understanding. Forced offload because 27B + vision "
            "encoder + KV does not fit RTX 5090 32 GB without it. "
            "~3 tok/s decode, ~120 s per caption. Use only when the "
            "task demands multi-step reasoning, long-clip coherence, "
            "or fine-grained nuance."
        ),
        relative_quality_pct=100,
    ),
}


# ---------------------------------------------------------------------------
# Task classifier — heuristic, deterministic, unit-testable
# ---------------------------------------------------------------------------


class VlmTaskClass(str, enum.Enum):
    """Coarse categories the heuristic classifier emits.

    Names map 1:1 to TRUST-2 ``rule_id`` values used in the receipt
    so a reviewer can grep the audit log for routing decisions.
    """

    QUICK_DESCRIBE = "quick_describe"
    OCR_DOMINANT = "ocr_dominant"
    DETAILED_ANALYSIS = "detailed_analysis"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    LONG_FORM = "long_form"
    FORENSIC = "forensic"


# Word-class regex patterns — tuned against typical user prompts in
# DE + EN. Patterns are conservative (false negatives → fast tier,
# which is acceptable; false positives → premium tier, costs latency).
# Add new patterns here, never inline-mutate them in `classify_vlm_task`.
_OCR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bread\b",
        r"\btext\b.*\bsay",
        r"\bocr\b",
        r"\bsubtitle",
        r"\blesen\b",
        r"\bschrift\b",
        r"\bwas steht\b",
    )
)

_REASONING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bcalculate\b",
        r"\bestimate\b",
        r"\bexplain\b.*\bwhy\b",
        r"\binfer\b",
        r"\breason\b",
        r"\banalyze\b",
        r"\banalyse\b",
        r"\bvergleiche\b",
        r"\berklär",
        r"\bberechne\b",
        r"\banalysier",
        r"\bschlussfolgere\b",
        r"\bwarum\b.*(geschieht|passiert|geht|läuft)",
    )
)

_FORENSIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bforensic",
        r"\bevidence\b",
        r"\bdetect\b.*\b(anomal|tamper|fake|deepfake)",
        r"\bbeweis\b",
        r"\bmanipulation\b",
        r"\bauthentic",
        r"\bauthentiz",
    )
)

_DETAILED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdetailed\b",
        r"\bcomprehensive\b",
        r"\btimeline\b",
        r"\bschritt für schritt\b",
        r"\bausführlich\b",
        r"\bdetailliert\b",
    )
)

# Above this many words, prompts are usually doing more than asking
# "what is this?" — escalate one tier. Empirical, derived from a
# sample of 200 video-related prompts in our skill catalog.
_LONG_PROMPT_WORD_THRESHOLD = 60


def classify_vlm_task(
    prompt: str,
    *,
    video_seconds: float | None = None,
    explicit_class: str | None = None,
) -> VlmTaskClass:
    """Classify what kind of video task the prompt is asking for.

    Pure function — no I/O, no globals, deterministic. Same input
    always returns the same output, which keeps the TRUST-2 receipt
    reproducible at audit time.

    Args:
        prompt: The user's question or instruction about the video.
            Empty string is treated as ``QUICK_DESCRIBE``.
        video_seconds: Clip duration in seconds. If known, drives the
            ``LONG_FORM`` escalation. ``None`` skips that branch.
        explicit_class: Caller-provided override (e.g. set by an MCP
            tool that *knows* it needs forensic analysis). When given
            and valid, returned verbatim — bypassing all heuristics.

    Returns:
        One of the :class:`VlmTaskClass` values. The mapping to a
        :class:`VlmProfile` is done by :class:`VlmRouter`, not here —
        keeping classification orthogonal to model selection so the
        router can be swapped or A/B-tested without touching the
        classifier.
    """
    if explicit_class:
        try:
            return VlmTaskClass(explicit_class)
        except ValueError:
            log.warning(
                "vlm_explicit_class_unknown",
                given=explicit_class,
                falling_back="heuristic",
            )

    text = (prompt or "").strip()
    if not text:
        return VlmTaskClass.QUICK_DESCRIBE

    # Forensic patterns trump everything else — privacy / legal context.
    for pattern in _FORENSIC_PATTERNS:
        if pattern.search(text):
            return VlmTaskClass.FORENSIC

    # Long-form clips imply temporal reasoning even without a complex
    # prompt — automatic escalation past the ``fast`` tier.
    if video_seconds is not None and video_seconds > 60:
        return VlmTaskClass.LONG_FORM

    for pattern in _REASONING_PATTERNS:
        if pattern.search(text):
            return VlmTaskClass.MULTI_STEP_REASONING

    for pattern in _OCR_PATTERNS:
        if pattern.search(text):
            return VlmTaskClass.OCR_DOMINANT

    # A long prompt usually means the user wants depth.
    word_count = len(text.split())
    if word_count >= _LONG_PROMPT_WORD_THRESHOLD:
        return VlmTaskClass.DETAILED_ANALYSIS

    for pattern in _DETAILED_PATTERNS:
        if pattern.search(text):
            return VlmTaskClass.DETAILED_ANALYSIS

    return VlmTaskClass.QUICK_DESCRIBE


# Mapping from task class to default profile. Centralised so
# external A/B harnesses can swap the table without forking the
# classifier.
_DEFAULT_TASK_PROFILE_MAP: dict[VlmTaskClass, str] = {
    VlmTaskClass.QUICK_DESCRIBE: "fast",
    VlmTaskClass.OCR_DOMINANT: "fast",
    VlmTaskClass.DETAILED_ANALYSIS: "balanced",
    VlmTaskClass.MULTI_STEP_REASONING: "premium",
    VlmTaskClass.LONG_FORM: "premium",
    VlmTaskClass.FORENSIC: "premium",
}


# ---------------------------------------------------------------------------
# Router — ContextVar isolation, scope, TRUST-2 explanation
# ---------------------------------------------------------------------------


_vlm_quality_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_vlm_quality", default=None
)


@dataclasses.dataclass(frozen=True)
class VlmRoutingDecision:
    """The full record the router emits — ready for TRUST-2 receipt.

    Every routing decision is one of these. The Receipt sidebar shows
    ``rule_id`` + ``rule_source`` + ``matched_pattern`` so a reviewer
    can replay the decision: "the prompt contained the word 'compare',
    therefore the multi-step-reasoning rule fired, therefore premium".
    """

    profile: VlmProfile
    task_class: VlmTaskClass
    rule_id: str
    rule_source: str
    matched_pattern: str | None
    overridden_by: str | None  # "context_var", "config", "explicit", or None


class VlmRouter:
    """Heuristic + override-aware VLM selection.

    Three precedence layers, evaluated top-to-bottom:

    1. **ContextVar override** — :meth:`quality_scope` /
       :meth:`set_quality` force a profile for the lifetime of an
       ``async with``. Highest priority.
    2. **Config override** — ``config.vllm.quality_default`` (string
       in :data:`VLM_PROFILES`) sets a global default the user can
       pin in ``~/.cognithor/config.yaml``.
    3. **Heuristic** — :func:`classify_vlm_task` runs against the
       prompt + clip metadata, then :data:`_DEFAULT_TASK_PROFILE_MAP`
       picks the profile.

    All three layers emit a :class:`VlmRoutingDecision` so the audit
    chain can replay why a particular VLM was chosen. There is no
    "silent default" — every code path tags its rule_id.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config

    # ----- override surface -------------------------------------------------

    def set_quality(self, name: str) -> None:
        """Pin a quality tier for downstream calls in this async context.

        Raises:
            ValueError: if ``name`` is not in :data:`VLM_PROFILES`.
        """
        if name not in VLM_PROFILES:
            raise ValueError(f"Unknown VLM quality {name!r}. Valid options: {sorted(VLM_PROFILES)}")
        _vlm_quality_var.set(name)
        log.info("vlm_quality_pinned", quality=name)

    def get_quality(self) -> str | None:
        """Return the active pinned quality tier, or ``None``."""
        return _vlm_quality_var.get()

    def clear_quality(self) -> None:
        """Remove the pinned quality tier."""
        current = _vlm_quality_var.get()
        if current:
            log.info("vlm_quality_cleared", was=current)
        _vlm_quality_var.set(None)

    @contextlib.contextmanager
    def quality_scope(self, name: str) -> Iterator[None]:
        """Temporarily pin a quality tier; restore on exit (even on raise).

        Mirrors :meth:`ModelRouter.context_profile_scope` precisely so
        callers fluent in one are fluent in the other.
        """
        if name not in VLM_PROFILES:
            raise ValueError(f"Unknown VLM quality {name!r}. Valid options: {sorted(VLM_PROFILES)}")
        token = _vlm_quality_var.set(name)
        log.info("vlm_quality_scope_entered", quality=name)
        try:
            yield
        finally:
            _vlm_quality_var.reset(token)
            log.info("vlm_quality_scope_exited", quality=name)

    # ----- selection --------------------------------------------------------

    def select_profile(
        self,
        prompt: str,
        *,
        video_seconds: float | None = None,
        explicit_class: str | None = None,
    ) -> VlmProfile:
        """Return the chosen :class:`VlmProfile` (no explanation).

        Convenience wrapper around :meth:`select_profile_explained`
        for callers that don't need the audit surface.
        """
        return self.select_profile_explained(
            prompt,
            video_seconds=video_seconds,
            explicit_class=explicit_class,
        ).profile

    def select_profile_explained(
        self,
        prompt: str,
        *,
        video_seconds: float | None = None,
        explicit_class: str | None = None,
    ) -> VlmRoutingDecision:
        """Select profile + return the full decision record for TRUST-2.

        The return value carries enough information for a Receipt
        sidebar to render: which profile, which task class, which
        rule_id, which exact regex matched (for heuristic decisions),
        and which override layer (if any) trumped the heuristic.
        """
        # Layer 1: ContextVar override (highest precedence)
        pinned = _vlm_quality_var.get()
        if pinned is not None:
            profile = VLM_PROFILES.get(pinned)
            if profile is not None:
                return VlmRoutingDecision(
                    profile=profile,
                    task_class=VlmTaskClass.QUICK_DESCRIBE,  # not used
                    rule_id="vlm_override_contextvar",
                    rule_source="contextvars._vlm_quality",
                    matched_pattern=None,
                    overridden_by="context_var",
                )

        # Layer 2: Config override
        config_default = None
        if self._config is not None:
            try:
                config_default = getattr(self._config.vllm, "quality_default", None)
            except AttributeError:
                config_default = None
        if config_default and config_default in VLM_PROFILES:
            profile = VLM_PROFILES[config_default]
            return VlmRoutingDecision(
                profile=profile,
                task_class=VlmTaskClass.QUICK_DESCRIBE,  # not used
                rule_id="vlm_override_config",
                rule_source="config.vllm.quality_default",
                matched_pattern=None,
                overridden_by="config",
            )

        # Layer 3: heuristic classification
        task_class = classify_vlm_task(
            prompt,
            video_seconds=video_seconds,
            explicit_class=explicit_class,
        )
        profile_name = _DEFAULT_TASK_PROFILE_MAP[task_class]
        profile = VLM_PROFILES[profile_name]
        matched = self._first_match_for(task_class, prompt or "", video_seconds)

        return VlmRoutingDecision(
            profile=profile,
            task_class=task_class,
            rule_id=f"vlm_heuristic_{task_class.value}",
            rule_source="vlm_router.classify_vlm_task",
            matched_pattern=matched,
            overridden_by="explicit" if explicit_class else None,
        )

    @staticmethod
    def _first_match_for(
        task_class: VlmTaskClass,
        prompt: str,
        video_seconds: float | None,
    ) -> str | None:
        """Find the regex/condition that triggered this task class.

        Used for TRUST-2 explainability — a reviewer can see "the
        word ``compare`` matched, and that promoted you to premium".
        Returns ``None`` for default cases (QUICK_DESCRIBE without
        any pattern) so the receipt can read "no rule matched →
        default fast".
        """
        if task_class == VlmTaskClass.LONG_FORM:
            if video_seconds is not None:
                return f"video_seconds={video_seconds:.1f}>60"
            return None

        pattern_groups: dict[VlmTaskClass, tuple[re.Pattern[str], ...]] = {
            VlmTaskClass.FORENSIC: _FORENSIC_PATTERNS,
            VlmTaskClass.MULTI_STEP_REASONING: _REASONING_PATTERNS,
            VlmTaskClass.OCR_DOMINANT: _OCR_PATTERNS,
            VlmTaskClass.DETAILED_ANALYSIS: _DETAILED_PATTERNS,
        }
        for pattern in pattern_groups.get(task_class, ()):
            match = pattern.search(prompt)
            if match:
                return match.group(0)

        if task_class == VlmTaskClass.DETAILED_ANALYSIS:
            words = len(prompt.split())
            if words >= _LONG_PROMPT_WORD_THRESHOLD:
                return f"prompt_words={words}>={_LONG_PROMPT_WORD_THRESHOLD}"

        return None


__all__ = [
    "VLM_PROFILES",
    "VlmProfile",
    "VlmRouter",
    "VlmRoutingDecision",
    "VlmTaskClass",
    "classify_vlm_task",
]
