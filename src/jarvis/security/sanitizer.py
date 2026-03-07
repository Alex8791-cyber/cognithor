"""Input-Sanitizer: Schutz gegen Prompt-Injection und Path-Traversal.

Prüft und bereinigt alle externen Inhalte, bevor sie dem Planner
(LLM) vorgelegt werden. Externe Inhalte werden in
<external_content>-Tags gewrappt, bekannte Injection-Patterns
werden entfernt oder neutralisiert.

Zusätzlich: Validierung von Voice-/Modellnamen gegen Path-Traversal
(CWE-22) — verhindert das Einschleusen von ../../-Sequenzen in
Dateinamen, die als Pfade verwendet werden (z.B. Piper TTS Modelle).

Sicherheitsgarantien:
  - Externe Inhalte sind IMMER markiert
  - Bekannte Injection-Patterns werden neutralisiert
  - Instruction-Hierarchie wird durchgesetzt
  - System-Prompts sind nicht überschreibbar
  - Voice-/Modellnamen sind gegen Path-Traversal validiert

Bibel-Referenz: §11.3 (Input-Sanitization)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jarvis.models import SanitizeResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Voice/Model Name Validation (CWE-22 Path Traversal Prevention)
# ============================================================================

# Piper voice names follow pattern: lang_REGION-name-quality
# e.g. "de_DE-thorsten-high", "de_DE-thorsten_emotional-medium", "en_US-lessac-low"
# Allowed: ASCII letters, digits, underscores, hyphens, dots (for version suffixes)
# Blocked: path separators (/ \), parent traversal (..), null bytes, spaces, special chars
_VOICE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$")

# Maximum length for voice/model names (prevents abuse via extremely long names)
_MAX_VOICE_NAME_LENGTH = 128


def validate_voice_name(voice: str) -> str:
    """Validates a voice/model name against path traversal attacks (CWE-22).

    Ensures the name contains only safe characters and cannot be used
    to construct paths outside the intended directory.

    Args:
        voice: The voice/model name to validate.

    Returns:
        The validated voice name (unchanged if valid).

    Raises:
        ValueError: If the voice name is invalid or potentially malicious.
    """
    if not voice:
        raise ValueError("Voice name must not be empty")

    if len(voice) > _MAX_VOICE_NAME_LENGTH:
        raise ValueError(
            f"Voice name too long ({len(voice)} chars, max {_MAX_VOICE_NAME_LENGTH})"
        )

    # Null byte injection check (can bypass string checks in C-level path APIs)
    if "\x00" in voice:
        log.warning("voice_name_null_byte_blocked", voice=repr(voice))
        raise ValueError("Voice name contains null byte")

    # Path separator check (explicit, before regex, for clear error message)
    if "/" in voice or "\\" in voice:
        log.warning("voice_name_path_separator_blocked", voice=voice)
        raise ValueError(
            f"Voice name contains path separator: {voice!r}"
        )

    # Parent directory traversal check
    if ".." in voice:
        log.warning("voice_name_traversal_blocked", voice=voice)
        raise ValueError(
            f"Voice name contains directory traversal sequence: {voice!r}"
        )

    # Whitelist regex: only safe characters
    if not _VOICE_NAME_PATTERN.match(voice):
        log.warning("voice_name_invalid_chars_blocked", voice=voice)
        raise ValueError(
            f"Voice name contains invalid characters: {voice!r}. "
            f"Allowed: letters, digits, underscores, hyphens, dots."
        )

    return voice


def validate_model_path_containment(
    model_path: Path,
    allowed_dir: Path,
) -> Path:
    """Defense-in-depth: verifies a constructed model path stays within its directory.

    Even after voice name validation, this ensures the resolved path
    is contained within the expected directory. Catches edge cases
    like symlink escapes or OS-specific path normalization quirks.

    Args:
        model_path: The constructed model file path.
        allowed_dir: The directory the path must reside in.

    Returns:
        The resolved, validated path.

    Raises:
        ValueError: If the resolved path escapes the allowed directory.
    """
    try:
        resolved = model_path.resolve()
        allowed_resolved = allowed_dir.resolve()
        resolved.relative_to(allowed_resolved)
    except (ValueError, OSError) as exc:
        log.warning(
            "model_path_containment_violation",
            model_path=str(model_path),
            allowed_dir=str(allowed_dir),
        )
        raise ValueError(
            f"Model path {model_path} escapes allowed directory {allowed_dir}"
        ) from exc
    return resolved


# ============================================================================
# Injection-Pattern-Definitionen
# ============================================================================


@dataclass(frozen=True)
class InjectionPattern:
    """Ein erkanntes Prompt-Injection-Pattern."""

    name: str
    pattern: re.Pattern[str]
    severity: str = "high"  # high | medium | low


# Bekannte Prompt-Injection-Patterns
_INJECTION_PATTERNS: list[InjectionPattern] = [
    # Direkte Instruction-Override
    InjectionPattern(
        name="system_override",
        pattern=re.compile(
            r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions?|prompts?|rules?|context)",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    InjectionPattern(
        name="new_instructions",
        pattern=re.compile(
            r"(?:new|updated|revised)\s+(?:system\s+)?instructions?:?\s",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    InjectionPattern(
        name="role_switch",
        pattern=re.compile(
            r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|evil|unrestricted)",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    InjectionPattern(
        name="jailbreak_dan",
        pattern=re.compile(
            r"(?:DAN|do\s+anything\s+now|developer\s+mode|DUDE)\s*(?:mode|prompt)?",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    # System-Prompt Extraktion
    InjectionPattern(
        name="prompt_leak",
        pattern=re.compile(
            r"(?:print|show|reveal|repeat|output|display)\s+(?:your\s+)?"
            r"(?:system\s+)?(?:prompt|instructions?|rules?|configuration)",
            re.IGNORECASE,
        ),
        severity="medium",
    ),
    # XML/Tag Injection
    InjectionPattern(
        name="xml_injection",
        pattern=re.compile(
            r"<\s*/?(?:system|assistant|user|instruction|prompt|tool_result)\s*>",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    # Encoding-Umgehung
    InjectionPattern(
        name="base64_injection",
        pattern=re.compile(
            r"(?:decode|eval|execute)\s+(?:this\s+)?base64",
            re.IGNORECASE,
        ),
        severity="medium",
    ),
    # Delimiter-Confusion
    InjectionPattern(
        name="delimiter_escape",
        pattern=re.compile(
            r"```\s*(?:system|end_turn|human_turn|<\|)",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    # Community-Skill-spezifisch: Tool-Override-Versuche
    InjectionPattern(
        name="tools_required_bypass",
        pattern=re.compile(
            r"(?:ignore|bypass|skip|disable)\s+(?:the\s+)?tools?_required",
            re.IGNORECASE,
        ),
        severity="high",
    ),
    InjectionPattern(
        name="gatekeeper_bypass",
        pattern=re.compile(
            r"(?:ignore|bypass|skip|disable)\s+(?:the\s+)?gatekeeper",
            re.IGNORECASE,
        ),
        severity="high",
    ),
]


# ============================================================================
# Sanitizer
# ============================================================================


class InputSanitizer:
    """Sanitizer für externe Inhalte. [B§11.3]

    Prüft Text auf bekannte Prompt-Injection-Patterns und
    neutralisiert sie. Externe Inhalte werden in sichere
    Tags gewrappt.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[InjectionPattern] | None = None,
        strict: bool = True,
    ) -> None:
        """Initialisiert den Sanitizer.

        Args:
            extra_patterns: Zusätzliche Injection-Patterns.
            strict: Bei 'high' Severity blocken statt neutralisieren.
        """
        self._patterns = list(_INJECTION_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        self._strict = strict
        self._stats = _SanitizerStats()

    def sanitize_external(self, text: str, source: str = "unknown") -> SanitizeResult:
        """Sanitisiert externen Content und wrappt ihn.

        Externe Inhalte (Web-Scraping, Tool-Output, Datei-Inhalt)
        werden in <external_content>-Tags gewrappt und auf
        Injection-Patterns geprüft.

        Args:
            text: Der externe Text.
            source: Herkunft des Textes (z.B. 'web', 'file', 'tool').

        Returns:
            SanitizeResult mit bereinigtem Text.
        """
        if not text:
            return SanitizeResult(
                original_length=0,
                sanitized_length=0,
                was_modified=False,
                sanitized_text="",
            )

        found_patterns: list[str] = []

        # 0. Unicode normalization — prevents bypass via zero-width chars,
        #    homoglyphs, and NFKD decomposition tricks
        import unicodedata
        sanitized = unicodedata.normalize("NFKC", text)
        # Strip zero-width characters that could hide injection payloads
        sanitized = sanitized.replace("\u200b", "").replace("\u200c", "").replace(
            "\u200d", "").replace("\ufeff", "").replace("\u00ad", "")

        # 1. Injection-Patterns prüfen
        for ip in self._patterns:
            matches = ip.pattern.findall(sanitized)
            if matches:
                found_patterns.append(f"{ip.name}({ip.severity})")
                self._stats.patterns_detected += 1

                if ip.severity == "high" and self._strict:
                    # High-Severity: Match komplett entfernen
                    sanitized = ip.pattern.sub("[BLOCKED_INJECTION]", sanitized)
                else:
                    # Medium/Low: In Kommentar neutralisieren
                    sanitized = ip.pattern.sub(
                        lambda m, _ip=ip: f"[NEUTRALIZED: {_ip.name}]",  # type: ignore[misc]
                        sanitized,
                    )

        # 2. XML-Tags neutralisieren (außer erlaubte)
        sanitized = self._neutralize_xml_tags(sanitized)

        # 3. In external_content-Tag wrappen
        safe_source = source.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        wrapped = f'<external_content source="{safe_source}">\n{sanitized}\n</external_content>'

        was_modified = sanitized != text
        if found_patterns:
            self._stats.texts_modified += 1
            log.warning(
                "injection_patterns_detected",
                source=source,
                patterns=found_patterns,
                strict=self._strict,
            )

        self._stats.texts_processed += 1

        return SanitizeResult(
            original_length=len(text),
            sanitized_length=len(wrapped),
            patterns_found=found_patterns,
            was_modified=was_modified,
            sanitized_text=wrapped,
        )

    def sanitize_user_input(self, text: str) -> SanitizeResult:
        """Sanitisiert User-Input (weniger strikt).

        User-Input wird NICHT gewrappt (ist vertrauenswürdig),
        aber auf XML-Tag-Injection geprüft, die die
        Instruction-Hierarchie brechen könnte.

        Args:
            text: Der User-Input.

        Returns:
            SanitizeResult.
        """
        if not text:
            return SanitizeResult(
                original_length=0,
                sanitized_length=0,
                was_modified=False,
                sanitized_text="",
            )

        found_patterns: list[str] = []
        sanitized = text

        # Nur XML-Tag-Injection prüfen (User darf sonst alles)
        xml_pattern = next((p for p in self._patterns if p.name == "xml_injection"), None)
        if xml_pattern and xml_pattern.pattern.search(sanitized):
            found_patterns.append("xml_injection(high)")
            sanitized = xml_pattern.pattern.sub("[TAG_REMOVED]", sanitized)

        self._stats.texts_processed += 1
        was_modified = sanitized != text
        if was_modified:
            self._stats.texts_modified += 1

        return SanitizeResult(
            original_length=len(text),
            sanitized_length=len(sanitized),
            patterns_found=found_patterns,
            was_modified=was_modified,
            sanitized_text=sanitized,
        )

    def scan_only(self, text: str) -> list[str]:
        """Scannt Text auf Injection-Patterns ohne Modifikation.

        Args:
            text: Zu scannender Text.

        Returns:
            Liste gefundener Pattern-Namen.
        """
        if not text:
            return []
        found: list[str] = []
        for ip in self._patterns:
            if ip.pattern.search(text):
                found.append(f"{ip.name}({ip.severity})")
        return found

    @property
    def stats(self) -> dict[str, int]:
        """Statistiken des Sanitizers."""
        return {
            "texts_processed": self._stats.texts_processed,
            "texts_modified": self._stats.texts_modified,
            "patterns_detected": self._stats.patterns_detected,
        }

    def _neutralize_xml_tags(self, text: str) -> str:
        """Neutralisiert gefährliche XML-Tags.

        Erlaubt: <b>, <i>, <code>, <pre>, <br>, <p>, <ul>, <li>, <ol>
        Blockiert: <system>, <assistant>, <user>, <instruction>,
                   <tool_result>, <prompt>
        """
        dangerous = re.compile(
            r"<\s*/?(?:system|assistant|user|instruction|prompt|"
            r"tool_result|tool_use|function_call|human_turn|ai_turn)\s*"
            r"(?:\s+[^>]*)?>",
            re.IGNORECASE,
        )
        return dangerous.sub(
            lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
            text,
        )


@dataclass
class _SanitizerStats:
    """Interne Statistiken."""

    texts_processed: int = 0
    texts_modified: int = 0
    patterns_detected: int = 0
