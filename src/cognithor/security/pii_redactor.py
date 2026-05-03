"""Local PII redaction for outbound LLM messages.

Runs inside the LLM client wrapper before a chat() request reaches the
provider. Default-off; enable via ``security.pii_redactor.enabled``.

Philosophy: local-first, zero telemetry — no external service calls.
Regex baseline handles ~70% of cases (emails, phone numbers, API keys,
credit cards, SSNs, IBANs, private-key blocks). Optional spaCy NER mode
handles names/orgs/locations when the user installs ``spacy`` + a model.

See also:
    - Issue #122 — community design discussion
    - ``docs/superpowers/specs/2026-04-22-pii-redactor-design.md`` (if
      the design brief is later committed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)


Category = Literal[
    "email",
    "phone",
    "api_key",
    "credit_card",
    "ssn",
    "iban",
    "private_key",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Each pattern is intentionally conservative — better false-negative than
# false-positive. A false positive would mangle legitimate user text.
# ---------------------------------------------------------------------------

_PATTERNS: dict[Category, re.Pattern[str]] = {
    # Simplified RFC 5322 — matches 99% of real-world addresses.
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # E.164 international or common local formats with 7+ digits so
    # short numerics (years, order IDs) don't trip it.
    "phone": re.compile(
        r"""(?x)
        (?<!\w)                        # left word boundary
        (?:
            \+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4})?
            |
            \(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}
        )
        (?!\w)                         # right word boundary
        """
    ),
    # Common API-key shapes. Keep the list anchored to known prefixes
    # to avoid redacting random 40-char base64 blobs in code reviews.
    "api_key": re.compile(
        r"""(?x)
        \b(?:
            sk-[A-Za-z0-9-]{20,}           # OpenAI (incl. sk-proj-*), Anthropic
            | sk-ant-[A-Za-z0-9-]{20,}     # Anthropic explicit
            | gho_[A-Za-z0-9]{30,}         # GitHub OAuth
            | ghp_[A-Za-z0-9]{30,}         # GitHub personal access
            | ghs_[A-Za-z0-9]{30,}         # GitHub server-to-server
            | github_pat_[A-Za-z0-9_]{70,} # GitHub fine-grained PAT
            | AKIA[0-9A-Z]{16}             # AWS access key ID
            | AIza[0-9A-Za-z_-]{35,}       # Google API key (>=39 chars total)
            | xox[baprs]-[A-Za-z0-9-]{10,} # Slack tokens
            | hf_[A-Za-z0-9]{30,}          # Hugging Face tokens
        )\b
        """
    ),
    # 13-19 digit sequence with optional separators, Luhn-validated
    # downstream to rule out false positives.
    "credit_card": re.compile(r"(?<!\d)(?:\d[\s-]?){12,18}\d(?!\d)"),
    # US SSN — 3-2-4 with hyphens.
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    # European IBAN: 2-letter country + 2 check digits + up to 30 alnum.
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    # PEM-encoded private-key blocks — single-line & multi-line.
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        re.MULTILINE,
    ),
}


@dataclass(frozen=True)
class RedactionMatch:
    """One thing we redacted. Returned for logging/auditing, never the raw value."""

    category: Category
    start: int
    end: int
    length: int


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum for credit-card candidates."""
    only_digits = [int(d) for d in digits if d.isdigit()]
    if len(only_digits) < 13 or len(only_digits) > 19:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(only_digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class PIIRedactor:
    """Regex-based redactor. Spacy NER is a future mode (see ``mode``).

    Thread-safe: stateless instance methods, compiled patterns shared.
    """

    def __init__(
        self,
        categories: Iterable[Category] | None = None,
        replacement_template: str = "[REDACTED:{category}]",
    ) -> None:
        self._categories: list[Category] = (
            list(categories) if categories is not None else list(_PATTERNS.keys())
        )
        self._replacement_template = replacement_template

    def redact(self, text: str) -> tuple[str, list[RedactionMatch]]:
        """Redact PII from ``text``. Returns (sanitized_text, matches).

        Matches are returned in original-text offset order, useful for
        logging. The sanitized text uses the replacement template.
        """
        if not text or not self._categories:
            return text, []

        # First pass: collect all match spans per category.
        raw_matches: list[RedactionMatch] = []
        for cat in self._categories:
            pattern = _PATTERNS.get(cat)
            if pattern is None:
                continue
            for m in pattern.finditer(text):
                if cat == "credit_card" and not _luhn_valid(m.group(0)):
                    continue
                raw_matches.append(
                    RedactionMatch(
                        category=cat,
                        start=m.start(),
                        end=m.end(),
                        length=m.end() - m.start(),
                    )
                )

        if not raw_matches:
            return text, []

        # Sort by start asc, then by length desc so longer overlapping
        # matches win (e.g., a credit card that happens to overlap with
        # a phone number pattern).
        raw_matches.sort(key=lambda m: (m.start, -m.length))

        # Drop overlaps: if this match starts inside a previously kept
        # match, skip it.
        kept: list[RedactionMatch] = []
        last_end = -1
        for m in raw_matches:
            if m.start >= last_end:
                kept.append(m)
                last_end = m.end

        # Build sanitized output by slicing around kept matches.
        out: list[str] = []
        cursor = 0
        for m in kept:
            out.append(text[cursor : m.start])
            out.append(self._replacement_template.format(category=m.category))
            cursor = m.end
        out.append(text[cursor:])

        return "".join(out), kept

    def redact_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], list[RedactionMatch]]:
        """Redact PII from all ``content`` fields in a chat-message list.

        Returns a new list (does not mutate the input) and a flat list
        of every match across all messages.
        """
        if not messages or not self._categories:
            return messages, []

        out_messages: list[dict[str, str]] = []
        all_matches: list[RedactionMatch] = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                out_messages.append(msg)
                continue
            sanitized, matches = self.redact(content)
            if matches:
                new_msg = dict(msg)
                new_msg["content"] = sanitized
                out_messages.append(new_msg)
                all_matches.extend(matches)
            else:
                out_messages.append(msg)
        return out_messages, all_matches
