"""``Domain`` Protocol + ``DomainMetadata`` data class (Sprint-26 §26.1).

Each Sprint-26 domain (SQL, JSON, Datetime, AST, BinaryData, Float,
Image) implements the ``Domain`` protocol and is registered with
``DomainRegistry``. The protocol is intentionally minimal so that
adding a new domain in Sprint-27+ is a one-file change plus a
fixture-load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cognithor.channels.program_synthesis.dsl.registry import (
        PrimitiveRegistry,
    )


class DomainCapability(StrEnum):
    """Capability bits a domain may declare on registration.

    The capability set drives which pipeline stages can run for a
    domain. ``synthesise`` is the bare minimum (all Sprint-22 domains
    have it). ``execute`` is required for any domain whose verifier
    actually runs the synthesised program (SQL, AST). ``property``
    means the domain ships hypothesis-based property tests beyond
    example-equality.
    """

    SYNTHESISE = "synthesise"
    EXECUTE = "execute"
    PROPERTY = "property"
    BRIDGE = "bridge"
    SANDBOX = "sandbox"


@dataclass(frozen=True)
class DomainMetadata:
    """Static metadata describing one domain.

    Frozen so that the registry can hash it for de-duplication and
    treat it as a stable description across the synthesis run.

    Attributes
    ----------
    name:
        Stable lowercase identifier used as registry key, prompts/
        directory, and scorecard column. Must match
        ``[a-z][a-z0-9_]*``.
    display_name:
        Human-readable name for UI / scorecard headers.
    description:
        One-line description used by the LLM prior's system prompt.
    capabilities:
        Capability bits — see :class:`DomainCapability`.
    type_tags:
        Set of canonical type-tag strings the domain introduces. Used
        by :mod:`bridges` to type-check cross-domain compositions.
    benchmark_name:
        Externally-recognised benchmark this domain is scored against
        (``"spider-easy"``, ``"humaneval-plus"``, …). Empty for
        domains that only have a custom suite.
    benchmark_target:
        Target score on ``benchmark_name`` at sprint-end. Stored as a
        normalised 0..1 float (e.g. ``0.30`` for "30 % EX on Spider").
    few_shot_bank_path:
        Filesystem path (relative to repo root) of the JSONL few-shot
        bank used by :class:`DomainAwareLLMPrior`. Empty if the domain
        has no LLM prior wired yet.
    """

    name: str
    display_name: str
    description: str
    capabilities: frozenset[DomainCapability]
    type_tags: frozenset[str] = field(default_factory=frozenset)
    benchmark_name: str = ""
    benchmark_target: float = 0.0
    few_shot_bank_path: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid domain name: {self.name!r}"
            raise ValueError(msg)
        if self.name != self.name.lower():
            msg = f"Domain name must be lowercase: {self.name!r}"
            raise ValueError(msg)
        if not 0.0 <= self.benchmark_target <= 1.0:
            msg = f"benchmark_target must be in [0.0, 1.0], got {self.benchmark_target}"
            raise ValueError(msg)


@runtime_checkable
class Domain(Protocol):
    """Protocol every Sprint-26 domain implements.

    A domain bundles the *static* shape of a synthesis target. The
    protocol is intentionally read-only — verifiers and runtime caches
    live on separate objects so the registry can stay free of mutable
    per-call state.
    """

    @property
    def metadata(self) -> DomainMetadata:
        """Return the immutable metadata for this domain."""

    def primitives(self) -> PrimitiveRegistry:
        """Return the primitive registry scoped to this domain."""

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Return True when ``program`` reproduces every example.

        Implementations may execute the program (SQL, AST sandboxes),
        evaluate it symbolically (Datetime arithmetic), or run a
        property suite. Return value is binary; a failing verifier
        must raise to surface the *reason* via the audit log.
        """


def domain_relative_path(metadata: DomainMetadata, *parts: str) -> Path:
    """Helper: build ``prompts/pse/<name>/<parts>`` from metadata.

    Centralised so that prompt-bank lookups stay consistent across
    LLM-prior, scorecard, and few-shot-export tooling.
    """
    return Path("prompts/pse") / metadata.name / Path(*parts)
