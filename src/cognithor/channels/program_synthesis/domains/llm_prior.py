"""``DomainAwareLLMPrior`` — system-prompt + few-shot bank per domain.

Owner-Decision D4 (Sprint-26 memo): one model (Qwen3.6:27B) drives all
LLM-prior calls; per-domain customisation happens via a *system prompt
swap* + *few-shot bank load*. No per-domain models — keeps VRAM free
for embeddings, browser-agent, voice.

A few-shot bank is a JSONL file under
``prompts/pse/<domain>/examples.jsonl`` with one ``FewShotExample`` per
line. The system prompt is at ``prompts/pse/<domain>/system.md``. Both
files are read lazily and cached in-process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.channels.program_synthesis.domains.base import (
        DomainMetadata,
    )

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Few-shot bank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotExample:
    """One (input → expected output) pair for the LLM prior.

    The format is intentionally minimal: a JSON-serialisable input
    object (typically a list of ``{input, output}`` examples), and a
    target program string (the synthesised solution the LLM should
    produce). Optional ``rationale`` lets us add chain-of-thought to
    the prompt without polluting the JSONL schema for non-rationale
    cases.
    """

    input_examples: tuple[dict[str, Any], ...]
    target_program: str
    rationale: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FewShotExample:
        examples = raw.get("input_examples") or raw.get("examples") or []
        if not isinstance(examples, list):
            msg = f"FewShotExample.input_examples must be a list, got {type(examples).__name__}"
            raise ValueError(msg)
        target = raw.get("target_program") or raw.get("program") or ""
        if not isinstance(target, str):
            msg = "FewShotExample.target_program must be a string"
            raise ValueError(msg)
        rationale = raw.get("rationale", "")
        if not isinstance(rationale, str):
            msg = "FewShotExample.rationale must be a string"
            raise ValueError(msg)
        return cls(
            input_examples=tuple(dict(ex) for ex in examples),
            target_program=target,
            rationale=rationale,
        )


class FewShotBank:
    """Lazy reader for ``prompts/pse/<domain>/examples.jsonl``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._examples: tuple[FewShotExample, ...] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[FewShotExample, ...]:
        """Read the JSONL bank into memory once and cache.

        Missing file → empty bank, logged at INFO. Malformed lines
        skip with a warning so a single bad row doesn't kill a synthesis
        run.
        """
        if self._examples is not None:
            return self._examples
        if not self._path.is_file():
            log.info("fewshot_bank_missing", path=str(self._path))
            self._examples = ()
            return self._examples

        out: list[FewShotExample] = []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("fewshot_bank_read_error", path=str(self._path), error=str(exc))
            self._examples = ()
            return self._examples

        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
                out.append(FewShotExample.from_dict(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning(
                    "fewshot_bank_skip_line",
                    path=str(self._path),
                    lineno=lineno,
                    error=str(exc),
                )
        self._examples = tuple(out)
        log.debug(
            "fewshot_bank_loaded",
            path=str(self._path),
            count=len(self._examples),
        )
        return self._examples


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


class DomainAwareLLMPrior:
    """Build a (system_prompt, messages) pair for a synthesis request.

    Single-model + system-prompt-switching policy (Owner-Decision D4).
    The prior **does not** call the LLM itself — it constructs the
    payload that the central VLLMBackend / OllamaClient consumes via
    the existing Sprint-21 wiring. This keeps the prior cheap to test
    (no GPU dependency) and lets the Sprint-22 LLMPriorClient stay
    the single LLM caller.
    """

    def __init__(
        self,
        prompts_root: Path | None = None,
        *,
        max_few_shot: int = 5,
    ) -> None:
        # ``prompts_root`` is configurable so tests can point at a
        # tmp_path bank without writing to the real prompts/pse/ tree.
        self._prompts_root = prompts_root or Path("prompts/pse")
        self._max_few_shot = max_few_shot
        self._bank_cache: dict[str, FewShotBank] = {}

    @property
    def prompts_root(self) -> Path:
        return self._prompts_root

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self, metadata: DomainMetadata) -> str:
        """Return the system prompt for ``metadata.name``.

        Resolution order:
        1. ``prompts/pse/<name>/system.md`` (file content)
        2. fallback string built from the metadata description.
        """
        path = self._prompts_root / metadata.name / "system.md"
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                log.warning(
                    "system_prompt_read_error",
                    domain=metadata.name,
                    error=str(exc),
                )
        return self._fallback_system_prompt(metadata)

    @staticmethod
    def _fallback_system_prompt(metadata: DomainMetadata) -> str:
        """Build a generic system prompt when no file is available."""
        return (
            f"You are the Cognithor Program Synthesis Engine running in "
            f"the {metadata.display_name} domain.\n\n"
            f"{metadata.description}\n\n"
            "Given a list of (input, output) examples, return ONE "
            "deterministic program that reproduces every example. "
            "Output the program as a single JSON object — no prose, "
            "no markdown fence."
        )

    # ------------------------------------------------------------------
    # Few-shot bank
    # ------------------------------------------------------------------

    def few_shot_bank(self, metadata: DomainMetadata) -> FewShotBank:
        """Return a (cached) :class:`FewShotBank` for ``metadata``."""
        if metadata.name in self._bank_cache:
            return self._bank_cache[metadata.name]
        path = self._prompts_root / metadata.name / "examples.jsonl"
        if metadata.few_shot_bank_path:
            # Allow metadata to override the canonical location, e.g.
            # for shared banks across related domains.
            path = Path(metadata.few_shot_bank_path)
        bank = FewShotBank(path)
        self._bank_cache[metadata.name] = bank
        return bank

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_messages(
        self,
        metadata: DomainMetadata,
        user_examples: Iterable[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Return chat-completion-style messages for the given request.

        The shape matches both the Ollama and vLLM /chat/completions
        bodies the rest of Cognithor uses, so callers can hand the
        return value to the existing LLMBackend layer unchanged.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt(metadata)},
        ]
        bank = self.few_shot_bank(metadata).load()
        for shot in bank[: self._max_few_shot]:
            user_block = json.dumps(
                {"examples": list(shot.input_examples)},
                ensure_ascii=False,
            )
            assistant_block: str
            if shot.rationale:
                assistant_block = f"<reasoning>{shot.rationale}</reasoning>\n{shot.target_program}"
            else:
                assistant_block = shot.target_program
            messages.append({"role": "user", "content": user_block})
            messages.append({"role": "assistant", "content": assistant_block})

        user_payload = json.dumps({"examples": list(user_examples)}, ensure_ascii=False)
        messages.append({"role": "user", "content": user_payload})
        return messages
