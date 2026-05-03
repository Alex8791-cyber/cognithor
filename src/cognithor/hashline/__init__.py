# Copyright 2024-2026 Cognithor Contributors
# Licensed under the Apache License, Version 2.0
"""Hashline Guard — line-level integrity for safe file editing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.hashline.exceptions import (
    BinaryFileError,
    CacheFullError,
    FileTooLargeError,
    HashlineError,
    HashMismatchError,
    MaxRetriesExceededError,
    StaleReadError,
)
from cognithor.hashline.models import (
    EditIntent,
    EditResult,
    HashlinedFile,
    HashlinedLine,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.hashline.audit import HashlineAuditor
    from cognithor.hashline.cache import HashlineCache
    from cognithor.hashline.config import HashlineConfig
    from cognithor.hashline.editor import HashlineEditor
    from cognithor.hashline.formatter import HashlineFormatter
    from cognithor.hashline.hasher import LineHasher
    from cognithor.hashline.recovery import HashlineRecovery
    from cognithor.hashline.tagger import HashlineTagger
    from cognithor.hashline.validator import HashlineValidator


class HashlineGuard:
    """Main facade for all Hashline operations.

    Provides a unified interface for reading files with hash tags,
    editing with hash verification, and automatic recovery on mismatch.
    """

    def __init__(
        self,
        config: HashlineConfig,
        hasher: LineHasher,
        cache: HashlineCache,
        tagger: HashlineTagger,
        validator: HashlineValidator,
        editor: HashlineEditor,
        recovery: HashlineRecovery,
        formatter: HashlineFormatter,
        auditor: HashlineAuditor | None = None,
    ) -> None:
        self._config = config
        self._hasher = hasher
        self._cache = cache
        self._tagger = tagger
        self._validator = validator
        self._editor = editor
        self._recovery = recovery
        self._formatter = formatter
        self._auditor = auditor

    @classmethod
    def create(
        cls,
        config: HashlineConfig | None = None,
        data_dir: Path | None = None,
    ) -> HashlineGuard:
        """Factory method creating all components.

        Args:
            config: HashlineConfig to use. Defaults to HashlineConfig.default().
            data_dir: Base directory for audit and cache files.

        Returns:
            A fully wired HashlineGuard instance.
        """
        from cognithor.hashline.audit import HashlineAuditor
        from cognithor.hashline.cache import HashlineCache
        from cognithor.hashline.config import HashlineConfig as HLConfig
        from cognithor.hashline.editor import HashlineEditor
        from cognithor.hashline.formatter import HashlineFormatter
        from cognithor.hashline.hasher import LineHasher
        from cognithor.hashline.recovery import HashlineRecovery
        from cognithor.hashline.tagger import HashlineTagger
        from cognithor.hashline.validator import HashlineValidator

        cfg = config or HLConfig.default()
        hasher = LineHasher(cfg)
        cache = HashlineCache(cfg)
        tagger = HashlineTagger(hasher, cache, cfg)
        validator = HashlineValidator(hasher, cache, cfg)
        editor = HashlineEditor(validator, cache, hasher, cfg)
        recovery = HashlineRecovery(tagger, editor, cache, cfg)
        formatter = HashlineFormatter()

        auditor = None
        if cfg.audit_enabled:
            auditor = HashlineAuditor(data_dir=data_dir)

        return cls(
            config=cfg,
            hasher=hasher,
            cache=cache,
            tagger=tagger,
            validator=validator,
            editor=editor,
            recovery=recovery,
            formatter=formatter,
            auditor=auditor,
        )

    def read_file(self, path: Path) -> str:
        """Read a file and return tagged output.

        Args:
            path: Path to the file to read.

        Returns:
            Formatted string with hash-tagged line numbers.
        """
        data = self._tagger.read_and_tag(path)
        if self._auditor:
            self._auditor.log_read(path, len(data.lines), "system")
        return self._formatter.format_file(data)

    def read_range(self, path: Path, start: int, end: int) -> str:
        """Read a specific line range and return tagged output.

        Args:
            path: Path to the file.
            start: First line number (1-based, inclusive).
            end: Last line number (1-based, inclusive).

        Returns:
            Formatted string with hash-tagged line numbers for the range.
        """
        data = self._tagger.read_range(path, start, end)
        if self._auditor:
            self._auditor.log_read(path, len(data.lines), "system")
        return self._formatter.format_file(data)

    def edit(self, intent: EditIntent) -> EditResult:
        """Edit a file with validation and automatic recovery on mismatch.

        Args:
            intent: The edit operation to perform.

        Returns:
            EditResult indicating success or failure.
        """
        result = self._recovery.attempt_with_recovery(intent)
        if self._auditor:
            self._auditor.log_edit(result, intent, "system")
        return result

    def edit_batch(self, intents: list[EditIntent]) -> list[EditResult]:
        """Execute a batch of edits.

        Args:
            intents: List of edit intents.

        Returns:
            List of EditResult for each edit.
        """
        results = self._editor.execute_batch(intents)
        if self._auditor:
            for result, intent in zip(results, intents, strict=False):
                self._auditor.log_edit(result, intent, "system")
        return results

    def invalidate(self, path: Path) -> None:
        """Clear cache for a specific file.

        Args:
            path: Path to invalidate.
        """
        self._cache.invalidate(path)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with hits, misses, evictions, and size.
        """
        s = self._cache.stats
        return {
            "hits": s.hits,
            "misses": s.misses,
            "evictions": s.evictions,
            "size": s.size,
        }


__all__ = [
    "BinaryFileError",
    "CacheFullError",
    "EditIntent",
    "EditResult",
    "FileTooLargeError",
    "HashMismatchError",
    "HashlineError",
    "HashlineGuard",
    "HashlinedFile",
    "HashlinedLine",
    "MaxRetriesExceededError",
    "StaleReadError",
]
