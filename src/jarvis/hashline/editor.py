# Copyright 2024-2026 Cognithor Contributors
# Licensed under the Apache License, Version 2.0
"""Hash-gesicherte Edit-Operationen."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from jarvis.hashline.cache import HashlineCache
from jarvis.hashline.config import HashlineConfig
from jarvis.hashline.hasher import LineHasher
from jarvis.hashline.models import EditIntent, EditResult
from jarvis.hashline.validator import HashlineValidator
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class HashlineEditor:
    """Performs hash-verified file edit operations.

    All edits are atomic (write to temp file, then os.replace) and preserve
    the original file's encoding, newline style, and permissions.

    Args:
        validator: HashlineValidator for pre-edit validation.
        cache: HashlineCache for invalidation and rebuild.
        hasher: LineHasher for recomputing hashes after edit.
        config: HashlineConfig for settings.
    """

    def __init__(
        self,
        validator: HashlineValidator,
        cache: HashlineCache,
        hasher: LineHasher,
        config: HashlineConfig,
    ) -> None:
        self._validator = validator
        self._cache = cache
        self._hasher = hasher
        self._config = config

    def execute_edit(self, intent: EditIntent) -> EditResult:
        """Execute a single edit with validation.

        Steps:
            1. Validate the intent's hash against the current file.
            2. If invalid, return a failed EditResult.
            3. Perform the edit with atomic write.
            4. Invalidate and rebuild cache.
            5. Return success EditResult.

        Args:
            intent: The edit operation to perform.

        Returns:
            EditResult indicating success or failure.
        """
        resolved = intent.file_path.resolve()

        # Validate
        vr = self._validator.validate_edit(intent)
        if not vr.valid:
            return EditResult(
                success=False,
                operation=intent.operation,
                file_path=resolved,
                line_number=intent.target_line,
                old_content=vr.current_content,
                new_content=intent.new_content,
                audit_hash="",
                error=vr.reason,
            )

        # Read file, detect encoding and newline style
        raw_bytes, encoding, newline = self._read_raw(resolved)
        lines = self._split_lines(raw_bytes.decode(encoding), newline)

        old_content = lines[intent.target_line - 1] if intent.target_line <= len(lines) else None

        # Apply operation
        lines = self._apply_operation(lines, intent)

        # Write atomically
        self._atomic_write(resolved, lines, encoding, newline)

        # Cache invalidate + rebuild
        self._cache.invalidate(resolved)
        self._rebuild_cache(resolved)

        # Compute audit hash
        audit_hash = self._hasher.hash_file(resolved)

        log.info(
            "edit_applied",
            path=str(resolved),
            line=intent.target_line,
            op=intent.operation,
        )

        return EditResult(
            success=True,
            operation=intent.operation,
            file_path=resolved,
            line_number=intent.target_line,
            old_content=old_content,
            new_content=intent.new_content,
            audit_hash=audit_hash,
        )

    def execute_batch(self, intents: list[EditIntent]) -> list[EditResult]:
        """Execute multiple edits as one atomic write.

        Edits are sorted by line number descending (highest first) so that
        insertions and deletions do not shift the line numbers of subsequent
        edits.

        Args:
            intents: List of edit intents to apply.

        Returns:
            List of EditResult in the same order as the sorted intents.
        """
        if not intents:
            return []

        # Sort highest line first
        sorted_intents = sorted(intents, key=lambda i: i.target_line, reverse=True)

        # Validate all first
        results: list[EditResult] = []
        all_valid = True
        for intent in sorted_intents:
            vr = self._validator.validate_edit(intent)
            if not vr.valid:
                all_valid = False
                results.append(
                    EditResult(
                        success=False,
                        operation=intent.operation,
                        file_path=intent.file_path.resolve(),
                        line_number=intent.target_line,
                        old_content=vr.current_content,
                        new_content=intent.new_content,
                        audit_hash="",
                        error=vr.reason,
                    )
                )
            else:
                results.append(None)  # type: ignore[arg-type]  # placeholder

        if not all_valid:
            # Fill in placeholders with failures
            final = []
            for i, r in enumerate(results):
                if r is not None:
                    final.append(r)
                else:
                    intent = sorted_intents[i]
                    final.append(
                        EditResult(
                            success=False,
                            operation=intent.operation,
                            file_path=intent.file_path.resolve(),
                            line_number=intent.target_line,
                            old_content=None,
                            new_content=intent.new_content,
                            audit_hash="",
                            error="Batch aborted: one or more intents failed validation",
                        )
                    )
            return final

        # All valid — apply all edits in one atomic write
        resolved = sorted_intents[0].file_path.resolve()
        raw_bytes, encoding, newline = self._read_raw(resolved)
        lines = self._split_lines(raw_bytes.decode(encoding), newline)

        final_results: list[EditResult] = []
        for intent in sorted_intents:
            old_content = (
                lines[intent.target_line - 1] if intent.target_line <= len(lines) else None
            )
            lines = self._apply_operation(lines, intent)
            final_results.append(
                EditResult(
                    success=True,
                    operation=intent.operation,
                    file_path=resolved,
                    line_number=intent.target_line,
                    old_content=old_content,
                    new_content=intent.new_content,
                    audit_hash="",  # will be set after write
                )
            )

        # Atomic write
        self._atomic_write(resolved, lines, encoding, newline)

        # Cache invalidate + rebuild
        self._cache.invalidate(resolved)
        self._rebuild_cache(resolved)

        # Set audit hash
        audit_hash = self._hasher.hash_file(resolved)
        for r in final_results:
            r.audit_hash = audit_hash

        return final_results

    @staticmethod
    def _apply_operation(lines: list[str], intent: EditIntent) -> list[str]:
        """Apply a single edit operation to a list of lines.

        Args:
            lines: Mutable list of line strings.
            intent: The edit to apply.

        Returns:
            The modified list of lines.
        """
        idx = intent.target_line - 1  # convert to 0-based

        if intent.operation == "replace":
            if 0 <= idx < len(lines):
                lines[idx] = intent.new_content or ""
        elif intent.operation == "insert_after":
            insert_pos = min(idx + 1, len(lines))
            lines.insert(insert_pos, intent.new_content or "")
        elif intent.operation == "insert_before":
            insert_pos = max(idx, 0)
            lines.insert(insert_pos, intent.new_content or "")
        elif intent.operation == "delete":
            if 0 <= idx < len(lines):
                lines.pop(idx)

        return lines

    @staticmethod
    def _read_raw(path: Path) -> tuple[bytes, str, str]:
        """Read raw bytes and detect encoding and newline style.

        Returns:
            Tuple of (raw_bytes, encoding, newline_char).
        """
        raw = path.read_bytes()

        # Detect newline style
        if b"\r\n" in raw:
            newline = "\r\n"
        else:
            newline = "\n"

        # Detect encoding
        try:
            raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                raw.decode("latin-1")
                encoding = "latin-1"
            except UnicodeDecodeError:
                encoding = "utf-8"

        return raw, encoding, newline

    @staticmethod
    def _split_lines(content: str, newline: str) -> list[str]:
        """Split content into lines using the detected newline style.

        Args:
            content: File content as string.
            newline: The newline character(s) to split on.

        Returns:
            List of line strings without newline terminators.
        """
        # Use splitlines to handle mixed endings gracefully
        if newline == "\r\n":
            return content.split("\r\n")
        return content.split("\n")

    @staticmethod
    def _atomic_write(
        path: Path,
        lines: list[str],
        encoding: str,
        newline: str,
    ) -> None:
        """Write lines to a file atomically, preserving encoding and newlines.

        Uses a temporary file + os.replace for atomicity. Preserves
        file permissions via shutil.copystat.

        Args:
            path: Target file path.
            lines: Lines to write.
            encoding: Encoding to use.
            newline: Newline character(s) to join with.
        """
        content = newline.join(lines)
        raw_bytes = content.encode(encoding)

        # Preserve permissions
        try:
            original_stat = path.stat()
            has_stat = True
        except OSError:
            has_stat = False

        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".hashline_edit_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Restore permissions
        if has_stat:
            try:
                os.chmod(str(path), original_stat.st_mode)
            except OSError:
                pass

    def _rebuild_cache(self, path: Path) -> None:
        """Re-read and cache the file after an edit."""
        try:
            from jarvis.hashline.tagger import HashlineTagger

            tagger = HashlineTagger(self._hasher, self._cache, self._config)
            tagger.read_and_tag(path)
        except Exception:
            log.debug("cache_rebuild_failed", path=str(path), exc_info=True)
