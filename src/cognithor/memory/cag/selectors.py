from __future__ import annotations

from typing import Any


class CAGSelector:
    """Decides which memory content qualifies for CAG prefix caching."""

    def select(self, core_memory_text: str) -> list[dict[str, Any]]:
        """Return candidate entries if the core memory is substantial enough.

        A minimum of 50 tokens (words) is required to justify caching.
        """
        if not core_memory_text or not core_memory_text.strip():
            return []

        token_count = len(core_memory_text.split())
        if token_count < 50:
            return []

        return [
            {
                "cache_id": "core_memory",
                "content": core_memory_text,
                "source_tier": "core",
                "priority": 1,
            }
        ]
