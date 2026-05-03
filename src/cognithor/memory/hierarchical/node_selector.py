"""LLM-driven node selection for hierarchical document retrieval."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from cognithor.memory.hierarchical.models import DocumentTree, SelectedNode
from cognithor.memory.hierarchical.prompts import format_selection_prompt

logger = logging.getLogger(__name__)


class LLMNodeSelector:
    """Select relevant nodes from a :class:`DocumentTree` using an LLM."""

    def __init__(
        self,
        llm_fn: Callable[..., Any],
        language: str = "de",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._llm_fn = llm_fn
        self._language = language
        self._timeout = timeout_seconds

    async def select_nodes(
        self,
        query: str,
        tree: DocumentTree,
        max_nodes: int = 5,
        max_tokens_per_node: int = 2000,
    ) -> list[SelectedNode]:
        """Traverse *tree* top-down, selecting nodes via LLM."""
        results: list[SelectedNode] = []
        consecutive_failures = 0

        async def _recurse(node_id: str, depth: int) -> None:
            nonlocal consecutive_failures

            if len(results) >= max_nodes:
                return
            if consecutive_failures >= 3:
                return

            node = tree.nodes[node_id]
            children_ids = node.children_ids

            if not children_ids:
                # Leaf node — add to results
                content = self._trim_content(node.content, max_tokens_per_node)
                results.append(
                    SelectedNode(
                        node=tree.nodes[node_id]._replace(content=content)  # type: ignore[attr-defined]
                        if hasattr(tree.nodes[node_id], "_replace")
                        else tree.nodes[node_id],
                        depth=depth,
                        reasoning="leaf node",
                        score=0.0,
                    )
                )
                return

            # Build children info for prompt
            children_info: list[tuple[str, str, str]] = []
            for cid in children_ids:
                cnode = tree.nodes.get(cid)
                if cnode:
                    children_info.append((cid, cnode.title, cnode.summary))

            if not children_info:
                return

            prompt = format_selection_prompt(query, children_info, self._language)
            response = await self._llm_fn(prompt)

            parsed = self._parse_response(response)
            if parsed is None:
                consecutive_failures += 1
                logger.warning("Failed to parse LLM response (%d/3)", consecutive_failures)
                return

            consecutive_failures = 0
            selected_ids = parsed.get("selected_node_ids", [])
            reasoning = parsed.get("reasoning", "")

            if not selected_ids:
                return

            for sid in selected_ids:
                if len(results) >= max_nodes:
                    break
                if sid not in tree.nodes:
                    continue

                selected_node = tree.nodes[sid]
                if not selected_node.children_ids:
                    # Leaf — add directly
                    content = self._trim_content(selected_node.content, max_tokens_per_node)
                    from dataclasses import replace

                    trimmed = replace(selected_node, content=content)
                    results.append(
                        SelectedNode(
                            node=trimmed,
                            depth=depth + 1,
                            reasoning=reasoning,
                            score=0.0,
                        )
                    )
                else:
                    # Non-leaf — recurse
                    await _recurse(sid, depth + 1)

        root = tree.nodes[tree.root_node_id]
        if root.children_ids:
            await _recurse(tree.root_node_id, 0)
        else:
            # Single root, no children
            content = self._trim_content(root.content, max_tokens_per_node)
            from dataclasses import replace

            trimmed = replace(root, content=content)
            results.append(SelectedNode(node=trimmed, depth=0, reasoning="only node", score=0.0))

        return results[:max_nodes]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, with regex fallback."""
        try:
            return cast("dict[str, Any] | None", json.loads(response))

        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return cast("dict[str, Any] | None", json.loads(match.group()))

            except (json.JSONDecodeError, TypeError):
                pass

        return None

    @staticmethod
    def _trim_content(content: str, max_tokens: int) -> str:
        """Keep first 40% + last 40% of allowed tokens, trim middle."""
        words = content.split()
        # Rough: tokens ~ words * 1.3, so max_words ~ max_tokens / 1.3
        max_words = int(max_tokens / 1.3)
        if len(words) <= max_words:
            return content

        front = int(max_words * 0.4)
        back = int(max_words * 0.4)
        return " ".join(words[:front]) + " [...] " + " ".join(words[-back:])
