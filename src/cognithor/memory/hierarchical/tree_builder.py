"""Build a DocumentTree from a source file using parsers and LLM summaries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from cognithor.memory.hierarchical.models import DocumentTree, RawSection, TreeNode
from cognithor.memory.hierarchical.parsers import get_parser

logger = logging.getLogger(__name__)


def _token_estimate(text: str) -> int:
    """Rough token count: word count * 1.3."""
    return int(len(text.split()) * 1.3)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class DocumentTreeBuilder:
    """Construct a :class:`DocumentTree` from a source file."""

    def __init__(
        self,
        llm_fn: Callable[..., Any],
        max_parallel_summaries: int = 10,
        node_split_threshold: int = 4000,
        max_branching_factor: int = 50,
        max_depth: int = 8,
    ) -> None:
        self._llm_fn = llm_fn
        self._max_parallel = max_parallel_summaries
        self._split_threshold = node_split_threshold
        self._max_branch = max_branching_factor
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def build(
        self,
        source_path: Path,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DocumentTree:
        doc_id = document_id or uuid.uuid4().hex
        parser = get_parser(source_path)
        parser_name = type(parser).__name__

        content: str | bytes
        if source_path.suffix.lower() in (".pdf", ".docx"):
            content = source_path.read_bytes()
        else:
            content = source_path.read_text(encoding="utf-8", errors="replace")

        sections = parser.parse(content, source_path)

        if not sections:
            # Empty document — single root node
            root_id = f"{doc_id}-root"
            root = TreeNode(
                node_id=root_id,
                document_id=doc_id,
                parent_id=None,
                level=0,
                title=source_path.stem,
                summary="",
                content="",
                content_hash=_content_hash(""),
                token_count=0,
                children_ids=(),
                position=0,
            )
            return DocumentTree(
                document_id=doc_id,
                source_path=source_path,
                source_hash=_content_hash(
                    content
                    if isinstance(content, str)
                    else content.decode("utf-8", errors="replace")
                ),
                title=source_path.stem,
                root_node_id=root_id,
                nodes={root_id: root},
                created_at=datetime.now(UTC),
                parser_used=parser_name,
                total_tokens=0,
            )

        total_text = " ".join(s.content for s in sections)
        total_tokens = _token_estimate(total_text)
        if total_tokens < 100:
            logger.info("Short document (%d tokens): %s", total_tokens, source_path)

        # Build hierarchy from flat sections
        nodes_data = self._build_hierarchy(sections, doc_id)

        # Split large nodes
        nodes_data = self._split_large_nodes(nodes_data, doc_id)

        # Enforce branching factor
        nodes_data = self._enforce_branching(nodes_data, doc_id)

        # Enforce max depth
        nodes_data = self._enforce_depth(nodes_data)

        # Generate summaries bottom-up
        await self._generate_summaries(nodes_data, progress_callback)

        # Build final TreeNode objects
        nodes: dict[str, TreeNode] = {}
        for nd in nodes_data:
            child_ids = tuple(c["node_id"] for c in nodes_data if c["parent_id"] == nd["node_id"])
            nodes[nd["node_id"]] = TreeNode(
                node_id=nd["node_id"],
                document_id=doc_id,
                parent_id=nd["parent_id"],
                level=nd["level"],
                title=nd["title"],
                summary=nd.get("summary", ""),
                content=nd["content"],
                content_hash=_content_hash(nd["content"]),
                token_count=_token_estimate(nd["content"]),
                children_ids=child_ids,
                position=nd["position"],
                page_number=nd.get("page"),
                metadata=metadata or {},
            )

        root_id = nodes_data[0]["node_id"]
        doc_title = sections[0].title if sections else source_path.stem
        source_text = (
            content if isinstance(content, str) else content.decode("utf-8", errors="replace")
        )

        return DocumentTree(
            document_id=doc_id,
            source_path=source_path,
            source_hash=_content_hash(source_text),
            title=doc_title,
            root_node_id=root_id,
            nodes=nodes,
            created_at=datetime.now(UTC),
            parser_used=parser_name,
            total_tokens=sum(n.token_count for n in nodes.values()),
        )

    # ------------------------------------------------------------------
    # Hierarchy construction
    # ------------------------------------------------------------------

    def _build_hierarchy(self, sections: list[RawSection], doc_id: str) -> list[dict[str, Any]]:
        """Convert flat sections into a parent-child hierarchy."""
        root_id = f"{doc_id}-root"
        nodes: list[dict[str, Any]] = [
            {
                "node_id": root_id,
                "parent_id": None,
                "level": 0,
                "title": "Document",
                "content": "",
                "position": 0,
                "page": None,
            }
        ]

        # parent_stack: list of (level, node_id)
        parent_stack: list[tuple[int, str]] = [(0, root_id)]
        position = 1

        for section in sections:
            level = section.level
            if level == 0:
                # Preamble content — attach to root
                nodes[0]["content"] = section.content
                continue

            # Insert virtual headings for level jumps
            if parent_stack:
                current_parent_level = parent_stack[-1][0]
                while level > current_parent_level + 1:
                    current_parent_level += 1
                    virtual_id = f"{doc_id}-virtual-{uuid.uuid4().hex[:8]}"
                    nodes.append(
                        {
                            "node_id": virtual_id,
                            "parent_id": parent_stack[-1][1],
                            "level": current_parent_level,
                            "title": f"(Level {current_parent_level})",
                            "content": "",
                            "position": position,
                            "page": None,
                            "virtual": True,
                        }
                    )
                    parent_stack.append((current_parent_level, virtual_id))
                    position += 1

            # Pop stack to find correct parent
            while len(parent_stack) > 1 and parent_stack[-1][0] >= level:
                parent_stack.pop()

            node_id = f"{doc_id}-{uuid.uuid4().hex[:8]}"
            nodes.append(
                {
                    "node_id": node_id,
                    "parent_id": parent_stack[-1][1],
                    "level": level,
                    "title": section.title,
                    "content": section.content,
                    "position": position,
                    "page": section.page,
                }
            )
            parent_stack.append((level, node_id))
            position += 1

        return nodes

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _split_large_nodes(self, nodes: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
        """Split nodes exceeding the token threshold into parts."""
        result: list[dict[str, Any]] = []
        for nd in nodes:
            tokens = _token_estimate(nd["content"])
            if tokens <= self._split_threshold:
                result.append(nd)
                continue

            # Split content into roughly equal parts
            words = nd["content"].split()
            chunk_size = len(words) // 2
            parts = [
                " ".join(words[:chunk_size]),
                " ".join(words[chunk_size:]),
            ]

            # Original becomes a container
            nd_copy = dict(nd)
            nd_copy["content"] = ""
            result.append(nd_copy)

            for i, part in enumerate(parts, 1):
                part_id = f"{doc_id}-part-{uuid.uuid4().hex[:8]}"
                result.append(
                    {
                        "node_id": part_id,
                        "parent_id": nd["node_id"],
                        "level": nd["level"] + 1,
                        "title": f"{nd['title']} — Part {i}",
                        "content": part,
                        "position": nd["position"],
                        "page": nd.get("page"),
                    }
                )

        return result

    def _enforce_branching(self, nodes: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
        """If a parent has >max_branching_factor children, insert group nodes."""
        # Build children map
        children_map: dict[str, list[dict[str, Any]]] = {}
        node_map: dict[str, dict[str, Any]] = {}
        for nd in nodes:
            node_map[nd["node_id"]] = nd
            pid = nd["parent_id"]
            if pid is not None:
                children_map.setdefault(pid, []).append(nd)

        result: list[dict[str, Any]] = []
        for nd in nodes:
            children = children_map.get(nd["node_id"], [])
            if len(children) <= self._max_branch:
                result.append(nd)
                continue

            result.append(nd)
            # Group children into chunks
            for gi in range(0, len(children), self._max_branch):
                group = children[gi : gi + self._max_branch]
                if gi == 0 and len(children) <= self._max_branch:
                    # No grouping needed
                    break
                group_id = f"{doc_id}-group-{uuid.uuid4().hex[:8]}"
                group_node = {
                    "node_id": group_id,
                    "parent_id": nd["node_id"],
                    "level": nd["level"] + 1,
                    "title": f"Group {gi // self._max_branch + 1}",
                    "content": "",
                    "position": nd["position"],
                    "page": None,
                }
                result.append(group_node)
                for child in group:
                    child["parent_id"] = group_id

        # Add any children that were regrouped (already in nodes list)
        # Re-collect to ensure consistency
        seen = {nd["node_id"] for nd in result}
        for nd in nodes:
            if nd["node_id"] not in seen:
                result.append(nd)

        return result

    def _enforce_depth(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten nodes beyond max_depth into their parent."""
        # Compute depths
        depth_map: dict[str, int] = {}
        node_map: dict[str, dict[str, Any]] = {nd["node_id"]: nd for nd in nodes}

        def _depth(node_id: str) -> int:
            if node_id in depth_map:
                return depth_map[node_id]
            nd = node_map[node_id]
            if nd["parent_id"] is None:
                depth_map[node_id] = 0
                return 0
            d = _depth(nd["parent_id"]) + 1
            depth_map[node_id] = d
            return d

        for nd in nodes:
            _depth(nd["node_id"])

        result: list[dict[str, Any]] = []
        for nd in nodes:
            d = depth_map[nd["node_id"]]
            if d > self._max_depth:
                # Flatten: reparent to grandparent
                parent = node_map.get(nd["parent_id"])
                if parent and parent["parent_id"] is not None:
                    nd["parent_id"] = parent["parent_id"]
                    nd["level"] = node_map[parent["parent_id"]]["level"] + 1
            result.append(nd)

        return result

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    async def _generate_summaries(
        self,
        nodes: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None,
    ) -> None:
        """Generate summaries bottom-up via llm_fn."""
        # Sort by depth (deepest first) for bottom-up order
        node_map: dict[str, dict[str, Any]] = {nd["node_id"]: nd for nd in nodes}

        def _depth(node_id: str) -> int:
            nd = node_map[node_id]
            if nd["parent_id"] is None:
                return 0
            return _depth(nd["parent_id"]) + 1

        depth_sorted = sorted(nodes, key=lambda nd: -_depth(nd["node_id"]))
        total = len(depth_sorted)
        done = 0
        sem = asyncio.Semaphore(self._max_parallel)

        async def _summarize(nd: dict[str, Any]) -> None:
            nonlocal done
            text = nd["content"]
            if not text.strip():
                # Aggregate children summaries for container nodes
                child_summaries = [
                    n.get("summary", "")
                    for n in nodes
                    if n["parent_id"] == nd["node_id"] and n.get("summary")
                ]
                text = " ".join(child_summaries)

            if text.strip():
                prompt = f"Summarize in 1-2 sentences:\n\n{text[:2000]}"
                async with sem:
                    nd["summary"] = await self._llm_fn(prompt)
            else:
                nd["summary"] = ""

            done += 1
            if progress_callback:
                progress_callback(done, total)

        tasks = [_summarize(nd) for nd in depth_sorted]
        await asyncio.gather(*tasks)
