"""Conversation Tree — full branching chat history.

Each message is a node with parentId/childIds. Conversations are trees,
not linear lists. Supports forking, path computation, and SQLite persistence.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ConversationTree"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT '',
    active_leaf_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS chat_nodes (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_id TEXT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    branch_index INTEGER DEFAULT 0,
    agent_name TEXT DEFAULT 'jarvis',
    model_used TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_conv ON chat_nodes(conversation_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON chat_nodes(parent_id);
"""


class ConversationTree:
    """SQLite-backed conversation tree with branching support."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with encrypted_connect(str(self._db_path)) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = encrypted_connect(str(self._db_path))
        conn.row_factory = compatible_row_factory()
        return conn

    # ── Conversations ──────────────────────────────────────────────

    def create_conversation(self, title: str = "") -> str:
        """Create a new conversation. Returns conversation ID."""
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)",
                (conv_id, title, time.time()),
            )
        return conv_id

    def set_active_leaf(self, conversation_id: str, node_id: str) -> None:
        """Set the active leaf node for a conversation."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE conversations SET active_leaf_id=?, updated_at=? WHERE id=?",
                (node_id, time.time(), conversation_id),
            )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation metadata."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── Nodes ──────────────────────────────────────────────────────

    def add_node(
        self,
        conversation_id: str,
        role: str,
        text: str,
        parent_id: str | None = None,
        agent_name: str = "jarvis",
        model_used: str = "",
        duration_ms: int = 0,
    ) -> str:
        """Add a message node to the tree. Returns node ID."""
        node_id = f"node_{uuid.uuid4().hex[:12]}"

        # Determine branch_index (position among siblings)
        branch_index = 0
        if parent_id:
            with self._conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM chat_nodes WHERE parent_id=?",
                    (parent_id,),
                ).fetchone()[0]
                branch_index = count  # 0-based, next available

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_nodes "
                "(id, conversation_id, parent_id, role, text, branch_index, "
                " agent_name, model_used, duration_ms, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    node_id,
                    conversation_id,
                    parent_id,
                    role,
                    text,
                    branch_index,
                    agent_name,
                    model_used,
                    duration_ms,
                    time.time(),
                ),
            )

        # Update conversation active leaf
        self.set_active_leaf(conversation_id, node_id)

        return node_id

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a single node by ID."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chat_nodes WHERE id=?", (node_id,)).fetchone()
        return dict(row) if row else None

    def get_children(self, node_id: str) -> list[dict[str, Any]]:
        """Get all children of a node, ordered by branch_index."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_nodes WHERE parent_id=? ORDER BY branch_index",
                (node_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_branch_index(self, node_id: str) -> int:
        """Get the branch index of a node among its siblings."""
        node = self.get_node(node_id)
        if not node:
            return 0
        return node.get("branch_index", 0)

    # ── Path Computation ──────────────────────────────────────────

    def get_path_to_root(self, node_id: str) -> list[dict[str, Any]]:
        """Get the path from root to this node (inclusive, ordered root->node)."""
        path = []
        current_id: str | None = node_id
        with self._conn() as conn:
            while current_id:
                row = conn.execute("SELECT * FROM chat_nodes WHERE id=?", (current_id,)).fetchone()
                if not row:
                    break
                path.append(dict(row))
                current_id = row["parent_id"]
        path.reverse()
        return path

    def get_active_path(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get the currently active path (root to active leaf)."""
        conv = self.get_conversation(conversation_id)
        if not conv or not conv.get("active_leaf_id"):
            return []
        return self.get_path_to_root(conv["active_leaf_id"])

    # ── Fork Points ───────────────────────────────────────────────

    def get_fork_points(self, conversation_id: str) -> dict[str, int]:
        """Get all nodes that have multiple children (fork points).

        Returns: {node_id: child_count} for nodes with 2+ children.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT parent_id, COUNT(*) as cnt FROM chat_nodes "
                "WHERE conversation_id=? AND parent_id IS NOT NULL "
                "GROUP BY parent_id HAVING cnt > 1",
                (conversation_id,),
            ).fetchall()
        return {r["parent_id"]: r["cnt"] for r in rows}

    # ── Tree Structure ────────────────────────────────────────────

    def get_tree_structure(self, conversation_id: str) -> dict[str, Any]:
        """Get the full tree structure for visualization."""
        with self._conn() as conn:
            nodes = conn.execute(
                "SELECT * FROM chat_nodes WHERE conversation_id=? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        conv = self.get_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "active_leaf_id": conv.get("active_leaf_id") if conv else None,
            "nodes": [dict(n) for n in nodes],
            "fork_points": self.get_fork_points(conversation_id),
        }

    # ── Utility ───────────────────────────────────────────────────

    def get_messages_for_replay(self, conversation_id: str, leaf_id: str) -> list[dict[str, Any]]:
        """Get ordered messages from root to leaf for WM replay.

        Returns only role + text, suitable for rebuilding WorkingMemory.
        """
        path = self.get_path_to_root(leaf_id)
        return [
            {
                "role": n["role"],
                "text": n["text"],
                "agent_name": n.get("agent_name", ""),
            }
            for n in path
        ]
