"""KanbanStore — encrypted SQLite persistence for Kanban tasks."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from cognithor.kanban.models import (
    Task,
    TaskHistory,
    TaskPriority,
    TaskSource,
    TaskStatus,
    _now_iso,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    priority TEXT NOT NULL DEFAULT 'medium',
    assigned_agent TEXT DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    source_ref TEXT DEFAULT '',
    parent_id TEXT DEFAULT '',
    labels TEXT DEFAULT '[]',
    sort_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT DEFAULT '',
    created_by TEXT NOT NULL DEFAULT 'user',
    result_summary TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    changed_by TEXT,
    changed_at TEXT NOT NULL,
    note TEXT DEFAULT '',
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(assigned_agent);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_history_task ON task_history(task_id);
"""


def _dict_row_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> sqlite3.Row:
    """Row factory that returns dict-like objects compatible with ``row["col"]`` access.

    Used as a fallback when ``sqlcipher3.Row`` is not available.
    """
    # Build a lightweight object that supports both index and key access
    desc = cursor.description
    fields = {col[0]: row[idx] for idx, col in enumerate(desc)}
    return type(  # type: ignore[no-any-return]
        "Row",
        (),
        {
            "__getitem__": lambda self, k: fields[k] if isinstance(k, str) else row[k],
            "keys": lambda self: fields.keys(),
        },
    )()


class KanbanStore:
    """SQLite-backed store for Kanban tasks with optional SQLCipher encryption."""

    def __init__(self, db_path: str, use_encryption: bool = True) -> None:
        self._db_path = db_path
        self._use_encryption = use_encryption
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        _is_sqlcipher = False
        if self._use_encryption:
            try:
                from cognithor.security.encrypted_db import encrypted_connect

                self._conn = encrypted_connect(self._db_path)
                _is_sqlcipher = True
            except Exception:
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        # sqlcipher3 cursors are incompatible with sqlite3.Row — use
        # the matching Row class or fall back to a dict factory.
        if _is_sqlcipher:
            try:
                import sqlcipher3

                self._conn.row_factory = sqlcipher3.Row
            except (ImportError, AttributeError):
                self._conn.row_factory = _dict_row_factory
        else:
            self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def create(self, task: Task) -> Task:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO tasks (id, title, description, status, priority,
               assigned_agent, source, source_ref, parent_id, labels,
               sort_order, created_at, updated_at, completed_at,
               created_by, result_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task.id,
                task.title,
                task.description,
                task.status.value if isinstance(task.status, TaskStatus) else task.status,
                task.priority.value if isinstance(task.priority, TaskPriority) else task.priority,
                task.assigned_agent,
                task.source.value if isinstance(task.source, TaskSource) else task.source,
                task.source_ref,
                task.parent_id,
                task.labels_json,
                task.sort_order,
                task.created_at,
                task.updated_at,
                task.completed_at,
                task.created_by,
                task.result_summary,
            ),
        )
        conn.commit()
        return task

    def get(self, task_id: str) -> Task | None:
        row = self._get_conn().execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        agent: str | None = None,
        priority: TaskPriority | None = None,
        source: TaskSource | None = None,
        parent_id: str | None = None,
        label: str | None = None,
    ) -> list[Task]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []

        if status is not None:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, TaskStatus) else status)
        if agent is not None:
            query += " AND assigned_agent = ?"
            params.append(agent)
        if priority is not None:
            query += " AND priority = ?"
            params.append(priority.value if isinstance(priority, TaskPriority) else priority)
        if source is not None:
            query += " AND source = ?"
            params.append(source.value if isinstance(source, TaskSource) else source)
        if parent_id is not None:
            query += " AND parent_id = ?"
            params.append(parent_id)
        if label is not None:
            query += " AND labels LIKE ?"
            params.append(f'%"{label}"%')

        query += " ORDER BY sort_order ASC, created_at DESC"
        rows = self._get_conn().execute(query, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    _ALLOWED_COLUMNS = frozenset(
        {
            "title",
            "description",
            "status",
            "priority",
            "assigned_agent",
            "source",
            "source_ref",
            "parent_id",
            "labels",
            "sort_order",
            "created_at",
            "updated_at",
            "completed_at",
            "created_by",
            "result_summary",
        }
    )

    def update(self, task_id: str, **fields: Any) -> Task | None:
        if not fields:
            return self.get(task_id)
        # Whitelist column names to prevent SQL injection
        fields = {k: v for k, v in fields.items() if k in self._ALLOWED_COLUMNS}
        if not fields:
            return self.get(task_id)
        fields["updated_at"] = _now_iso()
        if "labels" in fields and isinstance(fields["labels"], list):
            fields["labels"] = json.dumps(fields["labels"])
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = [*list(fields.values()), task_id]
        self._get_conn().execute(f"UPDATE tasks SET {sets} WHERE id = ?", vals)
        self._get_conn().commit()
        return self.get(task_id)

    def delete(self, task_id: str, cascade: bool = True) -> None:
        conn = self._get_conn()
        if cascade:
            conn.execute("DELETE FROM tasks WHERE parent_id = ?", (task_id,))
        conn.execute("DELETE FROM task_history WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()

    def move(self, task_id: str, new_status: str, sort_order: int = 0) -> Task | None:
        now = _now_iso()
        completed = now if new_status == TaskStatus.DONE.value else ""
        self._get_conn().execute(
            "UPDATE tasks SET status=?, sort_order=?, updated_at=?, completed_at=? WHERE id=?",
            (new_status, sort_order, now, completed, task_id),
        )
        self._get_conn().commit()
        return self.get(task_id)

    def get_subtasks(self, parent_id: str) -> list[Task]:
        return self.list_tasks(parent_id=parent_id)

    def record_history(
        self,
        task_id: str,
        old_status: str,
        new_status: str,
        changed_by: str,
        note: str = "",
    ) -> None:
        self._get_conn().execute(
            """INSERT INTO task_history (task_id, old_status, new_status,
               changed_by, changed_at, note) VALUES (?,?,?,?,?,?)""",
            (task_id, old_status, new_status, changed_by, _now_iso(), note),
        )
        self._get_conn().commit()

    def get_history(self, task_id: str) -> list[TaskHistory]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT * FROM task_history WHERE task_id = ? ORDER BY id ASC",
                (task_id,),
            )
            .fetchall()
        )
        return [
            TaskHistory(
                id=r["id"],
                task_id=r["task_id"],
                old_status=r["old_status"],
                new_status=r["new_status"],
                changed_by=r["changed_by"],
                changed_at=r["changed_at"],
                note=r["note"] or "",
            )
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        by_status: dict[str, int] = {}
        for row in conn.execute("SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"):
            by_status[row["status"]] = row["cnt"]
        by_agent: dict[str, int] = {}
        for row in conn.execute(
            "SELECT assigned_agent, COUNT(*) as cnt FROM tasks "
            "WHERE assigned_agent != '' GROUP BY assigned_agent"
        ):
            by_agent[row["assigned_agent"]] = row["cnt"]
        by_source: dict[str, int] = {}
        for row in conn.execute("SELECT source, COUNT(*) as cnt FROM tasks GROUP BY source"):
            by_source[row["source"]] = row["cnt"]
        return {
            "total": total,
            "by_status": by_status,
            "by_agent": by_agent,
            "by_source": by_source,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        labels = row["labels"]
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except (json.JSONDecodeError, TypeError):
                labels = []
        return Task(
            id=row["id"],
            title=row["title"],
            description=row["description"] or "",
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            assigned_agent=row["assigned_agent"] or "",
            source=TaskSource(row["source"]),
            source_ref=row["source_ref"] or "",
            parent_id=row["parent_id"] or "",
            labels=labels,
            sort_order=row["sort_order"] or 0,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"] or "",
            created_by=row["created_by"] or "user",
            result_summary=row["result_summary"] or "",
        )
