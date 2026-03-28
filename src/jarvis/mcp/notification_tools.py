"""Notification & Reminder Tools for Jarvis.

MCP-Tools for scheduling reminders and sending desktop notifications.

Tools:
  - set_reminder: Schedule a future reminder with optional repeat
  - list_reminders: List active/pending/past reminders
  - send_notification: Send an immediate desktop notification

All reminders are persisted in SQLite (~/.jarvis/reminders.db) and
re-scheduled on gateway startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Maximum number of concurrently scheduled asyncio tasks (prevent memory leaks)
_MAX_ACTIVE_REMINDERS = 100

__all__ = [
    "NotificationTools",
    "register_notification_tools",
    "restore_pending_reminders",
]

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    repeat TEXT DEFAULT 'none',
    status TEXT DEFAULT 'pending',
    fired_at TEXT
);
"""


# ---------------------------------------------------------------------------
# Desktop notification helpers
# ---------------------------------------------------------------------------


def _try_plyer_notification(title: str, message: str, sound: bool) -> bool:
    """Attempt to send notification via plyer (cross-platform)."""
    try:
        from plyer import notification as plyer_notif  # type: ignore[import-untyped]

        plyer_notif.notify(
            title=title,
            message=message,
            app_name="Jarvis",
            timeout=10,
        )
        return True
    except Exception:
        return False


def _try_powershell_notification(title: str, message: str) -> bool:
    """Attempt to send notification via PowerShell MessageBox."""
    try:
        escaped_title = title.replace("'", "''")
        escaped_msg = message.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.MessageBox]::Show('{escaped_msg}', '{escaped_title}')"
        )
        subprocess.Popen(
            [
                "powershell",
                "-WindowStyle",
                "Hidden",
                "-Command",
                cmd,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception:
        return False


def _try_winsound_fallback() -> bool:
    """Attempt to play a system beep via winsound."""
    try:
        import winsound  # type: ignore[import-untyped]

        winsound.MessageBeep(winsound.MB_ICONASTERISK)
        return True
    except Exception:
        return False


async def _send_desktop_notification(title: str, message: str, sound: bool) -> str:
    """Send a desktop notification using the best available backend.

    Fallback chain: plyer -> PowerShell MessageBox -> winsound + log.
    """
    loop = asyncio.get_running_loop()

    # 1) plyer
    ok = await loop.run_in_executor(None, _try_plyer_notification, title, message, sound)
    if ok:
        return "Notification sent via plyer."

    # 2) PowerShell (Windows only)
    if sys.platform == "win32":
        ok = await loop.run_in_executor(None, _try_powershell_notification, title, message)
        if ok:
            if sound:
                await loop.run_in_executor(None, _try_winsound_fallback)
            return "Notification sent via PowerShell."

    # 3) Fallback: beep + log
    if sound and sys.platform == "win32":
        await loop.run_in_executor(None, _try_winsound_fallback)

    log.info("desktop_notification_fallback", title=title, message=message)
    return f"Notification logged (no desktop backend): {title} - {message}"


# ---------------------------------------------------------------------------
# NotificationTools class
# ---------------------------------------------------------------------------


class NotificationTools:
    """Manages reminders in SQLite and schedules asyncio tasks."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = encrypted_connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        # Active asyncio tasks keyed by reminder id
        self._tasks: dict[int, asyncio.Task[None]] = {}

    # -- public API ---------------------------------------------------------

    async def set_reminder(
        self,
        text: str,
        delay_minutes: int | None = None,
        at: str | None = None,
        repeat: str = "none",
    ) -> dict[str, Any]:
        """Schedule a new reminder.

        Either *delay_minutes* (relative) or *at* (ISO datetime, absolute) must
        be provided.  *repeat* can be ``none``, ``daily``, or ``weekly``.
        """
        if repeat not in ("none", "daily", "weekly"):
            return {"error": f"Invalid repeat value: {repeat!r}. Use none/daily/weekly."}

        # Enforce max active reminders
        pending = self._count_pending()
        if pending >= _MAX_ACTIVE_REMINDERS:
            return {
                "error": (
                    f"Maximum active reminders reached ({_MAX_ACTIVE_REMINDERS}). "
                    "Cancel some reminders first."
                ),
            }

        now = datetime.now(UTC)

        if at is not None:
            try:
                due = datetime.fromisoformat(at)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=UTC)
            except ValueError:
                return {"error": f"Invalid ISO datetime: {at!r}"}
        elif delay_minutes is not None:
            if delay_minutes < 0:
                return {"error": "delay_minutes must be >= 0"}
            due = now + timedelta(minutes=delay_minutes)
        else:
            return {"error": "Provide either 'delay_minutes' or 'at'."}

        reminder_id = self._insert_reminder(text, now, due, repeat)
        self._schedule_task(reminder_id, text, due, repeat)

        return {
            "id": reminder_id,
            "text": text,
            "due_at": due.isoformat(),
            "repeat": repeat,
            "status": "pending",
        }

    async def list_reminders(self, include_past: bool = False) -> list[dict[str, Any]]:
        """Return a list of reminders."""
        if include_past:
            rows = self._conn.execute(
                "SELECT id, text, due_at, repeat, status, fired_at FROM reminders ORDER BY due_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, text, due_at, repeat, status, fired_at FROM reminders "
                "WHERE status = 'pending' ORDER BY due_at"
            ).fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "due_at": r[2],
                "repeat": r[3],
                "status": r[4],
                "fired_at": r[5],
            }
            for r in rows
        ]

    async def send_notification(self, title: str, message: str, sound: bool = True) -> str:
        """Send an immediate desktop notification."""
        return await _send_desktop_notification(title, message, sound)

    async def cancel_reminder(self, reminder_id: int) -> dict[str, Any]:
        """Cancel a pending reminder."""
        task = self._tasks.pop(reminder_id, None)
        if task is not None and not task.done():
            task.cancel()
        self._conn.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (reminder_id,))
        self._conn.commit()
        return {"id": reminder_id, "status": "cancelled"}

    async def restore_pending(self) -> int:
        """Reload pending reminders from DB and reschedule tasks.

        Returns the number of reminders restored.
        """
        now = datetime.now(UTC)
        rows = self._conn.execute(
            "SELECT id, text, due_at, repeat FROM reminders WHERE status = 'pending'"
        ).fetchall()

        restored = 0
        for rid, text, due_str, repeat in rows:
            due = datetime.fromisoformat(due_str)
            if due.tzinfo is None:
                due = due.replace(tzinfo=UTC)

            if due <= now:
                # Overdue -- fire immediately
                await self._fire_reminder(rid, text, repeat)
            else:
                self._schedule_task(rid, text, due, repeat)
            restored += 1

        log.info("reminders_restored", count=restored)
        return restored

    def shutdown(self) -> None:
        """Cancel all scheduled tasks and close DB."""
        for task in self._tasks.values():
            if not task.done():
                with contextlib.suppress(RuntimeError):
                    task.cancel()
        self._tasks.clear()
        self._conn.close()

    # -- internal helpers ---------------------------------------------------

    def _count_pending(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE status = 'pending'"
        ).fetchone()
        return row[0] if row else 0

    def _insert_reminder(
        self,
        text: str,
        created: datetime,
        due: datetime,
        repeat: str,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO reminders (text, created_at, due_at, repeat, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (text, created.isoformat(), due.isoformat(), repeat),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def _schedule_task(
        self,
        reminder_id: int,
        text: str,
        due: datetime,
        repeat: str,
    ) -> None:
        now = datetime.now(UTC)
        delay_seconds = max((due - now).total_seconds(), 0)

        async def _wait_and_fire() -> None:
            try:
                await asyncio.sleep(delay_seconds)
                await self._fire_reminder(reminder_id, text, repeat)
            except asyncio.CancelledError:
                pass

        task = asyncio.ensure_future(_wait_and_fire())
        self._tasks[reminder_id] = task

    async def _fire_reminder(self, reminder_id: int, text: str, repeat: str) -> None:
        """Mark a reminder as fired, notify, and handle repeat."""
        now = datetime.now(UTC)
        self._conn.execute(
            "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ?",
            (now.isoformat(), reminder_id),
        )
        self._conn.commit()
        self._tasks.pop(reminder_id, None)

        # Send desktop notification
        await _send_desktop_notification("Jarvis Reminder", text, sound=True)
        log.info("reminder_fired", id=reminder_id, text=text)

        # Handle repeats
        if repeat == "daily":
            due_next = now + timedelta(days=1)
            new_id = self._insert_reminder(text, now, due_next, repeat)
            self._schedule_task(new_id, text, due_next, repeat)
            log.info("reminder_repeated", old_id=reminder_id, new_id=new_id, repeat=repeat)
        elif repeat == "weekly":
            due_next = now + timedelta(weeks=1)
            new_id = self._insert_reminder(text, now, due_next, repeat)
            self._schedule_task(new_id, text, due_next, repeat)
            log.info("reminder_repeated", old_id=reminder_id, new_id=new_id, repeat=repeat)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def register_notification_tools(
    mcp_client: Any,
    config: Any,
) -> NotificationTools:
    """Register notification/reminder MCP tools.

    Args:
        mcp_client: JarvisMCPClient instance.
        config: JarvisConfig instance.

    Returns:
        NotificationTools instance (needed for restore_pending_reminders).
    """
    jarvis_home = getattr(config, "jarvis_home", Path.home() / ".jarvis")
    db_path = Path(jarvis_home) / "reminders.db"
    tools = NotificationTools(db_path)

    # -- set_reminder -------------------------------------------------------
    async def _set_reminder(**kwargs: Any) -> str:
        text = kwargs.get("text", "")
        if not text:
            return "Error: 'text' is required."
        delay_minutes = kwargs.get("delay_minutes")
        at = kwargs.get("at")
        repeat = kwargs.get("repeat", "none")
        result = await tools.set_reminder(
            text=text,
            delay_minutes=int(delay_minutes) if delay_minutes is not None else None,
            at=at,
            repeat=repeat,
        )
        if "error" in result:
            return f"Error: {result['error']}"
        return (
            f'Reminder #{result["id"]} set: "{result["text"]}" '
            f"due at {result['due_at']} (repeat: {result['repeat']})"
        )

    mcp_client.register_builtin_handler(
        "set_reminder",
        _set_reminder,
        description=(
            "Set a reminder for the future. Provide 'text' and either "
            "'delay_minutes' (relative) or 'at' (ISO datetime). "
            "Optional 'repeat': none/daily/weekly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to remind about",
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": "Minutes from now (e.g. 30)",
                },
                "at": {
                    "type": "string",
                    "description": "Absolute ISO datetime (e.g. 2026-03-12T15:00:00)",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["none", "daily", "weekly"],
                    "description": "Repeat schedule (default: none)",
                    "default": "none",
                },
            },
            "required": ["text"],
        },
    )

    # -- list_reminders -----------------------------------------------------
    async def _list_reminders(**kwargs: Any) -> str:
        include_past = kwargs.get("include_past", False)
        if isinstance(include_past, str):
            include_past = include_past.lower() in ("true", "1", "yes")
        reminders = await tools.list_reminders(include_past=bool(include_past))
        if not reminders:
            return "No reminders found."
        lines = []
        for r in reminders:
            fired = f" (fired: {r['fired_at']})" if r["fired_at"] else ""
            lines.append(
                f'#{r["id"]} [{r["status"]}] "{r["text"]}" '
                f"due: {r['due_at']} repeat: {r['repeat']}{fired}"
            )
        return "\n".join(lines)

    mcp_client.register_builtin_handler(
        "list_reminders",
        _list_reminders,
        description=(
            "List active/pending reminders. Set include_past=true to see fired/cancelled ones."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "include_past": {
                    "type": "boolean",
                    "description": "Include fired/cancelled reminders (default: false)",
                    "default": False,
                },
            },
        },
    )

    # -- send_notification --------------------------------------------------
    async def _send_notification(**kwargs: Any) -> str:
        title = kwargs.get("title", "Jarvis")
        message = kwargs.get("message", "")
        if not message:
            return "Error: 'message' is required."
        sound = kwargs.get("sound", True)
        if isinstance(sound, str):
            sound = sound.lower() in ("true", "1", "yes")
        return await tools.send_notification(title=title, message=message, sound=bool(sound))

    mcp_client.register_builtin_handler(
        "send_notification",
        _send_notification,
        description="Send an immediate desktop notification with a title and message.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notification title",
                    "default": "Jarvis",
                },
                "message": {
                    "type": "string",
                    "description": "Notification body text",
                },
                "sound": {
                    "type": "boolean",
                    "description": "Play notification sound (default: true)",
                    "default": True,
                },
            },
            "required": ["message"],
        },
    )

    log.info(
        "notification_tools_registered",
        tools=["set_reminder", "list_reminders", "send_notification"],
    )
    return tools


async def restore_pending_reminders(notification_tools: NotificationTools) -> None:
    """Restore pending reminders from DB after gateway restart.

    Call this after register_notification_tools() in init_tools.
    """
    try:
        count = await notification_tools.restore_pending()
        if count:
            log.info("pending_reminders_restored", count=count)
    except Exception:
        log.warning("pending_reminders_restore_failed", exc_info=True)
