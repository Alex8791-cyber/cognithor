"""Cognithor · Gateway session management — extracted from `gateway.py`.

In-memory session-and-working-memory bookkeeping. The TTLDicts and lock
objects live on the Gateway instance (`gw._sessions`, `gw._working_memories`,
`gw._session_lock`); these helpers operate on them through the `gw` parameter.

Persistent session storage lives in `gateway/session_store.py` (separate
sub-system); this module only handles in-memory cache + cleanup.

Part of the staged `gateway.py` split — see
`project_v0960_refactor_backlog.md` and the architect blueprint.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from cognithor.models import SessionContext, WorkingMemory
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway

log = get_logger(__name__)


def cleanup_stale_sessions(gw: Gateway) -> None:
    """Remove sessions that have not been accessed for more than _SESSION_TTL_SECONDS.

    This is called periodically (guarded by _CLEANUP_INTERVAL_SECONDS) to
    prevent unbounded growth of the in-memory session and working-memory dicts.

    When a VideoCleanupWorker is configured, each evicted session_id is also
    dispatched to :meth:`VideoCleanupWorker.on_session_close` so that any
    uploaded video files registered against that session are deleted
    immediately, instead of waiting for the 24 h TTL sweep.
    """
    now = time.monotonic()
    evicted_session_ids: list[str] = []
    with gw._session_lock:
        stale_keys = [
            key
            for key, last_ts in gw._session_last_accessed.items()
            if (now - last_ts) > gw._SESSION_TTL_SECONDS
        ]
        for key in stale_keys:
            session = gw._sessions.pop(key, None)
            if session:
                gw._working_memories.pop(session.session_id, None)
                evicted_session_ids.append(session.session_id)
            gw._session_last_accessed.pop(key, None)
    if stale_keys:
        log.info("stale_sessions_cleaned", count=len(stale_keys))
    # Session-lifetime video cleanup: fire VideoCleanupWorker.on_session_close
    # for each evicted session so registered uploads are deleted now rather
    # than waiting up to 24 h for the TTL sweep. on_session_close is a
    # coroutine; schedule it as a background task if we are on an event
    # loop, otherwise the TTL sweep remains a safety net.
    video_cleanup = getattr(gw, "_video_cleanup", None)
    if evicted_session_ids and video_cleanup is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called from a thread or context with no running loop —
            # rely on the TTL sweep to reclaim the orphaned uploads.
            loop = None
        if loop is not None:
            background_tasks = getattr(gw, "_background_tasks", None)
            for sid in evicted_session_ids:
                try:
                    task = loop.create_task(video_cleanup.on_session_close(sid))
                    # Track to keep a strong reference and avoid "Task was
                    # destroyed but it is pending" warnings.
                    if background_tasks is not None:
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)
                except Exception:
                    log.debug(
                        "video_cleanup_schedule_failed",
                        session_id=sid,
                        exc_info=True,
                    )
    gw._last_session_cleanup = now


def maybe_cleanup_sessions(gw: Gateway) -> None:
    """Trigger stale session cleanup if enough time has passed since the last sweep."""
    now = time.monotonic()
    if (now - gw._last_session_cleanup) >= gw._CLEANUP_INTERVAL_SECONDS:
        cleanup_stale_sessions(gw)
        # GDPR retention: also clean up persisted sessions & channel mappings
        if gw._session_store:
            try:
                gw._session_store.cleanup_old_sessions(max_age_days=30)
                gw._session_store.cleanup_channel_mappings(max_age_days=30)
            except Exception as exc:
                log.warning("gdpr_retention_cleanup_failed", error=str(exc))


async def run_retention_enforcement(gw: Gateway) -> None:
    """GDPR: enforce retention policies via cron."""
    try:
        from cognithor.security.gdpr import GDPRComplianceManager

        if hasattr(gw, "_compliance_engine") and gw._compliance_engine:
            mgr = getattr(gw, "_gdpr_compliance_manager", None)
            if mgr and isinstance(mgr, GDPRComplianceManager):
                result = mgr.enforce_retention()
                log.info("gdpr_retention_enforced", result=result)
            else:
                log.debug("gdpr_retention_enforcement_skipped_no_manager")
    except Exception:
        log.debug("gdpr_retention_enforcement_failed", exc_info=True)


def get_or_create_session(
    gw: Gateway,
    channel: str,
    user_id: str,
    agent_name: str = "jarvis",
) -> SessionContext:
    """Laedt oder erstellt eine Session fuer Channel+User+Agent.

    Per-Agent-Isolation: Jeder Agent hat seine eigene Session.
    Das verhindert dass Working Memories vermischt werden.

    Reihenfolge:
      0. Periodic stale-session cleanup
      1. Im RAM-Cache nachschauen
      2. Aus SQLite laden (Session-Persistenz)
      3. Neue Session erstellen
    """
    # 0. Periodically clean up stale sessions
    gw._maybe_cleanup_sessions()

    key = f"{channel}:{user_id}:{agent_name}"

    with gw._session_lock:
        # 1. RAM-Cache
        if key in gw._sessions:
            gw._session_last_accessed[key] = time.monotonic()
            return gw._sessions[key]

        # 2. SQLite-Persistenz
        if gw._session_store:
            stored = gw._session_store.load_session(channel, user_id, agent_name)
            if stored and stored.agent_name == agent_name:
                gw._sessions[key] = stored
                gw._session_last_accessed[key] = time.monotonic()
                log.info(
                    "session_restored",
                    session=stored.session_id[:8],
                    channel=channel,
                    agent=agent_name,
                    messages=stored.message_count,
                )
                return stored

        # 3. Neue Session
        session = SessionContext(
            user_id=user_id,
            channel=channel,
            agent_name=agent_name,
            max_iterations=gw._config.security.max_iterations,
        )
        gw._sessions[key] = session
        gw._session_last_accessed[key] = time.monotonic()

    # Persist (outside lock, does not block other sessions)
    if gw._session_store:
        gw._session_store.save_session(session)

    log.info(
        "session_created",
        session=session.session_id[:8],
        channel=channel,
        agent=agent_name,
    )
    return session


def get_or_create_working_memory(gw: Gateway, session: SessionContext) -> WorkingMemory:
    """Laedt oder erstellt Working Memory fuer eine Session.

    Bei existierenden Sessions wird die Chat-History aus SQLite geladen.
    """
    with gw._session_lock:
        if session.session_id in gw._working_memories:
            return gw._working_memories[session.session_id]

    # Create outside lock (I/O operations do not block other sessions)
    wm = WorkingMemory(
        session_id=session.session_id,
        max_tokens=gw._config.models.planner.context_window,
    )

    # Core Memory laden (wenn vorhanden)
    core_path = gw._config.core_memory_path
    if core_path.exists():
        try:
            wm.core_memory_text = core_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("core_memory_load_failed", error=str(exc))

    # CAG prefix injection
    # CAG prefix is prepared in handle_message() (async context), not here

    # Chat-History aus SessionStore wiederherstellen
    if gw._session_store:
        try:
            history_limit = getattr(
                getattr(gw._config, "session", None),
                "chat_history_limit",
                100,
            )
            history = gw._session_store.load_chat_history(
                session.session_id,
                limit=history_limit,
            )
            if history:
                wm.chat_history = history
                log.info(
                    "chat_history_restored",
                    session=session.session_id[:8],
                    messages=len(history),
                )
        except Exception as exc:
            log.warning("chat_history_load_failed", error=str(exc))

    with gw._session_lock:
        # Double-check: another thread may have been faster
        if session.session_id not in gw._working_memories:
            gw._working_memories[session.session_id] = wm
        return gw._working_memories[session.session_id]


def check_and_compact(gw: Gateway, wm: WorkingMemory, session: SessionContext) -> None:
    """Prueft Token-Budget und kompaktiert Chat-History wenn noetig.

    Nutzt den WorkingMemoryManager fuer sprachbewusste Token-Schaetzung
    und FIFO-Entfernung alter Nachrichten.
    """
    from cognithor.memory.working import WorkingMemoryManager

    mem_cfg = gw._config.memory
    mgr = WorkingMemoryManager(config=mem_cfg, max_tokens=wm.max_tokens)
    mgr._memory = wm  # Manager auf aktuelle WM zeigen

    if mgr.needs_compaction:
        result = mgr.compact()
        if result.messages_removed > 0:
            log.info(
                "auto_compaction",
                session=session.session_id[:8],
                messages_removed=result.messages_removed,
                tokens_freed=result.tokens_freed,
                usage_after=f"{mgr.usage_ratio:.0%}",
            )
