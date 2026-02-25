"""Session Manager -- Cookie/State-Persistierung für Browser-Use v17.

Speichert und lädt:
  - Cookies pro Domain
  - Local Storage Snapshots
  - Zuletzt besuchte URLs
  - Formular-Daten (optional, verschlüsselt)

Persistierung in ~/.jarvis/browser/sessions/
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SessionSnapshot:
    """Snapshot einer Browser-Session."""
    session_id: str
    domain: str
    cookies: list[dict[str, Any]] = field(default_factory=list)
    local_storage: dict[str, str] = field(default_factory=dict)
    last_url: str = ""
    last_title: str = ""
    created_at: str = ""
    updated_at: str = ""
    visit_count: int = 0

    def __post_init__(self) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.created_at:
            self.created_at = now
        self.updated_at = now


class SessionManager:
    """Verwaltet Browser-Sessions mit Cookie-Persistierung."""

    def __init__(self, storage_dir: str | Path = "") -> None:
        if storage_dir:
            self._storage_dir = Path(storage_dir)
        else:
            self._storage_dir = Path.home() / ".jarvis" / "browser" / "sessions"
        self._sessions: dict[str, SessionSnapshot] = {}
        self._loaded = False

    def _ensure_dir(self) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._storage_dir / f"{safe_id}.json"

    # ── CRUD ─────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> SessionSnapshot | None:
        """Lädt Session aus Cache oder Disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._load_from_disk(session_id)

    def save_session(self, snapshot: SessionSnapshot) -> bool:
        """Speichert Session auf Disk."""
        self._sessions[snapshot.session_id] = snapshot
        snapshot.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            self._ensure_dir()
            path = self._session_path(snapshot.session_id)
            data = {
                "session_id": snapshot.session_id,
                "domain": snapshot.domain,
                "cookies": snapshot.cookies,
                "local_storage": snapshot.local_storage,
                "last_url": snapshot.last_url,
                "last_title": snapshot.last_title,
                "created_at": snapshot.created_at,
                "updated_at": snapshot.updated_at,
                "visit_count": snapshot.visit_count,
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            return True
        except Exception as exc:
            log.warning("session_save_error", session_id=snapshot.session_id, error=str(exc))
            return False

    def delete_session(self, session_id: str) -> bool:
        """Löscht eine Session."""
        self._sessions.pop(session_id, None)
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[SessionSnapshot]:
        """Listet alle gespeicherten Sessions."""
        self._load_all()
        return list(self._sessions.values())

    # ── Playwright Integration ───────────────────────────────────

    async def save_from_page(self, page: Any, session_id: str) -> SessionSnapshot:
        """Extrahiert Cookies + Storage aus einer Playwright-Page."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(page.url).netloc or "unknown"
        except Exception:
            domain = "unknown"

        existing = self.get_session(session_id)
        snapshot = existing or SessionSnapshot(session_id=session_id, domain=domain)
        snapshot.domain = domain
        snapshot.last_url = page.url

        try:
            snapshot.last_title = await page.title()
        except Exception:
            log.debug("session_page_title_skipped", exc_info=True)

        # Cookies
        try:
            context = page.context
            cookies = await context.cookies()
            snapshot.cookies = [
                {k: v for k, v in c.items() if k in (
                    "name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite"
                )}
                for c in cookies
            ]
        except Exception as exc:
            log.debug("session_cookies_error", error=str(exc))

        # Local Storage
        try:
            storage = await page.evaluate("""
                () => {
                    const items = {};
                    for (let i = 0; i < localStorage.length && i < 50; i++) {
                        const key = localStorage.key(i);
                        items[key] = localStorage.getItem(key);
                    }
                    return items;
                }
            """)
            snapshot.local_storage = storage or {}
        except Exception:
            log.debug("session_local_storage_skipped", exc_info=True)

        snapshot.visit_count += 1
        self.save_session(snapshot)
        return snapshot

    async def restore_to_context(self, context: Any, session_id: str) -> bool:
        """Stellt Cookies in einem Playwright-BrowserContext wieder her."""
        snapshot = self.get_session(session_id)
        if not snapshot or not snapshot.cookies:
            return False

        try:
            await context.add_cookies(snapshot.cookies)
            log.info("session_restored", session_id=session_id, cookies=len(snapshot.cookies))
            return True
        except Exception as exc:
            log.warning("session_restore_error", session_id=session_id, error=str(exc))
            return False

    async def restore_local_storage(self, page: Any, session_id: str) -> bool:
        """Stellt Local Storage auf einer Seite wieder her."""
        snapshot = self.get_session(session_id)
        if not snapshot or not snapshot.local_storage:
            return False

        try:
            for key, value in snapshot.local_storage.items():
                await page.evaluate(
                    "([k, v]) => localStorage.setItem(k, v)",
                    [key, value],
                )
            return True
        except Exception:
            return False

    # ── Disk I/O ─────────────────────────────────────────────────

    def _load_from_disk(self, session_id: str) -> SessionSnapshot | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            snapshot = SessionSnapshot(
                session_id=data["session_id"], domain=data.get("domain", ""),
                cookies=data.get("cookies", []),
                local_storage=data.get("local_storage", {}),
                last_url=data.get("last_url", ""),
                last_title=data.get("last_title", ""),
                created_at=data.get("created_at", ""),
                visit_count=data.get("visit_count", 0),
            )
            if "updated_at" in data:
                snapshot.updated_at = data["updated_at"]
            self._sessions[session_id] = snapshot
            return snapshot
        except Exception as exc:
            log.warning("session_load_error", session_id=session_id, error=str(exc))
            return None

    def _load_all(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._storage_dir.exists():
            return
        for path in self._storage_dir.glob("*.json"):
            sid = path.stem
            if sid not in self._sessions:
                self._load_from_disk(sid)

    def cleanup(self, max_age_days: int = 30) -> int:
        """Löscht Sessions die älter als max_age_days sind."""
        self._load_all()
        import calendar
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        for sid in list(self._sessions):
            snap = self._sessions[sid]
            try:
                ts = calendar.timegm(time.strptime(snap.updated_at, "%Y-%m-%dT%H:%M:%SZ"))
                if ts < cutoff:
                    self.delete_session(sid)
                    removed += 1
            except (ValueError, OverflowError):
                pass
        return removed

    def stats(self) -> dict[str, Any]:
        self._load_all()
        return {
            "total_sessions": len(self._sessions),
            "storage_dir": str(self._storage_dir),
            "domains": list({s.domain for s in self._sessions.values()}),
        }
