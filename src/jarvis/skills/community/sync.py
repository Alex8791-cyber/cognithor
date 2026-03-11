"""RegistrySync: Periodische Synchronisation mit dem Community-Registry.

Aufgaben:
  - registry.json periodisch aktualisieren
  - Recall-Checks durchfuehren (aktive Recalls sofort anwenden)
  - Installierte Community-Skills gegen Recalls pruefen
  - Optional: Auto-Update installierter Skills

Bible reference: §6.2 (Skills), §14 (Marketplace)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# HTTP-Timeout (Sekunden)
_HTTP_TIMEOUT_S = 30


@dataclass
class SyncResult:
    """Ergebnis einer Registry-Synchronisation."""

    success: bool
    registry_skills: int = 0
    new_recalls: list[str] = field(default_factory=list)
    deactivated_skills: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sync_time: float = 0.0


class RegistrySync:
    """Periodische Synchronisation mit dem Community-Skill-Registry.

    Usage::

        sync = RegistrySync(
            registry_url="https://raw.githubusercontent.com/cognithor/skill-registry/main",
            community_dir=Path.home() / ".jarvis" / "skills" / "community",
            check_interval=3600,  # 1 Stunde
        )
        await sync.sync_once()
        # oder:
        await sync.start_periodic()  # Laeuft im Hintergrund
    """

    def __init__(
        self,
        *,
        registry_url: str = "",
        community_dir: Path | None = None,
        check_interval: int = 3600,
        marketplace_store: Any | None = None,
        skill_registry: Any | None = None,
    ) -> None:
        self._registry_url = registry_url or (
            "https://raw.githubusercontent.com/cognithor/skill-registry/main"
        )
        self._community_dir = community_dir or (Path.home() / ".jarvis" / "skills" / "community")
        self._check_interval = check_interval
        self._marketplace_store = marketplace_store
        self._skill_registry = skill_registry

        self._last_sync: float = 0.0
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()

    # ====================================================================
    # Sync
    # ====================================================================

    async def sync_once(self) -> SyncResult:
        """Fuehrt eine einzelne Synchronisation durch.

        1. registry.json herunterladen
        2. recalls/active.json herunterladen
        3. Lokale Skills gegen Recalls pruefen
        4. Recalled Skills deaktivieren

        Returns:
            SyncResult mit Details.
        """
        async with self._sync_lock:
            return await self._sync_once_inner()

    async def _sync_once_inner(self) -> SyncResult:
        """Innere Sync-Logik (Lock wird von sync_once gehalten)."""
        start = time.monotonic()
        result = SyncResult(success=False)

        try:
            # 1. Registry herunterladen
            registry_url = f"{self._registry_url}/registry.json"
            registry_data = await self._fetch_json(registry_url)
            skills = registry_data.get("skills", [])
            result.registry_skills = len(skills)

            # 2. Recalls herunterladen
            recalls_url = f"{self._registry_url}/recalls/active.json"
            try:
                recalls_data = await self._fetch_json(recalls_url)
                active_recalls = recalls_data.get("recalls", [])
            except Exception as _recalls_exc:
                log.debug("recalls_fetch_failed", url=recalls_url, error=str(_recalls_exc))
                active_recalls = []

            # 3. Lokale installierte Skills pruefen
            installed = self._get_installed_skills()
            recalled_names = {r.get("skill_name", "") for r in active_recalls}

            for skill_name in installed:
                if skill_name in recalled_names:
                    result.new_recalls.append(skill_name)

                    # Skill lokal deaktivieren
                    self._deactivate_skill(skill_name)
                    result.deactivated_skills.append(skill_name)

                    # In MarketplaceStore persistieren
                    if self._marketplace_store is not None:
                        recall_entry = next(
                            (r for r in active_recalls if r.get("skill_name") == skill_name),
                            {},
                        )
                        self._marketplace_store.save_remote_recall(recall_entry)

            # 4. In SkillRegistry deaktivieren
            if self._skill_registry is not None:
                for skill_name in result.deactivated_skills:
                    self._skill_registry.disable(skill_name)

            self._last_sync = time.time()
            result.success = True
            result.sync_time = time.monotonic() - start

            log.info(
                "registry_sync_complete",
                skills=result.registry_skills,
                new_recalls=len(result.new_recalls),
                deactivated=len(result.deactivated_skills),
                duration_ms=round(result.sync_time * 1000),
            )

        except Exception as exc:
            result.errors.append(str(exc))
            result.sync_time = time.monotonic() - start
            log.error("registry_sync_failed", error=str(exc), exc_info=True)

        return result

    # ====================================================================
    # Periodische Synchronisation
    # ====================================================================

    async def start_periodic(self) -> None:
        """Startet die periodische Synchronisation im Hintergrund."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._periodic_loop())
        log.info(
            "registry_sync_started",
            interval_seconds=self._check_interval,
        )

    async def stop(self) -> None:
        """Stoppt die periodische Synchronisation."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("registry_sync_stopped")

    async def _periodic_loop(self) -> None:
        """Hintergrund-Loop fuer periodische Syncs."""
        while self._running:
            try:
                await self.sync_once()
            except Exception as exc:
                log.error("periodic_sync_error", error=str(exc))

            await asyncio.sleep(self._check_interval)

    # ====================================================================
    # Hilfsmethoden
    # ====================================================================

    def _get_installed_skills(self) -> list[str]:
        """Gibt eine Liste installierter Community-Skill-Namen zurueck."""
        if not self._community_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self._community_dir.iterdir())
            if d.is_dir() and (d / "skill.md").exists()
        ]

    def _deactivate_skill(self, skill_name: str) -> None:
        """Markiert einen Skill als deaktiviert (Recall-Marker-Datei).

        Erstellt eine ``.recalled``-Datei im Skill-Verzeichnis.
        Die SkillRegistry laedt recalled Skills nicht.
        """
        skill_dir = (self._community_dir / skill_name).resolve()
        if not skill_dir.is_relative_to(self._community_dir.resolve()):
            log.error("path_traversal_blocked", skill_name=skill_name)
            return
        if skill_dir.exists():
            recall_marker = skill_dir / ".recalled"
            try:
                recall_marker.write_text(
                    json.dumps({"recalled_at": time.time()}),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.error(
                    "skill_recall_marker_write_failed",
                    skill=skill_name,
                    path=str(recall_marker),
                    error=str(exc),
                )
                raise
            log.warning(
                "skill_recalled_locally",
                skill=skill_name,
            )

    @property
    def last_sync(self) -> float:
        """Zeitstempel des letzten Syncs (Unix timestamp)."""
        return self._last_sync

    @property
    def is_running(self) -> bool:
        """Ob die periodische Synchronisation laeuft."""
        return self._running

    # ====================================================================
    # HTTP
    # ====================================================================

    async def _fetch_json(self, url: str) -> dict[str, Any]:
        """Laedt JSON von einer URL."""
        text = await self._fetch_text(url)
        return json.loads(text)

    async def _fetch_text(self, url: str) -> str:
        """Laedt Text von einer URL.

        Nutzt aiohttp wenn verfuegbar und funktional, sonst urllib-Fallback.
        """
        aiohttp_available = False
        try:
            import aiohttp

            aiohttp_available = True
        except ImportError:
            pass

        if aiohttp_available:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.text()
            except Exception as aio_exc:
                log.debug(
                    "aiohttp_fetch_failed_falling_back_to_urllib", url=url, error=str(aio_exc)
                )

        # Fallback: urllib (synchron im Executor)
        import urllib.request

        def _sync_fetch() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Jarvis-RegistrySync/1.0"})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                return resp.read().decode("utf-8")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_fetch)
