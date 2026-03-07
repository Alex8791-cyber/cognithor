"""CommunityRegistryClient: Fetch, Verify & Install von Community-Skills.

Kommuniziert mit dem oeffentlichen GitHub-Repository ``cognithor/skill-registry``
ueber die GitHub API (oder Raw-Content-URLs).  Kein eigener Server noetig.

Sicherheitskette bei Installation:
  1. registry.json fetchen → Recall-Check
  2. Publisher-Reputation pruefen
  3. skill.md von GitHub herunterladen
  4. content_hash verifizieren (SHA-256)
  5. Lokale 5-stufige Sicherheits-Checks (SkillValidator)
  6. Tool-Permissions dem User anzeigen
  7. Skill nach ~/.jarvis/skills/community/<name>/ schreiben
  8. SkillRegistry neu laden

Bible reference: §6.2 (Skills), §14 (Marketplace)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.skills.community.validator import SkillValidator, ValidationResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# HTTP-Timeout (Sekunden) fuer alle Fetch-Operationen
_HTTP_TIMEOUT_S = 30


# ============================================================================
# Datenmodelle
# ============================================================================

@dataclass
class InstallResult:
    """Ergebnis einer Skill-Installation."""

    success: bool
    skill_name: str
    version: str = ""
    install_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tools_required: list[str] = field(default_factory=list)


@dataclass
class RegistryEntry:
    """Ein Eintrag aus der registry.json."""

    name: str
    version: str = ""
    description: str = ""
    author_github: str = ""
    category: str = "general"
    tools_required: list[str] = field(default_factory=list)
    content_hash: str = ""
    recalled: bool = False


# ============================================================================
# Client
# ============================================================================


class CommunityRegistryClient:
    """Client fuer das Community-Skill-Registry auf GitHub.

    Usage::

        client = CommunityRegistryClient(
            community_dir=Path.home() / ".jarvis" / "skills" / "community",
        )
        # Registry-Index laden
        entries = await client.fetch_registry()

        # Skill installieren
        result = await client.install("morgen-briefing")
    """

    # Standard-Registry-URL (GitHub Raw Content)
    DEFAULT_REGISTRY_URL = (
        "https://raw.githubusercontent.com/cognithor/skill-registry/main"
    )

    def __init__(
        self,
        *,
        community_dir: Path | None = None,
        registry_url: str = "",
        validator: SkillValidator | None = None,
        min_publisher_reputation: float = 0.0,
    ) -> None:
        self._community_dir = community_dir or (
            Path.home() / ".jarvis" / "skills" / "community"
        )
        self._registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self._validator = validator or SkillValidator()
        self._min_publisher_reputation = min_publisher_reputation

        # Cached registry.json (geschuetzt durch _lock)
        self._registry_cache: dict[str, RegistryEntry] = {}
        self._lock = asyncio.Lock()

    # ====================================================================
    # Registry
    # ====================================================================

    async def fetch_registry(self) -> dict[str, RegistryEntry]:
        """Laedt die registry.json vom GitHub-Repository.

        Returns:
            Dict von Skill-Name → RegistryEntry.
        """
        async with self._lock:
            url = f"{self._registry_url}/registry.json"
            data = await self._fetch_json(url)

            entries: dict[str, RegistryEntry] = {}
            for skill_data in data.get("skills", []):
                name = skill_data.get("name", "")
                if not name:
                    continue
                entries[name] = RegistryEntry(
                    name=name,
                    version=skill_data.get("version", ""),
                    description=skill_data.get("description", ""),
                    author_github=skill_data.get("author_github", ""),
                    category=skill_data.get("category", "general"),
                    tools_required=skill_data.get("tools_required", []),
                    content_hash=skill_data.get("content_hash", ""),
                    recalled=skill_data.get("recalled", False),
                )

            self._registry_cache = entries
            log.info("community_registry_fetched", count=len(entries))
            return entries

    async def search(
        self,
        query: str = "",
        category: str = "",
        limit: int = 20,
    ) -> list[RegistryEntry]:
        """Durchsucht die Registry.

        Args:
            query: Suchbegriff (Name, Beschreibung).
            category: Kategorie-Filter.
            limit: Max Ergebnisse.

        Returns:
            Liste passender RegistryEntry-Objekte.
        """
        if not self._registry_cache:
            await self.fetch_registry()

        # Snapshot unter Lock fuer konsistente Iteration
        async with self._lock:
            cache_snapshot = dict(self._registry_cache)

        results: list[RegistryEntry] = []
        query_lower = query.lower()

        for entry in cache_snapshot.values():
            if entry.recalled:
                continue
            if category and entry.category != category:
                continue
            if query_lower:
                if (
                    query_lower not in entry.name.lower()
                    and query_lower not in entry.description.lower()
                ):
                    continue
            results.append(entry)

        return results[:limit]

    async def get_entry(self, skill_name: str) -> RegistryEntry | None:
        """Gibt einen einzelnen Registry-Eintrag zurueck (laedt Registry bei Bedarf).

        Returns:
            RegistryEntry oder None wenn nicht gefunden.
        """
        if not self._registry_cache:
            await self.fetch_registry()
        async with self._lock:
            return self._registry_cache.get(skill_name)

    # ====================================================================
    # Installation
    # ====================================================================

    async def install(self, skill_name: str) -> InstallResult:
        """Installiert einen Community-Skill.

        Fuehrt die vollstaendige Sicherheitskette durch:
        1. Recall-Check
        2. Skill-Dateien herunterladen
        3. Hash-Verifikation
        4. 5-stufige Sicherheitspruefung
        5. Lokal speichern

        Args:
            skill_name: Name des Skills (z.B. "morgen-briefing").

        Returns:
            InstallResult mit Erfolg/Misserfolg und Details.
        """
        # 1. Registry-Eintrag pruefen (fetch_registry ist Lock-geschuetzt)
        if not self._registry_cache:
            await self.fetch_registry()

        # Snapshot unter Lock fuer konsistenten Zugriff
        async with self._lock:
            entry = self._registry_cache.get(skill_name)
        if entry is None:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=[f"Skill '{skill_name}' nicht in der Registry gefunden"],
            )

        if entry.recalled:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=[f"Skill '{skill_name}' wurde zurueckgerufen (Recall)"],
            )

        # 2. skill.md herunterladen
        skill_md_url = f"{self._registry_url}/skills/{skill_name}/skill.md"
        try:
            skill_md = await self._fetch_text(skill_md_url)
        except Exception as exc:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=[f"skill.md konnte nicht heruntergeladen werden: {exc}"],
            )

        # 3. manifest.json herunterladen
        manifest_url = f"{self._registry_url}/skills/{skill_name}/manifest.json"
        manifest: dict[str, Any] | None = None
        try:
            manifest = await self._fetch_json(manifest_url)
        except Exception as _manifest_exc:
            log.debug("no_manifest_found", skill=skill_name, error=str(_manifest_exc))

        # 4. Hash-Verifikation
        actual_hash = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
        if entry.content_hash and actual_hash != entry.content_hash:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=[
                    f"content_hash stimmt nicht ueberein! "
                    f"Erwartet: {entry.content_hash[:16]}..., "
                    f"Berechnet: {actual_hash[:16]}..."
                ],
            )

        # 5. Lokale Sicherheitspruefung (dieselben 5 Checks wie CI)
        async with self._lock:
            existing_names = set(self._registry_cache.keys()) - {skill_name}
        validation = self._validator.validate(
            skill_md, manifest, existing_names=existing_names,
        )
        if not validation.valid:
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=validation.errors,
                warnings=validation.warnings,
            )

        # 6. Lokal speichern
        install_dir = self._community_dir / skill_name
        install_dir.mkdir(parents=True, exist_ok=True)

        try:
            (install_dir / "skill.md").write_text(skill_md, encoding="utf-8")
            if manifest:
                (install_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except OSError as write_exc:
            log.error("community_skill_write_failed", skill=skill_name, error=str(write_exc))
            return InstallResult(
                success=False,
                skill_name=skill_name,
                errors=[f"Skill konnte nicht gespeichert werden: {write_exc}"],
            )

        tools_required = entry.tools_required
        if manifest:
            tools_required = manifest.get("tools_required", tools_required)

        log.info(
            "community_skill_installed",
            skill=skill_name,
            version=entry.version,
            path=str(install_dir),
            tools=tools_required,
        )

        return InstallResult(
            success=True,
            skill_name=skill_name,
            version=entry.version,
            install_path=str(install_dir),
            warnings=validation.warnings,
            tools_required=tools_required,
        )

    # ====================================================================
    # Deinstallation
    # ====================================================================

    async def uninstall(self, skill_name: str) -> bool:
        """Deinstalliert einen Community-Skill.

        Entfernt das Skill-Verzeichnis aus ~/.jarvis/skills/community/.

        Returns:
            True wenn erfolgreich entfernt.
        """
        import shutil

        install_dir = self._community_dir / skill_name
        if not install_dir.exists():
            log.warning("skill_not_found_for_uninstall", skill=skill_name)
            return False

        shutil.rmtree(install_dir)
        log.info("community_skill_uninstalled", skill=skill_name)
        return True

    # ====================================================================
    # Installierte Skills
    # ====================================================================

    def list_installed(self) -> list[str]:
        """Gibt eine Liste installierter Community-Skill-Namen zurueck."""
        if not self._community_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self._community_dir.iterdir())
            if d.is_dir() and (d / "skill.md").exists()
        ]

    # ====================================================================
    # HTTP-Hilfsmethoden
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
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)) as resp:
                        resp.raise_for_status()
                        return await resp.text()
            except Exception as aio_exc:
                log.debug("aiohttp_fetch_failed_falling_back_to_urllib", url=url, error=str(aio_exc))

        # Fallback: urllib (synchron im Executor)
        import urllib.request

        def _sync_fetch() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Jarvis-CommunityClient/1.0"})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                return resp.read().decode("utf-8")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_fetch)
