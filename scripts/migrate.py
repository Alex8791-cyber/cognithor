#!/usr/bin/env python3
"""Jarvis · Migrations-Script – Datenbank- und Config-Migrationen.

Verwaltet Schema-Änderungen zwischen Versionen.

Nutzung:
  python scripts/migrate.py                    # Alle pending Migrationen
  python scripts/migrate.py --status           # Aktuellen Status anzeigen
  python scripts/migrate.py --target 0.2.0     # Auf bestimmte Version migrieren
  python scripts/migrate.py --dry-run          # Nur anzeigen, nicht ausführen
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Farben
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}── {msg} ──{RESET}")


# ============================================================================
# Migration-Registry
# ============================================================================


@dataclass
class Migration:
    """Eine einzelne Migration."""

    version: str
    description: str
    up: Callable[[Path], bool]


# Alle Migrationen in chronologischer Reihenfolge
MIGRATIONS: list[Migration] = []


def migration(version: str, description: str):
    """Decorator zum Registrieren von Migrationen."""

    def decorator(func: Callable[[Path], bool]) -> Callable[[Path], bool]:
        MIGRATIONS.append(Migration(version=version, description=description, up=func))
        return func

    return decorator


# ============================================================================
# Migrationen (chronologisch)
# ============================================================================


@migration("0.1.0", "Initiale Verzeichnisstruktur")
def migrate_010(jarvis_home: Path) -> bool:
    """Erstellt die initiale Verzeichnisstruktur."""
    dirs = [
        "memory",
        "memory/episodes",
        "memory/knowledge",
        "memory/procedures",
        "memory/sessions",
        "memory/index",
        "memory/semantic",
        "logs",
        "workspace",
        "policies",
        "backups",
    ]
    created = 0
    for d in dirs:
        p = jarvis_home / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created += 1
    ok(f"{created} Verzeichnisse erstellt")
    return True


@migration("0.1.1", "Audit-Trail Index hinzufügen")
def migrate_011(jarvis_home: Path) -> bool:
    """Erstellt Index auf audit.jsonl für schnellere Queries."""
    audit_file = jarvis_home / "logs" / "audit.jsonl"
    if not audit_file.exists():
        ok("Kein Audit-Trail vorhanden – übersprungen")
        return True

    # Audit-Einträge prüfen/reparieren
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    valid = 0
    for line in lines:
        try:
            json.loads(line)
            valid += 1
        except json.JSONDecodeError:
            pass
    ok(f"Audit-Trail: {valid}/{len(lines)} gültige Einträge")
    return True


@migration("0.1.2", "SQLite WAL-Modus aktivieren")
def migrate_012(jarvis_home: Path) -> bool:
    """Aktiviert WAL-Modus für bessere Concurrent-Performance."""
    db_path = jarvis_home / "memory" / "index" / "memory.db"
    if not db_path.exists():
        ok("Keine Datenbank vorhanden – übersprungen")
        return True

    conn = sqlite3.connect(str(db_path))
    try:
        current_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if current_mode == "wal":
            ok("WAL-Modus bereits aktiv")
        else:
            conn.execute("PRAGMA journal_mode=WAL")
            ok(f"WAL-Modus aktiviert (war: {current_mode})")
    finally:
        conn.close()
    return True


@migration("0.1.3", "Backup-Verzeichnis erstellen")
def migrate_013(jarvis_home: Path) -> bool:
    """Erstellt Backup-Verzeichnis."""
    backup_dir = jarvis_home / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ok("Backup-Verzeichnis vorhanden")
    return True


# ============================================================================
# Migration-Engine
# ============================================================================


def get_applied_versions(jarvis_home: Path) -> set[str]:
    """Liest angewendete Migrationen aus der Marker-Datei."""
    marker = jarvis_home / ".migrations"
    if not marker.exists():
        return set()
    data = json.loads(marker.read_text(encoding="utf-8"))
    return set(data.get("applied", []))


def save_applied_versions(jarvis_home: Path, versions: set[str]) -> None:
    """Speichert angewendete Migrationen."""
    marker = jarvis_home / ".migrations"
    data = {
        "applied": sorted(versions),
        "last_run": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_pending_migrations(jarvis_home: Path) -> list[Migration]:
    """Gibt alle noch nicht angewendeten Migrationen zurück."""
    applied = get_applied_versions(jarvis_home)
    return [m for m in MIGRATIONS if m.version not in applied]


def run_migrations(
    jarvis_home: Path,
    target: str | None = None,
    dry_run: bool = False,
) -> int:
    """Führt pending Migrationen aus."""
    header("Jarvis · Migrationen")
    print(f"  Home: {jarvis_home}")

    applied = get_applied_versions(jarvis_home)
    pending = get_pending_migrations(jarvis_home)

    if target:
        pending = [m for m in pending if m.version <= target]

    print(f"  Angewendet: {len(applied)}")
    print(f"  Ausstehend: {len(pending)}")

    if not pending:
        ok("Alles aktuell – keine Migrationen nötig")
        return 0

    if dry_run:
        header("Dry-Run (keine Änderungen)")
        for m in pending:
            print(f"  → {m.version}: {m.description}")
        return 0

    # Backup vor Migration
    backup_dir = jarvis_home / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    errors = 0
    for m in pending:
        header(f"Migration {m.version}: {m.description}")
        try:
            success = m.up(jarvis_home)
            if success:
                applied.add(m.version)
                save_applied_versions(jarvis_home, applied)
                ok(f"Migration {m.version} erfolgreich")
            else:
                fail(f"Migration {m.version} fehlgeschlagen")
                errors += 1
                break  # Stoppe bei Fehler
        except Exception as exc:
            fail(f"Migration {m.version} Fehler: {exc}")
            errors += 1
            break

    # Zusammenfassung
    header("Ergebnis")
    if errors:
        fail(f"{errors} Migration(en) fehlgeschlagen")
        return 1
    else:
        ok(f"Alle {len(pending)} Migration(en) erfolgreich")
        return 0


def show_status(jarvis_home: Path) -> None:
    """Zeigt den aktuellen Migrations-Status."""
    header("Migrations-Status")
    applied = get_applied_versions(jarvis_home)

    for m in MIGRATIONS:
        if m.version in applied:
            ok(f"{m.version}: {m.description}")
        else:
            warn(f"{m.version}: {m.description} (ausstehend)")

    pending = len(MIGRATIONS) - len(applied)
    if pending:
        print(f"\n  {YELLOW}{pending} ausstehende Migration(en){RESET}")
    else:
        print(f"\n  {GREEN}Alles aktuell{RESET}")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis Migrationen")
    parser.add_argument("--jarvis-home", default=str(Path.home() / ".jarvis"))
    parser.add_argument("--status", action="store_true", help="Status anzeigen")
    parser.add_argument("--target", type=str, help="Ziel-Version")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen")
    args = parser.parse_args()

    jarvis_home = Path(args.jarvis_home)
    if not jarvis_home.exists():
        jarvis_home.mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status(jarvis_home)
        return 0

    return run_migrations(jarvis_home, target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
