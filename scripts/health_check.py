#!/usr/bin/env python3
"""Jarvis · Health-Check – Laufzeit-Prüfung.

Für systemd Watchdog, Monitoring und Cron-Jobs.
Prüft ob alle Subsysteme funktionsfähig sind.

Nutzung:
  python scripts/health_check.py                  # Vollständig
  python scripts/health_check.py --quick           # Nur Ollama + Disk
  python scripts/health_check.py --json            # JSON-Ausgabe

Exit-Code: 0=healthy, 1=unhealthy, 2=degraded
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def check_ollama(url: str = "http://localhost:11434") -> dict:
    """Prüft Ollama-Erreichbarkeit und geladene Modelle."""
    try:
        import httpx

        resp = httpx.get(f"{url}/api/version", timeout=5)
        if resp.status_code != 200:
            return {"status": "error", "message": f"HTTP {resp.status_code}"}

        tags = httpx.get(f"{url}/api/tags", timeout=10)
        models = [m["name"] for m in tags.json().get("models", [])]

        return {
            "status": "ok",
            "version": resp.json().get("version", "?"),
            "models_count": len(models),
            "models": models[:10],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def check_disk(jarvis_home: str = "~/.jarvis") -> dict:
    """Prüft Speicherplatz und Verzeichnisse."""
    home = Path(jarvis_home).expanduser()
    result = {"status": "ok", "path": str(home)}

    if not home.exists():
        return {"status": "error", "message": f"{home} existiert nicht"}

    # Speicherplatz
    usage = shutil.disk_usage(home)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    used_pct = (usage.used / usage.total) * 100

    result["disk_free_gb"] = round(free_gb, 1)
    result["disk_total_gb"] = round(total_gb, 1)
    result["disk_used_pct"] = round(used_pct, 1)

    if free_gb < 1.0:
        result["status"] = "warning"
        result["message"] = f"Wenig Speicher: {free_gb:.1f} GB frei"

    # Jarvis-Verzeichnisse
    dirs_ok = 0
    dirs_missing = 0
    for d in ["memory", "memory/episodes", "memory/knowledge", "memory/index", "logs"]:
        if (home / d).exists():
            dirs_ok += 1
        else:
            dirs_missing += 1

    result["dirs_ok"] = dirs_ok
    result["dirs_missing"] = dirs_missing

    # DB-Größe
    db_path = home / "memory" / "index" / "memory.db"
    if db_path.exists():
        result["db_size_mb"] = round(db_path.stat().st_size / (1024**2), 2)

    # Log-Größe
    logs_dir = home / "logs"
    if logs_dir.exists():
        total_log_size = sum(f.stat().st_size for f in logs_dir.glob("*") if f.is_file())
        result["logs_size_mb"] = round(total_log_size / (1024**2), 2)

    return result


def check_memory(jarvis_home: str = "~/.jarvis") -> dict:
    """Prüft Memory-System Integrität."""
    try:
        from jarvis.config import JarvisConfig, ensure_directory_structure
        from jarvis.memory.manager import MemoryManager

        home = Path(jarvis_home).expanduser()
        config = JarvisConfig(jarvis_home=home)
        ensure_directory_structure(config)
        manager = MemoryManager(config)
        stats = manager.initialize()
        manager.close_sync()

        return {
            "status": "ok",
            "chunks": stats.get("chunks", 0),
            "entities": stats.get("entities", 0),
            "procedures": stats.get("procedures", 0),
            "core_loaded": stats.get("core_memory_loaded", False),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def check_audit(jarvis_home: str = "~/.jarvis") -> dict:
    """Prüft Audit-Trail Integrität."""
    try:
        from jarvis.security.audit import AuditTrail

        home = Path(jarvis_home).expanduser()
        logs_dir = home / "logs"
        if not logs_dir.exists():
            return {"status": "warning", "message": "Kein Audit-Log vorhanden"}

        audit = AuditTrail(log_dir=logs_dir)
        valid, total, broken_at = audit.verify_chain()

        result = {
            "status": "ok" if valid else "error",
            "entries": total,
            "chain_valid": valid,
        }
        if not valid:
            result["broken_at"] = broken_at
            result["message"] = f"Hash-Chain gebrochen bei Eintrag {broken_at}"

        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def run_health_check(jarvis_home: str, ollama_url: str, quick: bool = False) -> dict:
    """Führt alle Health-Checks aus."""
    start = time.monotonic()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    # Immer prüfen
    results["checks"]["ollama"] = check_ollama(ollama_url)
    results["checks"]["disk"] = check_disk(jarvis_home)

    if not quick:
        results["checks"]["memory"] = check_memory(jarvis_home)
        results["checks"]["audit"] = check_audit(jarvis_home)

    # Gesamtstatus
    statuses = [c["status"] for c in results["checks"].values()]
    if "error" in statuses:
        results["overall"] = "unhealthy"
    elif "warning" in statuses:
        results["overall"] = "degraded"
    else:
        results["overall"] = "healthy"

    results["duration_ms"] = round((time.monotonic() - start) * 1000)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis Health-Check")
    parser.add_argument("--jarvis-home", default=str(Path.home() / ".jarvis"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--quick", action="store_true", help="Nur Ollama + Disk")
    parser.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    args = parser.parse_args()

    results = run_health_check(args.jarvis_home, args.ollama_url, args.quick)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        status_colors = {"ok": GREEN, "warning": YELLOW, "error": RED}

        print(f"\n{BOLD}Jarvis Health-Check{RESET}  ({results['timestamp'][:19]})")
        print("─" * 45)
        for name, check in results["checks"].items():
            color = status_colors.get(check["status"], "")
            print(f"  {color}●{RESET} {name:12s} {check['status']:8s}  {check.get('message', '')}")
        print("─" * 45)

        overall = results["overall"]
        color = {"healthy": GREEN, "degraded": YELLOW, "unhealthy": RED}[overall]
        print(f"  {color}{BOLD}{overall.upper()}{RESET}  ({results['duration_ms']}ms)\n")

    return {"healthy": 0, "degraded": 2, "unhealthy": 1}[results["overall"]]


if __name__ == "__main__":
    sys.exit(main())
