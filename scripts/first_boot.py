#!/usr/bin/env python3
"""Jarvis · First Boot — Erster Start mit echtem Ollama.

Dieses Skript validiert den kompletten Jarvis-Stack auf deiner Maschine:
  1. System-Check (Python, Ollama, VRAM)
  2. Modell-Verfügbarkeit (Planner, Executor, Embeddings)
  3. LLM-Rauchtest (einfacher Chat)
  4. Embedding-Test (Vektor generieren)
  5. Memory-Initialisierung (CORE.md, Prozeduren, Index)
  6. Agent-Loop (komplette Anfrage durch PGE-Zyklus)
  7. Reflexion (Session-Auswertung)

Aufruf:
  python scripts/first_boot.py           # Alle Tests
  python scripts/first_boot.py --quick   # Nur Checks 1–4 (ohne Agent-Loop)
  python scripts/first_boot.py --fix     # Versucht Probleme automatisch zu beheben

Voraussetzung: Ollama läuft (ollama serve) und Modelle sind geladen.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Farben und Ausgabe
# ============================================================================

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {BLUE}→{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


@dataclass
class BootResult:
    """Sammelt Ergebnisse aller Checks."""

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def add_pass(self, name: str) -> None:
        self.passed.append(name)
        ok(name)

    def add_fail(self, name: str, detail: str = "") -> None:
        self.failed.append(name)
        fail(f"{name}" + (f" — {detail}" if detail else ""))

    def add_warn(self, name: str) -> None:
        self.warnings.append(name)
        warn(name)


# ============================================================================
# 1. System-Check
# ============================================================================


def check_system(result: BootResult) -> None:
    """Prüft Python-Version, Ollama-Binary und VRAM."""
    header("1. System-Check")

    # Python
    v = sys.version_info
    if v >= (3, 12):
        result.add_pass(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        result.add_fail(f"Python {v.major}.{v.minor} — benötigt 3.12+")

    # Jarvis installiert?
    try:
        import jarvis  # noqa: F401

        result.add_pass("jarvis Package importierbar")
    except ImportError:
        result.add_fail("jarvis nicht installiert — pip install -e '.[all]'")
        return

    # Ollama Binary
    ollama_path = shutil.which("ollama")
    if ollama_path:
        result.add_pass(f"Ollama Binary gefunden: {ollama_path}")
    else:
        result.add_fail("Ollama nicht im PATH — https://ollama.ai installieren")

    # VRAM (nvidia-smi)
    try:
        nvidia_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        for line in nvidia_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpu_name, total_mb, free_mb = parts[0], int(parts[1]), int(parts[2])
                total_gb = total_mb / 1024
                free_gb = free_mb / 1024
                result.add_pass(f"GPU: {gpu_name} — {total_gb:.0f} GB total, {free_gb:.1f} GB frei")
                if free_gb < 8:
                    result.add_warn(
                        f"Wenig freier VRAM ({free_gb:.1f} GB) — "
                        "möglicherweise Modell-Loading langsam"
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result.add_warn("nvidia-smi nicht verfügbar — VRAM-Check übersprungen")
    except Exception as exc:
        result.add_warn(f"GPU-Check fehlgeschlagen: {exc}")


# ============================================================================
# 2. Ollama-Verbindung & Modelle
# ============================================================================


async def check_ollama(result: BootResult, fix: bool = False) -> dict[str, bool]:
    """Prüft Ollama-Erreichbarkeit und Modell-Verfügbarkeit."""
    header("2. Ollama & Modelle")

    from jarvis.config import load_config
    from jarvis.core.model_router import OllamaClient

    config = load_config()
    client = OllamaClient(config)

    models_available: dict[str, bool] = {}

    try:
        available = await client.is_available()
        if available:
            result.add_pass(f"Ollama erreichbar: {config.ollama.base_url}")
        else:
            result.add_fail(
                "Ollama nicht erreichbar",
                f"Starte mit: ollama serve (URL: {config.ollama.base_url})",
            )
            await client.close()
            return models_available
    except Exception as exc:
        result.add_fail(f"Ollama Verbindungsfehler: {exc}")
        await client.close()
        return models_available

    # Modelle prüfen
    try:
        available_models = await client.list_models()
        result.add_pass(f"{len(available_models)} Modelle geladen")

        required = {
            "Planner": config.models.planner.name,
            "Executor": config.models.executor.name,
            "Coder": config.models.coder.name,
            "Embedding": config.models.embedding.name,
        }

        for role, model_name in required.items():
            if model_name in available_models:
                result.add_pass(f"{role}: {model_name}")
                models_available[role] = True
            else:
                if fix:
                    info(f"{role}: {model_name} wird heruntergeladen...")
                    try:
                        subprocess.run(
                            ["ollama", "pull", model_name],
                            check=True,
                            timeout=600,
                        )
                        result.add_pass(f"{role}: {model_name} (heruntergeladen)")
                        models_available[role] = True
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        result.add_fail(f"{role}: {model_name} Download fehlgeschlagen")
                        models_available[role] = False
                else:
                    result.add_fail(
                        f"{role}: {model_name} nicht geladen",
                        f"ollama pull {model_name}",
                    )
                    models_available[role] = False
    except Exception as exc:
        result.add_fail(f"Modell-Listing fehlgeschlagen: {exc}")

    await client.close()
    return models_available


# ============================================================================
# 3. LLM-Rauchtest
# ============================================================================


async def check_llm(result: BootResult, models: dict[str, bool]) -> None:
    """Sendet eine Test-Nachricht an das Planner-Modell."""
    header("3. LLM-Rauchtest")

    if not models.get("Planner"):
        result.add_warn("Planner-Modell nicht verfügbar — LLM-Test übersprungen")
        return

    from jarvis.config import load_config
    from jarvis.core.model_router import OllamaClient

    config = load_config()
    client = OllamaClient(config)

    test_messages = [
        {"role": "system", "content": "Du bist Jarvis. Antworte kurz auf Deutsch."},
        {"role": "user", "content": "Wer bist du? Antworte in maximal 2 Sätzen."},
    ]

    t0 = time.perf_counter()
    try:
        response = await client.chat(
            model=config.models.planner.name,
            messages=test_messages,
            temperature=0.3,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        content = response.get("message", {}).get("content", "")
        if content and len(content) > 10:
            result.add_pass(f"Planner antwortet ({elapsed:.1f}s)")
            info(f"Antwort: {content[:120]}{'...' if len(content) > 120 else ''}")
            result.timings["llm_chat"] = elapsed
        else:
            result.add_fail("Planner-Antwort leer oder zu kurz")
    except Exception as exc:
        result.add_fail(f"LLM-Chat fehlgeschlagen: {exc}")

    # Executor-Test (schnelleres Modell)
    if models.get("Executor"):
        t0 = time.perf_counter()
        try:
            response = await client.chat(
                model=config.models.executor.name,
                messages=[
                    {"role": "system", "content": "Antworte nur mit 'OK'."},
                    {"role": "user", "content": "Ping"},
                ],
                temperature=0.0,
            )
            t1 = time.perf_counter()
            content = response.get("message", {}).get("content", "")
            if content:
                result.add_pass(f"Executor antwortet ({t1 - t0:.1f}s)")
                result.timings["executor_chat"] = t1 - t0
        except Exception as exc:
            result.add_fail(f"Executor-Chat fehlgeschlagen: {exc}")

    await client.close()


# ============================================================================
# 4. Embedding-Test
# ============================================================================


async def check_embeddings(result: BootResult, models: dict[str, bool]) -> None:
    """Generiert einen Test-Embedding-Vektor."""
    header("4. Embedding-Test")

    if not models.get("Embedding"):
        result.add_warn("Embedding-Modell nicht verfügbar — Test übersprungen")
        return

    from jarvis.config import load_config
    from jarvis.core.model_router import OllamaClient

    config = load_config()
    client = OllamaClient(config)

    test_text = "Projektmanagement-Tools für Entwicklerteams vergleichen"

    t0 = time.perf_counter()
    try:
        embedding = await client.embed(
            model=config.models.embedding.name,
            text=test_text,
        )
        t1 = time.perf_counter()

        if isinstance(embedding, list) and len(embedding) > 100:
            result.add_pass(f"Embedding generiert: {len(embedding)} Dimensionen ({t1 - t0:.2f}s)")
            result.timings["embedding"] = t1 - t0
        else:
            result.add_fail(f"Embedding unerwartet: {type(embedding)}, Länge {len(embedding)}")

        # Batch-Test
        t0 = time.perf_counter()
        batch_result = await client.embed_batch(
            model=config.models.embedding.name,
            texts=["Test Eins", "Test Zwei", "Test Drei"],
        )
        t1 = time.perf_counter()
        if len(batch_result) == 3:
            result.add_pass(f"Batch-Embedding: 3 Vektoren ({t1 - t0:.2f}s)")
        else:
            result.add_fail(f"Batch-Embedding: erwartet 3, bekommen {len(batch_result)}")

    except Exception as exc:
        result.add_fail(f"Embedding fehlgeschlagen: {exc}")

    await client.close()


# ============================================================================
# 5. Memory-Initialisierung
# ============================================================================


def check_memory_init(result: BootResult) -> None:
    """Prüft ob Verzeichnisse, CORE.md und Prozeduren korrekt erstellt werden."""
    header("5. Memory-Initialisierung")

    from jarvis.config import JarvisConfig, ensure_directory_structure

    # Temporäres Home für Test (oder echtes wenn gewünscht)
    jarvis_home = Path.home() / ".jarvis"

    config = JarvisConfig(jarvis_home=jarvis_home)
    created = ensure_directory_structure(config)

    if created:
        result.add_pass(f"{len(created)} neue Pfade erstellt")
        for p in created[:5]:
            info(f"  {p}")
        if len(created) > 5:
            info(f"  ... und {len(created) - 5} weitere")
    else:
        result.add_pass("Alle Pfade existieren bereits")

    # CORE.md
    if config.core_memory_file.exists():
        content = config.core_memory_file.read_text(encoding="utf-8")
        if "Jarvis" in content and "Identität" in content:
            result.add_pass(f"CORE.md vorhanden ({len(content)} Zeichen)")
        else:
            result.add_warn("CORE.md existiert, aber Inhalt prüfen")
    else:
        result.add_fail("CORE.md nicht erstellt")

    # Prozeduren
    procs = list(config.procedures_dir.glob("*.md"))
    if procs:
        result.add_pass(f"{len(procs)} Starter-Prozeduren installiert")
        for p in procs:
            info(f"  {p.name}")
    else:
        result.add_warn("Keine Prozeduren installiert — data/procedures/ prüfen")

    # Default Policy
    default_policy = config.policies_dir / "default.yaml"
    if default_policy.exists():
        result.add_pass("Default-Policies vorhanden")
    else:
        result.add_fail("default.yaml fehlt in policies/")

    # SQLite-Index (wird beim ersten Memory-Zugriff erstellt)
    index_dir = config.index_dir
    if index_dir.exists():
        result.add_pass(f"Index-Verzeichnis: {index_dir}")
    else:
        result.add_warn("Index-Verzeichnis fehlt (wird beim ersten Zugriff erstellt)")


# ============================================================================
# 6. Agent-Loop (komplette Anfrage)
# ============================================================================


async def check_agent_loop(result: BootResult, models: dict[str, bool]) -> None:
    """Sendet eine echte Anfrage durch den kompletten PGE-Zyklus."""
    header("6. Agent-Loop (Full PGE)")

    if not models.get("Planner"):
        result.add_warn("Planner nicht verfügbar — Agent-Loop übersprungen")
        return

    from jarvis.gateway.gateway import Gateway
    from jarvis.models import IncomingMessage

    gateway = Gateway()
    try:
        info("Gateway wird initialisiert...")
        t0 = time.perf_counter()
        await gateway.initialize()
        t_init = time.perf_counter() - t0
        result.add_pass(f"Gateway initialisiert ({t_init:.1f}s)")
        result.timings["gateway_init"] = t_init
    except Exception as exc:
        result.add_fail(f"Gateway-Initialisierung fehlgeschlagen: {exc}")
        return

    # Test 1: Direkte Antwort (OPTION A)
    info("Test: Direkte Antwort (Wissensfrage)...")
    t0 = time.perf_counter()
    try:
        msg = IncomingMessage(
            channel="test",
            user_id="alexander",
            text="Was ist der Unterschied zwischen BU und Grundfähigkeitsversicherung?",
        )
        response = await gateway.handle_message(msg)
        t1 = time.perf_counter()

        if response and response.text and len(response.text) > 20:
            result.add_pass(
                f"Direkte Antwort erhalten ({t1 - t0:.1f}s, {len(response.text)} Zeichen)"
            )
            info(f"Antwort: {response.text[:150]}...")
            result.timings["direct_response"] = t1 - t0
        else:
            result.add_fail(
                "Direkte Antwort leer",
                f"Response: {response.text[:100] if response else 'None'}",
            )
    except Exception as exc:
        result.add_fail(f"Direkte Antwort fehlgeschlagen: {exc}")

    # Test 2: Tool-Plan (OPTION B)
    info("Test: Tool-Plan (Dateisystem)...")
    t0 = time.perf_counter()
    try:
        msg2 = IncomingMessage(
            channel="test",
            user_id="alexander",
            text="Liste mir die Dateien im Jarvis-Workspace auf.",
        )
        response2 = await gateway.handle_message(msg2)
        t1 = time.perf_counter()

        if response2 and response2.text:
            result.add_pass(f"Tool-Plan ausgeführt ({t1 - t0:.1f}s)")
            info(f"Antwort: {response2.text[:150]}...")
            result.timings["tool_plan"] = t1 - t0
        else:
            result.add_warn(
                "Tool-Plan: Antwort leer (Planner hat möglicherweise direkt geantwortet)"
            )
    except Exception as exc:
        result.add_fail(f"Tool-Plan fehlgeschlagen: {exc}")

    # Cleanup
    with contextlib.suppress(Exception):
        await gateway.shutdown()


async def check_memory_roundtrip(result: BootResult, models: dict[str, bool]) -> None:
    """Testet Memory: Schreiben → Lesen → Verifizieren."""
    header("6b. Memory-Roundtrip")

    if not models.get("Planner"):
        result.add_warn("Planner nicht verfügbar — Memory-Test übersprungen")
        return

    from jarvis.gateway.gateway import Gateway
    from jarvis.models import IncomingMessage

    gateway = Gateway()
    try:
        await gateway.initialize()
    except Exception as exc:
        result.add_fail(f"Gateway für Memory-Test: {exc}")
        return

    # Schreiben
    info("Memory: Speichere Testkunden-Daten...")
    t0 = time.perf_counter()
    try:
        msg_write = IncomingMessage(
            channel="test",
            user_id="testuser",
            text=(
                "Merke dir bitte: Kontakt Firma TechGmbH, Ansprechpartner "
                "Stefan Weber, 42 Jahre, IT-Leiter, interessiert an "
                "Zusammenarbeit im Cloud-Bereich für seine 15 Mitarbeiter."
            ),
        )
        resp_write = await gateway.handle_message(msg_write)
        t1 = time.perf_counter()

        if resp_write and resp_write.text and len(resp_write.text) > 10:
            result.add_pass(f"Memory-Schreiben ({t1 - t0:.1f}s)")
            result.timings["memory_write"] = t1 - t0
        else:
            result.add_fail("Memory-Schreiben: Leere Antwort")
    except Exception as exc:
        result.add_fail(f"Memory-Schreiben: {exc}")
        with contextlib.suppress(Exception):
            await gateway.shutdown()
        return

    # Lesen
    info("Memory: Rufe Testkunden-Daten ab...")
    t0 = time.perf_counter()
    try:
        msg_read = IncomingMessage(
            channel="test",
            user_id="alexander",
            text="Was weißt du über die Firma TechGmbH und Stefan Weber?",
        )
        resp_read = await gateway.handle_message(msg_read)
        t1 = time.perf_counter()

        if resp_read and resp_read.text:
            # Prüfe ob gespeicherte Daten in der Antwort auftauchen
            answer = resp_read.text.lower()
            keywords = ["techgmbh", "weber", "stefan", "bav", "bu", "mitarbeiter", "it"]
            found = sum(1 for kw in keywords if kw in answer)

            if found >= 2:
                result.add_pass(f"Memory-Lesen: {found}/7 Keywords gefunden ({t1 - t0:.1f}s)")
            else:
                result.add_warn(
                    f"Memory-Lesen: Nur {found}/7 Keywords — Daten möglicherweise nicht gespeichert"
                )
            info(f"Antwort: {resp_read.text[:150]}...")
            result.timings["memory_read"] = t1 - t0
        else:
            result.add_fail("Memory-Lesen: Leere Antwort")
    except Exception as exc:
        result.add_fail(f"Memory-Lesen: {exc}")

    with contextlib.suppress(Exception):
        await gateway.shutdown()


async def check_procedure_match(result: BootResult, models: dict[str, bool]) -> None:
    """Testet ob Prozedur-Trigger erkannt werden."""
    header("6c. Prozedur-Matching")

    if not models.get("Planner"):
        result.add_warn("Planner nicht verfügbar — Prozedur-Test übersprungen")
        return

    from jarvis.gateway.gateway import Gateway
    from jarvis.models import IncomingMessage

    gateway = Gateway()
    try:
        await gateway.initialize()
    except Exception as exc:
        result.add_fail(f"Gateway für Prozedur-Test: {exc}")
        return

    info("Trigger: Meeting-Vorbereitung für einen Termin erstellen...")
    t0 = time.perf_counter()
    try:
        msg = IncomingMessage(
            channel="test",
            user_id="alexander",
            text=(
                "Ich muss ein Meeting mit einem neuen Kontakt vorbereiten. "
                "Was brauchst du von mir an Informationen?"
            ),
        )
        resp = await gateway.handle_message(msg)
        t1 = time.perf_counter()

        if resp and resp.text and len(resp.text) > 30:
            # Meeting-Prozedur sollte nach Gesprächsdetails fragen
            answer = resp.text.lower()
            keywords = ["beruf", "alter", "geburt", "rente", "einkommen", "daten", "brauche"]
            found = sum(1 for kw in keywords if kw in answer)

            if found >= 1:
                result.add_pass(
                    f"Prozedur-Rückfrage: {found} relevante Nachfragen ({t1 - t0:.1f}s)"
                )
            else:
                result.add_warn("Prozedur erkannt, aber keine typischen Meeting-Rückfragen")
            info(f"Antwort: {resp.text[:200]}...")
            result.timings["procedure_match"] = t1 - t0
        else:
            result.add_fail("Prozedur-Test: Leere Antwort")
    except Exception as exc:
        result.add_fail(f"Prozedur-Test: {exc}")

    with contextlib.suppress(Exception):
        await gateway.shutdown()


# ============================================================================
# 7. Zusammenfassung
# ============================================================================


def print_summary(result: BootResult) -> None:
    """Gibt eine Zusammenfassung aus."""
    header("Zusammenfassung")

    total = len(result.passed) + len(result.failed)

    if result.success:
        print(f"\n  {GREEN}{BOLD}✓ FIRST BOOT ERFOLGREICH{RESET}")
        print(f"  {total}/{total} Checks bestanden")
    else:
        print(f"\n  {RED}{BOLD}✗ FIRST BOOT UNVOLLSTÄNDIG{RESET}")
        print(
            f"  {len(result.passed)}/{total} Checks bestanden, {len(result.failed)} fehlgeschlagen"
        )
        print(f"\n  {RED}Fehlgeschlagen:{RESET}")
        for f in result.failed:
            print(f"    ✗ {f}")

    if result.warnings:
        print(f"\n  {YELLOW}Warnungen:{RESET}")
        for w in result.warnings:
            print(f"    ⚠ {w}")

    if result.timings:
        print(f"\n  {BLUE}Timings:{RESET}")
        for name, t in sorted(result.timings.items()):
            print(f"    {name}: {t:.2f}s")

    print()

    if result.success:
        print(f"  {GREEN}Jarvis ist bereit!{RESET}")
        print(f"  Starte mit: {BOLD}python -m jarvis{RESET}")
    else:
        print("  Behebe die Fehler oben und starte erneut:")
        print(f"  {BOLD}python scripts/first_boot.py --fix{RESET}")

    print()


# ============================================================================
# Main
# ============================================================================


async def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis · First Boot Validierung")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Nur System-Check + Ollama + LLM (kein Agent-Loop)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Versucht fehlende Modelle automatisch zu laden",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║         Jarvis · First Boot                  ║{RESET}")
    print(f"{BOLD}║         Erster Start mit echtem Ollama        ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════╝{RESET}")

    result = BootResult()

    # 1. System
    check_system(result)

    # 2. Ollama & Modelle
    models = await check_ollama(result, fix=args.fix)

    # 3. LLM
    await check_llm(result, models)

    # 4. Embeddings
    await check_embeddings(result, models)

    # 5. Memory-Init
    check_memory_init(result)

    if not args.quick:
        # 6. Agent-Loop
        await check_agent_loop(result, models)

        # 6b. Memory-Roundtrip
        await check_memory_roundtrip(result, models)

        # 6c. Prozedur-Matching
        await check_procedure_match(result, models)

    # Zusammenfassung
    print_summary(result)

    return 0 if result.success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
