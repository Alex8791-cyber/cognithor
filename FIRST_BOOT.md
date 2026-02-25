# Jarvis · First Boot Guide

> Von der Installation zum ersten echten Gespräch.

## Überblick

Dieses Dokument führt dich durch den ersten Start von Jarvis auf deiner Maschine.
Am Ende hast du ein funktionierendes Agent-System, das Fragen beantwortet,
Dateien verwaltet und individuelle Workflows unterstützt.

## Voraussetzungen

| Komponente | Minimum | Empfohlen (dein Setup) |
|-----------|---------|----------------------|
| GPU | 8 GB VRAM | RTX 5090 (32 GB) |
| CPU | 8 Kerne | Ryzen 9 9950X3D |
| RAM | 16 GB | 64 GB+ |
| Python | 3.12+ | 3.12+ |
| Ollama | 0.3+ | Aktuellste Version |
| Disk | 50 GB frei | 100 GB+ (für Modelle) |

## Schritt 1: Installation

```bash
# Repository klonen
git clone <repo-url> jarvis
cd jarvis

# Empfohlen: Interaktiver Installer
./install.sh

# Oder manuell:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## Schritt 2: Ollama vorbereiten

```bash
# Ollama starten (falls nicht als Service)
ollama serve &

# Modelle laden
ollama pull qwen3:32b           # Planner — 20 GB, ~2 Min
ollama pull qwen3:8b            # Executor — 6 GB, ~30 Sek
ollama pull qwen3-coder:32b     # Coder — 20 GB, ~2 Min
ollama pull nomic-embed-text    # Embeddings — 300 MB, ~5 Sek
```

**RTX 5090 Tipp:** Planner (20 GB) + Executor (6 GB) = 26 GB → passen gleichzeitig
in 32 GB VRAM. Qwen3-Coder teilt sich den Speicher mit dem Planner (Ollama
entlädt automatisch).

## Schritt 3: First Boot

```bash
# Vollständige Validierung (empfohlen beim ersten Mal)
python scripts/first_boot.py

# Kurztest (nur System + Ollama + LLM, kein Agent-Loop)
python scripts/first_boot.py --quick

# Automatisch fehlende Modelle nachladen
python scripts/first_boot.py --fix
```

Der First-Boot-Test prüft:

| Check | Was wird getestet |
|-------|------------------|
| System | Python-Version, GPU, Ollama-Binary |
| Modelle | Planner, Executor, Coder, Embedding verfügbar |
| LLM | Chat mit Planner + Executor (Antwortzeit) |
| Embeddings | Vektor-Generierung (Einzel + Batch) |
| Memory | CORE.md, Prozeduren, Policies, Verzeichnisse |
| Agent-Loop | Komplette PGE-Anfrage (Plan → Gate → Execute) |

**Erwartete Ausgabe:**

```
╔══════════════════════════════════════════════╗
║         Jarvis · First Boot                  ║
║         Erster Start mit echtem Ollama        ║
╚══════════════════════════════════════════════╝

──────────────────────────────────────────────────────────
  1. System-Check
──────────────────────────────────────────────────────────
  ✓ Python 3.12.x
  ✓ jarvis Package importierbar
  ✓ Ollama Binary gefunden
  ✓ GPU: NVIDIA GeForce RTX 5090 — 32 GB total, 28.5 GB frei

  ...

  ✓ FIRST BOOT ERFOLGREICH
  12/12 Checks bestanden

  Jarvis ist bereit!
  Starte mit: python -m jarvis
```

## Schritt 4: Erster Start

```bash
python -m jarvis
```

Du siehst das CLI-REPL:

```
╭──────────────────────────────────────╮
│  Jarvis · Agent OS                   │
│  Lokaler KI-Assistent                │
╰──────────────────────────────────────╯

jarvis> 
```

### Erste Gespräche zum Ausprobieren

**Direkte Antwort (Option A):**
```
jarvis> Was ist eine REST-API?
jarvis> Erkläre mir den Unterschied zwischen Docker und Podman.
jarvis> Wie geht es dir?
```

**Tool-Plan (Option B):**
```
jarvis> Zeig mir die Dateien in meinem Workspace.
jarvis> Was weißt du über mich?
jarvis> Erstelle mir eine Datei mit einer Checkliste für Neukundengespräche.
```

**Prozedur-Trigger:**
```
jarvis> Ich habe morgen einen Termin mit einem neuen Lead: Markus Weber, IT-Unternehmer.
jarvis> Bereite ein Meeting mit der TechCorp GmbH vor — Thema Cloud-Migration.
jarvis> Was steht heute an?
```

## Schritt 5: Konfiguration anpassen

Die Konfiguration liegt in `~/.jarvis/config.yaml`:

```yaml
# Wichtigste Einstellungen:
ollama:
  base_url: http://localhost:11434    # Ollama-Server URL
  timeout_seconds: 120                 # Timeout für lange Planungen

planner:
  max_iterations: 10                   # Max. Schritte pro Anfrage

memory:
  chunk_size_tokens: 400               # Chunk-Größe für Indexierung
  search_top_k: 6                      # Anzahl Memory-Treffer
```

## Schritt 6: Identität verfeinern

Jarvis' Persönlichkeit und Regeln stehen in `~/.jarvis/CORE.md`.
Du kannst diese Datei jederzeit bearbeiten:

```bash
# Mit deinem Editor öffnen
nano ~/.jarvis/CORE.md
```

Oder direkt über Jarvis:
```
jarvis> Ändere in der CORE.md: Füge unter Fachgebiet hinzu dass ich auch Sachversicherungen berate.
```

## Architektur im Überblick

Was bei einer Anfrage passiert:

```
Du: "Bereite das Meeting mit Kontakt Schmidt vor"
 │
 ▼
[CLI Channel] → IncomingMessage
 │
 ▼
[Gateway] → Session erstellen/laden
 │
 ▼
[Memory Manager] → CORE.md laden, relevante Erinnerungen suchen
 │
 ▼
[Planner (qwen3:32b)] → Plan erstellen:
 │  1. search_memory("Kunde Schmidt")
 │  2. write_file("bu-vergleich.md", ...)
 │
 ▼
[Gatekeeper] → Jeden Step prüfen:
 │  ✓ search_memory → ALLOW (sicher)
 │  ✓ write_file → INFORM (Datei wird geschrieben)
 │
 ▼
[Executor] → Tools ausführen:
 │  1. MCP: search_memory → Kundendaten gefunden
 │  2. MCP: write_file → Datei erstellt
 │
 ▼
[Planner (replan)] → Ergebnisse interpretieren → Antwort formulieren
 │
 ▼
[Reflector] → Session auswerten, Fakten extrahieren, Prozeduren lernen
 │
 ▼
Du: "Hier ist die Meeting-Vorbereitung für Kontakt Schmidt: ..."
```

## Troubleshooting

### Ollama antwortet nicht
```bash
# Ist Ollama gestartet?
curl http://localhost:11434/api/tags

# Neustart:
pkill ollama
ollama serve &
```

### Modell-Loading langsam
Beim ersten Aufruf eines Modells lädt Ollama es in den VRAM. Das kann
30–60 Sekunden dauern. Danach sind Antworten in 1–5 Sekunden da.

```bash
# Modelle vorladen:
ollama run qwen3:32b "Hallo" --keepalive 30m
ollama run qwen3:8b "Hallo" --keepalive 30m
```

### Planner erstellt keinen Plan (antwortet immer direkt)
Das passiert wenn das Modell die Tool-Liste nicht erkennt. Prüfe:
1. Sind MCP-Tools registriert? → Log prüfen: `~/.jarvis/logs/jarvis.log`
2. Ist der System-Prompt zu lang? → `memory.search_top_k` reduzieren
3. Ist die Temperatur zu niedrig? → In `config.yaml`: `planner.temperature: 0.7`

### Memory-Suche findet nichts
Memory muss erst gefüllt werden. Beim ersten Start ist die Datenbank leer.
```
jarvis> Merke dir: Mein wichtigster Kunde ist Firma Müller GmbH, Maschinenbau, 50 Mitarbeiter.
jarvis> Was weißt du über Firma Müller?
```

## Nächste Schritte

1. **Kundendaten einspeisen** — Bestehende Kunden-Notizen in `~/.jarvis/memory/knowledge/kunden/` ablegen
2. **Telegram einrichten** — Für mobile Nutzung: `~/.jarvis/config.yaml` → `telegram_enabled: true`
3. **Cron aktivieren** — Automatisches Morgen-Briefing: `~/.jarvis/cron/jobs.yaml`
4. **Eigene Prozeduren** — Wiederkehrende Workflows als Prozeduren in `~/.jarvis/memory/procedures/`
