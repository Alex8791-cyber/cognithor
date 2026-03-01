# Quickstart — Vom Klonen zum ersten Gespräch

Diese Anleitung bringt Jarvis in 10 Minuten zum Laufen.

## 1. Voraussetzungen

- **Python 3.12+** — `python3 --version`
- **Ollama** — [ollama.ai](https://ollama.ai) installieren
- **GPU empfohlen** — RTX 3090+ (24 GB VRAM) oder RTX 5090 (32 GB VRAM)

## 2. Installation

```bash
git clone <repo-url> jarvis
cd jarvis
./install.sh
```

Der Installer erkennt dein System und fragt nach dem gewünschten Modus:
- **Minimal** — Core-Funktionen, CLI
- **Full** — Alle Features (Telegram, Cron, Web-Suche)
- **Systemd** — Full + Autostart als Service
- **Docker** — Container-Build

Alternativ manuell:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## 3. Ollama-Modelle laden

```bash
# Pflicht
ollama pull qwen3:32b            # Planner — das „Gehirn" (20 GB VRAM)
ollama pull qwen3:8b             # Executor — schnelle Ausführung (6 GB VRAM)
ollama pull nomic-embed-text     # Embeddings — Vektor-Suche (300 MB VRAM)

# Optional (für Code-Aufgaben)
ollama pull qwen3-coder:32b      # Code-Spezialist (20 GB VRAM)
```

Ollama starten (falls nicht automatisch):
```bash
ollama serve
```

## 4. First Boot — System validieren

```bash
python scripts/first_boot.py
```

Dieses Skript prüft:
1. ✓ Python-Version und Imports
2. ✓ Ollama erreichbar, Modelle geladen
3. ✓ LLM antwortet (Planner + Executor)
4. ✓ Embeddings funktionieren
5. ✓ CORE.md und Prozeduren erstellt
6. ✓ Kompletter Agent-Loop (echte Konversation)
7. ✓ Memory-Roundtrip (Schreiben + Lesen)
8. ✓ Prozedur-Matching (Keyword-Trigger)

Schnelltest (nur LLM, ohne Agent-Loop):
```bash
python scripts/first_boot.py --quick
```

Fehlende Modelle automatisch laden:
```bash
python scripts/first_boot.py --fix
```

## 5. Jarvis starten

### Option A: One-Click (empfohlen)

```
Doppelklick auf  start_cognithor.bat  →  Browser öffnet sich  →  "Power On" klicken  →  Fertig.
```

Das Batch-File startet das Control Center UI, das den Python-Backend-Prozess automatisch verwaltet (Start, Stop, Health-Checks, Orphan-Cleanup). Keine Terminal-Kenntnisse nötig.

> **Tipp:** Auf dem Desktop liegt eine Verknüpfung namens **Cognithor** — einfach doppelklicken.

### Option B: CLI

```bash
python -m jarvis
```

Du siehst das CLI-REPL:
```
┌──────────────────────────────────┐
│  Jarvis · Agent OS v0.7.0       │
│  Modell: qwen3:32b              │
│  Tools: 12 registriert          │
└──────────────────────────────────┘

User > _
```

## 6. Erste Gespräche

### Direkte Antwort (Option A)
```
User > Was ist der Unterschied zwischen REST und GraphQL?
```
Jarvis antwortet direkt aus seinem Wissen — kein Tool-Call nötig.

### Tool-Plan (Option B)
```
User > Liste mein Workspace-Verzeichnis auf.
```
Jarvis erstellt einen Plan → Gatekeeper prüft → Executor führt `list_directory` aus.

### Memory nutzen
```
User > Merke dir: Kontakt Müller, Softwareentwickler, Firma TechCorp.
```
Jarvis speichert die Daten im Semantic Memory.

```
User > Was weißt du über Kontakt Müller?
```
Jarvis durchsucht Memory und gibt die gespeicherten Infos zurück.

### Prozedur-Trigger
```
User > Bereite das Meeting mit TechCorp morgen vor.
```
Jarvis erkennt das Meeting-Muster, lädt die `meeting-vorbereitung` Prozedur und sammelt systematisch Hintergrundinformationen.

### Morgen-Briefing
```
User > Was steht heute an?
```
Jarvis lädt die gestrigen Episoden, offene Aufgaben und erstellt einen Tagesüberblick.

## 7. Konfiguration anpassen

```bash
# Hauptkonfiguration
nano ~/.jarvis/config.yaml

# Identität & Regeln
nano ~/.jarvis/memory/CORE.md

# Prozeduren bearbeiten/hinzufügen
ls ~/.jarvis/memory/procedures/
```

Wichtige Config-Optionen:
```yaml
ollama:
  base_url: http://localhost:11434    # Ollama-URL (Standard)
  timeout_seconds: 120                 # Timeout pro Anfrage

models:
  planner:
    name: qwen3:32b                    # Oder kleineres Modell bei wenig VRAM
    context_window: 32768

security:
  allowed_paths:                        # Dateizugriff nur hier
    - ~/.jarvis
    - ~/Dokumente
```

## 8. Monitoring

```bash
make smoke        # 26 Installations-Checks
make health       # Laufzeit-Check (Ollama, Disk, Memory)
make test         # 1.236 Tests ausführen
```

Logs:
```bash
tail -f ~/.jarvis/logs/jarvis.log
```

## Nächste Schritte

- **Telegram-Bot** einrichten → Token in `~/.jarvis/.env` → `JARVIS_TELEGRAM_TOKEN=...`
- **Cron-Jobs** aktivieren → Morning Briefing, Weekly Review
- **Eigene Prozeduren** anlegen → `~/.jarvis/memory/procedures/mein-workflow.md`
- **CORE.md** personalisieren → Eigene Regeln und Präferenzen ergänzen

Bei Problemen: `python scripts/first_boot.py --fix` erneut ausführen.
