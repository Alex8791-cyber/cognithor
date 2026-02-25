# Jarvis · Deployment Guide

## Schnellstart

```bash
# 1. Repo klonen
git clone https://github.com/team-soellner/jarvis.git
cd jarvis

# 2. Interaktive Installation
./install.sh

# 3. Starten
jarvis
```

## Installationsmodi

| Modus | Befehl | Beschreibung |
|-------|--------|--------------|
| Interaktiv | `./install.sh` | Alles inkl. Systemd, Ollama-Check |
| Minimal | `./install.sh --minimal` | Nur Core, kein Web/Telegram |
| Vollständig | `./install.sh --full` | Alles inkl. Voice |
| Nur Systemd | `./install.sh --systemd` | Nur Service-Dateien |
| Deinstallation | `./install.sh --uninstall` | Entfernt Installation (nicht Daten) |

## Verzeichnisstruktur

```
~/.jarvis/
├── config.yaml           # Hauptkonfiguration
├── .env                  # Umgebungsvariablen (optional)
├── venv/                 # Python Virtual Environment
├── memory/
│   ├── CORE.md           # Jarvis-Identität + Regeln
│   ├── episodes/         # Tageslog (episodisches Gedächtnis)
│   ├── knowledge/        # Wissensdateien (semantisch)
│   ├── procedures/       # Gelernte Abläufe
│   ├── sessions/         # Session-Daten
│   └── index/
│       └── memory.db     # SQLite-Index + Embeddings
├── logs/
│   ├── jarvis.log        # Strukturierte Logs
│   └── audit.jsonl       # Audit-Trail (Hash-Chain)
├── workspace/            # Sandbox für Tool-Ausführung
└── policies/             # Gatekeeper-Policies
```

## Systemd-Services

### Jarvis Core (CLI)

```bash
systemctl --user start jarvis       # Starten
systemctl --user stop jarvis        # Stoppen
systemctl --user restart jarvis     # Neustart
systemctl --user enable jarvis      # Autostart aktivieren
systemctl --user status jarvis      # Status
journalctl --user -u jarvis -f      # Live-Logs
```

### Web-UI (Optional)

```bash
systemctl --user start jarvis-webui
systemctl --user enable jarvis-webui
# Erreichbar unter http://localhost:8080
```

## Konfiguration

### config.yaml

Die wichtigsten Einstellungen:

```yaml
# Ollama-Server (wenn auf anderem Rechner)
ollama:
  base_url: "http://192.168.1.50:11434"

# Modelle an VRAM anpassen
models:
  planner:
    name: "qwen3:32b"      # RTX 5090: 32B
  executor:
    name: "qwen3:8b"       # Immer 8B

# Channels aktivieren
channels:
  cli_enabled: true
  webui_enabled: true
  webui_port: 8080
  telegram_enabled: false
```

### Umgebungsvariablen

Überschreiben `config.yaml`:

```bash
export JARVIS_OLLAMA_BASE_URL=http://192.168.1.50:11434
export JARVIS_LOGGING_LEVEL=DEBUG
```

## Ollama-Modelle

### Pflicht

```bash
ollama pull qwen3:8b           # Executor (6 GB VRAM)
ollama pull nomic-embed-text   # Embeddings (0.5 GB VRAM)
```

### Empfohlen

```bash
ollama pull qwen3:32b          # Planner (20 GB VRAM)
ollama pull qwen3-coder:32b    # Code-Generierung (20 GB VRAM)
```

### VRAM-Profile

| GPU | VRAM | Planner | Executor | Qualität |
|-----|------|---------|----------|----------|
| RTX 5090 | 32 GB | qwen3:32b | qwen3:8b | ★★★★★ |
| RTX 4090 | 24 GB | qwen3:32b | qwen3:8b | ★★★★☆ |
| RTX 3090 | 24 GB | qwen3:32b-q4 | qwen3:8b | ★★★☆☆ |
| RTX 4070 | 12 GB | qwen3:14b | qwen3:8b | ★★★☆☆ |
| 8 GB | 8 GB | qwen3:8b | qwen3:8b | ★★☆☆☆ |

## Monitoring

```bash
# Smoke-Test (nach Installation)
make smoke

# Health-Check (Laufzeit)
make health

# Health-Check als JSON (für Monitoring)
make health-json

# Projekt-Statistiken
make stats
```

## Make-Targets

```bash
make help          # Alle Targets anzeigen
make install       # Installation
make dev           # Entwicklungsmodus
make run           # Jarvis starten
make test          # Alle Tests
make test-cov      # Tests mit Coverage
make lint          # Code-Prüfung
make check         # Alles prüfen (lint + types + tests)
make clean         # Aufräumen
```

## Troubleshooting

### Ollama nicht erreichbar

```bash
# Prüfe ob Ollama läuft
systemctl status ollama
ollama serve  # Manuell starten

# Prüfe Verbindung
curl http://localhost:11434/api/version
```

### Import-Fehler

```bash
# venv aktivieren
source ~/.jarvis/venv/bin/activate

# Neu installieren
pip install -e ".[all,dev]"
```

### Permission-Fehler

```bash
# Verzeichnis-Rechte prüfen
ls -la ~/.jarvis/
chmod -R u+rw ~/.jarvis/
```

### Logs prüfen

```bash
# Jarvis-Logs
tail -f ~/.jarvis/logs/jarvis.log

# Systemd-Logs
journalctl --user -u jarvis --since "1 hour ago"

# Audit-Trail
cat ~/.jarvis/logs/audit.jsonl | python3 -m json.tool
```
