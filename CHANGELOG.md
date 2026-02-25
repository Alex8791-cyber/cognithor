# Changelog

Alle wichtigen Änderungen an Jarvis Agent OS.

Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).
Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

## [0.1.0] – 2026-02-22

### Hinzugefügt

**Core-Architektur**
- PGE Trinity: Planner → Gatekeeper → Executor Agent-Loop
- Multi-Modell Model-Router (Planner/Executor/Coder-Routing)
- Reflector für Post-Execution-Analyse und Lernschleifen
- Gateway als zentraler Message-Bus mit Session-Management

**5-Tier Memory-System**
- Core Memory (CORE.md): Identität, Regeln, Persönlichkeit
- Episodic Memory: Tageslog mit Append-Only-Schreibweise
- Semantic Memory: Wissensgraph mit Entities + Relations
- Procedural Memory: Gelernte Abläufe mit Trigger-Matching
- Working Memory: Session-Kontext mit Auto-Compaction
- Hybrid-Suche: BM25 + Vektor-Embedding + Graph-Queries
- Markdown-Chunker mit Section-Awareness

**Sicherheit**
- Gatekeeper mit 4-Stufen-Risiko-Klassifizierung (GREEN/YELLOW/ORANGE/RED)
- 6 eingebaute Security-Policies
- Input-Sanitizer gegen Prompt-Injection
- Credential-Store mit AES-256-Verschlüsselung
- Audit-Trail mit SHA-256-Hash-Chain
- Dateisystem-Sandbox mit Pfad-Whitelist

**MCP-Tools**
- Dateisystem: read_file, write_file, edit_file, list_directory
- Shell: exec_command (mit Gatekeeper-Schutz)
- Web: web_search, web_fetch, search_and_read
- Memory: memory_search, memory_write, entity_create

**Channels**
- CLI-Channel mit Rich-Terminal-UI
- API-Channel (FastAPI REST)
- WebUI-Channel mit WebSocket-Support
- Telegram-Bot-Channel
- Voice-Channel (Whisper STT + Piper TTS)

**Cron-System**
- Wiederkehrende Aufgaben mit Cron-Syntax
- Proaktive Erinnerungen und Checks

**Multi-Agent**
- Agent-Delegation zwischen Spezialisten
- Shared Context für Agent-Kommunikation

**Deployment**
- install.sh mit interaktiver Installation
- Systemd-Services (User-Level)
- Docker + Docker Compose
- Smoke-Test und Health-Check Scripts
- Backup/Restore mit Rotations-Management
- Versions-Migrationen
- Logrotate-Konfiguration

**Qualität**
- 1.060 automatisierte Tests
- 14.878 Zeilen Source Code
- 12.700+ Zeilen Test Code
- Cross-Module Integration Tests
- Structured Logging mit structlog

### Technische Details
- Python 3.12+
- Ollama als LLM-Backend (qwen3:32b/8b)
- SQLite + sqlite-vec für Memory-Index
- Pydantic v2 für Datenvalidierung
- 100% lokal – keine Cloud-Abhängigkeiten
