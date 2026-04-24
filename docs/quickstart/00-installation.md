# 00 · Installation

Cognithor läuft auf deinem eigenen Rechner — kein Cloud-Account, kein API-Key nötig.

**Voraussetzungen**
- Python **3.12 oder neuer** (`python --version`)
- **Ollama** ≥ 0.4.0 (lokale LLM-Runtime) — [ollama.com](https://ollama.com)
- ~2 GB freier Plattenplatz (Modell `qwen3:8b`)
- Internet-Verbindung für den einmaligen `pip install` + Modell-Download

**Zeitbedarf:** 3 Minuten
**Endzustand:** `cognithor --version` druckt `0.93.0`, `ollama list` zeigt mindestens `qwen3:8b`.

---

## Option A — Windows One-Click-Installer (empfohlen für Windows)

1. Lade `CognithorSetup-0.93.0-x64.exe` von [github.com/Alex8791-cyber/cognithor/releases](https://github.com/Alex8791-cyber/cognithor/releases) herunter.
2. Doppelklick, "Weiter", "Installieren". Der Installer bringt eine portable Python-Runtime + ffmpeg + Ollama-Setup-Link mit.
3. Start über Startmenü-Eintrag **Cognithor** oder `cognithor` in der Eingabeaufforderung.

Kein separater Python-Install nötig.

---

## Option B — `pip install` (Linux, macOS, Windows-DIY)

```bash
# 1. Ollama installieren (einmalig)
#    Linux:  curl -fsSL https://ollama.com/install.sh | sh
#    macOS:  brew install ollama
#    Win:    Installer von https://ollama.com

# 2. Ollama-Modell pullen
ollama pull qwen3:8b

# 3. Cognithor installieren
pip install "cognithor[all]"

# 4. Verifikation
cognithor --version
# → Cognithor 0.93.0
```

**Virtuelle Umgebung empfohlen:**

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# oder
.venv\Scripts\activate       # Windows
pip install "cognithor[all]"
```

---

## Option C — Docker Compose (headless / Server)

```bash
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor
docker compose up -d
# → Ollama + Cognithor API auf http://localhost:8741
```

Health-Check:

```bash
curl http://localhost:8741/health
# → {"status":"ok","version":"0.93.0"}
```

---

## Verifikation

Alle drei Wege sollten am Ende folgende Kommandos beantworten:

```bash
cognithor --version
# → Cognithor 0.93.0

ollama list
# → NAME        ID           SIZE    MODIFIED
# → qwen3:8b    abc123def    4.7 GB  ...

cognithor tools list | head -5
# → Listet die ersten fünf verfügbaren MCP-Tools
```

Fehler? → Siehe [07 · Troubleshooting](07-troubleshooting.md).

---

**Next:** [01 · Erste Crew](01-first-crew.md)
