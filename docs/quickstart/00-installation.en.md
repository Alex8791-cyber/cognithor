# 00 · Installation

Cognithor runs on your own machine — no cloud account, no API key required.

**Prerequisites**
- Python **3.12 or newer** (`python --version`)
- **Ollama** ≥ 0.4.0 (local LLM runtime) — [ollama.com](https://ollama.com)
- ~2 GB of free disk space (for the `qwen3:8b` model)
- Internet connection for the one-time `pip install` + model download

**Time:** 3 minutes
**End state:** `cognithor --version` prints `0.93.0`, and `ollama list` shows at least `qwen3:8b`.

---

## Option A — Windows One-Click Installer (recommended on Windows)

1. Download `CognithorSetup-0.93.0-x64.exe` from [github.com/Alex8791-cyber/cognithor/releases](https://github.com/Alex8791-cyber/cognithor/releases).
2. Double-click, "Next", "Install". The installer bundles a portable Python runtime + ffmpeg + an Ollama-setup link.
3. Launch via the **Cognithor** Start-menu entry or `cognithor` at the command prompt.

No separate Python install required.

---

## Option B — `pip install` (Linux, macOS, DIY Windows)

```bash
# 1. Install Ollama (one-time)
#    Linux:  curl -fsSL https://ollama.com/install.sh | sh
#    macOS:  brew install ollama
#    Win:    installer from https://ollama.com

# 2. Pull the Ollama model
ollama pull qwen3:8b

# 3. Install Cognithor
pip install "cognithor[all]"

# 4. Verify
cognithor --version
# → Cognithor 0.93.0
```

**Virtual environment recommended:**

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or
.venv\Scripts\activate       # Windows
pip install "cognithor[all]"
```

---

## Option C — Docker Compose (headless / server)

```bash
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor
docker compose up -d
# → Ollama + Cognithor API on http://localhost:8741
```

Health check:

```bash
curl http://localhost:8741/health
# → {"status":"ok","version":"0.93.0"}
```

---

## Verification

All three options should eventually answer these commands:

```bash
cognithor --version
# → Cognithor 0.93.0

ollama list
# → NAME        ID           SIZE    MODIFIED
# → qwen3:8b    abc123def    4.7 GB  ...

cognithor tools list | head -5
# → Lists the first five available MCP tools
```

Errors? → See [07 · Troubleshooting](07-troubleshooting.en.md).

---

**Next:** [01 · First Crew](01-first-crew.en.md)
