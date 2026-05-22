# Prerequisites

System-level dependencies required by Cognithor, organized by feature.

## Required

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **Python** | >= 3.12 | Runtime | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | >= 23.0 | Package manager | Included with Python |

## Required for Local LLM (Default)

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **Ollama** | >= 0.3 | Local LLM inference | [ollama.com/download](https://ollama.com/download) |

After installing Ollama, pull the required models:

```bash
ollama pull qwen3:32b          # Planner model
ollama pull qwen3:8b           # Executor model
ollama pull nomic-embed-text   # Embedding model
```

## Optional — by Feature

### Control Center UI

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **Node.js** | >= 18 | React UI build | [nodejs.org](https://nodejs.org/) |
| **npm** | >= 9 | Node package manager | Included with Node.js |

### Voice Mode

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **ffmpeg** | >= 5.0 | Audio format conversion | `apt install ffmpeg` / `brew install ffmpeg` / [ffmpeg.org](https://ffmpeg.org/download.html) |

Python packages (installed via `pip install cognithor[voice]`):
- `faster-whisper` — Speech-to-Text
- `piper-tts` — Text-to-Speech (Piper voices auto-download from HuggingFace)
- `sounddevice` — Audio I/O

### Browser Automation

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **Chromium** | (managed) | Headless browser | `playwright install chromium` |

### Sandbox Isolation (Linux only)

| Dependency | Purpose | Install |
|---|---|---|
| **bubblewrap** | L1 namespace isolation | `apt install bubblewrap` |
| **nsjail** | L1 namespace isolation (alternative) | [github.com/google/nsjail](https://github.com/google/nsjail) |
| **Docker** | L2 container isolation | [docs.docker.com](https://docs.docker.com/get-docker/) |

### Documents Export

Python packages (installed via `pip install cognithor[documents]`):
- `fpdf2` — PDF generation
- `python-docx` — DOCX generation

### PostgreSQL Backend (optional)

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| **PostgreSQL** | >= 14 | Alternative database backend | [postgresql.org](https://www.postgresql.org/download/) |

## Environment Variables

All configuration can be set via environment variables (prefix `JARVIS_`).
Place them in `~/.jarvis/.env` for automatic loading.

### LLM Provider Keys (set ONE for cloud backends)

```bash
# Ollama (default, no key needed — just install and run)
OLLAMA_HOST=http://localhost:11434    # Custom Ollama URL

# Cloud providers (pick one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
OPENROUTER_API_KEY=...
XAI_API_KEY=...
CEREBRAS_API_KEY=...
GITHUB_TOKEN=...
HUGGINGFACE_API_KEY=...
MOONSHOT_API_KEY=...
```

### Channel Tokens (optional, set to activate)

```bash
JARVIS_TELEGRAM_TOKEN=...
JARVIS_DISCORD_TOKEN=...
JARVIS_SLACK_TOKEN=...
JARVIS_WHATSAPP_TOKEN=...
JARVIS_SIGNAL_TOKEN=...
JARVIS_MATRIX_TOKEN=...
JARVIS_TEAMS_APP_ID=...
```

### API Security (optional)

```bash
JARVIS_API_TOKEN=...           # Protect the Control Center API
JARVIS_API_HOST=127.0.0.1     # Bind address (default: localhost only)
JARVIS_API_CORS_ORIGINS=...   # Comma-separated allowed origins
```

## Platform Notes

### Windows

- Cognithor runs natively on Windows 10/11 (no WSL required).
- Use the bootstrap script for automated setup: `python scripts/bootstrap_windows.py`
- **PATH:** After `pip install -e ".[all]"`, the `cognithor` command may not be found if Python's `Scripts` directory is not in your PATH. Use `python -m cognithor` as a reliable alternative that always works.
- To fix PATH permanently: add `%APPDATA%\Python\PythonXY\Scripts` (or the `Scripts` folder inside your venv) to your system PATH.
- Long path support: Enable via Group Policy or registry if paths exceed 260 chars.

### macOS

- Apple Silicon (M1+): Ollama runs natively with Metal acceleration.
- Install Xcode Command Line Tools: `xcode-select --install`
- Homebrew recommended: `brew install python@3.12 ollama ffmpeg`

### Linux

- Any distro with Python 3.12+ and systemd (for Ollama service).
- For full sandbox isolation, install `bubblewrap` or `Docker`.

## Quick Verification

Run the built-in health check after installation:

```bash
python scripts/preflight_check.py
```

This validates Python version, Ollama connectivity, required models,
directory permissions, and optional dependency availability.
