# Quickstart -- From Clone to First Conversation

This guide gets Cognithor running in 10 minutes.

## 1. Prerequisites

- **Python 3.12+** -- `python3 --version`
- **LLM Backend** (one of):
  - **Ollama** -- [ollama.ai](https://ollama.ai) (recommended, CLI-based)
  - **LM Studio** -- [lmstudio.ai](https://lmstudio.ai) (GUI, OpenAI-compatible API on port 1234)
- **GPU recommended** -- RTX 3090+ (24 GB VRAM) or RTX 5090 (32 GB VRAM)

## 2. Installation

```bash
git clone <repo-url> cognithor
cd cognithor
chmod +x install.sh
./install.sh
```

The installer detects your system and asks for the desired mode:
- **Minimal** -- Core features, CLI
- **Full** -- All features (Telegram, Cron, Web Search)
- **Systemd** -- Full + autostart as a service
- **Docker** -- Container build

Alternatively, install manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## 3. Download Ollama Models

```bash
# Required
ollama pull qwen3:32b            # Planner -- the "brain" (20 GB VRAM)
ollama pull qwen3:8b             # Executor -- fast execution (6 GB VRAM)
ollama pull qwen3-embedding:0.6b # Embeddings -- vector search (0.5 GB VRAM)

# Optional (for code tasks)
ollama pull qwen3-coder:30b      # Code specialist (20 GB VRAM)
```

Start Ollama (if not running automatically):
```bash
ollama serve
```

### Alternative: LM Studio Instead of Ollama

If you prefer LM Studio, download your models in the LM Studio GUI and set in `~/.cognithor/config.yaml`:

```yaml
llm_backend_type: "lmstudio"
# lmstudio_base_url: "http://localhost:1234/v1"  # Default
```

LM Studio requires no API key and runs completely locally (like Ollama).

## 4. First Boot -- Validate the System

```bash
python scripts/first_boot.py
```

This script checks:
1. Python version and imports
2. Ollama reachable, models loaded
3. LLM responds (Planner + Executor)
4. Embeddings working
5. CORE.md and procedures created
6. Complete agent loop (real conversation)
7. Memory roundtrip (write + read)
8. Procedure matching (keyword triggers)

Quick test (LLM only, no agent loop):
```bash
python scripts/first_boot.py --quick
```

Automatically download missing models:
```bash
python scripts/first_boot.py --fix
```

## 5. Start Cognithor

### Option A: One-Click (recommended)

```
Double-click  start_cognithor.bat  -->  Browser opens  -->  Click "Power On"  -->  Done.
```

The batch file starts the Control Center UI, which automatically manages the Python backend process (start, stop, health checks, orphan cleanup). No terminal knowledge required.

> **Tip:** There is a desktop shortcut called **Cognithor** -- just double-click it.

### Option B: CLI

```bash
python -m cognithor
```

You will see the CLI REPL:
```
+----------------------------------+
|  Cognithor - Agent OS v0.99.0    |
|  Model: qwen3:32b                |
|  Tools: 141 registered           |
+----------------------------------+

User > _
```

## 6. First Conversations

### Direct Answer (Option A)
```
User > What is the difference between REST and GraphQL?
```
Cognithor answers directly from its knowledge -- no tool call needed.

### Tool Plan (Option B)
```
User > List the files in my workspace directory.
```
Cognithor creates a plan -> Gatekeeper checks -> Executor runs `list_directory`.

### Using Memory
```
User > Remember this: Contact Mueller, software developer, company TechCorp.
```
Cognithor stores the data in Semantic Memory.

```
User > What do you know about Contact Mueller?
```
Cognithor searches memory and returns the stored information.

### Procedure Trigger
```
User > Prepare the meeting with TechCorp tomorrow.
```
Cognithor detects the meeting pattern, loads the `meeting-preparation` procedure, and systematically gathers background information.

### Morning Briefing
```
User > What's on the agenda today?
```
Cognithor loads yesterday's episodes, open tasks, and creates a daily overview.

## 7. Customize Configuration

```bash
# Main configuration
nano ~/.cognithor/config.yaml

# Identity & rules
nano ~/.cognithor/memory/CORE.md

# Edit/add procedures
ls ~/.cognithor/memory/procedures/
```

Important config options:
```yaml
ollama:
  base_url: http://localhost:11434    # Ollama URL (default)
  timeout_seconds: 120                 # Timeout per request

models:
  planner:
    name: qwen3:32b                    # Or a smaller model with limited VRAM
    context_window: 32768

security:
  allowed_paths:                        # File access restricted to these paths
    - ~/.cognithor
    - ~/Documents

personality:
  warmth: 0.7                            # How warm/empathetic Cognithor responds
  humor: 0.3                             # Humor level (0 = factual, 1 = playful)
  greeting_enabled: true                 # Time-of-day greetings
```

## 8. Monitoring

```bash
make smoke        # Installation checks (preflight)
make health       # Runtime check (Ollama, disk, memory)
make test         # Run the full backend regression (17,000+ test functions, ~13 min on a warm tree)
# Authoritative count: pytest --collect-only -q tests/ | tail -5
```

Logs:
```bash
tail -f ~/.cognithor/logs/cognithor.log
```

## 9. Server Deployment (optional)

Cognithor can also run on a server:

### Docker (Production)

```bash
cp .env.example .env   # Edit: set JARVIS_API_TOKEN
docker compose -f docker-compose.prod.yml up -d

# Optional: add PostgreSQL or Nginx
docker compose -f docker-compose.prod.yml --profile postgres --profile nginx up -d
```

### Bare-Metal (Ubuntu/Debian)

```bash
sudo bash deploy/install-server.sh --domain cognithor.example.com --email admin@example.com
```

See [`deploy/README.md`](deploy/README.md) for complete deployment documentation.

## Next Steps

- **Telegram bot** -- Set token in `~/.cognithor/.env`: `COGNITHOR_TELEGRAM_TOKEN=...`
- **Cron jobs** -- Enable morning briefing, weekly review
- **Custom procedures** -- Create workflows in `~/.cognithor/memory/procedures/`
- **CORE.md** -- Add your own rules and preferences
- **Server deployment** -- See `deploy/README.md` for Docker, bare-metal, TLS

If you run into problems: run `python scripts/first_boot.py --fix` again.
