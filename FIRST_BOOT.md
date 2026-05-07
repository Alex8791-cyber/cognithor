# Cognithor - First Boot Guide

> From installation to your first real conversation.

## Overview

This document walks you through starting Cognithor for the first time on your machine.
By the end, you will have a functioning agent system that answers questions,
manages files, and supports custom workflows.

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | RTX 5090 (32 GB) |
| CPU | 8 cores | Ryzen 9 9950X3D |
| RAM | 16 GB | 64 GB+ |
| Python | 3.12+ | 3.12+ |
| Ollama or LM Studio | 0.3+ / 0.3+ | Latest version |
| Disk | 50 GB free | 100 GB+ (for models) |

## Step 1: Installation

```bash
# Clone the repository
git clone <repo-url> cognithor
cd cognithor

# Recommended: Interactive installer
./install.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## Step 2: Prepare the LLM Backend

### Option A: Ollama (recommended)

```bash
# Start Ollama (if not running as a service)
ollama serve &

# Download models
ollama pull qwen3:32b           # Planner -- 20 GB, ~2 min
ollama pull qwen3:8b            # Executor -- 6 GB, ~30 sec
ollama pull qwen3-coder:30b     # Coder -- 20 GB, ~2 min
ollama pull qwen3-embedding:0.6b # Embeddings -- 0.5 GB, ~5 sec
```

**RTX 5090 tip:** Planner (20 GB) + Executor (6 GB) = 26 GB -- they fit simultaneously
in 32 GB VRAM. Qwen3-Coder shares memory with the Planner (Ollama
unloads automatically).

### Option B: LM Studio

1. Download and load models in the LM Studio GUI (e.g., `qwen/qwen3-32b`)
2. Start the server (runs by default on `http://localhost:1234`)
3. Set in `~/.cognithor/config.yaml`:

```yaml
llm_backend_type: "lmstudio"
```

LM Studio requires no API key and stays completely local.

## Step 3: First Boot

```bash
# Full validation (recommended for first time)
python scripts/first_boot.py

# Quick test (system + Ollama + LLM only, no agent loop)
python scripts/first_boot.py --quick

# Automatically download missing models
python scripts/first_boot.py --fix
```

The first boot test checks:

| Check | What Is Tested |
|-------|----------------|
| System | Python version, GPU, Ollama binary |
| Models | Planner, Executor, Coder, Embedding available |
| LLM | Chat with Planner + Executor (response time) |
| Embeddings | Vector generation (single + batch) |
| Memory | CORE.md, procedures, policies, directories |
| Agent Loop | Complete PGE request (Plan -> Gate -> Execute) |

**Expected output:**

```
+----------------------------------------------+
|         Cognithor - First Boot                |
|         First start with real Ollama          |
+----------------------------------------------+

--------------------------------------------------
  1. System Check
--------------------------------------------------
  [OK] Python 3.12.x
  [OK] jarvis package importable
  [OK] Ollama binary found
  [OK] GPU: NVIDIA GeForce RTX 5090 -- 32 GB total, 28.5 GB free

  ...

  [OK] FIRST BOOT SUCCESSFUL
  12/12 checks passed

  Cognithor is ready!
  Start with: start_cognithor.bat (or: python -m jarvis)
```

## Step 4: First Start

### Option A: One-Click (recommended)

```
Double-click  start_cognithor.bat  -->  Browser opens  -->  Click "Power On"  -->  Done.
```

The UI automatically starts the Python backend process. No terminal needed.

> **Desktop shortcut:** A shortcut called **Cognithor** is on the desktop.

### Option B: CLI

```bash
python -m jarvis
```

You will see the CLI REPL:

```
+--------------------------------------+
|  Cognithor - Agent OS                |
|  Local AI Assistant                  |
+--------------------------------------+

cognithor>
```

### First Conversations to Try

**Direct answer (Option A):**
```
cognithor> What is a REST API?
cognithor> Explain the difference between Docker and Podman.
cognithor> How are you?
```

**Tool plan (Option B):**
```
cognithor> Show me the files in my workspace.
cognithor> What do you know about me?
cognithor> Create a file with a checklist for new client meetings.
```

**Procedure trigger:**
```
cognithor> I have an appointment tomorrow with a new lead: John Smith, IT entrepreneur.
cognithor> Prepare a meeting with TechCorp -- topic: cloud migration.
cognithor> What's on the agenda today?
```

## Step 5: Customize Configuration

The configuration is at `~/.cognithor/config.yaml`:

```yaml
# Most important settings:
ollama:
  base_url: http://localhost:11434    # Ollama server URL
  timeout_seconds: 120                 # Timeout for long planning tasks

planner:
  max_iterations: 10                   # Max steps per request

memory:
  chunk_size_tokens: 400               # Chunk size for indexing
  search_top_k: 6                      # Number of memory search results
```

## Step 6: Refine Identity

Cognithor's personality and rules are in `~/.cognithor/CORE.md`.
You can edit this file at any time:

```bash
# Open with your editor
nano ~/.cognithor/CORE.md
```

Or directly through Cognithor:
```
cognithor> Update the CORE.md: Add under expertise that I also advise on property insurance.
```

## Architecture Overview

What happens during a request:

```
You: "Prepare the meeting with contact Smith"
 |
 v
[CLI Channel] -> IncomingMessage
 |
 v
[Gateway] -> Create/load session
 |
 v
[Memory Manager] -> Load CORE.md, search relevant memories
 |
 v
[Planner (qwen3:32b)] -> Create plan:
 |  1. search_memory("client Smith")
 |  2. write_file("meeting-prep.md", ...)
 |
 v
[Gatekeeper] -> Check each step:
 |  [OK] search_memory -> ALLOW (safe)
 |  [OK] write_file -> INFORM (file will be written)
 |
 v
[Executor] -> Execute tools:
 |  1. MCP: search_memory -> Client data found
 |  2. MCP: write_file -> File created
 |
 v
[Planner (replan)] -> Interpret results -> Formulate response
 |
 v
[Reflector] -> Evaluate session, extract facts, learn procedures
 |
 v
You: "Here is the meeting preparation for contact Smith: ..."
```

## Troubleshooting

### Ollama is not responding
```bash
# Is Ollama running?
curl http://localhost:11434/api/tags

# Restart:
pkill ollama
ollama serve &
```

### Model loading is slow
On the first call, Ollama loads the model into VRAM. This can take
30-60 seconds. After that, responses come in 1-5 seconds.

```bash
# Preload models:
ollama run qwen3:32b "Hello" --keepalive 30m
ollama run qwen3:8b "Hello" --keepalive 30m
```

### Planner does not create a plan (always answers directly)
This happens when the model does not recognize the tool list. Check:
1. Are MCP tools registered? Check log: `~/.cognithor/logs/cognithor.log`
2. Is the system prompt too long? Reduce `memory.search_top_k`
3. Is the temperature too low? In `config.yaml`: `planner.temperature: 0.7`

### Memory search finds nothing
Memory needs to be populated first. On first start the database is empty.
```
cognithor> Remember: My most important client is Mueller Corp, mechanical engineering, 50 employees.
cognithor> What do you know about Mueller Corp?
```

## Next Steps

1. **Import knowledge** -- Place existing notes in `~/.cognithor/memory/knowledge/`
2. **Set up Telegram** -- For mobile use: token in `~/.cognithor/.env`: `COGNITHOR_TELEGRAM_TOKEN=...`
3. **Enable cron** -- Automatic morning briefing: `~/.cognithor/cron/jobs.yaml`
4. **Custom procedures** -- Create recurring workflows in `~/.cognithor/memory/procedures/`
