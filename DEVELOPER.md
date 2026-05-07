# Cognithor Developer Guide

> How to extend, test, and contribute to the Cognithor agent OS.
> For architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).
> For user-facing setup, see [QUICKSTART.md](QUICKSTART.md).

## Table of Contents

- [Repository Layout](#repository-layout)
- [Development Setup](#development-setup)
- [Adding an MCP Tool](#adding-an-mcp-tool)
- [Adding a Channel](#adding-a-channel)
- [Creating a Skill](#creating-a-skill)
- [Configuration System](#configuration-system)
- [Testing](#testing)
- [Code Style & Linting](#code-style--linting)
- [Common Patterns](#common-patterns)

---

## Repository Layout

```
cognithor/
├── src/cognithor/              # Main source package
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # 3-layer config cascade
│   ├── models.py               # Shared Pydantic dataclasses
│   ├── core/                   # PGE-Trinity + supporting systems
│   │   ├── planner.py          # LLM reasoning engine
│   │   ├── gatekeeper.py       # Deterministic policy engine
│   │   ├── executor.py         # Sandboxed action runner
│   │   ├── sandbox.py          # Process isolation (bwrap/jobobject)
│   │   ├── reflector.py        # Post-action knowledge extraction
│   │   ├── model_router.py     # LLM selection by task type
│   │   ├── context_pipeline.py # Pre-Planner enrichment
│   │   ├── personality.py      # Tone, warmth, humor tuning
│   │   ├── sentiment.py        # Keyword-based sentiment detection
│   │   └── ...
│   ├── mcp/                    # MCP tool modules (29 modules)
│   │   ├── client.py           # MCP client + tool registry
│   │   ├── filesystem.py       # read_file, write_file, edit_file, ...
│   │   ├── shell.py            # exec_command (sandboxed)
│   │   ├── web.py              # web_search, web_fetch, http_request
│   │   ├── code_tools.py       # analyze_code, run_python
│   │   ├── media.py            # image/audio/document processing
│   │   ├── memory_server.py    # search_memory, save_to_memory, ...
│   │   ├── vault.py            # vault_search, vault_read, vault_save
│   │   └── ...
│   ├── channels/               # Communication channels (16+)
│   │   ├── base.py             # Channel ABC + StatusType enum
│   │   ├── cli.py              # CLI REPL
│   │   ├── telegram.py         # Telegram bot
│   │   ├── discord.py          # Discord bot
│   │   └── ...
│   ├── gateway/                # Orchestration layer
│   │   ├── gateway.py          # Main message loop + 8-phase init
│   │   ├── phases/             # Modular init (core, memory, security, ...)
│   │   ├── routes.py           # FastAPI REST endpoints
│   │   └── monitoring.py       # Health + metrics
│   ├── skills/                 # Skill registry + community marketplace
│   │   ├── registry.py         # SkillRegistry, keyword matching
│   │   ├── community/          # CommunityRegistryClient, validator
│   │   └── generator.py        # Self-improvement skill generator
│   ├── memory/                 # 5-tier memory system
│   │   ├── manager.py          # MemoryManager (tiers 1-5)
│   │   ├── search.py           # Hybrid BM25 + vector + graph
│   │   └── vector_index.py     # FAISS HNSW index
│   ├── i18n/                   # Internationalization
│   │   ├── __init__.py         # t(), set_locale(), get_locale()
│   │   └── locales/            # en.json, de.json, zh.json
│   ├── security/               # Token store, credential masking
│   ├── audit/                  # Immutable decision logging
│   └── learning/               # Prompt evolution, gap detection
├── tests/                      # 868+ test files, 17,000+ test functions (live: pytest --collect-only -q tests/ | tail -5)
├── ui/                         # React 19 + Vite 7.3 WebUI
├── apps/pwa/                   # Preact + Capacitor 7 PWA
├── skills/                     # Built-in skill definitions (.md)
├── data/procedures/            # Procedural skill library
├── deploy/                     # Docker, systemd, nginx configs
├── scripts/                    # Bootstrap, preflight, utilities
├── pyproject.toml              # Package metadata + tool config
└── requirements.txt            # Pinned dependencies
```

---

## Development Setup

```bash
# 1. Clone
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Install in development mode (all extras)
pip install -e ".[dev,search,documents,telegram,discord]"

# 4. Pull required Ollama models
ollama pull qwen3:8b               # Executor
ollama pull nomic-embed-text       # Embeddings

# 5. Run preflight check
python scripts/preflight_check.py

# 6. Start Cognithor
python -m cognithor
```

### UI Development

```bash
cd ui
npm install
npm run dev                        # Starts Vite + auto-launches backend
```

The Vite `cognithorLauncher` plugin manages the backend lifecycle on port 8741.

---

## Adding an MCP Tool

MCP tools follow a consistent registration pattern. Each tool module has:
1. A class implementing the tool methods
2. A `register_*_tools(mcp_client, config)` function

### Step-by-Step

**1. Create your tool module** in `src/cognithor/mcp/`:

```python
# src/cognithor/mcp/my_tools.py
from __future__ import annotations

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


class MyTools:
    def __init__(self, config):
        self._config = config

    async def hello_world(self, *, name: str = "World") -> str:
        """Greets someone by name."""
        return f"Hello, {name}!"


def register_my_tools(mcp_client, config) -> MyTools:
    tools = MyTools(config)

    mcp_client.register_builtin_handler(
        "hello_world",
        tools.hello_world,
        description="Greets someone by name.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                },
            },
            "required": ["name"],
        },
    )

    log.info("my_tools_registered", tools=["hello_world"])
    return tools
```

**2. Wire into gateway initialization** in `src/cognithor/gateway/phases/tools.py`:

```python
from cognithor.mcp.my_tools import register_my_tools

# Inside init_tools():
register_my_tools(mcp_client, config)
```

**3. Register with the Gatekeeper** in `src/cognithor/core/gatekeeper.py`:

Add your tool to the appropriate risk set in `_classify_risk()`:

```python
_GREEN_TOOLS = {"read_file", "list_directory", ..., "hello_world"}
```

Unknown tools default to **ORANGE** (requires user approval).

**4. Update test assertions** in `tests/test_tool_registration.py`.

### Key Rules

- Tool handlers **must be async** (`async def`)
- Return error strings — never raise exceptions
- File paths must use `.resolve()` + `.relative_to(root)` for validation
- Output is capped at 50 KB per tool call
- Timeout defaults to 30s (override in `model_router.py` `TOOL_TIMEOUTS`)

---

## Adding a Channel

All channels extend the `Channel` ABC from `channels/base.py`.

### Required Methods

```python
class Channel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self, handler: MessageHandler) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> None: ...

    @abstractmethod
    async def request_approval(self, session_id, action, reason) -> bool: ...

    @abstractmethod
    async def send_streaming_token(self, session_id, token) -> None: ...

    # Optional (has default no-op):
    async def send_status(self, session_id, status: StatusType, text) -> None: ...
```

### Minimal Example

```python
# src/cognithor/channels/my_channel.py
from cognithor.channels.base import Channel, StatusType

class MyChannel(Channel):
    @property
    def name(self) -> str:
        return "my_channel"

    async def start(self, handler):
        self._handler = handler
        # Your event loop (poll API, listen on socket, etc.)
        while True:
            text = await self._receive()
            msg = IncomingMessage(channel=self.name, user_id="user1", text=text)
            response = await self._handler(msg)
            await self.send(response)

    async def stop(self):
        pass  # Clean shutdown

    async def send(self, message):
        await self._deliver(message.text)

    async def request_approval(self, session_id, action, reason):
        return True  # Or prompt user

    async def send_streaming_token(self, session_id, token):
        pass  # Or stream to client

    async def send_status(self, session_id, status, text):
        pass  # Optional progress feedback
```

### StatusType Values

| Status | When |
|--------|------|
| `THINKING` | Planner is reasoning |
| `SEARCHING` | Web search in progress |
| `EXECUTING` | Running a tool |
| `RETRYING` | Retrying after error |
| `PROCESSING` | General processing |
| `FINISHING` | Wrapping up response |

---

## Creating a Skill

Skills are Markdown files with YAML frontmatter. The Planner reads the skill body
as instructions for how to handle the matched user query.

### Skill File Format

```markdown
---
name: my-skill
trigger_keywords: [keyword1, keyword2, keyword3]
tools_required: [web_search, write_file]
category: research
priority: 5
description: "Short description for the registry"
enabled: true
---
# Skill Title

## When to Apply
Describe the situations where this skill should activate.

## Steps
1. Step one...
2. Step two...

## Known Pitfalls
- Edge cases to watch for

## Quality Criteria
- How to evaluate success
```

### Where to Place Skills

| Location | Type | Loaded |
|----------|------|--------|
| `data/procedures/` | Built-in | Always |
| `~/.cognithor/skills/` | User custom | Always |
| `~/.cognithor/skills/community/` | Marketplace | If installed |

### How Matching Works

The `SkillRegistry.match(query)` method:
1. Extracts keywords from the user query
2. Matches against `trigger_keywords` (exact + fuzzy, 70% threshold)
3. Scores by overlap count + success rate bonus
4. Returns ranked `SkillMatch[]` — best match is injected into Working Memory

### Community Skills

Community skills have an additional `manifest.json` with `tools_required`.
The `ToolEnforcer` restricts community skills to only their declared tools,
preventing privilege escalation.

---

## Configuration System

Three-layer cascade (lowest to highest priority):

```
Code defaults  →  ~/.cognithor/config.yaml  →  COGNITHOR_* env vars (JARVIS_* still accepted as legacy alias)
```

### Key Config Sections

| Section | Key Fields |
|---------|------------|
| `ollama` | `base_url`, `timeout_seconds`, `keep_alive` |
| `models.planner` | `name` (default: `qwen3:32b`) |
| `models.executor` | `name` (default: `qwen3:8b`) |
| `gatekeeper` | `default_risk_level`, `max_blocked_retries` |
| `planner` | `max_iterations`, `temperature`, `response_token_budget` |
| `web` | `searxng_url`, `brave_api_key`, `duckduckgo_enabled` |
| `personality` | `warmth`, `humor`, `follow_up_questions` |
| `executor` | `default_timeout_seconds`, `max_retries`, `max_parallel_tools` |

### Environment Variable Mapping

Config keys map to env vars via prefix + underscore nesting:

```
models.planner.name  →  COGNITHOR_MODELS_PLANNER_NAME
web.searxng_url      →  COGNITHOR_WEB_SEARXNG_URL
personality.warmth   →  COGNITHOR_PERSONALITY_WARMTH
```

### Sensitive Values

Passwords and API keys are **never** stored in `config.yaml`. They use
env var references:

```yaml
email:
  password_env: "COGNITHOR_EMAIL_PASSWORD"  # Name of env var, not the password
```

---

## Testing

### Running Tests

```bash
# Full suite (17,000+ test functions, ~13 min on a warm tree)
pytest tests/

# Authoritative live count
pytest --collect-only -q tests/ | tail -5

# Fast (skip slow/integration)
pytest tests/ -m "not slow and not integration"

# Single module
pytest tests/test_planner.py -v

# With coverage
pytest tests/ --cov=src/cognithor --cov-report=html
```

### Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"              # All async tests run automatically
asyncio_default_fixture_loop_scope = "function"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests requiring external services",
]
```

### Test Organization

```
tests/
├── test_planner.py              # Core planner tests
├── test_gatekeeper.py           # Gatekeeper risk classification
├── test_executor.py             # Sandboxed execution
├── test_tool_registration.py    # Tool count assertions
├── test_web.py                  # Web search + fetch
├── test_memory/                 # Memory subsystem
├── test_channels/               # Channel implementations
├── test_skills/                 # Skill matching + validation
├── test_integration/            # Full pipeline tests
├── test_v036/                   # v0.36 feature tests
└── ...
```

### Testing Patterns

**Async tests** — just use `async def`. `asyncio_mode = "auto"` handles the rest:

```python
async def test_planner_creates_plan():
    planner = Planner(config, llm_mock)
    plan = await planner.plan("search for Python tutorials")
    assert plan.steps
```

**CircuitBreaker tests** — must wait for `recovery_timeout`:

```python
async def test_circuit_opens():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    for _ in range(2):
        cb.record_failure()
    assert cb.state == "open"
    await asyncio.sleep(0.15)
    assert cb.state == "half_open"
```

**Gatekeeper audit tests** — flush the buffer:

```python
def test_audit_writes():
    gk = Gatekeeper(config)
    gk.evaluate(action)
    gk._flush_audit_buffer()  # Buffer threshold is 10
    assert audit_file.exists()
```

**ContextVar isolation** — use an autouse fixture:

```python
@pytest.fixture(autouse=True)
def reset_coding_override():
    from cognithor.core.model_router import _coding_override_var
    token = _coding_override_var.set(None)
    yield
    _coding_override_var.reset(token)
```

---

## Code Style & Linting

### Ruff Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 110

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = ["E501"]  # Line length handled by formatter
```

### Running Linters

```bash
# Check
ruff check src/ tests/

# Auto-fix
ruff check src/ tests/ --fix

# Format
ruff format src/ tests/
```

### CI Pipeline

GitHub Actions runs on every push:
1. `ruff check` — lint errors fail the build
2. `ruff format --check` — formatting differences fail the build
3. `pytest tests/` — all tests must pass
4. CodeQL — security analysis (CWE-22 path traversal, etc.)

---

## Common Patterns

### Path Validation (CWE-22 Prevention)

```python
resolved = user_path.resolve()
try:
    resolved.relative_to(allowed_root.resolve())
except ValueError:
    raise SecurityError("Path traversal blocked")
```

### Structured Logging

```python
from cognithor.utils.logging import get_logger
log = get_logger(__name__)

log.info("tool_registered", name="hello_world", category="greeting")
log.error("tool_failed", name="hello_world", error=str(exc))
```

### i18n Translations

```python
from cognithor.i18n import t

msg = t("error.timeout")                          # Simple lookup
msg = t("error.rate_limited", service="Ollama")   # With interpolation
```

### Portability (Cross-Platform)

```python
import sys, os, tempfile

sys.executable          # Not "python"
os.sep                  # Not "/"
tempfile.gettempdir()   # Not "/tmp"
asyncio.get_running_loop()  # Not get_event_loop()
```
