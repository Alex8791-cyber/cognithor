# Cognithor Testing Guide

> How to run, write, and maintain tests for the Cognithor agent OS.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Configuration](#configuration)
- [Writing Tests](#writing-tests)
- [Common Patterns](#common-patterns)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Full suite (17,000+ test functions, ~13 minutes on a warm tree)
pytest tests/

# Authoritative count
pytest --collect-only -q tests/ | tail -5

# Quick check (stop on first failure)
pytest tests/ -x -q --tb=short

# Skip slow/integration tests
pytest tests/ -m "not slow and not integration"

# Single module
pytest tests/test_planner.py -v

# Single test
pytest tests/test_planner.py::TestPlanner::test_plan_creation

# With coverage
pytest tests/ --cov=src/cognithor --cov-report=html
```

### Property-based + nightly burn-in

- **Hypothesis property tests** under `tests/test_security/test_reflector_audit_completeness.py` and adjacent files exercise audit-chain invariants (NFC canonicalization idempotency, `prev_hash` linking, completeness guarantees). Run them like any other pytest module.
- **Nightly burn-in workflow** at `.github/workflows/nightly-burn-in.yml` runs `scripts/burn_in_compliance_spring.py --home <tmp>` daily at 03:00 UTC. The script generates synthetic load and asserts four green metrics (storage / audit-failures / atomic-rollbacks / behavioral-drift). Failures page the operator the next morning.

---

## Test Organization

```
tests/                          # 868+ files, 17,000+ test functions (counts grow over time — `pytest --collect-only` is authoritative)
├── conftest.py                 # Root fixtures (config, tmp_home, locale)
├── test_core/                  # (54) Planner, Gatekeeper, Executor, ...
├── test_security/              # (53) Security infrastructure, policies
├── test_channels/              # (50) CLI, API, Telegram, Discord, ...
├── test_memory/                # (34) Memory system, embeddings, vector
├── test_mcp/                   # (30) MCP tools & integrations
├── test_integration/           # (24) End-to-end pipeline tests
├── test_skills/                # (15) Skill matching, validation
├── test_gateway/               # (12) Gateway orchestration
├── test_v036/                  # (10) v0.36 feature tests
├── test_telemetry/             #  (8) Monitoring, metrics
├── test_utils/                 #  (7) Utilities, error handling
├── test_tools/                 #  (6) Tool helpers
├── test_learning/              #  (5) Learning, feedback
├── test_governance/            #  (5) Trust, permissions
├── test_coverage/              #  (5) Coverage validation
├── test_db/                    #  (3) Database persistence
├── test_audit/                 #  (3) Audit trail
├── test_cron/                  #  (3) Scheduled tasks
├── test_entrypoint/            #  (3) CLI entry points
├── test_forensics/             #  (3) Forensic analysis
├── test_proactive/             #  (3) Proactive behavior
├── test_browser/               #  (3) Browser automation
├── test_benchmark/             #  (2) Performance benchmarks
├── test_phase7/                #  (2) Phase 7 features
├── test_sdk/                   #  (2) SDK tests
└── *.py                        # (28) Root-level test files
```

---

## Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests requiring external services",
]
filterwarnings = [
    "ignore::PendingDeprecationWarning:starlette",
    "ignore:Exception ignored in.*Transport:pytest.PytestUnraisableExceptionWarning",
    "ignore:coroutine .* was never awaited:RuntimeWarning",
]
```

Key settings:
- **`asyncio_mode = "auto"`** — all `async def test_*` functions run automatically
- **`asyncio_default_fixture_loop_scope = "function"`** — fresh event loop per test
- **Warning filters** suppress known third-party noise

---

## Writing Tests

### Root Fixtures (`conftest.py`)

```python
@pytest.fixture(autouse=True)
def _set_test_locale():
    """All tests run with German locale (backwards compatibility)."""
    set_locale("de")
    yield
    set_locale("de")

@pytest.fixture
def tmp_jarvis_home(tmp_path: Path) -> Path:
    """Temporary Jarvis home directory."""
    return tmp_path / ".jarvis"

@pytest.fixture
def config(tmp_jarvis_home: Path) -> JarvisConfig:
    """JarvisConfig with temporary home."""
    return JarvisConfig(jarvis_home=tmp_jarvis_home)

@pytest.fixture
def initialized_config(config: JarvisConfig) -> JarvisConfig:
    """Config with directory structure created."""
    ensure_directory_structure(config)
    return config
```

### Async Tests

Just write `async def` — `asyncio_mode = "auto"` handles the rest:

```python
async def test_planner_creates_plan():
    planner = Planner(config, llm_mock)
    plan = await planner.plan("search for Python tutorials")
    assert plan.steps
    assert plan.confidence > 0
```

### Mocking LLM Responses

```python
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": '{"steps": [{"tool": "web_search", "params": {"query": "test"}}]}'
        }
    }
    return llm
```

### Testing MCP Tools

```python
async def test_read_file(tmp_path):
    config = JarvisConfig(workspace_dir=tmp_path)
    fs = FileSystemTools(config)

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    result = await fs.read_file(path=str(test_file))
    assert "hello world" in result
```

### Testing Channels

```python
async def test_cli_send():
    cli = CliChannel()
    msg = OutgoingMessage(text="Hello!", channel="cli")
    await cli.send(msg)  # Should not raise
```

---

## Common Patterns

### ContextVar Isolation

The `model_router._coding_override_var` is a module-level ContextVar.
Tests **must** reset it to prevent cross-test contamination:

```python
@pytest.fixture(autouse=True)
def reset_coding_override():
    from jarvis.core.model_router import _coding_override_var
    token = _coding_override_var.set(None)
    yield
    _coding_override_var.reset(token)
```

### Gatekeeper Audit Buffer

Audit writes are buffered (threshold: 10). Flush in tests:

```python
def test_audit_logged():
    gk = Gatekeeper(config)
    gk.evaluate(action)
    gk._flush_audit_buffer()
    assert audit_file.exists()
```

### CircuitBreaker Timing

Must use real `asyncio.sleep()` — recovery_timeout is not mocked:

```python
async def test_circuit_opens_and_recovers():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"

    await asyncio.sleep(0.15)
    assert cb.state == "half_open"
```

### File Timestamp Ordering

Use explicit timestamps, not filesystem mtime (1-second resolution on Windows):

```python
# Bad — non-deterministic on Windows
cp1.save(path1)
cp2.save(path2)
latest = store.get_latest()  # Might be cp1!

# Good — deterministic
cp1 = Checkpoint(timestamp_utc="2026-01-01T00:00:00")
cp2 = Checkpoint(timestamp_utc="2026-01-01T00:00:01")
```

### Tool Count Assertions

When adding new MCP tools, update assertion counts:

```python
# tests/test_tool_registration.py
assert register_builtin_handler.call_count == 79  # Update this!
```

---

## CI/CD Pipeline

### Workflow: `ci.yml`

Runs on every push and PR. Matrix: **ubuntu-latest + windows-latest** x **Python 3.12 + 3.13**.

**Lint job:**
```bash
ruff check src/ tests/ --select=F821,F811 --no-fix
ruff format --check src/ tests/
```

**Test job:**
```bash
JARVIS_TEST_MODE=1 python -m pytest tests/ -x -q --tb=short \
  --ignore=tests/test_channels/test_voice_ws_bridge.py
```

Excluded: `test_voice_ws_bridge.py` (requires system audio devices).

### Optional Dependencies

CI installs extras with fallback on failure:

```
dev, memory, search, mcp, telegram, discord, slack, matrix,
web, documents, cron, vector, postgresql, irc, twitch
```

### CodeQL

GitHub CodeQL runs security analysis checking for:
- **CWE-22** — Path traversal
- **CWE-20** — URL substring matching
- **CWE-73/99** — Tainted file paths

---

## Troubleshooting

### Tests pass locally, fail in CI

- Check platform differences (Linux vs Windows)
- File path separators: use `os.sep` or `pathlib.Path`
- Temp directory: use `tempfile.gettempdir()`, not `/tmp`
- Subprocess: use `sys.executable`, not `"python"`

### Async test hangs

- Check for missing `await` on async calls
- Ensure fixtures use `async def` for async setup
- Verify no blocking I/O in async context

### Cross-test contamination

- Reset module-level state (ContextVars, caches)
- Use `tmp_path` for file operations
- Check for singleton patterns that persist between tests

### Flaky timing tests

- Use explicit timestamps instead of `time.time()` or `st_mtime`
- For CircuitBreaker: use `recovery_timeout=0.1` with `await asyncio.sleep(0.15)`
- For rate limiters: mock `time.monotonic()`

### Environment variable conflicts

Set `JARVIS_TEST_MODE=1` to enable test-safe behavior.
Clean up env vars in fixtures:

```python
@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("JARVIS_OLLAMA_BASE_URL", raising=False)
```
