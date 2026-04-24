# 07 · Troubleshooting

The most common stumbles from spec §8 and early user reviews — with a one-line fix wherever possible.

**Prerequisites:** none — this page is a reference.
**Time:** 3 minutes to the specific case.

---

## Ollama is not running

```
ConnectionError: Could not reach Ollama at http://127.0.0.1:11434
```

**Fix:**

```bash
# Linux/macOS:
ollama serve &

# Windows:
# 1. Start menu → launch "Ollama"
# 2. or as a service: net start ollama
```

Verify: `curl http://127.0.0.1:11434/api/version` returns JSON.

## Model pull failed

```
Error: model 'qwen3:8b' not found locally
```

**Fix:**

```bash
ollama pull qwen3:8b
# ~4.7 GB download — about 5 min on 50 Mbit/s

# Check disk space:
# Linux/macOS:  df -h /
# Windows:      wmic logicaldisk get size,freespace,caption
```

## Port conflict

```
OSError: [Errno 98] Address already in use — port 8741
```

**Fix:**

```bash
# Who owns the port?
# Linux/macOS:  lsof -i :8741
# Windows:      netstat -ano | findstr :8741

# Alternative: use a different port
export COGNITHOR_API_PORT=9000
cognithor
# or
cognithor --api-port 9000
```

## GuardrailFailure

```
cognithor.crew.errors.GuardrailFailure: Guardrail failed after 3 attempt(s): too long
```

**Causes:**
1. The agent consistently produces output that violates the guardrail (too long / PII / not schema-conformant).
2. The guardrail feedback is too vague — the agent doesn't understand what to change.

**Fix:**
- **Sharpen the feedback:** instead of "too long" → `f"Output has {count} words, max {max} allowed"`.
- **Increase `max_retries`:** default 2, with complex guardrails maybe 5.
- **Unit-test the guardrail:** write a test that only calls the guardrail function — otherwise you debug blind.

Details: [`docs/guardrails.md`](../guardrails.md).

## Unknown tool

```
cognithor.crew.errors.ToolNotFoundError: Tool 'web_seach' not found. Did you mean 'web_search'?
```

**Fix:**

```bash
# List all available tools
cognithor tools list

# Or via Python:
python -c "from cognithor.crew.tool_resolver import available_tool_names; \
           from cognithor.mcp.tool_registry_db import ToolRegistryDB; \
           print(available_tool_names(ToolRegistryDB('~/.cognithor/db/tool_registry.db')))"
```

The `did_you_mean` suggestion in the error often reveals the typo directly.

## Template name collision

```
Error: Directory 'my-project' already exists and is not empty
```

**Fix:**

```bash
# Pick a different name
cognithor init my-project-v2 --template research

# Or overwrite
cognithor init my-project --template research --force
```

Careful: `--force` does **not** delete — it merges. Your own changes can be overwritten — starting in an empty directory is safer.

## Planner model too slow

```
TimeoutError: Planner took longer than 60s
```

**Fix:**
- Use a smaller model: `COGNITHOR_MODEL_PLANNER=qwen3:8b` instead of `qwen3:32b`.
- Check GPU support: `ollama ps` shows load type (CPU vs GPU).
- `cognithor --lite` boots with fewer background services.

## "cognithor_home not initialized"

```
RuntimeWarning: cognithor config load failed (...); using temp-dir tool registry
```

**Fix:**

```bash
# One-time bootstrap initialization
cognithor --init-only

# Or: manually nuke if corrupted
rm -rf ~/.cognithor/.cognithor_initialized
cognithor --init-only
```

## Ollama installed as Windows service, `ollama` CLI not in PATH

```
'ollama' is not recognized as an internal or external command
```

**Fix:** open a new shell (PATH update requires a fresh session). If still missing: `where.exe ollama` — if empty, reinstall.

## More help

- [`docs/quickstart/EXTERNAL_REVIEW_RESULTS.md`](EXTERNAL_REVIEW_RESULTS.md) — currently known stumbles from external reviews.
- [GitHub Issues](https://github.com/Alex8791-cyber/cognithor/issues) — search the error message, often already resolved.

---

**End of Quickstart.** Back to the [overview](README.en.md).
