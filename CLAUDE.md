# CLAUDE.md

Context for AI coding assistants (Claude Code, Cursor, Codex, etc.) working in this repository.

## What this is

**Cognithor · Agent OS** — local-first autonomous agent operating system. Python backend (FastAPI gateway, PGE-Trinity orchestration; default backend is Ollama-served qwen3 models, with vLLM, OpenAI, Anthropic, Gemini and 14 other backends as opt-in alternatives; 127+ MCP tools across 30+ modules) + Flutter Command Center (~200 Dart files, 60+ screens spread across feature areas + a Config sub-tree, 27 providers, 4 locales).

Owner: Alexander Söllner. License: Apache 2.0. PyPI: `pip install cognithor`. Current release: see `pyproject.toml` (`[project] version`).

## Source layout

| Path | What lives here |
|---|---|
| `src/cognithor/` | Python source. 43 subpackages. Foundational: `core/`, `mcp/`, `channels/`, `crew/`, `gateway/`, `memory/`, `security/`, `skills/`. |
| `tests/` | Pytest, ~14 500 tests, `asyncio_mode=auto`. Mirror-structure under `tests/test_<package>/`. |
| `flutter_app/` | Flutter Command Center, package name `cognithor_ui`. UI lib at `lib/`, tests at `test/`, integration at `integration_test/`. |
| `docs/` | Architecture, ADRs, API reference, superpowers specs+plans. |
| `scripts/` | Bootstrapping + maintenance utilities (`preflight_check.py`, `bootstrap_windows.py`, etc.). |
| `cognithor_bench/` | In-monorepo benchmark scaffold with own `pyproject.toml`. |
| `examples/` | Reference impls (`insurance-agent-pack/`) + tutorial (`quickstart/`). |

## Architecture in one paragraph

User input enters via a **channel** (CLI / Telegram / WebUI / 14+ others) → gateway routes to **PGE-Trinity** (`Planner` formulates a plan, `Gatekeeper` classifies risk GREEN/YELLOW/ORANGE/RED, `Executor` runs the steps). All three are qwen3 models served by Ollama by default (vLLM is an opt-in alternative for NVIDIA GPUs). Persistence: 6-tier memory (`memory/`), audit-chain JSONL (`hashline/`), optional Postgres (`db/`). Live observability: `crew.trace_bus.TraceBus` → REST `/api/crew/*` + WebSocket `crew_*_subscribe` frames → Flutter Trace-UI screen. Skills, Agent Packs, and MCP tools are dynamically loaded.

## Critical conventions

- **Never run `git add -A` / `git add .`** — always stage exact paths. The repo has long-running pre-existing dirty state under `skills/` (auto-regen artefacts, see `git diff` carefully — frontmatter strips look destructive, do NOT commit them) and `docs/integrations/catalog.json` (timestamp-only regen, harmless to drop).
- **Never chain `git push` + `gh pr merge` + branch cleanup in one command** — Python's HTTPError swallows exit codes; a failed merge then deletes the branch. Always: merge in one turn, cleanup in the next.
- **Always run `ruff format` before commit** — CI has a separate `ruff format --check` step; lint-only passes mask format drift.
- **`from __future__ import annotations` + FastAPI**: types in endpoint signatures must be module-level importable, NOT forward-string-only.
- **Owner-gated APIs**: gate with `cognithor.security.owner.require_owner(user_id)`; reads `COGNITHOR_OWNER_USER_ID` env var.
- **Coverage gate**: `--cov-fail-under=89` in CI's full-regression step.
- **Win-py3.12 historically flaky**: `time.time()` resolution ~16ms on Windows; tests asserting strict ordering on back-to-back records need explicit insertion-seq tie-break.

## How to test

| What | Command |
|---|---|
| Full backend regression | `pytest tests/ -x -q` (~13 min, 14 448 tests at v0.95.0) |
| One backend module | `pytest tests/test_<pkg>/ -v` |
| Flutter unit tests | `cd flutter_app && flutter test test/` (~3s, 48 tests) |
| Flutter analyze | `cd flutter_app && flutter analyze` |
| Flutter format check | `cd flutter_app && dart format --output=none --set-exit-if-changed lib/ test/ integration_test/` |
| Backend lint | `ruff check src/cognithor/ tests/` |
| Backend format check | `ruff format --check src/cognithor/ tests/` |
| Backend type check | `mypy --strict src/cognithor/<file>.py` (per-file; whole-tree may have legacy issues) |

## Release flow (memorise)

1. Feature work on `feat/<name>` branches → PR to main → squash-merge.
2. After all feature PRs for a version: Direct-Commit on main bumps `pyproject.toml` + `src/cognithor/__init__.py` + `flutter_app/pubspec.yaml` + `flutter_app/lib/providers/connection_provider.dart` (`kFrontendVersion`) — all 4 must match.
3. Update `README.md` "What's New" + roll `CHANGELOG.md` `[Unreleased]` → `[X.Y.Z]`.
4. `git tag vX.Y.Z` + `git push origin vX.Y.Z`.
5. **Manually** `gh workflow run publish.yml --ref vX.Y.Z` — never auto-trigger from the tag push (loop-prevention, see v0.94.x lessons).

## Tooling

- gh CLI: `/c/Program Files/GitHub CLI/gh.exe` (Windows). Authed against `Alex8791-cyber/cognithor`.
- Ollama: `http://localhost:11434` by default. Override via `OLLAMA_HOST` env var.
- User home: `%USERPROFILE%\.cognithor\` (Windows) / `~/.cognithor/` (Linux/Mac).

## Pointers

- Architecture overview: `docs/ARCHITECTURE.md`.
- ADRs: `docs/adr/` (currently: ADR-0001 PGE-Trinity vs GroupChat).
- Spec & plan archive: `docs/superpowers/{specs,plans}/<YYYY-MM-DD-<name>>.md`.
- API reference: `docs/api/crew-traces.md` (more endpoints to come).
- Channel guide: `docs/CHANNELS_GUIDE.md`.

## House rules for AI assistants

- Plan before refactoring large files (`channels/config_routes.py` 6 600 LOC, `gateway/gateway.py` 5 800 LOC). Don't split them in a single drive-by commit.
- The Flutter test suite is thin (97 % screens, 89 % providers untested). Treat any new Flutter code as the only line of defence; write the test even if not requested.
- When the user says "weiter", read `~/.claude/projects/D--Jarvis/memory/MEMORY.md` for the live state of in-progress releases.
- Don't add fallback / legacy / "for backwards compat" code unless explicitly asked. Trust internal contracts. Validate at user-input + external-API boundaries only.
- Comments: only when WHY is non-obvious. Don't restate what the code does.
- Match existing conventions before introducing new ones (camelCase keys in ARB files, `_asyncio` aliasing decision in `webui.py`, etc.).
