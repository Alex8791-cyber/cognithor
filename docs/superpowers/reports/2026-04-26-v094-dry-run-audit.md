# v0.94.0 End-to-End Dry-Run Audit

**Date:** 2026-04-26
**Branch / HEAD:** `main` @ `0abb9d26` (recovery commit on top of 655fe704)
**Auditor:** Claude (Opus 4.7, 1M ctx)

## Verdict: WARNING

The v0.94.0 release surface is **functionally green** (imports OK, tests OK, lint/format/mypy clean, console-scripts wired correctly in `pyproject.toml`, all cross-refs resolve, all CI YAMLs parse), but the audit surfaced **two real wiring concerns** that ship to users:

1. **HIGH** — `cognithor_bench` AutoGenAdapter has a missing-runtime-dep / misleading error.
2. **MEDIUM** — sevDesk MCP tools appear in the integrations catalog but are not actually registered with the live MCP server.

Plus one local-env issue (LOW) for the developer machine. Details below.

---

## A. Package metadata correctness

- [x] [OK] `pyproject.toml` `version = "0.94.0"`.
- [x] [OK] `src/cognithor/__init__.py` `__version__ = "0.94.0"`.
- [x] [OK] `flutter_app/pubspec.yaml` `version: 0.94.0+1`.
- [x] [OK] `flutter_app/lib/providers/connection_provider.dart` `kFrontendVersion = '0.94.0'`.
- [x] [OK] `[project.optional-dependencies] dev` contains no `@ file:` refs (verified via `tomllib`). Comment block explains why.
- [x] [OK] `[project.optional-dependencies] autogen = ["autogen-agentchat==0.7.5"]` — single pin-point, exactly as spec §6.
- [x] [OK] `all` and `full` parse cleanly; no `[tool.hatch.metadata] allow-direct-references` block (correctly removed in `0abb9d26`).

## B. Public-API surface correctness

- [x] [OK] `from cognithor import __version__` returns `"0.94.0"`.
- [x] [OK] `from cognithor.crew import Crew, CrewAgent, CrewTask, CrewProcess` resolves.
- [x] [OK] `from cognithor.compat.autogen import …` (all 9 names) resolves; emits one `DeprecationWarning` on import as designed.
- [x] [OK] `inspect.signature(AssistantAgent.__init__)` has 18 parameters (17 fields + `self`), matching upstream `autogen-agentchat==0.7.5`.
- [x] [OK] `from cognithor_bench …` (3 imports) all resolve; `__version__ == "0.1.0"`.
- [x] [OK] `from insurance_agent_pack …` (3 imports) all resolve; `__version__ == "0.1.0"`.

## C. Console-script entry points

- [⚠️] [WARN] `cognithor --version` from the dev shell hits a **stale console-script binary** at `C:/Users/ArtiCall/AppData/Roaming/Python/Python313/Scripts/cognithor.exe`. It crashes with `ModuleNotFoundError: No module named 'jarvis.__main__'`. Root cause: the user's last `pip install -e .` was at v0.52.0 (when the entry point was `jarvis.__main__:main`). `pip show cognithor` confirms `Version: 0.52.0`. **The pyproject.toml is correct** (`cognithor.__main__:main`); only the local install is stale. `python -m cognithor --version` correctly prints `Cognithor v0.94.0`. Recommended fix on dev box: `pip install -e .` to refresh metadata + console-script.
- [x] [OK] `cognithor-bench --help` prints usage with `run` and `tabulate` subcommands.
- [x] [OK] `insurance-agent-pack --help` prints usage with `run` subcommand. (Description has one mojibake char `�34d` — German `§34d` corrupted to cp1252; cosmetic only, **LOW severity**.)

## D. Targeted regression test subsets

- [x] [OK] `tests/test_compat/test_autogen/` — 45 passed, 1 expected DeprecationWarning.
- [x] [OK] `cognithor_bench/tests/` — 30 passed, 4 xpassed.
- [x] [OK] `examples/insurance-agent-pack/tests/` — 58 passed, 1 skipped (slow marker).
- [x] [OK] `tests/test_docs/` — 6 passed.
- [x] [OK] `tests/test_crew/` — 166 passed, 1 skipped.

Total: 305 passed, 2 skipped, 4 xpassed across v0.94.0-relevant subsets. No regressions vs. 655fe704's green run.

## E. Lint + type-check

- [x] [OK] `ruff check` — *All checks passed!* (compat + tests/test_compat + cognithor_bench + insurance-agent-pack).
- [x] [OK] `ruff format --check` — 67 files already formatted.
- [x] [OK] `mypy --strict src/cognithor/compat` — Success: no issues found in 11 source files.

## F. Wiring sweep

1. **MCP tool registration — sevDesk** [⚠️ MEDIUM]
   `docs/integrations/catalog.json` lists `sevdesk_get_invoice` and `sevdesk_list_contacts`. They are decorated with `@mcp_tool`, but per `src/cognithor/mcp/sevdesk/tools.py:18-23`, **`mcp_tool` is a no-op marker decorator** for AST-based catalog scanning only. The module is also **not imported anywhere** except inside its own folder (verified via grep). So a running MCP server will not expose these tools. The module's own docstring admits this: *"actual runtime registration with the Cognithor MCP server happens via the existing `register_*` convention in sibling modules; hook those up when the connector is wired into a live Gateway."* The catalog over-promises capability.
2. **Channel registrations** [OK] 3 bundled `LeadSource` subclasses (`DiscordLeadSource`, `HnLeadSource`, `RssLeadSource`) — match the 3 bundled packs (`discord-lead-hunter`, `hn-lead-hunter`, `rss-lead-hunter`). No orphans, no missing translations.
3. **`@agent` / `@task` / `@crew` decorators** [OK] Defined in `src/cognithor/crew/decorators.py`. Templates correctly import them as `from cognithor.crew.decorators import agent, crew, task`. They are NOT re-exported in `cognithor.crew.__init__`, but no doc claims they are. Decorators import OK at runtime.
4. **PGE-Trinity wiring** [OK] `from cognithor.gateway.gateway import Gateway`, `Planner`, `Gatekeeper`, `Executor` all import without error. Gateway class resolves.
5. **`cognithor.compat.autogen → cognithor.crew` bridge** [OK] `_bridge.py` imports `from cognithor.crew import Crew, CrewAgent, CrewTask`; `run_single_task()` constructs a real `Crew(agents=[...], tasks=[...])` and awaits `kickoff_async`. End-to-end mock-LLM smoke test passes (covered by `test_hello_world_search_replace.py`).
6. **Insurance pack → cognithor.crew** [OK] `build_team()` returns a `Crew` with 4 `CrewAgent`s + 4 `CrewTask`s, `process=CrewProcess.SEQUENTIAL`. Verified by instantiating live: `team built: Crew agents: 4 tasks: 4 process: CrewProcess.SEQUENTIAL`.
7. **`cognithor-bench` AutoGen adapter ImportError path** [❌ HIGH] **BUG**. See "Bugs found" below.

## G. Documentation cross-references

- [x] [OK] README → `src/cognithor/compat/autogen/README.md` ✓ exists.
- [x] [OK] `docs/competitive-analysis/autogen.md` → `docs/adr/0001-pge-trinity-vs-group-chat.md` ✓ exists.
- [x] [OK] `docs/adr/0001-pge-trinity-vs-group-chat.md` → `docs/hashline-guard.md` ✓ exists.
- [x] [OK] `examples/insurance-agent-pack/README.md` → `docs/adr/0001-pge-trinity-vs-group-chat.md` ✓ exists.
- [x] [OK] `cognithor_bench/README.md` exists (cross-refs not exhaustively checked, but file is present).
- [x] [OK] `NOTICE` lines 87-94 contain BOTH CrewAI MIT attribution AND AutoGen MIT attribution + the `autogen-agentchat==0.7.5` reference.

## H. CI workflow correctness

- [x] [OK] `.github/workflows/ci.yml:74-77` installs `pip install -e ".[dev]"` first, then `pip install -e ./cognithor_bench` and `pip install -e ./examples/insurance-agent-pack` — exactly the fix from `0abb9d26`.
- [x] [OK] No leftover `@ file:` refs anywhere in `*.toml` (verified via `tomllib` introspection AND repo-wide grep).
- [x] [OK] All 12 workflow files (`ci.yml`, `publish.yml`, `release.yml`, `build-*.yml`, etc.) parse as valid YAML.

## I. Pre-existing dirty state (informational)

- 11 modified `skills/*` files (backup, gmail_sync, test, test_skill, wetter_abfrage) — user's local WIP, untouched.
- Untracked `docs/superpowers/plans/2026-04-25-cognithor-autogen-strategy.md` and `results/` — left in place.

---

## Bugs found

### BUG-1 — `cognithor[autogen]` extra is incomplete (HIGH)

**File:** `cognithor_bench/src/cognithor_bench/adapters/autogen_adapter.py:35`
**Also affects:** `cognithor_bench/pyproject.toml:34` and root `pyproject.toml:180-187` (`[autogen]` extra).

**Symptom:** A user follows the docs:
```bash
pip install cognithor[autogen]
cognithor-bench run --adapter autogen scenarios.jsonl
```
…and gets `ImportError: AutoGenAdapter requires pip install cognithor[autogen] (or pip install autogen-agentchat==0.7.5)` — even though `autogen-agentchat==0.7.5` IS installed.

**Reproduction (verified on this dev machine):**
```bash
pip show autogen-agentchat   # Version: 0.7.5  (installed)
pip show autogen-ext         # WARNING: Package(s) not found
python -c "from cognithor_bench.adapters.autogen_adapter import AutoGenAdapter; ..."
# → ImportError ... line 35: from autogen_ext.models.openai import (OpenAIChatCompletionClient, ...)
```

**Root cause:** `OpenAIChatCompletionClient` lives in the **`autogen-ext`** PyPI package, not in `autogen-agentchat`. Both `[autogen]` extras (root + bench sub-package) only pin `autogen-agentchat==0.7.5`. The `try/except ImportError` block in `autogen_adapter.py` catches the missing-`autogen_ext` ImportError but raises with a misleading "install autogen-agentchat" hint.

**Fix paths (any one):**
1. Add `"autogen-ext[openai]>=0.7,<0.8"` to the `[autogen]` extra in BOTH `pyproject.toml` files.
2. OR: change the adapter to use `cognithor.compat.autogen.OpenAIChatCompletionClient` (the local shim) instead of importing from `autogen_ext` directly. This eliminates the second dep but means the bench adapter is no longer a "pure AutoGen" comparison.
3. AND: update the error message to mention `autogen-ext` so future users can self-diagnose.

**Why tests didn't catch this:** `test_autogen_adapter_runs_when_import_succeeds` patches `sys.modules["autogen_ext.models.openai"]` with a `MagicMock`, so the test passes regardless of whether `autogen-ext` is actually installable. Recommend a lightweight integration check `pip install cognithor[autogen] && python -c "from autogen_ext.models.openai import OpenAIChatCompletionClient"` in CI.

### BUG-2 — sevDesk MCP tools listed in catalog but not registered (MEDIUM)

**Files:** `src/cognithor/mcp/sevdesk/tools.py`, `docs/integrations/catalog.json`.

**Symptom:** `docs/integrations/catalog.json` advertises 2 sevDesk tools with `dach_specific: true`. They appear on the `/integrations` site page (per memory). However, the `@mcp_tool` decorator they use is a **local no-op marker**, not the real `register_tool` mechanism used by other MCP modules. Nothing in `cognithor/gateway/`, `cognithor/mcp/server.py`, or `cognithor/mcp/bridge.py` imports `cognithor.mcp.sevdesk`, so a running MCP server will not expose these tools to the planner.

**Reproduction:**
```bash
grep -rn "from cognithor.mcp.sevdesk" src/cognithor/  # only the module's own internal imports
```

**Severity rationale:** Not a runtime crash; the tools are just silently absent. But the catalog promises capability the running server cannot fulfil — false advertising for the integrations page.

**Fix path:** Add `register_tool` calls in `src/cognithor/mcp/sevdesk/__init__.py` (currently empty) wired through whatever sibling module pattern (`tasks.py`, `email.py`, etc.) actually registers tools at server start. Alternative: remove the entries from `docs/integrations/catalog.json` until the connector is wired.

### BUG-3 — Mojibake in insurance-agent-pack CLI description (LOW)

**File:** `examples/insurance-agent-pack/src/insurance_agent_pack/cli.py` (or wherever `parser.description` is set).
**Symptom:** `insurance-agent-pack --help` prints `Reference Cognithor pack: �34d-NEUTRAL DACH insurance pre-advisory.` — `§34d` got encoded as `�34d` because Windows console default cp1252. Other DE strings inside the package render correctly (German backstories work), so this is just a CLI metadata string that needs an `\u00a7` escape or a `# -*- coding: utf-8 -*-` header / explicit UTF-8 stream wrap.

---

## Wiring concerns (not bugs)

1. **Stale local pip install (LOW)** — The dev's `Python313/Scripts/cognithor.exe` still points at `jarvis.__main__:main` because `pip install -e .` was last run at v0.52.0. `python -m cognithor` works, but the bare `cognithor` console script doesn't. Cosmetic for development; not a release defect.
2. **Decorators not re-exported** — `cognithor.crew.{agent, task, crew}` decorators are NOT in the public `__all__` of `cognithor.crew`. Templates import them via `cognithor.crew.decorators`, which is fine. Suggestion: add to `cognithor.crew.__init__.__all__` for symmetry with the rest of the crew surface, or don't.

---

## Recommended action list (sorted by severity)

1. **HIGH** — Fix BUG-1: either add `autogen-ext[openai]>=0.7,<0.8` to `[autogen]` extras (root + bench) **OR** rewire `AutoGenAdapter` to use `cognithor.compat.autogen.OpenAIChatCompletionClient`. Update the error message either way. Add a CI step that does a real `pip install cognithor[autogen] && python -c "from autogen_ext.models.openai import OpenAIChatCompletionClient"` to prevent regression.
2. **MEDIUM** — Fix BUG-2: wire sevDesk tools into the live MCP server **OR** remove from `docs/integrations/catalog.json` until v0.95.0 ships a real connector. Add a test that asserts every tool in `catalog.json` is actually returned by the live MCP server's tool-list endpoint.
3. **LOW** — Fix BUG-3: replace `§34d` in CLI description with `\u00a7 34d` or wrap stdout in UTF-8 on Windows.
4. **LOW** (dev box only) — `pip install -e .` to refresh stale 0.52.0 console-script. Not a release blocker.
5. **Optional** — Re-export `agent/task/crew` decorators in `cognithor.crew.__all__` for ergonomics.

---

## Summary

v0.94.0 is **shippable**. No CRITICAL bugs. The HIGH bug (`cognithor[autogen]` extra missing `autogen-ext`) only affects users who actively choose the optional `autogen` adapter, and the failure is a clean ImportError (not data loss / corruption / silent wrong behavior). The MEDIUM sevDesk catalog mismatch is documentation drift that affects the marketing `/integrations` page, not core agent behavior. Both are addressable in a v0.94.1 patch.
