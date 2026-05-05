# Cognithor VS Code Extension

Run [Cognithor Agent OS](https://cognithor.ai) plans from VS Code with
TRUST-1 receipt sidebar, structured Decision-Explanation hovers,
and live cost tracking.

## Supported IDEs

| IDE | Install path | Notes |
|---|---|---|
| VS Code 1.85+ | Marketplace or VSIX | Primary target |
| Cursor 0.40+ | VSIX | VS Code fork; same APIs |
| Windsurf 1.0+ | VSIX | VS Code fork; same APIs |
| Claude Desktop | `claude_desktop_config.json` | MCP-stdio only — see [`docs/IDE_COMPAT.md`](docs/IDE_COMPAT.md) |

See [`docs/IDE_COMPAT.md`](docs/IDE_COMPAT.md) for per-IDE install
walkthroughs and known limitations. Run
`bash scripts/check_ide_compat.sh` to verify the current build
still meets each target's minimum requirements.

## Status

| Feature | Sprint-27 PR | Status |
|---|---|---|
| Extension scaffold (TS + esbuild + vsce) | PR-D | ✅ |
| MCP-stdio bridge (`cognithor mcp --stdio` auto-spawn) | PR-E | ✅ |
| Palette: `Cognithor: Run Plan` + WS client | PR-F | ✅ |
| TRUST-1 Receipt sidebar | PR-G | ✅ |
| Decision-Explanation hover | PR-H | ✅ |
| Cost-budget gutter | PR-I | ✅ |
| Cursor + Windsurf + Claude-Desktop compat smoke | PR-J | ✅ this PR |
| End-to-end roundtrip smoke | PR-K | pending |

## Build

```bash
npm install
npm run codegen:events       # regenerate src/types/events.ts from the JSON Schema
npm run typecheck            # tsc --noEmit
npm run compile              # esbuild → dist/extension.js (dev build)
npm run package              # esbuild --production
npm run build:vsix           # produces cognithor-vscode-<version>.vsix
```

## Architecture (per Sprint-27 D5 decision)

The streaming protocol consumed by this extension is defined by:

- **`cognithor/streaming/schemas/v1/events.json`** — JSON Schema (Draft 2020-12), source of truth for the wire format.
- **`cognithor/streaming/event_emitter.py`** — single Producer.
- **`cognithor/streaming/sinks/jsonl_sink.py`** — stdout transport (PR-B).
- **`cognithor/streaming/sinks/ws_sink.py`** — WebSocket transport (PR-C).

The extension binds to the **WebSocket** transport via
`cognithor agent ws --port 8742`. Bearer-token auth from
`~/.cognithor/auth.token` is required on the upgrade request (H3).

## License

Apache 2.0. Free, including the Receipt-Sidebar and Cost-Gutter
features (per Sprint-27 D4 — no paywall on core trust visualization).

## Repository layout

This extension lives under `extensions/vscode/` in the main
[cognithor](https://github.com/Alex8791-cyber/cognithor) repo
(monorepo per Sprint-27 D1). All issues, PRs, and discussion go
in the parent repo.
