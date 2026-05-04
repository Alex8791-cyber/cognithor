# Cognithor VS Code Extension

Run [Cognithor Agent OS](https://cognithor.ai) plans from VS Code with
TRUST-1 receipt sidebar, structured Decision-Explanation hovers,
and live cost tracking.

> **Sprint-27 in progress.** This is PR-D — only the scaffold +
> activation skeleton is wired. Real features land in PR-E ... PR-K.
> See [`../../docs/superpowers/plans/2026-05-04-sprint27-ide-integration.md`](../../docs/superpowers/plans/2026-05-04-sprint27-ide-integration.md).

## Status

| Feature | Sprint-27 PR | Status |
|---|---|---|
| Extension scaffold (TS + esbuild + vsce) | PR-D | ✅ this PR |
| MCP-stdio bridge (`cognithor mcp --stdio` auto-spawn) | PR-E | pending |
| Palette: `Cognithor: Run Plan` + WS client | PR-F | pending |
| TRUST-1 Receipt sidebar | PR-G | pending |
| Decision-Explanation hover | PR-H | pending |
| Cost-budget gutter | PR-I | pending |
| Cursor + Windsurf + Claude-Desktop compat smoke | PR-J | pending |
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
