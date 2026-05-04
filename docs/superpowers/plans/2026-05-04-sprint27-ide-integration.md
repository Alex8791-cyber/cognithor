# Sprint-27 — IDE-Integration: VS-Code-Extension + CLI für Hybrid-Pipeline

**Date:** 2026-05-04
**Status:** Plan-Doku, Sprint noch nicht gestartet
**Owner:** Alexander Söllner
**Track:** Cognithor productisation — make the hybrid Planner→Gatekeeper→Executor + PSE flow accessible from a developer's IDE

---

## 1. Goal

Today's cognithor entry-points are:

- `cognithor` CLI (interactive REPL, `--api-port 8741`).
- Flutter Command Center (~200 Dart files, port 8741 to FastAPI).
- Telegram / Discord / 14 other channels.

A **VS Code developer** needs Cognithor's hybrid pipeline (PSE
synthesis + crew planning + receipt-grade audit) without leaving
the editor. Cursor / Continue / Claude Desktop's MCP integration
proves the demand. Sprint-27 ships the official Cognithor
VS-Code-Extension + a tightened headless CLI mode.

## 2. Scope (committed)

### Track A — VS-Code-Extension (`cognithor.vscode`)

1. **Extension bootstrap** — TypeScript project under
   `cognithor-vscode/` (separate repo, namespaced). Package
   metadata, `package.json` activation events, signing certificate.
2. **MCP-stdio bridge** — extension auto-spawns `cognithor mcp
   --stdio` (already exists, see `cognithor.mcp.server`) on
   workspace open. Reuses the existing 127+ MCP tools.
3. **Inline Plan-Run command** — palette command
   `Cognithor: Run Plan` opens a quick-pick of saved Procedural
   Memory entries (TRUST-9-tagged via #425). Selection triggers
   the headless CLI with structured streaming.
4. **TRUST-1 Receipt sidebar** — webview that polls
   `GET /api/crew/trace/{id}/receipt?include_trust=true` (#405)
   for the active run and renders the 6 trust sections inline.
   Equivalent of the Flutter `TrustBundlePanel` but VS-Code-native.
5. **Decision-Explanation hover** — when the agent blocks a
   tool call, the explanation `rule_id` / `rule_source` /
   `matched_pattern` (#377/#415/#418/#419/#423/#428) renders as a
   markdown hover at the offending line of the plan.
6. **Cost-budget gutter** — running cost in micro-USD per active
   run, shown as a gutter decoration. Pulls from the trust
   bundle's `cost.summary` block (#401).

### Track B — Headless CLI mode

1. **`cognithor agent run --plan FILE.json --stream`** — new
   subcommand that takes a serialised ActionPlan and streams
   per-step JSON-RPC events to stdout. The extension parses these
   to drive the receipt sidebar.
2. **`cognithor agent ws --port 8742`** — WebSocket variant of
   the same protocol, for IDEs that prefer long-lived sockets
   over stdio.
3. **Structured streaming protocol** — one-line-per-frame JSONL:
   `{"event":"plan_step","step":N,"action":...}` /
   `{"event":"gate_decision","status":...,"explanation":...}` /
   `{"event":"tool_result","ok":true,"chunks":...}` /
   `{"event":"run_complete","receipt":{...}}`. Schema-versioned;
   document under `docs/api/agent-streaming.md`.

### Track C — Cross-IDE compatibility

1. **JetBrains plugin scaffold** — same MCP-stdio bridge approach,
   IntelliJ-Platform plugin under `cognithor-jetbrains/`. Track-A
   webview reused via JetBrains JCEF browser.
2. **Cursor / Claude Desktop config snippets** — `docs/integrations/cursor.md`
   + `docs/integrations/claude-desktop.md` showing how to point
   their MCP client at `cognithor mcp --stdio`. Already works
   today but not documented.

## 3. Out of scope (Sprint-28+)

- Native Sublime Text / Vim plugins (low demand).
- Visual graph rendering of the plan tree (Sprint-28 — needs a
  graph-layout dependency choice).
- Two-way edit-collaboration (the IDE editing the plan back into
  cognithor's procedural memory). Track-A/B is read-output-only.

## 4. Plan-tasks (12 committed)

| # | Task | Owner | LOC est. |
|---|---|---|---|
| 1 | Scaffold `cognithor-vscode/` repo, TypeScript + esbuild, signing cert | Owner | 0 (infrastructure) |
| 2 | MCP-stdio bridge spawning `cognithor mcp` on activation | Claude | ~250 |
| 3 | Quick-pick palette command `Cognithor: Run Plan` | Claude | ~200 |
| 4 | Headless CLI: `cognithor agent run --plan FILE.json --stream` | Claude | ~300 |
| 5 | Headless CLI: `cognithor agent ws --port 8742` | Claude | ~250 |
| 6 | Streaming-protocol spec doc + JSON schema | Claude | 0 (docs) |
| 7 | TRUST-1 Receipt sidebar webview (poll `/api/crew/trace/.../receipt`) | Claude | ~400 |
| 8 | Decision-Explanation hover provider | Claude | ~150 |
| 9 | Cost-budget gutter decoration | Claude | ~120 |
| 10 | JetBrains plugin scaffold + MCP-stdio bridge | Owner+Claude | ~300 |
| 11 | Cursor + Claude Desktop integration docs | Claude | 0 (docs) |
| 12 | Smoke test: full Plan→Gate→Execute round-trip from VS Code | Claude | ~150 |

Total est. LOC: ~2120 across cognithor-vscode (~1370), cognithor-jetbrains (~300), cognithor (~550 backend agent CLI), docs (~3 files).

## 5. Dependencies

**On Cognithor backend:**
- `cognithor mcp --stdio` already exists (see `cognithor/__main__.py` MCP-server-mode). ✅
- `GET /api/crew/trace/{id}/receipt?include_trust=true` shipped #405. ✅
- TRUST-1 trust bundle composer shipped #402. ✅
- Procedural memory provenance shipped #425. ✅

**Tooling (must install):**
- Node 20+ + esbuild for TypeScript compilation.
- VS Code Extension API ≥ 1.85.
- IntelliJ Platform SDK 2024.2 for JetBrains plugin.

## 6. Risks

1. **MCP-stdio handshake instability** — cognithor's MCP server
   was built for batch use (Cursor opens it once, reuses).
   VS-Code's activate/deactivate cycle stresses the handshake.
   Mitigation: heartbeat ping every 30s, restart on no-pong.
2. **Cost-budget gutter UI noise** — updating gutter on every
   token streamed is too frequent. Mitigation: debounce 500ms,
   only render when delta > 0.001 USD.
3. **Receipt-sidebar polling cost** — every active run polls
   `/api/crew/trace/.../receipt`. With 10 active sessions × 1Hz
   that's 600 receipt-builds/min. Mitigation: WebSocket push
   from the gateway when a run produces a new audit entry, fall
   back to poll if WS fails.
4. **Signing cert + extension marketplace approval** — first
   release requires Microsoft Marketplace publisher account +
   code-signing certificate. Owner-side, ~1 week lead time.

## 7. Success criteria

- VS Code user types `Cmd+Shift+P` → "Cognithor: Run Plan" → picks
  a procedure → sees per-step plan execution streaming inline →
  sees the trust receipt sidebar populate in real time → can
  click a blocked tool call to see its `DecisionExplanation`
  hover.
- JetBrains user has the same flow via a single keystroke.
- Cursor user can drop a 5-line snippet into their `mcp.json` and
  reach all 127+ Cognithor tools.
- Extension marketplace listing live with screenshots and 50+ stars
  in the first week (proxy for "this matters to dev users").

## 8. Estimated timeline

| Phase | Tasks | Duration |
|---|---|---|
| Phase 1: backend + spec | 4, 5, 6 | 3 days |
| Phase 2: extension scaffold + bridge | 1, 2 | 2 days (1 day owner-side scaffold) |
| Phase 3: core commands + sidebar | 3, 7, 8, 9 | 5 days |
| Phase 4: cross-IDE + docs | 10, 11 | 3 days |
| Phase 5: smoke test + ship | 12 | 2 days |

**Total:** ~15 working days, parallelisable to ~10 calendar days
if backend + extension tracks run concurrently.

## 9. Owner decisions still required

- **Repo split** — `cognithor-vscode/` separate or under
  `cognithor/` monorepo? Marketplace listings prefer their own
  repo.
- **Signing cert** — purchase + provision. Owner-side admin task.
- **JetBrains plugin priority** — Sprint-27 or defer to Sprint-28?
  IntelliJ user-base is smaller but the integration is technically
  near-identical.
- **Free vs paid** — extension itself stays free (MIT/Apache-2),
  but should it require a paid Cognithor pack to unlock advanced
  features? Owner pricing-model decision.

## 10. Cross-references

- TRUST-1..10 reference: `docs/operational_trust.md` (#436).
- TRUST-1 trust bundle composer: `cognithor.security.trust_bundle` (#402).
- Receipt CLI: `cognithor receipt show / verify / list / export-all / diff` (#404, #424, #430, #431).
- Receipt REST: `GET /api/crew/trace/{id}/receipt` (#405).
- MCP-stdio entry: `cognithor.__main__:_run_mcp_server_mode`.
- Procedural memory: `cognithor.memory.procedural` (#425 TRUST-9 wired).

---

**Status:** Plan-Doku committed. Next step is the Sprint-27 kickoff — Owner approves the four open decisions in §9, then Phase-1 (backend + spec) starts.
