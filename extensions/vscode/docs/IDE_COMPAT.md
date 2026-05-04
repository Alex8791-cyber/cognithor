# Cognithor IDE Compatibility Guide (Sprint-27 PR-J)

The Cognithor VS Code extension targets the standard VS Code
extensibility API (engine `^1.85.0`). It also runs unmodified in
**Cursor**, **Windsurf**, and any other VS Code fork that keeps
the extension API surface in sync. This guide documents the
supported install path on each IDE plus the equivalent path for
**Claude Desktop**, which talks to Cognithor via MCP-stdio
directly without the VS Code extension layer.

## VS Code (primary target)

1. Install the latest [`cognithor`](https://pypi.org/project/cognithor/) Python package on PATH:
   ```bash
   pip install --upgrade cognithor
   ```
2. Open VS Code 1.85+.
3. Install the extension:
   - **From marketplace:** Search for "Cognithor" in the Extensions view. Publisher: `cognithor` (Apache-2.0).
   - **From VSIX:** download `cognithor-vscode-<version>.vsix` from the GitHub releases page and run `code --install-extension cognithor-vscode-<version>.vsix`.
4. Ensure `cognithor` is on PATH or set `cognithor.cliPath` in `Settings → Cognithor`.
5. Open a workspace, run **Cognithor: Run Plan** from the palette.

## Cursor

Cursor is a VS Code fork by Anysphere. The Cognithor extension
works unchanged.

1. Install the `cognithor` Python package as above.
2. Open Cursor 0.40+.
3. Install the extension via the VSIX route (Cursor's marketplace
   does not always mirror VS Code's; the VSIX is the reliable path):
   ```bash
   cursor --install-extension cognithor-vscode-<version>.vsix
   ```
4. Same palette command + sidebar. Cursor's chat side-panel does
   **not** consume our streaming events directly — it has its own
   inference layer. Use the Cognithor extension as a complementary
   surface for plan execution + receipts.

**Known limitations**

- Cursor's "Background Agent" feature owns its own working tree and
  stdio; do not run `cognithor agent ws` and a Cursor agent against
  the same workspace simultaneously.

## Windsurf

Windsurf is a VS Code fork by Codeium. Same install path as Cursor.

1. Install the `cognithor` Python package as above.
2. Open Windsurf 1.0+.
3. Install via VSIX:
   ```bash
   windsurf --install-extension cognithor-vscode-<version>.vsix
   ```
4. Settings live under `Settings → Cognithor` (same keys: `cliPath`,
   `wsPort`, `wsBind`, `apiHost`, `apiPort`, `receiptIncludeTrust`,
   `costThreshold*Micro`).

**Known limitations**

- Same as Cursor: Windsurf's Cascade flow owns its own stdio. Run
  Cognithor's WS server on a port that does not collide with
  Cascade's local listener.

## Claude Desktop

Claude Desktop talks to MCP servers directly via stdio. The Cognithor
VS Code extension is not used here — instead, register Cognithor as an
MCP server in Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cognithor": {
      "command": "cognithor",
      "args": ["mcp", "--stdio"]
    }
  }
}
```

Locations:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

After saving, restart Claude Desktop. The 141 Cognithor MCP tools
appear in the tool picker.

**Known limitations**

- Claude Desktop has no concept of the cost-gutter, receipt sidebar,
  or hover providers — those are VS Code-extension-only surfaces.
- The MCP-stdio surface is `cognithor mcp --stdio`; the
  WebSocket / TRUST-1 / Trace-UI surfaces are out-of-scope for
  Claude Desktop.

## Smoke check

`scripts/check_ide_compat.sh` exercises the manifest against the
declared engine + API minimums. Run it from the repo root:

```bash
bash extensions/vscode/scripts/check_ide_compat.sh
```

It validates:

1. `package.json::engines.vscode` is `^1.85.0` (the lowest version
   exposing all APIs the extension touches: webview, tree-view,
   hover provider, decoration ranges, status-bar, output channel).
2. `package.json::engines.node` is `>=20`.
3. The compiled `dist/extension.js` exists and is non-empty
   (regression guard for esbuild misconfigurations).
4. Required activation events declare the streaming surfaces:
   `onCommand:cognithor.runPlan`, `onLanguage:json`,
   `onLanguage:jsonc`, `onStartupFinished`.

If anything drifts (engine bump, missing dist, missing activation),
the script exits non-zero with the offending field.
