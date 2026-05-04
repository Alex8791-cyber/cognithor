/**
 * Cognithor VS Code extension — entry point.
 *
 * PR-D shipped the scaffold. PR-E (this file's update) wires the
 * MCP-stdio bridge so the extension auto-spawns
 * `cognithor mcp --stdio` on activation, heartbeats it every
 * 30 s, and restarts on no-pong within 5 s. Mitigates plan-doc
 * §6 risk-1 (MCP-stdio handshake instability for VS-Code's
 * activate/deactivate cycle).
 *
 * Subsequent wiring lands in:
 *
 *   - PR-F (#121): WebSocket client + plan picker (uses bridge
 *                  for procedural-memory queries)
 *   - PR-G (#122): TRUST-1 receipt sidebar webview
 *   - PR-H (#123): Decision-Explanation hover provider
 *   - PR-I (#124): Cost-budget gutter decoration
 */

import * as vscode from "vscode";
import { McpBridge, readBridgeConfig } from "./mcp_bridge";

let bridge: McpBridge | null = null;
let outputChannel: vscode.OutputChannel | null = null;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  outputChannel = vscode.window.createOutputChannel("Cognithor");
  context.subscriptions.push(outputChannel);
  outputChannel.appendLine("[cognithor] extension activated");

  // --------------------------------------------------------------------
  // MCP-stdio bridge (PR-E)
  // --------------------------------------------------------------------
  bridge = new McpBridge(readBridgeConfig());
  context.subscriptions.push(bridge);

  bridge.onLog((line) => outputChannel?.appendLine(`[mcp] ${line}`));
  bridge.onReady(() => {
    outputChannel?.appendLine("[mcp] bridge ready");
  });
  bridge.onCrash(({ code, restartCount }) => {
    outputChannel?.appendLine(
      `[mcp] crashed (exit code ${code}); restart attempt ${restartCount}`,
    );
  });

  // Best-effort start. If `cognithor` is not on PATH, the bridge
  // surfaces the spawn error via onLog + onCrash and the
  // extension keeps the user in a degraded-but-not-broken state
  // (the "Run Plan" command stub still works).
  try {
    await bridge.start();
  } catch (err) {
    const message = (err as Error).message;
    outputChannel?.appendLine(`[mcp] start failed: ${message}`);
    void vscode.window.showWarningMessage(
      `Cognithor MCP bridge failed to start: ${message}. ` +
        "Set cognithor.cliPath in settings to point at your cognithor executable.",
    );
  }

  // --------------------------------------------------------------------
  // Palette command stub (PR-F replaces this with the real flow)
  // --------------------------------------------------------------------
  const runPlan = vscode.commands.registerCommand(
    "cognithor.runPlan",
    async () => {
      void vscode.window.showInformationMessage(
        "Cognithor: Run Plan — wiring lands in Sprint-27 PR-F (#121).",
      );
    },
  );
  context.subscriptions.push(runPlan);
}

export function deactivate(): void {
  outputChannel?.appendLine("[cognithor] extension deactivated");
  void bridge?.stop();
  bridge = null;
}
