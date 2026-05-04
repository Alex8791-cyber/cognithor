/**
 * Cognithor VS Code extension — entry point.
 *
 * PR-D ships only the scaffold + activation skeleton + the
 * "Run Plan" command stub. The real wiring lands in:
 *
 *   - PR-E (#120): MCP-stdio bridge auto-spawned on activation
 *   - PR-F (#121): WebSocket client + plan picker
 *   - PR-G (#122): TRUST-1 receipt sidebar webview
 *   - PR-H (#123): Decision-Explanation hover provider
 *   - PR-I (#124): Cost-budget gutter decoration
 *
 * The streaming-event types under ./types/events.ts are
 * generated from src/cognithor/streaming/schemas/v1/events.json
 * via `npm run codegen:events` (per H5 of the decisions memo).
 */

import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext): void {
  console.log("[cognithor] extension activated");

  const runPlan = vscode.commands.registerCommand(
    "cognithor.runPlan",
    async () => {
      // PR-F will replace this stub with the WebSocket-backed plan picker.
      vscode.window.showInformationMessage(
        "Cognithor: Run Plan — wiring lands in Sprint-27 PR-F (#121).",
      );
    },
  );
  context.subscriptions.push(runPlan);
}

export function deactivate(): void {
  console.log("[cognithor] extension deactivated");
}
