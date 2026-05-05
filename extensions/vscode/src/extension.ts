/**
 * Cognithor VS Code extension — entry point.
 *
 * PR-D shipped the scaffold. PR-E wired the MCP-stdio bridge
 * (auto-spawn on activation, heartbeat every 30 s, restart on
 * no-pong within 5 s). PR-F (this update) ships the
 * "Cognithor: Run Plan" palette command + WebSocket client that
 * talks to the `cognithor agent ws` server (PR-C) and streams
 * events into the output channel.
 *
 * Subsequent wiring lands in:
 *
 *   - PR-G (#122): TRUST-1 receipt sidebar webview
 *   - PR-H (#123): Decision-Explanation hover provider
 *   - PR-I (#124): Cost-budget gutter decoration
 */

import * as vscode from "vscode";
import { registerCostGutter } from "./cost_gutter";
import { registerDecisionHover } from "./decision_hover";
import { McpBridge, readBridgeConfig } from "./mcp_bridge";
import { ReceiptTreeProvider, ReceiptViewer } from "./receipt_view";
import { WsClient } from "./ws_client";

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
  // Palette command: Cognithor: Run Plan (PR-F)
  // --------------------------------------------------------------------
  const runPlan = vscode.commands.registerCommand(
    "cognithor.runPlan",
    async () => {
      const log = outputChannel;
      const planUri = await pickPlanFile();
      if (!planUri) {
        return;
      }
      log?.show(true);
      log?.appendLine(`[plan] running: ${planUri.fsPath}`);

      const config = vscode.workspace.getConfiguration("cognithor");
      const port = config.get<number>("wsPort", 8742);
      const bind = config.get<string>("wsBind", "127.0.0.1");

      let client: WsClient;
      try {
        client = new WsClient({ bind, port });
      } catch (err) {
        const message = (err as Error).message;
        log?.appendLine(`[plan] WS client init failed: ${message}`);
        void vscode.window.showErrorMessage(`Cognithor: ${message}`);
        return;
      }

      client.onEvent((evt) => {
        // The streaming wire format uses `event` as the discriminator
        // (per src/cognithor/streaming/schemas/v1/events.json). The
        // earlier code logged `evt.type` which never exists on the
        // wire, so every entry showed as `?` in the output channel.
        const kind = typeof evt.event === "string" ? evt.event : "?";
        log?.appendLine(`[evt] ${kind} ${JSON.stringify(evt)}`);
      });
      client.onError((err) => {
        log?.appendLine(`[plan] error: ${err.message}`);
      });

      try {
        const info = await client.runPlan({ planPath: planUri.fsPath });
        log?.appendLine(
          `[plan] closed (code=${info.code}, clean=${info.wasClean})${info.reason ? ` — ${info.reason}` : ""}`,
        );
      } catch (err) {
        const message = (err as Error).message;
        log?.appendLine(`[plan] failed: ${message}`);
        void vscode.window.showErrorMessage(`Cognithor: ${message}`);
      }
    },
  );
  context.subscriptions.push(runPlan);

  // --------------------------------------------------------------------
  // TRUST-1 receipt viewer (PR-G)
  // --------------------------------------------------------------------
  const receiptViewer = new ReceiptViewer(context.workspaceState);
  const receiptTree = new ReceiptTreeProvider(receiptViewer);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("cognithor.receipts", receiptTree),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "cognithor.viewReceipt",
      async (traceIdArg?: string) => {
        let traceId = traceIdArg;
        if (typeof traceId !== "string" || traceId.length === 0) {
          const recent = receiptViewer.recentTraceIds();
          if (recent.length > 0) {
            const picked = await vscode.window.showQuickPick(
              [
                ...recent.map((id) => ({ label: id, value: id })),
                { label: "Enter a different trace id...", value: "__custom__" },
              ],
              { placeHolder: "Pick a recent trace id or enter a new one" },
            );
            if (!picked) return;
            traceId = picked.value === "__custom__" ? undefined : picked.value;
          }
          if (typeof traceId !== "string") {
            const entered = await vscode.window.showInputBox({
              prompt: "Cognithor trace id",
              placeHolder: "e.g. trace_abc123...",
            });
            if (!entered) return;
            traceId = entered;
          }
        }
        await receiptViewer.show(traceId);
        receiptTree.refresh();
      },
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("cognithor.refreshReceipts", () => {
      receiptTree.refresh();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("cognithor.forgetReceipts", async () => {
      const confirm = await vscode.window.showWarningMessage(
        "Forget all remembered Cognithor trace ids?",
        { modal: true },
        "Forget",
      );
      if (confirm === "Forget") {
        await receiptViewer.forgetAll();
        receiptTree.refresh();
      }
    }),
  );

  // --------------------------------------------------------------------
  // Decision-Explanation hover provider (PR-H)
  // --------------------------------------------------------------------
  registerDecisionHover(context);

  // --------------------------------------------------------------------
  // Cost-budget gutter decoration (PR-I)
  // --------------------------------------------------------------------
  registerCostGutter(context);
}

async function pickPlanFile(): Promise<vscode.Uri | undefined> {
  const editor = vscode.window.activeTextEditor;
  if (editor && editor.document.fileName.toLowerCase().endsWith(".json")) {
    const useActive = await vscode.window.showQuickPick(
      [
        { label: "Active file", description: editor.document.fileName, value: "active" },
        { label: "Pick a plan file...", description: "Browse for a plan JSON file", value: "browse" },
      ],
      { placeHolder: "Which plan should Cognithor run?" },
    );
    if (!useActive) return undefined;
    if (useActive.value === "active") {
      return editor.document.uri;
    }
  }

  const picked = await vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: { "Plan files": ["json"] },
    openLabel: "Run plan",
  });
  return picked?.[0];
}

export function deactivate(): void {
  outputChannel?.appendLine("[cognithor] extension deactivated");
  void bridge?.stop();
  bridge = null;
}
