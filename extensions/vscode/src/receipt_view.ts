/**
 * Cognithor TRUST-1 receipt webview (Sprint-27 PR-G).
 *
 * Renders the JSON run-receipt returned by
 * `GET /api/crew/trace/{trace_id}/receipt` (with optional
 * `?include_trust=true` for the TRUST-5..10 bundle) into a
 * VS Code webview panel.
 *
 * Self-contained — no dependency on the PR-F WS client. The
 * sidebar tree-view collects trace_ids the user has typed in;
 * a single command (`cognithor.viewReceipt`) prompts for a
 * trace_id and opens the panel. Trace IDs are persisted in
 * workspace memento storage so the sidebar survives reloads.
 */

import * as http from "node:http";
import * as vscode from "vscode";

const RECEIPT_HISTORY_KEY = "cognithor.receiptTraceIds";
const MAX_HISTORY = 20;

export interface ReceiptViewConfig {
  apiHost: string;
  apiPort: number;
  includeTrust: boolean;
}

export function readReceiptConfig(): ReceiptViewConfig {
  const config = vscode.workspace.getConfiguration("cognithor");
  return {
    apiHost: config.get<string>("apiHost", "127.0.0.1"),
    apiPort: config.get<number>("apiPort", 8741),
    includeTrust: config.get<boolean>("receiptIncludeTrust", true),
  };
}

interface FetchedReceipt {
  status: number;
  body: string;
}

function fetchReceipt(
  config: ReceiptViewConfig,
  traceId: string,
): Promise<FetchedReceipt> {
  // Tight URL construction: trace_id is path-segment encoded so a
  // user pasting whitespace or `/` doesn't accidentally redirect
  // the request elsewhere.
  const encoded = encodeURIComponent(traceId);
  const path =
    `/api/crew/trace/${encoded}/receipt` +
    (config.includeTrust ? "?include_trust=true" : "");

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: config.apiHost,
        port: config.apiPort,
        path,
        method: "GET",
        headers: { Accept: "application/json" },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            status: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf-8"),
          });
        });
        res.on("error", reject);
      },
    );
    req.setTimeout(5_000, () => {
      req.destroy(new Error(`receipt fetch timeout (${config.apiHost}:${config.apiPort})`));
    });
    req.on("error", reject);
    req.end();
  });
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderReceiptHtml(traceId: string, body: string, status: number): string {
  let pretty = body;
  let parsedOk = false;
  try {
    const parsed = JSON.parse(body) as unknown;
    pretty = JSON.stringify(parsed, null, 2);
    parsedOk = true;
  } catch {
    // Non-JSON body (e.g. error string) — render verbatim.
  }
  const heading = parsedOk
    ? `Run receipt for trace ${escapeHtml(traceId)}`
    : `Receipt fetch returned status ${status}`;

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';" />
  <title>${escapeHtml(heading)}</title>
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground);
           background: var(--vscode-editor-background); padding: 1rem; }
    h1 { font-size: 1.2rem; margin: 0 0 0.5rem; }
    .meta { opacity: 0.7; font-size: 0.85rem; margin-bottom: 1rem; }
    pre { background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border);
          padding: 0.75rem; overflow: auto; white-space: pre-wrap; word-break: break-word;
          font-family: var(--vscode-editor-font-family, monospace); font-size: 0.85rem; }
    .err { color: var(--vscode-errorForeground); }
  </style>
</head>
<body>
  <h1>${escapeHtml(heading)}</h1>
  <div class="meta">HTTP ${status} · ${escapeHtml(traceId)}</div>
  <pre class="${parsedOk ? "" : "err"}">${escapeHtml(pretty)}</pre>
</body>
</html>`;
}

export class ReceiptViewer {
  private panel: vscode.WebviewPanel | null = null;

  constructor(private readonly state: vscode.Memento) {}

  recentTraceIds(): string[] {
    return this.state.get<string[]>(RECEIPT_HISTORY_KEY, []);
  }

  rememberTraceId(traceId: string): Thenable<void> {
    const existing = this.recentTraceIds();
    const filtered = [traceId, ...existing.filter((t) => t !== traceId)];
    return this.state.update(
      RECEIPT_HISTORY_KEY,
      filtered.slice(0, MAX_HISTORY),
    );
  }

  forgetAll(): Thenable<void> {
    return this.state.update(RECEIPT_HISTORY_KEY, []);
  }

  async show(traceId: string, config?: ReceiptViewConfig): Promise<void> {
    const cfg = config ?? readReceiptConfig();
    const trimmed = traceId.trim();
    if (!trimmed) {
      void vscode.window.showWarningMessage("Cognithor: empty trace id");
      return;
    }

    let result: FetchedReceipt;
    try {
      result = await fetchReceipt(cfg, trimmed);
    } catch (err) {
      const message = (err as Error).message;
      void vscode.window.showErrorMessage(
        `Cognithor: receipt fetch failed — ${message}`,
      );
      return;
    }

    if (result.status === 200) {
      await this.rememberTraceId(trimmed);
    }

    const html = renderReceiptHtml(trimmed, result.body, result.status);
    if (this.panel === null || this.panel.visible === false) {
      this.panel = vscode.window.createWebviewPanel(
        "cognithor.receipt",
        `Cognithor Receipt · ${trimmed.slice(0, 12)}`,
        vscode.ViewColumn.Beside,
        { enableScripts: false, retainContextWhenHidden: true },
      );
      this.panel.onDidDispose(() => {
        this.panel = null;
      });
    } else {
      this.panel.title = `Cognithor Receipt · ${trimmed.slice(0, 12)}`;
      this.panel.reveal(vscode.ViewColumn.Beside, true);
    }
    this.panel.webview.html = html;
  }
}

class ReceiptTreeItem extends vscode.TreeItem {
  constructor(public readonly traceId: string) {
    super(traceId, vscode.TreeItemCollapsibleState.None);
    this.tooltip = `Open receipt for trace ${traceId}`;
    this.contextValue = "cognithor.receipt";
    this.iconPath = new vscode.ThemeIcon("file-binary");
    this.command = {
      command: "cognithor.viewReceipt",
      title: "View Receipt",
      arguments: [traceId],
    };
  }
}

export class ReceiptTreeProvider implements vscode.TreeDataProvider<ReceiptTreeItem> {
  private readonly emitter = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.emitter.event;

  constructor(private readonly viewer: ReceiptViewer) {}

  refresh(): void {
    this.emitter.fire();
  }

  getTreeItem(element: ReceiptTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ReceiptTreeItem[] {
    return this.viewer.recentTraceIds().map((id) => new ReceiptTreeItem(id));
  }
}
