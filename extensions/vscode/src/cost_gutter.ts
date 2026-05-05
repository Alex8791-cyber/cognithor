/**
 * Cognithor cost-budget gutter decoration (Sprint-27 PR-I).
 *
 * Highlights `cost_usd_micro` lines in receipt JSON / JSONC files
 * with a coloured gutter icon based on a configurable threshold,
 * and shows a status-bar item with the per-document totals while
 * a receipt is the active editor.
 *
 * Threshold ladder (defaults — overridable via settings):
 *
 *   < 1 000 µUSD  ($0.001)            → green   (negligible)
 *   < 10 000 µUSD ($0.01)             → yellow  (cheap)
 *   < 100 000 µUSD ($0.10)            → orange  (medium)
 *   ≥ 100 000 µUSD                    → red     (expensive)
 *
 * Self-contained — no network, no extension dependency. Activates
 * on the same JSON / JSONC surface as the PR-H hover provider.
 */

import * as vscode from "vscode";

interface CostThresholds {
  green: number; // upper bound of "green"
  yellow: number; // upper bound of "yellow"
  orange: number; // upper bound of "orange"; above this = red
}

interface DecorationSet {
  green: vscode.TextEditorDecorationType;
  yellow: vscode.TextEditorDecorationType;
  orange: vscode.TextEditorDecorationType;
  red: vscode.TextEditorDecorationType;
}

interface ScanResult {
  ranges: { range: vscode.Range; tier: keyof DecorationSet; valueMicro: number }[];
  totalMicro: number;
  count: number;
}

const COST_LINE_REGEX = /"cost_usd_micro"\s*:\s*(-?\d+(?:\.\d+)?)/g;
const RECEIPT_LANGUAGES = ["json", "jsonc"];

function readThresholds(): CostThresholds {
  const cfg = vscode.workspace.getConfiguration("cognithor");
  return {
    green: cfg.get<number>("costThresholdGreenMicro", 1_000),
    yellow: cfg.get<number>("costThresholdYellowMicro", 10_000),
    orange: cfg.get<number>("costThresholdOrangeMicro", 100_000),
  };
}

function tierFor(micro: number, t: CostThresholds): keyof DecorationSet {
  if (micro < t.green) return "green";
  if (micro < t.yellow) return "yellow";
  if (micro < t.orange) return "orange";
  return "red";
}

function isReceiptDocument(doc: vscode.TextDocument): boolean {
  if (doc.lineCount > 5_000) return false;
  if (!RECEIPT_LANGUAGES.includes(doc.languageId)) return false;
  const text = doc.getText();
  return text.includes('"cost_usd_micro"') && text.includes('"trace_id"');
}

function scanCosts(doc: vscode.TextDocument, t: CostThresholds): ScanResult {
  const text = doc.getText();
  const result: ScanResult = { ranges: [], totalMicro: 0, count: 0 };
  COST_LINE_REGEX.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = COST_LINE_REGEX.exec(text)) !== null) {
    const value = Number(match[1]);
    if (!Number.isFinite(value)) continue;
    const start = doc.positionAt(match.index);
    const end = doc.positionAt(match.index + match[0].length);
    result.ranges.push({
      range: new vscode.Range(start, end),
      tier: tierFor(value, t),
      valueMicro: value,
    });
    result.totalMicro += value;
    result.count += 1;
  }
  return result;
}

function buildDecorations(): DecorationSet {
  // Subtle gutter colour swatches via theme tokens — no PNG icons,
  // works on both light and dark themes, and is theme-aware.
  const make = (
    bg: string,
    border: string,
    overviewRulerLane?: vscode.OverviewRulerLane,
  ): vscode.TextEditorDecorationType =>
    vscode.window.createTextEditorDecorationType({
      isWholeLine: false,
      backgroundColor: bg,
      borderRadius: "2px",
      borderWidth: "0 0 0 3px",
      borderStyle: "solid",
      borderColor: border,
      overviewRulerColor: border,
      overviewRulerLane: overviewRulerLane ?? vscode.OverviewRulerLane.Right,
    });

  return {
    green: make("rgba(60,200,120,0.08)", "rgba(60,200,120,0.65)"),
    yellow: make("rgba(220,200,80,0.10)", "rgba(220,200,80,0.75)"),
    orange: make("rgba(240,160,60,0.12)", "rgba(240,160,60,0.85)"),
    red: make("rgba(240,80,80,0.14)", "rgba(240,80,80,0.95)"),
  };
}

function formatUsd(micro: number): string {
  const usd = micro / 1_000_000;
  if (Math.abs(usd) >= 1) {
    return `$${usd.toFixed(2)}`;
  }
  if (Math.abs(usd) >= 0.001) {
    return `$${usd.toFixed(4)}`;
  }
  return `${micro.toFixed(0)} µUSD`;
}

export class CostGutterController {
  private readonly decos = buildDecorations();
  private readonly statusItem: vscode.StatusBarItem;
  private disposed = false;

  constructor(context: vscode.ExtensionContext) {
    this.statusItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      90,
    );
    this.statusItem.tooltip = "Cognithor: total cost in this receipt file";
    context.subscriptions.push(this.statusItem);
    context.subscriptions.push(
      this.decos.green,
      this.decos.yellow,
      this.decos.orange,
      this.decos.red,
    );

    context.subscriptions.push(
      vscode.window.onDidChangeActiveTextEditor((ed) => this.update(ed)),
      vscode.workspace.onDidChangeTextDocument((evt) => {
        const ed = vscode.window.activeTextEditor;
        if (ed && evt.document === ed.document) {
          this.update(ed);
        }
      }),
      vscode.workspace.onDidChangeConfiguration((evt) => {
        if (
          evt.affectsConfiguration("cognithor.costThresholdGreenMicro") ||
          evt.affectsConfiguration("cognithor.costThresholdYellowMicro") ||
          evt.affectsConfiguration("cognithor.costThresholdOrangeMicro")
        ) {
          this.update(vscode.window.activeTextEditor);
        }
      }),
      { dispose: () => this.disposeInternal() },
    );

    this.update(vscode.window.activeTextEditor);
  }

  update(editor: vscode.TextEditor | undefined): void {
    if (this.disposed || !editor) {
      this.statusItem.hide();
      return;
    }
    if (!isReceiptDocument(editor.document)) {
      // Clear any leftover decorations from a previously-displayed receipt.
      editor.setDecorations(this.decos.green, []);
      editor.setDecorations(this.decos.yellow, []);
      editor.setDecorations(this.decos.orange, []);
      editor.setDecorations(this.decos.red, []);
      this.statusItem.hide();
      return;
    }

    const thresholds = readThresholds();
    const scan = scanCosts(editor.document, thresholds);
    const buckets: Record<keyof DecorationSet, vscode.Range[]> = {
      green: [],
      yellow: [],
      orange: [],
      red: [],
    };
    for (const item of scan.ranges) {
      buckets[item.tier].push(item.range);
    }
    editor.setDecorations(this.decos.green, buckets.green);
    editor.setDecorations(this.decos.yellow, buckets.yellow);
    editor.setDecorations(this.decos.orange, buckets.orange);
    editor.setDecorations(this.decos.red, buckets.red);

    const totalText = formatUsd(scan.totalMicro);
    const reds = buckets.red.length;
    const oranges = buckets.orange.length;
    const icon = reds > 0 ? "$(error)" : oranges > 0 ? "$(flame)" : "$(info)";
    this.statusItem.text = `${icon} Cognithor cost: ${totalText} (${scan.count})`;
    this.statusItem.tooltip = `Cognithor receipt costs in active document.\nEntries: ${scan.count} (red ${reds}, orange ${oranges})\nTotal: ${totalText}`;
    this.statusItem.show();
  }

  private disposeInternal(): void {
    this.disposed = true;
    this.statusItem.dispose();
  }
}

export function registerCostGutter(context: vscode.ExtensionContext): void {
  // Constructor wires editor + config listeners onto context.subscriptions.
  new CostGutterController(context);
}
