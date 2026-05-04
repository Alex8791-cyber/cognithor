/**
 * Cognithor Decision-Explanation hover provider (Sprint-27 PR-H).
 *
 * Surfaces TRUST-2 structured `decision_explanation` blocks as
 * VS Code hovers. Activates on JSON files that look like
 * Cognithor receipts (top-level `trace_id` + `decisions` array).
 *
 * The receipt shape — produced by
 * `cognithor.audit.AuditLogger.build_receipt_from_entries` —
 * embeds each gate decision under `decisions[i]` with fields
 * `tool`, `risk_level`, `status`, and an optional
 * `decision_explanation` carrying `rule_id`, `rule_source`, and
 * `matched_pattern`. When the user hovers over `tool` or
 * `risk_level` inside such an entry, we render the explanation
 * inline.
 *
 * Self-contained — no network calls, no other-extension
 * dependency. Works on any JSON file the user opens locally,
 * including receipts they've saved from PR-G's webview.
 */

import * as vscode from "vscode";

interface JsonObject {
  [key: string]: unknown;
}

interface DecisionExplanation {
  rule_id?: string;
  rule_source?: string;
  matched_pattern?: string;
  why?: string;
  reason?: string;
  [key: string]: unknown;
}

interface DecisionEntry {
  tool?: unknown;
  risk_level?: unknown;
  status?: unknown;
  decision_explanation?: DecisionExplanation;
  reason?: unknown;
  [key: string]: unknown;
}

const RECEIPT_LANGUAGES = ["json", "jsonc"];

function isReceiptDocument(doc: vscode.TextDocument): boolean {
  if (doc.lineCount > 5_000) return false; // bail on huge files
  const text = doc.getText();
  // Cheap-and-fast prefilter: don't try to JSON-parse files that
  // don't even mention the two top-level keys we rely on.
  return text.includes('"trace_id"') && text.includes('"decisions"');
}

function safeParse(text: string): JsonObject | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    return typeof parsed === "object" && parsed !== null ? (parsed as JsonObject) : null;
  } catch {
    return null;
  }
}

function locateDecisionEntry(
  doc: vscode.TextDocument,
  parsed: JsonObject,
  position: vscode.Position,
): DecisionEntry | null {
  const decisions = parsed.decisions;
  if (!Array.isArray(decisions) || decisions.length === 0) {
    return null;
  }

  // Strategy: find the byte-offset of the cursor, then walk the
  // raw text to find which JSON object's `{ ... }` brackets the
  // cursor sits inside, and return the parsed entry whose `tool`
  // matches that text region.
  //
  // For determinism we pair-match: each entry's serialised key
  // span is uniquely re-located via its `tool` value.
  const offset = doc.offsetAt(position);
  const text = doc.getText();

  let bestEntry: DecisionEntry | null = null;
  let bestSpanStart = -1;

  for (const candidate of decisions) {
    if (typeof candidate !== "object" || candidate === null) continue;
    const entry = candidate as DecisionEntry;
    if (typeof entry.tool !== "string") continue;

    // Find the position of `"tool": "<entry.tool>"` in the
    // source text. There may be many — pick the *latest* one
    // whose start is <= cursor offset (i.e. the entry the cursor
    // is inside).
    const probe = `"tool": "${entry.tool}"`;
    let searchFrom = 0;
    let entryStart = -1;
    while (true) {
      const idx = text.indexOf(probe, searchFrom);
      if (idx === -1) break;
      if (idx > offset) break;
      entryStart = idx;
      searchFrom = idx + probe.length;
    }
    if (entryStart === -1) continue;

    if (entryStart > bestSpanStart) {
      bestSpanStart = entryStart;
      bestEntry = entry;
    }
  }

  return bestEntry;
}

function renderHoverMarkdown(entry: DecisionEntry): vscode.MarkdownString {
  const md = new vscode.MarkdownString();
  md.isTrusted = false;
  md.supportThemeIcons = true;

  const tool = typeof entry.tool === "string" ? entry.tool : "?";
  const risk = typeof entry.risk_level === "string" ? entry.risk_level.toUpperCase() : "?";
  const status = typeof entry.status === "string" ? entry.status : "";
  const riskBadge = riskIcon(risk);

  md.appendMarkdown(`### ${riskBadge} Cognithor decision · \`${escapeMd(tool)}\`\n\n`);
  md.appendMarkdown(`**Risk:** ${escapeMd(risk)}`);
  if (status) {
    md.appendMarkdown(`  ·  **Status:** ${escapeMd(status)}`);
  }
  md.appendMarkdown("\n\n");

  const expl = entry.decision_explanation;
  if (expl !== undefined) {
    if (typeof expl.rule_id === "string" && expl.rule_id.length > 0) {
      md.appendMarkdown(`- **Rule:** \`${escapeMd(expl.rule_id)}\`\n`);
    }
    if (typeof expl.rule_source === "string" && expl.rule_source.length > 0) {
      md.appendMarkdown(`- **Source:** \`${escapeMd(expl.rule_source)}\`\n`);
    }
    if (typeof expl.matched_pattern === "string" && expl.matched_pattern.length > 0) {
      md.appendMarkdown(`- **Matched:** \`${escapeMd(expl.matched_pattern)}\`\n`);
    }
    const reasonText =
      (typeof expl.why === "string" && expl.why) ||
      (typeof expl.reason === "string" && expl.reason) ||
      "";
    if (reasonText) {
      md.appendMarkdown(`\n${escapeMd(reasonText)}\n`);
    }
  } else if (typeof entry.reason === "string" && entry.reason.length > 0) {
    md.appendMarkdown(`*Reason:* ${escapeMd(entry.reason)}\n`);
  } else {
    md.appendMarkdown("*No structured explanation attached.*\n");
  }

  return md;
}

function escapeMd(s: string): string {
  return s
    .replace(/\\/g, "\\\\")
    .replace(/`/g, "\\`")
    .replace(/\*/g, "\\*")
    .replace(/_/g, "\\_")
    .replace(/\[/g, "\\[")
    .replace(/\]/g, "\\]")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function riskIcon(risk: string): string {
  switch (risk) {
    case "GREEN":
      return "$(pass-filled)";
    case "YELLOW":
      return "$(warning)";
    case "ORANGE":
      return "$(flame)";
    case "RED":
      return "$(error)";
    default:
      return "$(question)";
  }
}

export class DecisionHoverProvider implements vscode.HoverProvider {
  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken,
  ): vscode.ProviderResult<vscode.Hover> {
    if (!RECEIPT_LANGUAGES.includes(document.languageId)) return null;
    if (!isReceiptDocument(document)) return null;

    const text = document.getText();
    const parsed = safeParse(text);
    if (parsed === null) return null;

    const entry = locateDecisionEntry(document, parsed, position);
    if (entry === null) return null;

    const md = renderHoverMarkdown(entry);
    return new vscode.Hover(md);
  }
}

export function registerDecisionHover(context: vscode.ExtensionContext): void {
  const provider = new DecisionHoverProvider();
  for (const lang of RECEIPT_LANGUAGES) {
    context.subscriptions.push(
      vscode.languages.registerHoverProvider(
        { language: lang, scheme: "file" },
        provider,
      ),
    );
    context.subscriptions.push(
      vscode.languages.registerHoverProvider(
        { language: lang, scheme: "untitled" },
        provider,
      ),
    );
  }
}
