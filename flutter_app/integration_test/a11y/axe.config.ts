/**
 * axe-core configuration for Cognithor a11y audits — Sprint 3.3.
 *
 * WCAG 2.1 AA scope. Covers:
 *   - Chat screen (default landing)
 *   - Settings (high-density form layout)
 *   - Dashboard (charts + status indicators)
 *
 * Per-screen rules can be tightened or relaxed via the `tags` and
 * `rules` overrides below. Disabling a rule requires a justification
 * comment + an issue link.
 */

import type { RunOptions } from "@axe-core/playwright";

export const defaultAxeOptions: RunOptions = {
  // WCAG 2.0/2.1 A + AA — the bar most public-sector procurement docs require.
  runOnly: {
    type: "tag",
    values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"],
  },
  rules: {
    // Flutter Web's canvas renderer can confuse colour-contrast detection
    // for dark-mode glyphs. Re-enable per-screen once the canvas-text
    // rendering pipeline emits proper aria roles. Tracked: #flutter-a11y-canvas
    "color-contrast": { enabled: true },
    // Flutter sometimes emits offscreen widgets without `aria-hidden`.
    // We allow during page-load animations but tighten on idle.
    "aria-hidden-focus": { enabled: true },
  },
};

export const screensToAudit: Array<{
  name: string;
  url: string;
  // Optional per-screen overrides — narrower scope or disabled rules
  overrides?: Partial<RunOptions>;
}> = [
  { name: "chat", url: "/" },
  { name: "settings", url: "/settings" },
  { name: "dashboard", url: "/dashboard" },
  { name: "leads", url: "/leads" },
  { name: "kanban", url: "/kanban" },
];
