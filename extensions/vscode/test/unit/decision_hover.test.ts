import { strict as assert } from "assert";
import { getMock, resetMockSpies } from "./vscode-mock";

import { DecisionHoverProvider } from "../../src/decision_hover";

interface FakeDocument {
  languageId: string;
  lineCount: number;
  getText(): string;
  offsetAt(pos: { line: number; character: number }): number;
}

function makeDoc(text: string, languageId = "json"): FakeDocument {
  return {
    languageId,
    lineCount: text.split("\n").length,
    getText: () => text,
    offsetAt: (pos) => pos.character,
  };
}

const SAMPLE_RECEIPT = JSON.stringify(
  {
    trace_id: "tr_demo",
    decisions: [
      {
        tool: "shell.exec",
        risk_level: "red",
        status: "blocked",
        decision_explanation: {
          rule_id: "GK-007",
          rule_source: "gatekeeper.default",
          matched_pattern: "rm -rf",
          why: "Destructive shell command on user-supplied path.",
        },
      },
      {
        tool: "fs.read",
        risk_level: "green",
        status: "ok",
        decision_explanation: {
          rule_id: "GK-001",
          rule_source: "gatekeeper.default",
        },
      },
    ],
  },
  null,
  2,
);

describe("DecisionHoverProvider", () => {
  beforeEach(() => {
    resetMockSpies();
  });

  it("returns null for non-receipt JSON documents", async () => {
    const provider = new DecisionHoverProvider();
    const doc = makeDoc(JSON.stringify({ unrelated: true }));
    const mock = getMock();
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, 5) as never,
      { isCancellationRequested: false } as never,
    );
    assert.equal(result, null);
  });

  it("returns null when the document language is not JSON/JSONC", async () => {
    const provider = new DecisionHoverProvider();
    const doc = makeDoc(SAMPLE_RECEIPT, "plaintext");
    const mock = getMock();
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, 0) as never,
      { isCancellationRequested: false } as never,
    );
    assert.equal(result, null);
  });

  it("renders hover content with rule + matched pattern + reason", async () => {
    const provider = new DecisionHoverProvider();
    const doc = makeDoc(SAMPLE_RECEIPT);
    const mock = getMock();
    // Aim the cursor past the first `"tool": "shell.exec"` token.
    const cursorOffset = SAMPLE_RECEIPT.indexOf('"tool": "shell.exec"') + 5;
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, cursorOffset) as never,
      { isCancellationRequested: false } as never,
    );
    assert.ok(result !== null && result !== undefined, "expected a Hover");
    const hover = result as unknown as { contents: { value: string } };
    const md = hover.contents.value;
    assert.ok(md.includes("shell.exec"), "tool name should be in the hover");
    assert.ok(md.includes("RED"), "risk level should be uppercased in the hover");
    assert.ok(md.includes("GK-007"), "rule id should be in the hover");
    assert.ok(md.includes("rm -rf"), "matched pattern should be in the hover");
    assert.ok(
      md.includes("Destructive shell command"),
      "reason text should be in the hover",
    );
  });

  it("returns null when the cursor is outside any decision entry", async () => {
    const provider = new DecisionHoverProvider();
    const doc = makeDoc(SAMPLE_RECEIPT);
    const mock = getMock();
    // Position 0 is before any `"tool": "..."` probe matches.
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, 0) as never,
      { isCancellationRequested: false } as never,
    );
    assert.equal(result, null);
  });

  it("degrades gracefully when no structured explanation is attached", async () => {
    const noExplanation = JSON.stringify(
      {
        trace_id: "tr_demo",
        decisions: [{ tool: "fs.read", risk_level: "green", status: "ok" }],
      },
      null,
      2,
    );
    const provider = new DecisionHoverProvider();
    const doc = makeDoc(noExplanation);
    const mock = getMock();
    const cursorOffset = noExplanation.indexOf('"tool": "fs.read"') + 5;
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, cursorOffset) as never,
      { isCancellationRequested: false } as never,
    );
    assert.ok(result !== null && result !== undefined);
    const hover = result as unknown as { contents: { value: string } };
    assert.ok(
      hover.contents.value.includes("No structured explanation attached"),
      "should display the graceful-degrade message",
    );
  });

  it("returns null on malformed JSON instead of throwing", async () => {
    const provider = new DecisionHoverProvider();
    const doc = makeDoc('{ "trace_id": "x", "decisions": [ INVALID }');
    const mock = getMock();
    const result = await provider.provideHover(
      doc as never,
      new mock.Position(0, 5) as never,
      { isCancellationRequested: false } as never,
    );
    assert.equal(result, null, "malformed JSON should yield null, not throw");
  });
});
