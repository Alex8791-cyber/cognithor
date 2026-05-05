import { strict as assert } from "assert";
import * as sinon from "sinon";
import { getMock, resetConfig, resetMockSpies, setConfig } from "./vscode-mock";

import { CostGutterController } from "../../src/cost_gutter";

interface FakeDocument {
  languageId: string;
  lineCount: number;
  getText(): string;
  positionAt(offset: number): { line: number; character: number };
}

interface FakeEditor {
  document: FakeDocument;
  setDecorations: sinon.SinonStub;
}

function makeContext(): { subscriptions: { dispose: () => void }[] } {
  return { subscriptions: [] };
}

function makeReceiptText(entries: number[]): string {
  const decisions = entries
    .map((micro, i) => `{ "tool": "t_${i}", "cost_usd_micro": ${micro} }`)
    .join(",\n");
  return `{ "trace_id": "tr_test", "decisions": [${decisions}] }`;
}

function makeEditor(text: string, languageId = "json"): FakeEditor {
  const doc: FakeDocument = {
    languageId,
    lineCount: text.split("\n").length,
    getText: () => text,
    positionAt: (offset: number) => ({ line: 0, character: offset }),
  };
  return {
    document: doc,
    setDecorations: sinon.stub(),
  };
}

describe("CostGutterController", () => {
  beforeEach(() => {
    resetConfig();
    resetMockSpies();
  });

  it("instantiates with a fresh context and registers four decoration types", () => {
    const mock = getMock();
    const ctx = makeContext();
    new CostGutterController(ctx as never);
    const stub = mock.window.createTextEditorDecorationType as sinon.SinonStub;
    assert.equal(stub.callCount, 4, "expected 4 tier decorations (green/yellow/orange/red)");
    assert.equal(ctx.subscriptions.length >= 5, true, "decorations + status item registered");
  });

  it("applies tiered decorations when costs exceed thresholds", () => {
    const ctrl = new CostGutterController(makeContext() as never);
    // 500 = green, 5_000 = yellow, 50_000 = orange, 500_000 = red.
    const editor = makeEditor(makeReceiptText([500, 5_000, 50_000, 500_000]));
    ctrl.update(editor as never);
    // Expect 4 setDecorations calls (one per tier), each with non-empty
    // ranges where its tier matched.
    const calls = editor.setDecorations.getCalls();
    assert.equal(calls.length, 4, "one setDecorations call per tier");
    const nonEmptyTiers = calls.filter((c) => (c.args[1] as unknown[]).length > 0).length;
    assert.equal(nonEmptyTiers, 4, "every tier should have one matching range");
  });

  it("clears decorations when the active document is not a receipt", () => {
    const ctrl = new CostGutterController(makeContext() as never);
    const editor = makeEditor("just plain prose, no trace_id here", "plaintext");
    ctrl.update(editor as never);
    const calls = editor.setDecorations.getCalls();
    // Non-receipt path still iterates over all four tiers but with empty arrays.
    assert.equal(calls.length, 4);
    for (const c of calls) {
      assert.deepEqual(c.args[1], [], "non-receipt should clear all decorations");
    }
  });

  it("respects custom thresholds from VS-Code configuration", () => {
    // Set the green threshold high enough to absorb a value that
    // would otherwise be yellow under the defaults.
    setConfig({
      "cognithor.costThresholdGreenMicro": 1_000_000,
      "cognithor.costThresholdYellowMicro": 1_000_001,
      "cognithor.costThresholdOrangeMicro": 1_000_002,
    });
    const ctrl = new CostGutterController(makeContext() as never);
    const editor = makeEditor(makeReceiptText([5_000])); // would be yellow under defaults
    ctrl.update(editor as never);
    const calls = editor.setDecorations.getCalls();
    // Find the call with a non-empty range — it must be the *first* (green) decoration type.
    const decorations = calls.map((c) => ({ deco: c.args[0], ranges: c.args[1] as unknown[] }));
    const nonEmpty = decorations.filter((d) => d.ranges.length > 0);
    assert.equal(nonEmpty.length, 1, "exactly one tier should contain the entry");
    // The first call (green decoration) is the one with the entry.
    assert.equal(decorations[0]?.ranges.length, 1, "entry should land in green tier");
  });

  it("treats a missing editor as a no-op (no crash, hides status)", () => {
    const ctrl = new CostGutterController(makeContext() as never);
    // Should not throw with undefined editor.
    assert.doesNotThrow(() => ctrl.update(undefined as never));
  });

  it("disposes cleanly when the context's dispose hook fires", () => {
    const ctx = makeContext();
    const ctrl = new CostGutterController(ctx as never);
    const editor = makeEditor(makeReceiptText([500]));
    ctrl.update(editor as never);
    // Find the dispose-only subscription (last entry pushed in constructor).
    const disposable = ctx.subscriptions[ctx.subscriptions.length - 1];
    assert.equal(typeof disposable.dispose, "function");
    disposable.dispose();
    // After disposal, further updates must be no-ops (no setDecorations).
    const editor2 = makeEditor(makeReceiptText([500]));
    ctrl.update(editor2 as never);
    assert.equal(editor2.setDecorations.callCount, 0, "post-dispose update should be inert");
  });
});
