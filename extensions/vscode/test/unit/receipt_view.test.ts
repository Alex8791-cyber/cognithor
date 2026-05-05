import { strict as assert } from "assert";
import * as http from "node:http";
import * as sinon from "sinon";
import { FakeMemento, FakeWebviewPanel, getMock, resetConfig, resetMockSpies, setConfig } from "./vscode-mock";

import {
  readReceiptConfig,
  ReceiptTreeProvider,
  ReceiptViewer,
} from "../../src/receipt_view";

interface CapturedRequest {
  path: string;
  host: string;
  port: number;
  method: string;
  headers: Record<string, string>;
}

function startStubServer(
  responder: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): Promise<{ port: number; close: () => Promise<void>; requests: CapturedRequest[] }> {
  const requests: CapturedRequest[] = [];
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      requests.push({
        path: req.url ?? "",
        host: "127.0.0.1",
        port: (server.address() as { port: number }).port,
        method: req.method ?? "GET",
        headers: req.headers as Record<string, string>,
      });
      responder(req, res);
    });
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as { port: number }).port;
      resolve({
        port,
        close: () =>
          new Promise<void>((res) => {
            server.close(() => res());
          }),
        requests,
      });
    });
  });
}

describe("ReceiptViewer + ReceiptTreeProvider", () => {
  beforeEach(() => {
    resetConfig();
    resetMockSpies();
  });

  it("readReceiptConfig pulls host/port/includeTrust from workspace settings", () => {
    setConfig({
      "cognithor.apiHost": "10.0.0.5",
      "cognithor.apiPort": 9999,
      "cognithor.receiptIncludeTrust": false,
    });
    const cfg = readReceiptConfig();
    assert.equal(cfg.apiHost, "10.0.0.5");
    assert.equal(cfg.apiPort, 9999);
    assert.equal(cfg.includeTrust, false);
  });

  it("show() warns and bails on empty trace id without firing HTTP", async () => {
    const memento = new FakeMemento();
    const viewer = new ReceiptViewer(memento as never);
    const mock = getMock();
    await viewer.show("   ");
    const warnStub = mock.window.showWarningMessage as sinon.SinonStub;
    assert.equal(warnStub.callCount, 1, "empty id should trigger warning");
    assert.equal(
      (mock.window.createWebviewPanel as sinon.SinonStub).callCount,
      0,
      "empty id must not open a webview",
    );
  });

  it("show() fetches receipt, opens a webview with rendered HTML, and remembers the trace id", async () => {
    const stub = await startStubServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ trace_id: "tr_abc", decisions: [] }));
    });
    const memento = new FakeMemento();
    const viewer = new ReceiptViewer(memento as never);
    const mock = getMock();
    try {
      await viewer.show("tr_abc", {
        apiHost: "127.0.0.1",
        apiPort: stub.port,
        includeTrust: true,
      });
    } finally {
      await stub.close();
    }
    const createCalls = (mock.window.createWebviewPanel as sinon.SinonStub).getCalls();
    assert.equal(createCalls.length, 1, "expected exactly one webview panel");
    const panel = createCalls[0]!.returnValue as FakeWebviewPanel;
    assert.ok(
      panel.webview.html.includes("Run receipt for trace tr_abc"),
      "rendered HTML should heading for parsed receipt",
    );
    assert.ok(panel.webview.html.includes("HTTP 200"), "status should appear in meta");
    assert.deepEqual(memento.store["cognithor.receiptTraceIds"], ["tr_abc"]);
    // include_trust=true should be on the wire.
    assert.equal(stub.requests.length, 1);
    assert.ok(stub.requests[0]!.path.endsWith("?include_trust=true"));
    assert.ok(stub.requests[0]!.path.includes("tr_abc"));
  });

  it("show() surfaces a fetch error via showErrorMessage and does not record the trace id", async () => {
    const memento = new FakeMemento();
    const viewer = new ReceiptViewer(memento as never);
    const mock = getMock();
    await viewer.show("tr_dead", {
      // unused port; immediate ECONNREFUSED.
      apiHost: "127.0.0.1",
      apiPort: 1,
      includeTrust: false,
    });
    const errStub = mock.window.showErrorMessage as sinon.SinonStub;
    assert.equal(errStub.callCount, 1, "fetch failure should surface an error toast");
    const msg = errStub.firstCall.args[0] as string;
    assert.ok(msg.startsWith("Cognithor: receipt fetch failed"), `unexpected: ${msg}`);
    assert.equal(
      memento.store["cognithor.receiptTraceIds"],
      undefined,
      "failed fetch must not pollute history",
    );
  });

  it("ReceiptViewer.rememberTraceId deduplicates and caps at MAX_HISTORY", async () => {
    const memento = new FakeMemento();
    const viewer = new ReceiptViewer(memento as never);
    for (let i = 0; i < 25; i++) {
      await viewer.rememberTraceId(`tr_${i}`);
    }
    // Re-add an existing one to confirm dedup + reorder.
    await viewer.rememberTraceId("tr_5");
    const stored = memento.store["cognithor.receiptTraceIds"] as string[];
    assert.equal(stored.length, 20, "should cap at 20 entries");
    assert.equal(stored[0], "tr_5", "most-recent should bubble to head");
    // No duplicates.
    assert.equal(new Set(stored).size, stored.length);
  });

  it("ReceiptTreeProvider exposes recent ids as TreeItems and refresh fires the change event", () => {
    const memento = new FakeMemento();
    memento.store["cognithor.receiptTraceIds"] = ["tr_first", "tr_second"];
    const viewer = new ReceiptViewer(memento as never);
    const provider = new ReceiptTreeProvider(viewer);
    const children = provider.getChildren();
    assert.equal(children.length, 2);
    assert.equal((children[0] as { label: string }).label, "tr_first");
    let fired = 0;
    provider.onDidChangeTreeData(() => {
      fired += 1;
    });
    provider.refresh();
    assert.equal(fired, 1, "refresh() should fire onDidChangeTreeData exactly once");
  });
});
