import { strict as assert } from "assert";
import { EventEmitter } from "events";
import * as Module from "module";

/**
 * Hijack the `ws` module before `ws_client.ts` imports it.
 * We register a require hook here that returns a fake WebSocket
 * class (and mark it as the default export so esModuleInterop's
 * `import WebSocket from "ws"` resolves to our fake constructor).
 */

interface FakeSendCall {
  data: unknown;
}

class FakeWebSocket extends EventEmitter {
  static lastInstance: FakeWebSocket | null = null;
  static instances: FakeWebSocket[] = [];

  public readonly url: string;
  public readonly options: Record<string, unknown>;
  public readonly sends: FakeSendCall[] = [];
  public terminated = false;
  public closed: { code: number; reason: string } | null = null;

  constructor(url: string, options: Record<string, unknown> = {}) {
    super();
    this.url = url;
    this.options = options;
    FakeWebSocket.lastInstance = this;
    FakeWebSocket.instances.push(this);
  }

  send(data: unknown): void {
    this.sends.push({ data });
  }

  terminate(): void {
    this.terminated = true;
  }

  close(code: number, reason: string): void {
    this.closed = { code, reason };
  }

  // Helpers used by tests:
  fireOpen(): void {
    this.emit("open");
  }
  fireMessage(data: unknown): void {
    this.emit("message", data);
  }
  fireError(err: Error): void {
    this.emit("error", err);
  }
  fireClose(code = 1000, reason = ""): void {
    this.emit("close", code, Buffer.from(reason));
  }
}

interface ProtoRequire {
  (this: NodeJS.Module, id: string): unknown;
}
const proto = (Module as unknown as { prototype: { require: ProtoRequire } }).prototype;
const originalRequire = proto.require;
proto.require = function (this: NodeJS.Module, id: string): unknown {
  if (id === "ws") {
    // esModuleInterop reads either `default` or the function itself.
    return Object.assign(FakeWebSocket as unknown as object, {
      default: FakeWebSocket,
      __esModule: true,
    });
  }
  return originalRequire.call(this, id);
};

// Now require the SUT — its `import WebSocket from "ws"` resolves to FakeWebSocket.
import { WsClient } from "../../src/ws_client";

function newClient(): WsClient {
  return new WsClient({
    bind: "127.0.0.1",
    port: 12345,
    token: "fake-token",
    connectTimeoutMs: 100,
  });
}

describe("WsClient", () => {
  beforeEach(() => {
    FakeWebSocket.lastInstance = null;
    FakeWebSocket.instances = [];
  });

  it("rejects runPlan() when neither planPath nor plan is provided", async () => {
    const client = newClient();
    await assert.rejects(client.runPlan({}), /requires either planPath or plan/);
  });

  it("opens with bearer auth and forwards plan_path on the wire", async () => {
    const client = newClient();
    const runPromise = client.runPlan({ planPath: "/abs/plan.json" });
    // Allow microtasks to schedule the WS construction.
    await new Promise((r) => setImmediate(r));
    const ws = FakeWebSocket.lastInstance!;
    assert.ok(ws, "expected a WebSocket instance");
    assert.equal(ws.url, "ws://127.0.0.1:12345/");
    assert.equal((ws.options.headers as Record<string, string>).Authorization, "Bearer fake-token");
    ws.fireOpen();
    await new Promise((r) => setImmediate(r));
    assert.equal(ws.sends.length, 1);
    const sent = JSON.parse(ws.sends[0]!.data as string) as Record<string, unknown>;
    assert.equal(sent.type, "run_plan");
    assert.equal(sent.plan_path, "/abs/plan.json");
    ws.fireClose(1000, "");
    await runPromise;
  });

  it("delivers parsed JSON event frames to onEvent listeners", async () => {
    const client = newClient();
    const events: Record<string, unknown>[] = [];
    client.onEvent((e) => events.push(e));
    const runPromise = client.runPlan({ plan: { steps: [] } });
    await new Promise((r) => setImmediate(r));
    const ws = FakeWebSocket.lastInstance!;
    ws.fireOpen();
    await new Promise((r) => setImmediate(r));
    ws.fireMessage(Buffer.from(JSON.stringify({ type: "step_start", id: "s1" })));
    ws.fireMessage('{"type":"step_end","id":"s1"}');
    ws.fireClose(1000, "ok");
    await runPromise;
    assert.equal(events.length, 2);
    assert.equal(events[0]!.type, "step_start");
    assert.equal(events[1]!.type, "step_end");
  });

  it("emits onError for malformed JSON frames without aborting the stream", async () => {
    const client = newClient();
    const errors: Error[] = [];
    const events: Record<string, unknown>[] = [];
    client.onError((e) => errors.push(e));
    client.onEvent((e) => events.push(e));
    const runPromise = client.runPlan({ plan: {} });
    await new Promise((r) => setImmediate(r));
    const ws = FakeWebSocket.lastInstance!;
    ws.fireOpen();
    await new Promise((r) => setImmediate(r));
    ws.fireMessage("{not valid json");
    ws.fireMessage('{"type":"recovered"}');
    ws.fireClose(1000, "");
    await runPromise;
    assert.equal(errors.length, 1, "exactly one parse error should surface");
    assert.match(errors[0]!.message, /malformed event frame/);
    assert.equal(events.length, 1, "post-error frames should still be delivered");
    assert.equal(events[0]!.type, "recovered");
  });

  it("rejects runPlan() with a clear message when the open times out", async () => {
    const client = new WsClient({
      bind: "127.0.0.1",
      port: 12345,
      token: "tok",
      connectTimeoutMs: 25,
    });
    await assert.rejects(client.runPlan({ plan: {} }), /WS open timeout/);
    // The dangling socket was torn down — terminate should have been called.
    const ws = FakeWebSocket.lastInstance!;
    assert.ok(ws.terminated, "open-timeout must terminate the dangling socket");
  });

  it("close() is idempotent and a no-op when no connection is active", () => {
    const client = newClient();
    assert.doesNotThrow(() => client.close());
    assert.doesNotThrow(() => client.close());
  });

  it("delivers CloseInfo with wasClean=true for code 1000 to onClose listeners", async () => {
    const client = newClient();
    const closes: { code: number; wasClean: boolean }[] = [];
    client.onClose((info) => closes.push({ code: info.code, wasClean: info.wasClean }));
    const runPromise = client.runPlan({ plan: {} });
    await new Promise((r) => setImmediate(r));
    const ws = FakeWebSocket.lastInstance!;
    ws.fireOpen();
    await new Promise((r) => setImmediate(r));
    ws.fireClose(1000, "done");
    const info = await runPromise;
    assert.equal(info.code, 1000);
    assert.equal(info.wasClean, true);
    assert.equal(info.reason, "done");
    assert.equal(closes.length, 1);
    assert.equal(closes[0]!.wasClean, true);
  });
});
