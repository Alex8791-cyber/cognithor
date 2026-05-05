import { strict as assert } from "assert";
import { EventEmitter, Readable, Writable } from "stream";
import * as Module from "module";
import * as actualChildProcess from "child_process";

/**
 * Hijack `child_process.spawn` before mcp_bridge.ts imports it.
 * The require hook returns a wrapper that delegates `spawn` to a
 * fake-process factory under our control. All other exports stay
 * real so transitive `child_process.exec` etc. (used by VS Code
 * dependencies indirectly) keep working.
 */

interface SpawnRecord {
  command: string;
  args: readonly string[];
  options: actualChildProcess.SpawnOptions;
  child: FakeChildProcess;
}

const spawnLog: SpawnRecord[] = [];

class FakeStdin extends Writable {
  public lines: string[] = [];
  override _write(
    chunk: Buffer | string,
    _enc: BufferEncoding,
    cb: (err?: Error | null) => void,
  ): void {
    const text = typeof chunk === "string" ? chunk : chunk.toString("utf-8");
    // mcp_bridge writes one JSON-RPC message per line.
    for (const line of text.split("\n")) {
      if (line !== "") this.lines.push(line);
    }
    cb();
  }
}

class FakeStdout extends Readable {
  override _read(): void {
    /* push happens externally via .push() */
  }
  pushLine(line: string): void {
    this.push(line + "\n");
  }
}

class FakeChildProcess extends EventEmitter {
  public stdin = new FakeStdin();
  public stdout = new FakeStdout();
  public stderr = new FakeStdout();
  public killed = false;
  public lastSignal: string | null = null;
  public pid = 12345;
  kill(signal?: string): boolean {
    this.killed = true;
    this.lastSignal = signal ?? "SIGTERM";
    return true;
  }
}

function fakeSpawn(
  command: string,
  args?: readonly string[],
  options?: actualChildProcess.SpawnOptions,
): FakeChildProcess {
  const child = new FakeChildProcess();
  spawnLog.push({
    command,
    args: args ?? [],
    options: options ?? {},
    child,
  });
  return child;
}

interface ProtoRequire {
  (this: NodeJS.Module, id: string): unknown;
}
const proto = (Module as unknown as { prototype: { require: ProtoRequire } }).prototype;
const originalRequire = proto.require;
proto.require = function (this: NodeJS.Module, id: string): unknown {
  if (id === "child_process") {
    return {
      ...actualChildProcess,
      spawn: fakeSpawn,
    };
  }
  return originalRequire.call(this, id);
};

// Now load the SUT under the hook.
import {
  DEFAULT_BRIDGE_CONFIG,
  McpBridge,
  readBridgeConfig,
  type BridgeConfig,
} from "../../src/mcp_bridge";
import { setConfig, resetConfig } from "./vscode-mock";

const fastConfig: BridgeConfig = {
  ...DEFAULT_BRIDGE_CONFIG,
  cliPath: "fake-cognithor",
  heartbeatMs: 60_000, // long — heartbeat won't fire in any test
  pongTimeoutMs: 1_000,
  restartBackoffMs: 50,
  maxRestarts: 2,
  requestTimeoutMs: 1_000,
};

function lastChild(): FakeChildProcess {
  return spawnLog[spawnLog.length - 1]!.child;
}

function flushIo(times = 4): Promise<void> {
  return new Promise((resolve) => {
    let n = times;
    const tick = (): void => {
      if (n-- <= 0) {
        resolve();
        return;
      }
      setImmediate(tick);
    };
    tick();
  });
}

describe("McpBridge", () => {
  beforeEach(() => {
    spawnLog.length = 0;
  });

  it("spawns the configured CLI with `mcp --stdio` args and reaches onReady after handshake", async () => {
    const bridge = new McpBridge(fastConfig);
    let ready = false;
    bridge.onReady(() => {
      ready = true;
    });
    const startPromise = bridge.start();
    await flushIo();
    const rec = spawnLog[0]!;
    assert.equal(rec.command, "fake-cognithor");
    assert.deepEqual(rec.args, ["mcp", "--stdio"]);
    // The handshake call serialised one JSON-RPC `initialize` message.
    const initLine = rec.child.stdin.lines[0]!;
    const parsed = JSON.parse(initLine) as Record<string, unknown>;
    assert.equal(parsed.method, "initialize");
    assert.equal(parsed.jsonrpc, "2.0");
    assert.equal(parsed.id, 1);
    // Respond so handshake completes.
    rec.child.stdout.pushLine(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { ok: true } }));
    await startPromise;
    assert.equal(ready, true);
    const stopPromise = bridge.stop();
    rec.child.emit("exit", 0);
    await stopPromise;
  });

  it("call() correlates response by id and resolves with the result", async () => {
    const bridge = new McpBridge(fastConfig);
    const startPromise = bridge.start();
    await flushIo();
    const child = lastChild();
    child.stdout.pushLine(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }));
    await startPromise;
    // Now make a real call and satisfy it.
    const callPromise = bridge.call("video_render", { run_id: "r1" });
    await flushIo();
    const lastLine = child.stdin.lines[child.stdin.lines.length - 1]!;
    const sent = JSON.parse(lastLine) as Record<string, unknown>;
    assert.equal(sent.method, "video_render");
    assert.equal(sent.id, 2);
    child.stdout.pushLine(
      JSON.stringify({ jsonrpc: "2.0", id: 2, result: { mp4_path: "/tmp/x.mp4" } }),
    );
    const result = (await callPromise) as { mp4_path: string };
    assert.equal(result.mp4_path, "/tmp/x.mp4");
    const stopPromise = bridge.stop();
    child.emit("exit", 0);
    await stopPromise;
  });

  it("call() rejects when the server returns a JSON-RPC error", async () => {
    const bridge = new McpBridge(fastConfig);
    const startPromise = bridge.start();
    await flushIo();
    const child = lastChild();
    child.stdout.pushLine(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }));
    await startPromise;
    const callPromise = bridge.call("broken_tool");
    await flushIo();
    child.stdout.pushLine(
      JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        error: { code: -32602, message: "invalid params" },
      }),
    );
    await assert.rejects(callPromise, /invalid params/);
    const stopPromise = bridge.stop();
    child.emit("exit", 0);
    await stopPromise;
  });

  it("ignores malformed stdout lines, surfaces them via onLog, and keeps working", async () => {
    const bridge = new McpBridge(fastConfig);
    const logs: string[] = [];
    bridge.onLog((s) => logs.push(s));
    const startPromise = bridge.start();
    await flushIo();
    const child = lastChild();
    child.stdout.pushLine("this is not json");
    child.stdout.pushLine(""); // blank — explicitly skipped
    child.stdout.pushLine(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }));
    await startPromise;
    assert.ok(
      logs.some((l) => l.startsWith("malformed stdout line")),
      "malformed line should surface via onLog",
    );
    const stopPromise = bridge.stop();
    child.emit("exit", 0);
    await stopPromise;
  });

  it("stop() rejects in-flight requests and SIGTERMs the child", async () => {
    const bridge = new McpBridge(fastConfig);
    const startPromise = bridge.start();
    await flushIo();
    const child = lastChild();
    child.stdout.pushLine(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }));
    await startPromise;
    const inflight = bridge.call("never_resolves");
    await flushIo();
    // Make stop() resolve quickly by emitting `exit` immediately.
    const stopPromise = bridge.stop();
    child.emit("exit", 0);
    await stopPromise;
    assert.equal(child.lastSignal, "SIGTERM");
    await assert.rejects(inflight, /MCP bridge stopped/);
  });

  it("readBridgeConfig pulls cliPath from workspace settings, defaults the rest", () => {
    resetConfig();
    setConfig({ "cognithor.cliPath": "/usr/local/bin/cognithor" });
    const cfg = readBridgeConfig();
    assert.equal(cfg.cliPath, "/usr/local/bin/cognithor");
    assert.equal(cfg.heartbeatMs, DEFAULT_BRIDGE_CONFIG.heartbeatMs);
    assert.equal(cfg.maxRestarts, DEFAULT_BRIDGE_CONFIG.maxRestarts);
  });
});
