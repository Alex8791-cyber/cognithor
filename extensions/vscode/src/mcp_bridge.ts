/**
 * Sprint-27 PR-E — MCP-stdio bridge.
 *
 * Auto-spawns `cognithor mcp --stdio` on workspace activation,
 * heartbeats every 30 s, and restarts the subprocess on no-pong
 * within 5 s. Mitigates plan-doc §6 risk-1 (MCP-stdio handshake
 * instability for VS-Code's activate/deactivate cycle).
 *
 * The bridge speaks JSON-RPC over the spawned process's stdin /
 * stdout. The Cognithor side ships 138+ MCP tools (per
 * `docs/integrations/catalog.json`), so once the bridge is up
 * the rest of the extension can call any of them via
 * `bridge.call("video_render", { run_id: ..., html_text: ... })`.
 *
 * Lifecycle hooks:
 *
 *   - `activate(context)` → constructs the bridge, registers
 *     dispose-on-deactivate.
 *   - `bridge.start()` → spawns the subprocess + handshake.
 *   - `bridge.stop()` → graceful shutdown (SIGTERM → 2 s grace
 *     → SIGKILL).
 *
 * Exposed events (via `vscode.EventEmitter`):
 *
 *   - `onReady`   — fired after successful handshake.
 *   - `onCrash`   — fired on unexpected exit (with restart
 *                   attempt count).
 *   - `onLog`     — child-process stderr lines (debug surface).
 */

import { spawn, type ChildProcess } from "child_process";
import * as readline from "readline";
import * as vscode from "vscode";

/** A pending in-flight JSON-RPC request awaiting its response. */
interface Pending {
  resolve: (result: unknown) => void;
  reject: (error: Error) => void;
  /** Wall-clock deadline for the request (epoch ms). */
  deadlineMs: number;
}

/** JSON-RPC 2.0 minimal message shape. */
interface JsonRpcMessage {
  jsonrpc: "2.0";
  id?: number | string | null;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

/** Configuration — pulled from the workspace `cognithor.*` settings. */
export interface BridgeConfig {
  cliPath: string;
  heartbeatMs: number;
  pongTimeoutMs: number;
  restartBackoffMs: number;
  maxRestarts: number;
  requestTimeoutMs: number;
}

export const DEFAULT_BRIDGE_CONFIG: BridgeConfig = {
  cliPath: "cognithor",
  heartbeatMs: 30_000,
  pongTimeoutMs: 5_000,
  restartBackoffMs: 2_000,
  maxRestarts: 5,
  requestTimeoutMs: 30_000,
};

export class McpBridge implements vscode.Disposable {
  private child: ChildProcess | null = null;
  private rl: readline.Interface | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  private heartbeatTimer: NodeJS.Timeout | null = null;
  private restartCount = 0;
  private stopped = false;

  private readonly readyEmitter = new vscode.EventEmitter<void>();
  private readonly crashEmitter = new vscode.EventEmitter<{
    code: number | null;
    restartCount: number;
  }>();
  private readonly logEmitter = new vscode.EventEmitter<string>();

  readonly onReady: vscode.Event<void> = this.readyEmitter.event;
  readonly onCrash: vscode.Event<{ code: number | null; restartCount: number }> =
    this.crashEmitter.event;
  readonly onLog: vscode.Event<string> = this.logEmitter.event;

  constructor(private readonly config: BridgeConfig = DEFAULT_BRIDGE_CONFIG) {}

  /** Spawn the subprocess + run the JSON-RPC initialise handshake. */
  async start(): Promise<void> {
    if (this.child !== null) {
      return; // idempotent
    }
    this.stopped = false;
    this.spawnChild();
    await this.handshake();
    this.startHeartbeat();
    this.readyEmitter.fire();
  }

  /** Graceful shutdown — SIGTERM, 2 s grace, then SIGKILL. */
  async stop(): Promise<void> {
    this.stopped = true;
    this.stopHeartbeat();
    if (this.child === null) {
      return;
    }
    const child = this.child;
    this.child = null;
    this.rl?.close();
    this.rl = null;
    // Reject any in-flight requests so the caller doesn't hang.
    for (const [id, pending] of this.pending) {
      pending.reject(new Error("MCP bridge stopped"));
      this.pending.delete(id);
    }
    if (!child.killed) {
      child.kill("SIGTERM");
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          if (!child.killed) {
            child.kill("SIGKILL");
          }
          resolve();
        }, 2_000);
        child.once("exit", () => {
          clearTimeout(timer);
          resolve();
        });
      });
    }
  }

  /** Send a JSON-RPC request and await the response. */
  async call(method: string, params?: unknown): Promise<unknown> {
    if (this.child === null) {
      throw new Error("MCP bridge is not running");
    }
    const id = this.nextId++;
    const message: JsonRpcMessage = { jsonrpc: "2.0", id, method, params };
    return new Promise<unknown>((resolve, reject) => {
      const deadlineMs = Date.now() + this.config.requestTimeoutMs;
      this.pending.set(id, { resolve, reject, deadlineMs });
      this.writeMessage(message);
      // Per-request deadline guard (cleared on response).
      setTimeout(() => {
        const pending = this.pending.get(id);
        if (pending !== undefined) {
          this.pending.delete(id);
          pending.reject(new Error(`MCP request ${method} timed out`));
        }
      }, this.config.requestTimeoutMs);
    });
  }

  dispose(): void {
    void this.stop();
    this.readyEmitter.dispose();
    this.crashEmitter.dispose();
    this.logEmitter.dispose();
  }

  // ------------------------------------------------------------------
  // Internals
  // ------------------------------------------------------------------

  private spawnChild(): void {
    const child = spawn(this.config.cliPath, ["mcp", "--stdio"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });
    this.child = child;
    if (child.stdout !== null) {
      this.rl = readline.createInterface({ input: child.stdout });
      this.rl.on("line", (line) => this.onStdoutLine(line));
    }
    if (child.stderr !== null) {
      const errRl = readline.createInterface({ input: child.stderr });
      errRl.on("line", (line) => this.logEmitter.fire(`stderr: ${line}`));
    }
    child.on("exit", (code) => this.onChildExit(code));
    child.on("error", (err) => this.logEmitter.fire(`spawn error: ${err.message}`));
  }

  private async handshake(): Promise<void> {
    // Minimal MCP `initialize` per the spec — name + version of
    // the client. The Cognithor server responds with its own
    // capability set; we don't need to inspect it here, but the
    // round-trip proves the subprocess is alive and parsing.
    await this.call("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "cognithor-vscode", version: "0.97.0" },
    });
  }

  private writeMessage(message: JsonRpcMessage): void {
    if (this.child === null || this.child.stdin === null) {
      throw new Error("MCP bridge: child stdin unavailable");
    }
    this.child.stdin.write(JSON.stringify(message) + "\n");
  }

  private onStdoutLine(line: string): void {
    if (line.trim() === "") {
      return;
    }
    let message: JsonRpcMessage;
    try {
      message = JSON.parse(line) as JsonRpcMessage;
    } catch (_err) {
      this.logEmitter.fire(`malformed stdout line: ${line.slice(0, 200)}`);
      return;
    }
    if (typeof message.id === "number") {
      const pending = this.pending.get(message.id);
      if (pending !== undefined) {
        this.pending.delete(message.id);
        if (message.error !== undefined) {
          pending.reject(new Error(message.error.message));
        } else {
          pending.resolve(message.result);
        }
      }
      return;
    }
    // Notifications + server-initiated messages — ignored for
    // PR-E's scope. PR-F and PR-G will subscribe to crew lifecycle
    // notifications for the receipt sidebar.
  }

  private onChildExit(code: number | null): void {
    this.child = null;
    this.rl?.close();
    this.rl = null;
    this.stopHeartbeat();
    if (this.stopped) {
      return; // intentional shutdown
    }
    this.restartCount += 1;
    this.crashEmitter.fire({ code, restartCount: this.restartCount });
    if (this.restartCount > this.config.maxRestarts) {
      this.logEmitter.fire(
        `MCP bridge: max restart count (${this.config.maxRestarts}) exceeded`,
      );
      return;
    }
    setTimeout(() => {
      if (!this.stopped) {
        // Best-effort restart. Failures will surface via the log
        // event + a subsequent crash event.
        void this.start().catch((err) => {
          this.logEmitter.fire(`restart failed: ${(err as Error).message}`);
        });
      }
    }, this.config.restartBackoffMs);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      void this.tick();
    }, this.config.heartbeatMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private async tick(): Promise<void> {
    if (this.child === null) {
      return;
    }
    // The MCP spec defines a `ping` request that returns an empty
    // result. Use that as our heartbeat; if the subprocess fails
    // to respond within `pongTimeoutMs`, treat it as crashed.
    try {
      await Promise.race([
        this.call("ping"),
        new Promise<never>((_resolve, reject) => {
          setTimeout(
            () => reject(new Error("ping timeout")),
            this.config.pongTimeoutMs,
          );
        }),
      ]);
    } catch (_err) {
      this.logEmitter.fire(
        `heartbeat: no pong within ${this.config.pongTimeoutMs}ms — restarting`,
      );
      // Synthesise a crash so the existing restart logic kicks in.
      this.child?.kill("SIGTERM");
    }
  }
}

/** Read the bridge config from VS Code workspace settings. */
export function readBridgeConfig(): BridgeConfig {
  const cfg = vscode.workspace.getConfiguration("cognithor");
  return {
    ...DEFAULT_BRIDGE_CONFIG,
    cliPath: cfg.get<string>("cliPath") ?? DEFAULT_BRIDGE_CONFIG.cliPath,
  };
}
