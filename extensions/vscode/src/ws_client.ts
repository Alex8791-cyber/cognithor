/**
 * Cognithor WebSocket client (Sprint-27 PR-F).
 *
 * Talks to the `cognithor agent ws` server (PR-C) at
 * `ws://127.0.0.1:8742/` with a bearer-token PSK loaded from
 * `~/.cognithor/auth.token` (or `COGNITHOR_HOME/auth.token`).
 *
 * Wire shape — matches `src/cognithor/streaming/server.py`:
 *
 *   1. Client opens WS with `Authorization: Bearer <token>`.
 *   2. Client sends first frame:
 *      `{"type": "run_plan", "plan_path": "<abs path>"}` OR
 *      `{"type": "run_plan", "plan": {...inline...}}`.
 *   3. Server streams JSON event frames (per
 *      `src/cognithor/streaming/schemas/v1/events.json`).
 *   4. Server closes the connection cleanly when the run finishes.
 *
 * The client surfaces three observers:
 *
 *   - onEvent: each parsed JSON frame from the server.
 *   - onError: connection-level errors (transport, auth, parse).
 *   - onClose: clean or abnormal close, with code + reason.
 *
 * Connection lifetime is one-shot per `runPlan()` call. The
 * extension is responsible for launching `cognithor agent ws`
 * out-of-band; this client does NOT spawn the server (that is
 * intentionally deferred to PR-K's end-to-end smoke).
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import WebSocket from "ws";

export interface WsClientOptions {
  bind?: string;
  port?: number;
  tokenPath?: string;
  /** Override for tests — receive a token directly instead of reading from disk. */
  token?: string;
  /** Maximum time in ms to wait for connection open before failing. */
  connectTimeoutMs?: number;
}

export interface RunPlanRequest {
  /** Absolute path to a plan JSON on the same filesystem as the server. */
  planPath?: string;
  /** Inline plan object (mutually exclusive with planPath). */
  plan?: Record<string, unknown>;
}

export interface CloseInfo {
  code: number;
  reason: string;
  wasClean: boolean;
}

export type EventListener = (event: Record<string, unknown>) => void;
export type ErrorListener = (err: Error) => void;
export type CloseListener = (info: CloseInfo) => void;

const DEFAULT_PORT = 8742;
const DEFAULT_BIND = "127.0.0.1";
const DEFAULT_CONNECT_TIMEOUT_MS = 5_000;

export function defaultTokenPath(): string {
  const override = process.env.COGNITHOR_HOME;
  const home = override ? path.resolve(override) : path.join(os.homedir(), ".cognithor");
  return path.join(home, "auth.token");
}

export function loadToken(tokenPath?: string): string {
  const target = tokenPath ?? defaultTokenPath();
  if (!fs.existsSync(target)) {
    throw new Error(
      `Cognithor auth token not found at ${target}. ` +
        "Start the agent ws server once (`cognithor agent ws`) — it generates the token on first run.",
    );
  }
  const raw = fs.readFileSync(target, "utf-8").trim();
  if (!raw) {
    throw new Error(`Cognithor auth token at ${target} is empty.`);
  }
  return raw;
}

/**
 * One-shot WebSocket client for `cognithor agent ws`.
 *
 * Usage:
 *
 *     const client = new WsClient({ port: 8742 });
 *     client.onEvent((evt) => console.log(evt));
 *     await client.runPlan({ planPath: "/abs/path/to/plan.json" });
 */
export class WsClient {
  private readonly bind: string;
  private readonly port: number;
  private readonly token: string;
  private readonly connectTimeoutMs: number;

  private ws: WebSocket | null = null;
  private readonly eventListeners = new Set<EventListener>();
  private readonly errorListeners = new Set<ErrorListener>();
  private readonly closeListeners = new Set<CloseListener>();

  constructor(opts: WsClientOptions = {}) {
    this.bind = opts.bind ?? DEFAULT_BIND;
    this.port = opts.port ?? DEFAULT_PORT;
    this.token = opts.token ?? loadToken(opts.tokenPath);
    this.connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
  }

  onEvent(listener: EventListener): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  onError(listener: ErrorListener): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  onClose(listener: CloseListener): () => void {
    this.closeListeners.add(listener);
    return () => this.closeListeners.delete(listener);
  }

  /**
   * Connect to the agent ws server, send a `run_plan` request,
   * and stream events until the server closes the connection.
   *
   * Resolves once the connection is closed (cleanly or otherwise).
   * Rejects only on connection-open failure; mid-stream errors
   * are reported via `onError` so partial event streams remain
   * visible to the caller.
   */
  async runPlan(req: RunPlanRequest): Promise<CloseInfo> {
    if (!req.planPath && !req.plan) {
      throw new Error("runPlan requires either planPath or plan");
    }

    const url = `ws://${this.bind}:${this.port}/`;
    const ws = new WebSocket(url, {
      headers: { Authorization: `Bearer ${this.token}` },
    });
    this.ws = ws;

    // Wait for open with timeout — surface a clear error if the
    // server isn't running rather than hanging silently.
    await new Promise<void>((resolve, reject) => {
      const onOpen = (): void => {
        cleanup();
        resolve();
      };
      const onError = (err: Error): void => {
        cleanup();
        try {
          ws.terminate();
        } catch {
          // best-effort
        }
        this.ws = null;
        reject(err);
      };
      const timer = setTimeout(() => {
        cleanup();
        // Tear the dangling socket down before rejecting so the
        // node process doesn't accumulate orphan TCP connections
        // on every "agent ws not running" attempt.
        try {
          ws.terminate();
        } catch {
          // best-effort
        }
        this.ws = null;
        reject(new Error(`WS open timeout after ${this.connectTimeoutMs} ms — is the agent ws server running on ${url}?`));
      }, this.connectTimeoutMs);
      const cleanup = (): void => {
        clearTimeout(timer);
        ws.off("open", onOpen);
        ws.off("error", onError);
      };
      ws.once("open", onOpen);
      ws.once("error", onError);
    });

    // Send the run_plan request — only one of planPath / plan
    // gets serialised so we don't send conflicting fields.
    const payload: Record<string, unknown> = { type: "run_plan" };
    if (req.planPath) {
      payload.plan_path = req.planPath;
    } else if (req.plan) {
      payload.plan = req.plan;
    }
    ws.send(JSON.stringify(payload));

    return new Promise<CloseInfo>((resolve) => {
      ws.on("message", (data) => {
        const text =
          typeof data === "string"
            ? data
            : Buffer.isBuffer(data)
              ? data.toString("utf-8")
              : Array.isArray(data)
                ? Buffer.concat(data).toString("utf-8")
                : Buffer.from(data as ArrayBuffer).toString("utf-8");
        let evt: Record<string, unknown>;
        try {
          evt = JSON.parse(text) as Record<string, unknown>;
        } catch (parseErr) {
          this.emitError(new Error(`malformed event frame: ${(parseErr as Error).message}`));
          return;
        }
        for (const listener of this.eventListeners) {
          try {
            listener(evt);
          } catch (handlerErr) {
            this.emitError(handlerErr as Error);
          }
        }
      });

      ws.on("error", (err) => {
        this.emitError(err);
      });

      ws.on("close", (code, reason) => {
        const info: CloseInfo = {
          code,
          reason: reason.toString("utf-8"),
          wasClean: code === 1000 || code === 1001,
        };
        for (const listener of this.closeListeners) {
          try {
            listener(info);
          } catch (handlerErr) {
            this.emitError(handlerErr as Error);
          }
        }
        this.ws = null;
        resolve(info);
      });
    });
  }

  /**
   * Force-close the connection if the user cancels.
   *
   * Idempotent — safe to call when there's no active connection.
   */
  close(): void {
    if (!this.ws) return;
    try {
      this.ws.close(1000, "client cancelled");
    } catch {
      // best-effort
    }
    this.ws = null;
  }

  private emitError(err: Error): void {
    for (const listener of this.errorListeners) {
      try {
        listener(err);
      } catch {
        // swallow listener errors — don't let one bad listener break the chain
      }
    }
  }
}
