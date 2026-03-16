import { useState, useRef, useCallback, useEffect } from "react";

const WS_RECONNECT_DELAY = 3000;
const WS_MAX_RETRIES = 10;
const WS_HEARTBEAT_INTERVAL = 30000;

// Fetch auth token from bootstrap endpoint (cached)
let _wsTokenCache = null;
async function getWsToken() {
  if (_wsTokenCache) return _wsTokenCache;
  try {
    const r = await fetch("/api/v1/bootstrap");
    if (r.ok) {
      const data = await r.json();
      _wsTokenCache = data?.token || null;
    }
  } catch {}
  return _wsTokenCache;
}

function makeSessionId() {
  // Persist session ID across page navigation so chat can resume
  const stored = sessionStorage.getItem("jarvis-session-id");
  if (stored) return stored;
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  const hex = Array.from(buf, b => b.toString(16).padStart(2, "0")).join("");
  const id = `web_${Date.now()}_${hex}`;
  sessionStorage.setItem("jarvis-session-id", id);
  return id;
}

function loadPersistedMessages() {
  try {
    const raw = sessionStorage.getItem("jarvis-messages");
    if (raw) return JSON.parse(raw);
  } catch {}
  return [];
}

export function useJarvisChat() {
  const [messages, setMessages] = useState(loadPersistedMessages);
  const [streamText, setStreamText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [canvasHtml, setCanvasHtml] = useState("");
  const [canvasTitle, setCanvasTitle] = useState("");
  const [activeTool, setActiveTool] = useState(null);
  const [pendingApproval, setPendingApproval] = useState(null);
  const [statusText, setStatusText] = useState("");
  const [pipelineState, setPipelineState] = useState(null);

  const wsRef = useRef(null);
  const retriesRef = useRef(0);
  const heartbeatRef = useRef(null);
  const sessionIdRef = useRef(makeSessionId());
  const streamAccRef = useRef("");
  const reconnectTimerRef = useRef(null);

  const addMessage = useCallback((role, text, meta) => {
    setMessages(prev => [...prev, {
      id: `${Date.now()}-${crypto.getRandomValues(new Uint32Array(1))[0]}`,
      role,
      text,
      time: new Date(),
      ...meta,
    }]);
  }, []);

  const handleWsMessage = useCallback((evt) => {
    let data;
    try { data = JSON.parse(evt.data); } catch { return; }

    switch (data.type) {
      case "assistant_message":
        setIsStreaming(false);
        setStreamText("");
        streamAccRef.current = "";
        addMessage("assistant", data.text || data.content || "");
        // Mark pipeline as complete — stays visible (collapsed) until next msg
        setPipelineState((prev) => {
          if (!prev) return prev;
          const iters = [...prev.iterations];
          const cur = iters[iters.length - 1];
          if (cur) {
            const updated = { ...cur, phases: { ...cur.phases } };
            for (const p of ["plan", "gate", "execute", "replan"]) {
              if (updated.phases[p].status === "running") {
                updated.phases[p] = { ...updated.phases[p], status: "done", durationMs: Date.now() - (updated.phases[p].startMs || Date.now()) };
              } else if (updated.phases[p].status === "pending") {
                updated.phases[p] = { status: "skipped" };
              }
            }
            iters[iters.length - 1] = updated;
          }
          return { ...prev, active: false, iterations: iters };
        });
        break;

      case "stream_token":
        setIsStreaming(true);
        streamAccRef.current += (data.token || data.content || "");
        setStreamText(streamAccRef.current);
        // First stream token = LLM is responding → advance pipeline to replan/finishing
        setPipelineState((prev) => {
          if (!prev || !prev.active) return prev;
          const iters = [...prev.iterations];
          const cur = { ...iters[iters.length - 1], phases: { ...iters[iters.length - 1].phases } };
          const now = Date.now();
          // Advance any pending/running phases to done
          if (cur.phases.plan.status === "running") {
            cur.phases.plan = { status: "done", startMs: cur.phases.plan.startMs, durationMs: now - (cur.phases.plan.startMs || now) };
          }
          if (cur.phases.gate.status === "pending") {
            cur.phases.gate = { status: "done", durationMs: 0 };
          }
          if (cur.phases.execute.status === "pending" || cur.phases.execute.status === "running") {
            cur.phases.execute = { status: "done", startMs: cur.phases.execute.startMs || now, durationMs: now - (cur.phases.execute.startMs || now) };
          }
          if (cur.phases.replan.status === "pending") {
            cur.phases.replan = { status: "running", startMs: now };
          }
          iters[iters.length - 1] = cur;
          return { ...prev, iterations: iters };
        });
        break;

      case "stream_end":
        setIsStreaming(false);
        if (streamAccRef.current) {
          addMessage("assistant", streamAccRef.current);
        }
        setStreamText("");
        streamAccRef.current = "";
        // Mark pipeline as complete (same as assistant_message)
        setPipelineState((prev) => {
          if (!prev) return prev;
          const iters = [...prev.iterations];
          const cur = iters[iters.length - 1];
          if (cur) {
            const updated = { ...cur, phases: { ...cur.phases } };
            for (const p of ["plan", "gate", "execute", "replan"]) {
              if (updated.phases[p].status === "running") {
                updated.phases[p] = { ...updated.phases[p], status: "done", durationMs: Date.now() - (updated.phases[p].startMs || Date.now()) };
              } else if (updated.phases[p].status === "pending") {
                updated.phases[p] = { status: "skipped" };
              }
            }
            iters[iters.length - 1] = updated;
          }
          return { ...prev, active: false, iterations: iters };
        });
        break;

      case "tool_start":
        setActiveTool({ name: data.tool || data.name, args: data.args });
        // Track sub-tool in pipeline state
        setPipelineState((prev) => {
          if (!prev) return prev;
          const iters = [...prev.iterations];
          const cur = iters[iters.length - 1];
          if (cur) {
            cur.tools = [...(cur.tools || []), { name: data.tool || data.name, status: "running", startMs: Date.now() }];
            iters[iters.length - 1] = { ...cur };
          }
          return { ...prev, iterations: iters };
        });
        break;

      case "tool_result":
        setActiveTool(null);
        // Mark sub-tool as done in pipeline state
        setPipelineState((prev) => {
          if (!prev) return prev;
          const iters = [...prev.iterations];
          const cur = iters[iters.length - 1];
          if (cur && cur.tools) {
            const toolName = data.tool || data.name || "";
            const idx = cur.tools.findIndex((t) => t.name === toolName && t.status === "running");
            if (idx >= 0) {
              const updated = [...cur.tools];
              updated[idx] = { ...updated[idx], status: "done", durationMs: Date.now() - updated[idx].startMs };
              cur.tools = updated;
              iters[iters.length - 1] = { ...cur };
            }
          }
          return { ...prev, iterations: iters };
        });
        break;

      case "approval_request":
        setPendingApproval({
          id: data.id || data.request_id,
          tool: data.tool || data.name,
          reason: data.reason || "",
          params: data.params || data.args || {},
        });
        break;

      case "pipeline_event":
        setPipelineState((prev) => {
          const phase = data.phase;
          const status = data.status;
          const iteration = data.iteration || 1;
          const now = Date.now();

          // Initialize on first event
          if (!prev || phase === "iteration" && status === "start") {
            const existing = prev ? prev.iterations : [];
            return {
              active: true,
              iterations: [
                ...existing,
                {
                  number: iteration,
                  phases: {
                    plan: { status: "pending" },
                    gate: { status: "pending" },
                    execute: { status: "pending" },
                    replan: { status: "pending" },
                  },
                  tools: [],
                },
              ],
            };
          }

          // Complete event — mark inactive, skip remaining pending phases
          if (phase === "complete") {
            const iters = [...prev.iterations];
            const cur = iters[iters.length - 1];
            if (cur) {
              const updated = { ...cur, phases: { ...cur.phases } };
              for (const p of ["plan", "gate", "execute", "replan"]) {
                if (updated.phases[p] && updated.phases[p].status === "pending") {
                  updated.phases[p] = { status: "skipped" };
                }
              }
              iters[iters.length - 1] = updated;
            }
            return { ...prev, active: false, toolsUsed: data.tools_used, iterations: iters };
          }

          // Update phase status in current iteration
          const iters = [...prev.iterations];
          let cur = iters[iters.length - 1];
          if (!cur) return prev;
          cur = { ...cur, phases: { ...cur.phases } };

          if (status === "start") {
            cur.phases[phase] = { status: "running", startMs: now };
          } else if (status === "done") {
            const existing = cur.phases[phase] || {};
            const durationMs = existing.startMs ? now - existing.startMs : 0;
            cur.phases[phase] = {
              status: "done",
              startMs: existing.startMs,
              durationMs,
              ...(data.tools ? { tools: data.tools } : {}),
              ...(data.success !== undefined ? { success: data.success, failed: data.failed } : {}),
              ...(data.blocked !== undefined ? { blocked: data.blocked, allowed: data.allowed } : {}),
            };
          } else if (status === "error") {
            cur.phases[phase] = { ...cur.phases[phase], status: "error" };
          }

          iters[iters.length - 1] = cur;
          return { ...prev, iterations: iters };
        });
        break;

      case "canvas_push":
        setCanvasHtml(data.html || data.content || "");
        setCanvasTitle(data.title || "Canvas");
        break;

      case "canvas_reset":
        setCanvasHtml("");
        setCanvasTitle("");
        break;

      case "canvas_eval":
        // handled in ChatCanvas via postMessage
        break;

      case "transcription":
        // Backend transcribed audio — update the last user message
        setMessages(prev => {
          const updated = [...prev];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].role === "user" && updated[i].text === "[Voice message]") {
              updated[i] = { ...updated[i], text: data.text };
              break;
            }
          }
          return updated;
        });
        break;

      case "status_update":
        setStatusText(data.text || data.status || "");
        // Derive pipeline state from status_update events (fallback when
        // pipeline_event is not available — works with existing backend)
        setPipelineState((prev) => {
          const now = Date.now();
          const st = data.status || "";
          const txt = data.text || "";

          // Create initial state on first status_update
          if (!prev) {
            prev = {
              active: true,
              iterations: [{
                number: 1,
                phases: {
                  plan: { status: "pending" },
                  gate: { status: "pending" },
                  execute: { status: "pending" },
                  replan: { status: "pending" },
                },
                tools: [],
              }],
            };
          }
          if (!prev.active) return prev;

          const iters = [...prev.iterations];
          const cur = { ...iters[iters.length - 1], phases: { ...iters[iters.length - 1].phases } };
          iters[iters.length - 1] = cur;

          if (st === "thinking") {
            // Plan phase
            if (cur.phases.plan.status === "pending") {
              cur.phases.plan = { status: "running", startMs: now };
            }
          } else if (st === "executing") {
            // Mark plan as done, gate as done, execute as running
            if (cur.phases.plan.status === "running") {
              cur.phases.plan = { status: "done", startMs: cur.phases.plan.startMs, durationMs: now - (cur.phases.plan.startMs || now) };
            }
            if (cur.phases.gate.status === "pending") {
              cur.phases.gate = { status: "done", durationMs: 0 };
            }
            if (cur.phases.execute.status === "pending") {
              cur.phases.execute = { status: "running", startMs: now, toolText: txt };
            }
          } else if (st === "finishing") {
            // Mark execute as done, replan as running
            if (cur.phases.execute.status === "running") {
              cur.phases.execute = { status: "done", startMs: cur.phases.execute.startMs, durationMs: now - (cur.phases.execute.startMs || now) };
            }
            if (cur.phases.replan.status === "pending") {
              cur.phases.replan = { status: "running", startMs: now };
            }
          }

          return { ...prev, iterations: iters };
        });
        break;

      case "error":
        addMessage("system", data.message || data.error || "An error occurred.");
        setIsStreaming(false);
        setStreamText("");
        streamAccRef.current = "";
        break;

      case "pong":
        // heartbeat response, ignore
        break;

      default:
        break;
    }
  }, [addMessage]);

  const startHeartbeat = useCallback((ws) => {
    if (heartbeatRef.current) clearInterval(heartbeatRef.current);
    heartbeatRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, WS_HEARTBEAT_INTERVAL);
  }, []);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  // Use refs for stable connect/disconnect to avoid useEffect re-triggering
  const handleWsMsgRef = useRef(handleWsMessage);
  useEffect(() => { handleWsMsgRef.current = handleWsMessage; }, [handleWsMessage]);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws/${sessionIdRef.current}`;

    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.error("[WS] Failed to create WebSocket:", err);
      return;
    }
    wsRef.current = ws;

    ws.onopen = async () => {
      console.log("[WS] Connected to", url);
      // Send auth token as first message before any chat traffic
      const token = await getWsToken();
      if (token) {
        ws.send(JSON.stringify({ type: "auth", token }));
      }
      setIsConnected(true);
      retriesRef.current = 0; // reset retries on successful connection
      startHeartbeat(ws);
    };

    ws.onmessage = (evt) => handleWsMsgRef.current(evt);

    ws.onclose = (ev) => {
      setIsConnected(false);
      stopHeartbeat();
      wsRef.current = null;
      // 4001 = auth rejected — invalidate cached token so reconnect fetches a fresh one
      if (ev.code === 4001) {
        _wsTokenCache = null;
      }
      // Auto-reconnect with backoff
      if (retriesRef.current < WS_MAX_RETRIES) {
        retriesRef.current++;
        const delay = WS_RECONNECT_DELAY * Math.min(retriesRef.current, 3);
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${retriesRef.current}/${WS_MAX_RETRIES})`);
        reconnectTimerRef.current = setTimeout(connect, delay);
      } else {
        console.warn("[WS] Max retries reached. Click to reconnect or reload page.");
      }
    };

    ws.onerror = () => {
      // onclose will fire after onerror
    };
  }, [startHeartbeat, stopHeartbeat]);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    stopHeartbeat();
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent reconnect from cleanup
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, [stopHeartbeat]);

  // Persist messages to sessionStorage for cross-page-nav recovery
  useEffect(() => {
    try {
      // Keep last 100 messages to avoid storage bloat
      const toStore = messages.slice(-100);
      sessionStorage.setItem("jarvis-messages", JSON.stringify(toStore));
    } catch {}
  }, [messages]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sendMessage = useCallback((text) => {
    if (!text.trim()) return;
    // Start pipeline visualization immediately on send
    setPipelineState({
      active: true,
      startMs: Date.now(),
      iterations: [{
        number: 1,
        phases: {
          plan: { status: "running", startMs: Date.now() },
          gate: { status: "pending" },
          execute: { status: "pending" },
          replan: { status: "pending" },
        },
        tools: [],
      }],
    });
    addMessage("user", text.trim());
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "user_message",
        text: text.trim(),
        session_id: sessionIdRef.current,
      }));
    }
  }, [addMessage]);

  const sendFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      addMessage("user", `[File: ${file.name}]`);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: "user_message",
          text: `[File: ${file.name}]`,
          session_id: sessionIdRef.current,
          metadata: {
            file_name: file.name,
            file_type: file.type,
            file_base64: base64,
          },
        }));
      }
    };
    reader.readAsDataURL(file);
  }, [addMessage]);

  const sendVoice = useCallback((blob) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      addMessage("user", "[Voice message]");
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: "user_message",
          text: "[Voice message]",
          session_id: sessionIdRef.current,
          metadata: {
            audio_base64: base64,
            audio_type: blob.type || "audio/webm",
          },
        }));
      }
    };
    reader.readAsDataURL(blob);
  }, [addMessage]);

  const respondApproval = useCallback((id, approved) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "approval_response",
        id,
        approved,
        session_id: sessionIdRef.current,
      }));
    }
    setPendingApproval(null);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStreamText("");
    streamAccRef.current = "";
    setIsStreaming(false);
    setActiveTool(null);
    setPendingApproval(null);
    setCanvasHtml("");
    setCanvasTitle("");
    setStatusText("");
    setPipelineState(null);
    try { sessionStorage.removeItem("jarvis-messages"); } catch {}
  }, []);

  return {
    messages,
    streamText,
    isStreaming,
    isConnected,
    canvasHtml,
    canvasTitle,
    activeTool,
    pendingApproval,
    statusText,
    pipelineState,
    sendMessage,
    sendFile,
    sendVoice,
    respondApproval,
    clearMessages,
  };
}
