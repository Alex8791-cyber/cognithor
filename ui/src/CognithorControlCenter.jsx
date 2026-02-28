import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════════════
// Cognithor · Control Center v2 — UX-Rewrite mit allen 23 Fixes
// ═══════════════════════════════════════════════════════════════════════
// Fix #1:  Dirty-state tracking (hasChanges)
// Fix #2:  Parallel saves via Promise.all + per-section error tracking
// Fix #3:  Version field read-only
// Fix #4:  Structured deep-setter (no JSON.parse roundtrip)
// Fix #5:  Loading spinner until API responds
// Fix #6:  Generic updateAgent/updateBinding helpers
// Fix #7:  Input validation + red border + error messages
// Fix #8:  JSON editor with validation feedback
// Fix #9:  Reset-to-default per prompt textarea
// Fix #10: Cron human-readable preview
// Fix #11: Memory weight sum permanent display
// Fix #12: Auto-open cards when enabled toggle is activated
// Fix #13: Search/filter for providers
// Fix #14: Preset details expandable
// Fix #15: Export includes prompts
// Fix #16: Import button
// Fix #17: Keyboard nav hints in sidebar
// Fix #18: Safe-area aware save bar
// Fix #19: Discord channel ID as string
// Fix #20: Toast/Snackbar system
// Fix #21: Slider with manual number input
// Fix #22: Font-display swap (no render block)
// Fix #23: Ctrl+S keyboard shortcut
// ═══════════════════════════════════════════════════════════════════════

const API = "/api/v1";

// ── API Helper ─────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(`${API}${path}`, opts);
    if (!r.ok) return { error: `HTTP ${r.status}`, status: r.status };
    const text = await r.text();
    if (!text) return {};
    // Fix #19: Prevent precision loss for large integers (like Discord IDs)
    const safeText = text.replace(/:\s*([0-9]{16,})\b/g, ':"$1"');
    return JSON.parse(safeText);
  } catch (e) {
    console.error(`API ${method} ${path}:`, e);
    return { error: e.message };
  }
}

// ── Icons (inline SVG) ─────────────────────────────────────────────────
const I = {
  home: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M3 12l9-9 9 9M5 10v10a1 1 0 001 1h3a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1h3a1 1 0 001-1V10"/></svg>,
  llm: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>,
  model: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>,
  brain: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M9.5 2A5.5 5.5 0 005 7.5a5.5 5.5 0 001 3.1A5.5 5.5 0 005 14.5 5.5 5.5 0 009.5 20h1a1 1 0 001-1V3a1 1 0 00-1-1zM14.5 2A5.5 5.5 0 0120 7.5a5.5 5.5 0 01-1 3.1 5.5 5.5 0 011 3.9 5.5 5.5 0 01-5.5 5.5h-1a1 1 0 01-1-1V3a1 1 0 011-1z"/></svg>,
  mem: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><circle cx="6" cy="6" r="1"/><circle cx="6" cy="18" r="1"/></svg>,
  ch: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>,
  shield: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  web: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
  plug: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>,
  clock: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>,
  db: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
  file: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>,
  bot: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4M8 16h0M16 16h0"/></svg>,
  link: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>,
  restart: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/></svg>,
  save: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/></svg>,
  check: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>,
  plus: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>,
  trash: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>,
  eye: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>,
  eyeOff: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24M1 1l22 22"/></svg>,
  gear: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.32 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/></svg>,
  upload: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>,
  reset: <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M3 12a9 9 0 019-9 9.75 9.75 0 016.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 01-9 9 9.75 9.75 0 01-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>,
  search: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>,
  x: <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>,
  terminal: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>,
  play: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>,
  stop: <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16"></rect></svg>,
};

// ── Navigation ─────────────────────────────────────────────────────────
const PAGES = [
  { id: "general", label: "Allgemein", icon: I.home, key: "1" },
  { id: "providers", label: "LLM Provider", icon: I.llm, key: "2" },
  { id: "models", label: "Modelle", icon: I.model, key: "3" },
  { id: "planner", label: "PGE Trinity", icon: I.brain, key: "4" },
  { id: "memory", label: "Memory", icon: I.mem, key: "5" },
  { id: "channels", label: "Channels", icon: I.ch, key: "6" },
  { id: "security", label: "Sicherheit", icon: I.shield, key: "7" },
  { id: "web", label: "Web-Tools", icon: I.web, key: "8" },
  { id: "mcp", label: "MCP & A2A", icon: I.plug, key: "9" },
  { id: "cron", label: "Cron & Heartbeat", icon: I.clock, key: "0" },
  { id: "database", label: "Datenbank", icon: I.db, key: null },
  { id: "logging", label: "Logging", icon: I.terminal, key: null },
  { id: "prompts", label: "Prompts & Policies", icon: I.file, key: null },
  { id: "agents", label: "Agenten", icon: I.bot, key: null },
  { id: "bindings", label: "Bindings", icon: I.link, key: null },
  { id: "system", label: "System", icon: I.gear, key: null },
];

// ── Default Config ─────────────────────────────────────────────────────
const defaults = () => ({
  owner_name: "User", version: "0.22.0",
  operation_mode: "auto", llm_backend_type: "ollama",
  cost_tracking_enabled: true, daily_budget_usd: 0, monthly_budget_usd: 0,
  openai_api_key: "", openai_base_url: "https://api.openai.com/v1",
  anthropic_api_key: "", anthropic_max_tokens: 4096,
  gemini_api_key: "", groq_api_key: "", deepseek_api_key: "",
  mistral_api_key: "", together_api_key: "", openrouter_api_key: "",
  xai_api_key: "", cerebras_api_key: "", github_api_key: "",
  bedrock_api_key: "", huggingface_api_key: "", moonshot_api_key: "",
  vision_model: "openbmb/minicpm-v4.5", vision_model_detail: "qwen3-vl:32b",
  ollama: { base_url: "http://localhost:11434", timeout_seconds: 120, keep_alive: "30m" },
  models: {
    planner: { name: "qwen3:32b", context_window: 32768, vram_gb: 20, strengths: ["reasoning","planning","reflection","german"], speed: "medium" },
    executor: { name: "qwen3:8b", context_window: 32768, vram_gb: 6, strengths: ["tool-calling","simple-tasks"], speed: "fast" },
    coder: { name: "qwen2.5-coder:32b", context_window: 32768, vram_gb: 20, strengths: ["code-generation","debugging","testing"], speed: "medium" },
    embedding: { name: "nomic-embed-text", context_window: 8192, vram_gb: 0.5, strengths: ["semantic-search"], speed: "fast" },
  },
  model_overrides: { skill_models: {} },
  planner: { max_iterations: 10, escalation_after: 3, temperature: 0.7, response_token_budget: 3000 },
  gatekeeper: { policies_dir: "policies", default_risk_level: "yellow", max_blocked_retries: 3 },
  sandbox: { level: "process", timeout_seconds: 30, max_memory_mb: 512, max_cpu_seconds: 10, allowed_paths: ["~/.jarvis/workspace/","/tmp/jarvis/"], network_access: false, env_vars: {} },
  memory: { chunk_size_tokens: 400, chunk_overlap_tokens: 80, search_top_k: 6, weight_vector: 0.5, weight_bm25: 0.3, weight_graph: 0.2, recency_half_life_days: 30, compaction_threshold: 0.8, compaction_keep_last_n: 4, episodic_retention_days: 365, dynamic_weighting: false },
  channels: {
    cli_enabled: true, webui_enabled: false, webui_port: 8080,
    telegram_enabled: false, telegram_whitelist: [],
    slack_enabled: false, slack_default_channel: "",
    discord_enabled: false, discord_channel_id: 0,
    whatsapp_enabled: false, whatsapp_default_chat: "", whatsapp_phone_number_id: "", whatsapp_webhook_port: 8443, whatsapp_verify_token: "", whatsapp_allowed_numbers: [],
    signal_enabled: false, signal_default_user: "",
    matrix_enabled: false, matrix_homeserver: "", matrix_user_id: "",
    teams_enabled: false, teams_default_channel: "",
    imessage_enabled: false, imessage_device_id: "",
    google_chat_enabled: false, google_chat_credentials_path: "", google_chat_allowed_spaces: [],
    mattermost_enabled: false, mattermost_url: "", mattermost_token: "", mattermost_channel: "",
    feishu_enabled: false, feishu_app_id: "", feishu_app_secret: "",
    irc_enabled: false, irc_server: "", irc_port: 6667, irc_nick: "JarvisBot", irc_channels: [],
    twitch_enabled: false, twitch_token: "", twitch_channel: "", twitch_allowed_users: [],
    voice_enabled: false,
    voice_config: { tts_backend: "piper", elevenlabs_api_key: "", elevenlabs_voice_id: "hJAaR77ekN23CNyp0byH", elevenlabs_model: "eleven_multilingual_v2", wake_word_enabled: false, wake_word: "jarvis", wake_word_backend: "vosk", talk_mode_enabled: false, talk_mode_auto_listen: false },
  },
  security: { max_iterations: 10, allowed_paths: ["~/.jarvis/","/tmp/jarvis/"], blocked_commands: ["rm\\s+-rf\\s+/","mkfs\\b","dd\\s+if=/dev"], credential_patterns: ["sk-[a-zA-Z0-9]{20,}","token_[a-zA-Z0-9]+"] },
  web: { searxng_url: "", brave_api_key: "", duckduckgo_enabled: true },
  logging: { level: "INFO", json_logs: false, console: true },
  database: { backend: "sqlite", pg_host: "localhost", pg_port: 5432, pg_dbname: "jarvis", pg_user: "jarvis", pg_password: "", pg_pool_min: 2, pg_pool_max: 10 },
  dashboard: { enabled: false, port: 9090 },
  heartbeat: { enabled: false, interval_minutes: 30, checklist_file: "HEARTBEAT.md", channel: "cli", model: "qwen3:8b" },
  plugins: { skills_dir: "skills", auto_update: false },
});

const defaultPrompts = () => ({
  coreMd: "", plannerSystem: "", replanPrompt: "",
  escalationPrompt: "", policyYaml: "", heartbeatMd: "",
});

// ── Fix #10: Cron human-readable ────────────────────────────────────────
function cronToHuman(expr) {
  if (!expr || typeof expr !== "string") return "";
  const parts = expr.trim().split(/\s+/);
  if (parts.length < 5) return "Ungültiger Ausdruck";
  const [min, hour, dom, mon, dow] = parts;
  const dayNames = { 0: "So", 1: "Mo", 2: "Di", 3: "Mi", 4: "Do", 5: "Fr", 6: "Sa", 7: "So" };
  const monNames = { 1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez" };
  let time = "";
  if (hour !== "*" && min !== "*") time = `um ${hour.padStart(2, "0")}:${min.padStart(2, "0")}`;
  else if (hour !== "*") time = `zur Stunde ${hour}`;
  else time = "jede Minute";
  let days = "";
  if (dow !== "*") {
    const ranges = dow.split(",").map(d => {
      if (d.includes("-")) {
        const [a, b] = d.split("-");
        return `${dayNames[a] || a}–${dayNames[b] || b}`;
      }
      return dayNames[d] || d;
    });
    days = ranges.join(", ");
  }
  let months = "";
  if (mon !== "*") {
    months = mon.split(",").map(m => monNames[m] || m).join(", ");
  }
  let domStr = "";
  if (dom !== "*") domStr = `am ${dom}.`;
  let result = time;
  if (days) result += ` (${days})`;
  if (domStr) result += ` ${domStr}`;
  if (months) result += ` in ${months}`;
  return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Fix #20: Toast System
// ═══════════════════════════════════════════════════════════════════════
function ToastContainer({ toasts, onDismiss }) {
  return (
    <div className="cc-toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`cc-toast cc-toast-${t.type}`}>
          <span>{t.type === "error" ? "✖" : t.type === "success" ? "✔" : "⚠"}</span>
          <span className="cc-toast-msg">{t.message}</span>
          <button className="cc-toast-close" onClick={() => onDismiss(t.id)}>{I.x}</button>
        </div>
      ))}
    </div>
  );
}

function useToast() {
  const [toasts, setToasts] = useState([]);
  const idRef = useRef(0);
  const add = useCallback((message, type = "info") => {
    const id = ++idRef.current;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);
  const dismiss = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), []);
  return { toasts, add, dismiss };
}

// ═══════════════════════════════════════════════════════════════════════
// UI Components (improved)
// ═══════════════════════════════════════════════════════════════════════

function Toggle({ label, value, onChange, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-field-row" onClick={() => onChange(!value)} style={{ cursor: "pointer" }}>
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        <div className={`cc-toggle ${value ? "on" : ""}`}><div className="cc-toggle-dot" /></div>
      </div>
    </div>
  );
}

// Fix #7: validation prop for inputs
function TextInput({ label, value, onChange, desc, placeholder, type = "text", mono, error, disabled }) {
  const [show, setShow] = useState(false);
  const isSecret = type === "password";
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-input-wrap">
        <input
          className={`cc-input ${mono ? "mono" : ""} ${error ? "cc-error" : ""} ${disabled ? "cc-input-disabled" : ""}`}
          type={isSecret && !show ? "password" : "text"}
          value={value || ""}
          onChange={e => !disabled && onChange(e.target.value)}
          placeholder={placeholder || ""}
          readOnly={disabled}
          tabIndex={disabled ? -1 : 0}
        />
        {isSecret && (
          <button className="cc-eye-btn" onClick={() => setShow(!show)} type="button">{show ? I.eyeOff : I.eye}</button>
        )}
      </div>
      {error && <div className="cc-field-error">{error}</div>}
    </div>
  );
}

function NumberInput({ label, value, onChange, desc, min, max, step = 1, error }) {
  const localErr = (value !== undefined && value !== null) && ((min !== undefined && value < min) || (max !== undefined && value > max));
  const displayErr = error || (localErr ? `Wert muss zwischen ${min} und ${max} liegen.` : null);
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <input
        className={`cc-input ${displayErr ? "cc-error" : ""}`}
        type="number"
        value={value ?? ""}
        onChange={e => onChange(e.target.value === "" ? null : Number(e.target.value))}
        min={min} max={max} step={step}
      />
      {displayErr && <div className="cc-field-error">{displayErr}</div>}
    </div>
  );
}

// Fix #21: Slider with editable value
function SliderInput({ label, value, onChange, min = 0, max = 1, step = 0.01, desc }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const commit = () => {
    const n = parseFloat(draft);
    if (!isNaN(n) && n >= min && n <= max) onChange(n);
    setEditing(false);
  };
  return (
    <div className="cc-field">
      <div className="cc-field-row">
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        {editing ? (
          <input
            className="cc-slider-edit"
            autoFocus
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={e => { if (e.key === "Enter") commit(); if (e.key === "Escape") setEditing(false); }}
          />
        ) : (
          <span
            className="cc-slider-val"
            onClick={() => { setDraft(typeof value === "number" ? value.toFixed(2) : String(value)); setEditing(true); }}
            title="Klicken zum manuellen Eingeben"
          >
            {typeof value === "number" ? value.toFixed(2) : value}
          </span>
        )}
      </div>
      <input type="range" className="cc-slider" value={value ?? min} onChange={e => onChange(Number(e.target.value))} min={min} max={max} step={step} />
    </div>
  );
}

function SelectInput({ label, value, onChange, options, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <select className="cc-select" value={value || ""} onChange={e => onChange(e.target.value)}>
        {options.map(o => typeof o === "string" ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function ListInput({ label, value = [], onChange, desc, placeholder }) {
  const [draft, setDraft] = useState("");
  const add = () => { if (draft.trim()) { onChange([...value, draft.trim()]); setDraft(""); } };
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-list-items">
        {value.map((v, i) => (
          <div key={i} className="cc-list-item">
            <span className="mono">{v}</span>
            <button className="cc-btn-icon" onClick={() => onChange(value.filter((_, j) => j !== i))} type="button">{I.trash}</button>
          </div>
        ))}
      </div>
      <div className="cc-list-add">
        <input className="cc-input" value={draft} onChange={e => setDraft(e.target.value)} placeholder={placeholder || "Hinzufügen…"} onKeyDown={e => e.key === "Enter" && add()} />
        <button className="cc-btn-sm" onClick={add} type="button">{I.plus}</button>
      </div>
    </div>
  );
}

// Fix #8: JSON editor with validation + Fix #9: reset button
function TextArea({ label, value, onChange, desc, rows = 8, mono, error, onReset, resetLabel }) {
  return (
    <div className="cc-field">
      <div className="cc-field-row">
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        {onReset && (
          <button className="cc-btn-reset" onClick={onReset} title={resetLabel || "Auf Standard zurücksetzen"} type="button">
            {I.reset} <span>Reset</span>
          </button>
        )}
      </div>
      <textarea
        className={`cc-textarea ${mono ? "mono" : ""} ${error ? "cc-error" : ""}`}
        rows={rows}
        value={value || ""}
        onChange={e => onChange(e.target.value)}
      />
      {error && <div className="cc-field-error">{error}</div>}
    </div>
  );
}

// Fix #8: Dedicated JSON textarea with live validation
// B7: Prevents cursor-jump by not syncing raw from parent during active editing
function JsonEditor({ label, value, onChange, desc, rows = 6, onValidationError }) {
  const [raw, setRaw] = useState(() => typeof value === "string" ? value : JSON.stringify(value || {}, null, 2));
  const [err, setErr] = useState(null);
  const editingRef = useRef(false);
  const onValidationErrorRef = useRef(onValidationError);
  
  useEffect(() => {
    onValidationErrorRef.current = onValidationError;
  }, [onValidationError]);

  useEffect(() => {
    // Only sync from parent when NOT actively editing (e.g. external reset)
    if (!editingRef.current) {
      setRaw(typeof value === "string" ? value : JSON.stringify(value || {}, null, 2));
      setErr(null);
      if (onValidationErrorRef.current) onValidationErrorRef.current(null);
    }
  }, [value]);

  const handleChange = (txt) => {
    editingRef.current = true;
    setRaw(txt);
    try {
      const parsed = JSON.parse(txt);
      setErr(null);
      if (onValidationErrorRef.current) onValidationErrorRef.current(null);
      onChange(parsed);
    } catch (e) {
      const errorMsg = `JSON-Fehler: ${e.message.replace(/^JSON\.parse: /, "")}`;
      setErr(errorMsg);
      if (onValidationErrorRef.current) onValidationErrorRef.current(errorMsg);
    }
    // Reset editing flag after a short debounce
    clearTimeout(editingRef._timer);
    editingRef._timer = setTimeout(() => { editingRef.current = false; }, 500);
  };
  return (
    <TextArea
      label={label}
      value={raw}
      onChange={handleChange}
      desc={desc}
      rows={rows}
      mono
      error={err}
    />
  );
}

// Fix #3: Read-only info display
function ReadOnly({ label, value, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-readonly">{value || "—"}</div>
    </div>
  );
}

// Fix #12: Card that can be externally forced open
function Card({ title, children, open: initOpen = true, badge, forceOpen }) {
  const [open, setOpen] = useState(initOpen);
  const prevForce = useRef(forceOpen);
  useEffect(() => {
    if (forceOpen && !prevForce.current) setOpen(true);
    prevForce.current = forceOpen;
  }, [forceOpen]);
  return (
    <div className="cc-card">
      <div className="cc-card-head" onClick={() => setOpen(!open)}>
        <span className="cc-card-title">{title}</span>
        <div className="cc-card-right">
          {badge && <span className={`cc-badge ${badge}`}>{badge}</span>}
          <span className={`cc-chevron ${open ? "open" : ""}`}>▾</span>
        </div>
      </div>
      {open && <div className="cc-card-body">{children}</div>}
    </div>
  );
}

function Section({ title, desc }) {
  return (
    <div className="cc-section-head">
      <h2 className="cc-section-title">{title}</h2>
      {desc && <p className="cc-section-desc">{desc}</p>}
    </div>
  );
}

// Fix #5: Loading spinner
function Spinner() {
  return (
    <div className="cc-spinner-wrap">
      <div className="cc-spinner" />
      <span className="cc-spinner-text">Konfiguration wird geladen…</span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Pages
// ═══════════════════════════════════════════════════════════════════════

// ── General ────────────────────────────────────────────────────────────
function GeneralPage({ cfg, set }) {
  return (<>
    <Section title="Allgemein" desc="Identität, Betriebsmodus, Kosten und Dashboard" />
    <Card title="Identität">
      <TextInput label="Besitzer-Name" value={cfg.owner_name} onChange={v => set("owner_name", v)} desc="Wird in Prompts und CORE.md verwendet" error={!cfg.owner_name ? "Pflichtfeld" : null} />
      <SelectInput label="Betriebsmodus" value={cfg.operation_mode} onChange={v => set("operation_mode", v)} options={[{value:"auto",label:"Auto-Detect"},{value:"offline",label:"Offline (nur lokal)"},{value:"online",label:"Online (Cloud-APIs)"},{value:"hybrid",label:"Hybrid"}]} desc="auto erkennt den Modus aus vorhandenen API-Keys" />
      <ReadOnly label="Version" value={cfg.version} desc="Systemversion (nur lesbar)" />
    </Card>
    <Card title="Kosten-Tracking">
      <Toggle label="Kosten-Tracking aktiviert" value={cfg.cost_tracking_enabled} onChange={v => set("cost_tracking_enabled", v)} />
      <NumberInput label="Tageslimit (USD)" value={cfg.daily_budget_usd} onChange={v => set("daily_budget_usd", v)} min={0} step={0.5} desc="0 = kein Limit" />
      <NumberInput label="Monatslimit (USD)" value={cfg.monthly_budget_usd} onChange={v => set("monthly_budget_usd", v)} min={0} step={1} desc="0 = kein Limit" />
    </Card>
    <Card title="Dashboard">
      <Toggle label="Dashboard aktiviert" value={cfg.dashboard?.enabled} onChange={v => set("dashboard.enabled", v)} desc="Web-basiertes Monitoring-Dashboard" />
      <NumberInput label="Port" value={cfg.dashboard?.port} onChange={v => set("dashboard.port", v)} min={1024} max={65535} />
    </Card>
  </>);
}

// ── Providers ──────────────────────────────────────────────────────────
const PROVIDERS = [
  { id: "ollama", name: "Ollama (Lokal)", keyField: null },
  { id: "openai", name: "OpenAI", keyField: "openai_api_key", extra: ["openai_base_url"] },
  { id: "anthropic", name: "Anthropic Claude", keyField: "anthropic_api_key", extra: ["anthropic_max_tokens"] },
  { id: "gemini", name: "Google Gemini", keyField: "gemini_api_key" },
  { id: "groq", name: "Groq", keyField: "groq_api_key" },
  { id: "deepseek", name: "DeepSeek", keyField: "deepseek_api_key" },
  { id: "mistral", name: "Mistral AI", keyField: "mistral_api_key" },
  { id: "together", name: "Together AI", keyField: "together_api_key" },
  { id: "openrouter", name: "OpenRouter", keyField: "openrouter_api_key" },
  { id: "xai", name: "xAI (Grok)", keyField: "xai_api_key" },
  { id: "cerebras", name: "Cerebras", keyField: "cerebras_api_key" },
  { id: "github", name: "GitHub Models", keyField: "github_api_key" },
  { id: "bedrock", name: "AWS Bedrock", keyField: "bedrock_api_key" },
  { id: "huggingface", name: "Hugging Face", keyField: "huggingface_api_key" },
  { id: "moonshot", name: "Moonshot/Kimi", keyField: "moonshot_api_key" },
];

// Fix #13: Search/filter for providers
function ProvidersPage({ cfg, set }) {
  const [filter, setFilter] = useState("");
  const filtered = PROVIDERS.filter(p =>
    !filter || p.name.toLowerCase().includes(filter.toLowerCase()) || p.id.includes(filter.toLowerCase())
  );
  return (<>
    <Section title="LLM Provider" desc="15 Backends — API-Keys, Endpunkte, Backend-Auswahl" />
    <Card title="Aktives Backend">
      <SelectInput label="LLM Backend" value={cfg.llm_backend_type} onChange={v => set("llm_backend_type", v)} options={PROVIDERS.map(p => ({ value: p.id, label: p.name }))} desc="Alle Modelle nutzen dieses Backend" />
    </Card>
    <div className="cc-search-bar">
      <span className="cc-search-icon">{I.search}</span>
      <input className="cc-search-input" placeholder="Provider filtern…" value={filter} onChange={e => setFilter(e.target.value)} />
      {filter && <button className="cc-search-clear" onClick={() => setFilter("")}>{I.x}</button>}
    </div>
    {filtered.map(p => {
      if (p.id === "ollama") return (
        <Card key="ollama" title="Ollama (Lokal)" open={cfg.llm_backend_type === "ollama"}>
          <TextInput label="Base URL" value={cfg.ollama?.base_url} onChange={v => set("ollama.base_url", v)} mono />
          <NumberInput label="Timeout (Sek.)" value={cfg.ollama?.timeout_seconds} onChange={v => set("ollama.timeout_seconds", v)} min={10} max={600} />
          <TextInput label="Keep-Alive" value={cfg.ollama?.keep_alive} onChange={v => set("ollama.keep_alive", v)} desc="Wie lange Modelle im VRAM bleiben (z.B. 30m)" />
        </Card>
      );
      return (
        <Card key={p.id} title={p.name} open={cfg.llm_backend_type === p.id}>
          <TextInput label="API Key" value={cfg[p.keyField]} onChange={v => set(p.keyField, v)} type="password" placeholder={`${p.id}-api-key...`} />
          {p.extra?.includes("openai_base_url") && <TextInput label="Base URL" value={cfg.openai_base_url} onChange={v => set("openai_base_url", v)} mono desc="Auch für Together, Groq, vLLM, LM Studio" />}
          {p.extra?.includes("anthropic_max_tokens") && <NumberInput label="Max Output Tokens" value={cfg.anthropic_max_tokens} onChange={v => set("anthropic_max_tokens", v)} min={256} max={32768} />}
        </Card>
      );
    })}
    {filtered.length === 0 && <div className="cc-empty">Kein Provider gefunden für „{filter}"</div>}
  </>);
}

// ── Models ─────────────────────────────────────────────────────────────
const ROLES = ["planner","executor","coder","embedding"];
const SPEEDS = ["slow","medium","fast"];

function ModelsPage({ cfg, set, setValidationErrors }) {
  return (<>
    <Section title="Modelle" desc="Modell-Zuordnung pro Rolle, Vision-Modelle, Skill-Overrides" />
    {ROLES.map(r => (
      <Card key={r} title={`${r.charAt(0).toUpperCase() + r.slice(1)}-Modell`}>
        <TextInput label="Modell-Name" value={cfg.models?.[r]?.name} onChange={v => set(`models.${r}.name`, v)} mono />
        <NumberInput label="Context Window" value={cfg.models?.[r]?.context_window} onChange={v => set(`models.${r}.context_window`, v)} min={1024} max={131072} step={1024} />
        <NumberInput label="VRAM (GB)" value={cfg.models?.[r]?.vram_gb} onChange={v => set(`models.${r}.vram_gb`, v)} min={0} max={80} step={0.5} />
        <SliderInput label="Temperature" value={cfg.models?.[r]?.temperature} onChange={v => set(`models.${r}.temperature`, v)} min={0} max={2} step={0.05} desc="Kreativität (0=deterministisch, 2=wild)" />
        <SliderInput label="Top P" value={cfg.models?.[r]?.top_p} onChange={v => set(`models.${r}.top_p`, v)} min={0} max={1} step={0.05} desc="Nucleus Sampling" />
        <SelectInput label="Speed" value={cfg.models?.[r]?.speed} onChange={v => set(`models.${r}.speed`, v)} options={SPEEDS} />
        <ListInput label="Stärken" value={cfg.models?.[r]?.strengths} onChange={v => set(`models.${r}.strengths`, v)} placeholder="z.B. reasoning" />
      </Card>
    ))}
    <Card title="Vision-Modelle">
      <TextInput label="Standard Vision" value={cfg.vision_model} onChange={v => set("vision_model", v)} mono desc="Schnelles Vision-Modell" />
      <TextInput label="Detail Vision" value={cfg.vision_model_detail} onChange={v => set("vision_model_detail", v)} mono desc="Höchste Qualität" />
    </Card>
    <Card title="Skill-Overrides" open={false}>
      <div className="cc-desc" style={{marginBottom:8}}>Pro-Skill Modell-Zuordnung (JSON Key-Value)</div>
      <JsonEditor 
        label="skill_models (JSON)" 
        value={cfg.model_overrides?.skill_models || {}} 
        onChange={v => set("model_overrides.skill_models", v)} 
        onValidationError={err => setValidationErrors(prev => ({ ...prev, skill_models: err }))}
        rows={4} 
      />
    </Card>
  </>);
}

// ── PGE Trinity ────────────────────────────────────────────────────────
function PlannerPage({ cfg, set, setValidationErrors }) {
  return (<>
    <Section title="PGE Trinity" desc="Planner, Gatekeeper, Executor — Kern-Engine" />
    <Card title="Planner (Denker)">
      <NumberInput label="Max Iterationen" value={cfg.planner?.max_iterations} onChange={v => set("planner.max_iterations", v)} min={1} max={50} desc="Max Agent-Loop-Zyklen pro Anfrage" />
      <NumberInput label="Eskalation nach" value={cfg.planner?.escalation_after} onChange={v => set("planner.escalation_after", v)} min={1} max={10} desc="Nach X Schritten den User informieren" />
      <SliderInput label="Temperature" value={cfg.planner?.temperature} onChange={v => set("planner.temperature", v)} min={0} max={2} step={0.05} desc="Kreativität (0=deterministisch, 2=wild)" />
      <NumberInput label="Response Token Budget" value={cfg.planner?.response_token_budget} onChange={v => set("planner.response_token_budget", v)} min={500} max={8000} />
    </Card>
    <Card title="Gatekeeper (Wächter)">
      <TextInput label="Policies-Verzeichnis" value={cfg.gatekeeper?.policies_dir} onChange={v => set("gatekeeper.policies_dir", v)} mono desc="Relativ zu jarvis_home" />
      <SelectInput label="Standard-Risikostufe" value={cfg.gatekeeper?.default_risk_level} onChange={v => set("gatekeeper.default_risk_level", v)} options={[{value:"green",label:"🟢 Green"},{value:"yellow",label:"🟡 Yellow"},{value:"orange",label:"🟠 Orange"},{value:"red",label:"🔴 Red"}]} />
      <NumberInput label="Max. blockierte Wiederholungen" value={cfg.gatekeeper?.max_blocked_retries} onChange={v => set("gatekeeper.max_blocked_retries", v)} min={1} max={10} />
    </Card>
    <Card title="Sandbox (Executor)">
      <SelectInput label="Sandbox Level" value={cfg.sandbox?.level} onChange={v => set("sandbox.level", v)} options={["process","namespace","container","jobobject"]} desc="Isolationsgrad der Code-Ausführung" />
      <NumberInput label="Timeout (Sek.)" value={cfg.sandbox?.timeout_seconds} onChange={v => set("sandbox.timeout_seconds", v)} min={1} max={600} />
      <NumberInput label="Max Memory (MB)" value={cfg.sandbox?.max_memory_mb} onChange={v => set("sandbox.max_memory_mb", v)} min={64} max={8192} />
      <NumberInput label="Max CPU (Sek.)" value={cfg.sandbox?.max_cpu_seconds} onChange={v => set("sandbox.max_cpu_seconds", v)} min={1} max={300} />
      <Toggle label="Netzwerkzugriff" value={cfg.sandbox?.network_access} onChange={v => set("sandbox.network_access", v)} />
      <ListInput label="Erlaubte Pfade" value={cfg.sandbox?.allowed_paths} onChange={v => set("sandbox.allowed_paths", v)} placeholder="/pfad/..." />
      <JsonEditor 
        label="Umgebungsvariablen (JSON)" 
        value={cfg.sandbox?.env_vars || {}} 
        onChange={v => set("sandbox.env_vars", v)} 
        onValidationError={err => setValidationErrors(prev => ({ ...prev, sandbox_env_vars: err }))}
        rows={4} 
      />
    </Card>
  </>);
}

// ── Memory ─────────────────────────────────────────────────────────────
// Fix #11: Permanent weight sum display with color feedback
function MemoryPage({ cfg, set }) {
  const m = cfg.memory || {};
  const sum = (m.weight_vector||0)+(m.weight_bm25||0)+(m.weight_graph||0);
  const sumColor = Math.abs(sum - 1.0) < 0.01 ? "var(--success)" : sum > 1.01 ? "var(--danger)" : "var(--warn)";
  return (<>
    <Section title="Memory System" desc="5-Tier Memory: Indexierung, Hybrid-Suche, Recency, Compaction" />
    <Card title="Indexierung">
      <NumberInput label="Chunk Size (Tokens)" value={m.chunk_size_tokens} onChange={v => set("memory.chunk_size_tokens", v)} min={100} max={2000} />
      <NumberInput label="Chunk Overlap" value={m.chunk_overlap_tokens} onChange={v => set("memory.chunk_overlap_tokens", v)} min={0} max={500} />
      <NumberInput label="Search Top-K" value={m.search_top_k} onChange={v => set("memory.search_top_k", v)} min={1} max={20} />
    </Card>
    <Card title="Hybrid-Suche Gewichtung">
      <div className="cc-weight-sum" style={{ borderColor: sumColor }}>
        <span>Summe:</span>
        <span className="cc-weight-sum-val" style={{ color: sumColor }}>{sum.toFixed(2)}</span>
        <span className="cc-weight-sum-hint">{Math.abs(sum - 1.0) < 0.01 ? "✓ Perfekt" : sum > 1.01 ? "⚠ Wird normalisiert" : "⚠ Unter 1.0"}</span>
      </div>
      <SliderInput label="Vector (Embedding)" value={m.weight_vector} onChange={v => set("memory.weight_vector", v)} />
      <SliderInput label="BM25 (Keyword)" value={m.weight_bm25} onChange={v => set("memory.weight_bm25", v)} />
      <SliderInput label="Graph (Wissen)" value={m.weight_graph} onChange={v => set("memory.weight_graph", v)} />
      <Toggle label="Dynamische Gewichtung" value={m.dynamic_weighting} onChange={v => set("memory.dynamic_weighting", v)} desc="Gewichte automatisch je nach Anfrage anpassen" />
    </Card>
    <Card title="Recency & Compaction">
      <NumberInput label="Recency Halbwertszeit (Tage)" value={m.recency_half_life_days} onChange={v => set("memory.recency_half_life_days", v)} min={1} max={365} />
      <SliderInput label="Compaction Schwelle" value={m.compaction_threshold} onChange={v => set("memory.compaction_threshold", v)} min={0.5} max={0.95} />
      <NumberInput label="Letzte N behalten" value={m.compaction_keep_last_n} onChange={v => set("memory.compaction_keep_last_n", v)} min={2} max={20} />
      <NumberInput label="Episodic Retention (Tage)" value={m.episodic_retention_days} onChange={v => set("memory.episodic_retention_days", v)} min={1} max={3650} desc="Wie viele Tage an Tageslogs behalten" />
    </Card>
  </>);
}

// ── Channels ───────────────────────────────────────────────────────────
const CHANNEL_DEFS = [
  { id: "cli", label: "CLI (Terminal)", fields: [] },
  { id: "webui", label: "Web UI", fields: [{ k: "webui_port", l: "Port", t: "num", min: 1024, max: 65535 }] },
  { id: "telegram", label: "Telegram", fields: [{ k: "telegram_whitelist", l: "Whitelist (User-IDs)", t: "list" }] },
  { id: "slack", label: "Slack", fields: [{ k: "slack_default_channel", l: "Default Channel", t: "text" }] },
  { id: "discord", label: "Discord", fields: [{ k: "discord_channel_id", l: "Channel ID", t: "num" }] },
  { id: "whatsapp", label: "WhatsApp", fields: [
    { k: "whatsapp_default_chat", l: "Default Chat", t: "text" },
    { k: "whatsapp_phone_number_id", l: "Phone Number ID", t: "text" },
    { k: "whatsapp_webhook_port", l: "Webhook Port", t: "num", min: 1024, max: 65535 },
    { k: "whatsapp_verify_token", l: "Verify Token", t: "secret" },
    { k: "whatsapp_allowed_numbers", l: "Erlaubte Nummern", t: "list" },
  ]},
  { id: "signal", label: "Signal", fields: [{ k: "signal_default_user", l: "Default User", t: "text" }] },
  { id: "matrix", label: "Matrix", fields: [{ k: "matrix_homeserver", l: "Homeserver URL", t: "text" }, { k: "matrix_user_id", l: "User ID", t: "text" }] },
  { id: "teams", label: "Microsoft Teams", fields: [{ k: "teams_default_channel", l: "Default Channel", t: "text" }] },
  { id: "imessage", label: "iMessage", fields: [{ k: "imessage_device_id", l: "Device ID", t: "text" }] },
  { id: "google_chat", label: "Google Chat", fields: [
    { k: "google_chat_credentials_path", l: "Credentials Pfad", t: "text" },
    { k: "google_chat_allowed_spaces", l: "Erlaubte Spaces", t: "list" },
  ]},
  { id: "mattermost", label: "Mattermost", fields: [
    { k: "mattermost_url", l: "Server URL", t: "text" },
    { k: "mattermost_token", l: "Token", t: "secret" },
    { k: "mattermost_channel", l: "Channel", t: "text" },
  ]},
  { id: "feishu", label: "Feishu/Lark", fields: [{ k: "feishu_app_id", l: "App ID", t: "text" }, { k: "feishu_app_secret", l: "App Secret", t: "secret" }] },
  { id: "irc", label: "IRC", fields: [
    { k: "irc_server", l: "Server", t: "text" },
    { k: "irc_port", l: "Port", t: "num", min: 1, max: 65535 },
    { k: "irc_nick", l: "Nickname", t: "text" },
    { k: "irc_channels", l: "Channels", t: "list" },
  ]},
  { id: "twitch", label: "Twitch", fields: [
    { k: "twitch_token", l: "Token", t: "secret" },
    { k: "twitch_channel", l: "Channel", t: "text" },
    { k: "twitch_allowed_users", l: "Erlaubte User", t: "list" },
  ]},
  { id: "voice", label: "Voice (TTS/STT)", fields: [] },
];

// Fix #12: channels auto-open when enabled
function ChannelsPage({ cfg, set }) {
  const ch = cfg.channels || {};
  return (<>
    <Section title="Channels" desc="17 Kommunikationskanäle — aktivieren, konfigurieren" />
    {CHANNEL_DEFS.map(cd => (
      <Card key={cd.id} title={cd.label} open={!!ch[`${cd.id}_enabled`]} forceOpen={!!ch[`${cd.id}_enabled`]}>
        <Toggle label={`${cd.label} aktivieren`} value={ch[`${cd.id}_enabled`]} onChange={v => set(`channels.${cd.id}_enabled`, v)} />
        {cd.fields.map(f => {
          if (f.t === "text") return <TextInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} />;
          if (f.t === "secret") return <TextInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} type="password" />;
          if (f.t === "num") return <NumberInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} min={f.min} max={f.max} />;
          if (f.t === "list") return <ListInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} />;
          return null;
        })}
      </Card>
    ))}
    <Card title="Voice-Konfiguration" open={!!ch.voice_enabled} forceOpen={!!ch.voice_enabled}>
      <SelectInput label="TTS Backend" value={ch.voice_config?.tts_backend} onChange={v => set("channels.voice_config.tts_backend", v)} options={["piper","espeak","elevenlabs"]} />
      <TextInput label="ElevenLabs API Key" value={ch.voice_config?.elevenlabs_api_key} onChange={v => set("channels.voice_config.elevenlabs_api_key", v)} type="password" />
      <TextInput label="ElevenLabs Voice ID" value={ch.voice_config?.elevenlabs_voice_id} onChange={v => set("channels.voice_config.elevenlabs_voice_id", v)} mono />
      <TextInput label="ElevenLabs Model" value={ch.voice_config?.elevenlabs_model} onChange={v => set("channels.voice_config.elevenlabs_model", v)} mono />
      <Toggle label="Wake-Word aktiviert" value={ch.voice_config?.wake_word_enabled} onChange={v => set("channels.voice_config.wake_word_enabled", v)} />
      <TextInput label="Wake Word" value={ch.voice_config?.wake_word} onChange={v => set("channels.voice_config.wake_word", v)} />
      <SelectInput label="Wake-Word Backend" value={ch.voice_config?.wake_word_backend} onChange={v => set("channels.voice_config.wake_word_backend", v)} options={["vosk","porcupine"]} />
      <Toggle label="Talk-Mode" value={ch.voice_config?.talk_mode_enabled} onChange={v => set("channels.voice_config.talk_mode_enabled", v)} />
      <Toggle label="Auto-Listen" value={ch.voice_config?.talk_mode_auto_listen} onChange={v => set("channels.voice_config.talk_mode_auto_listen", v)} />
    </Card>
  </>);
}

// ── Security ───────────────────────────────────────────────────────────
function SecurityPage({ cfg, set }) {
  const s = cfg.security || {};
  return (<>
    <Section title="Sicherheit" desc="Iterations-Limits, Pfade, blockierte Befehle, Credential-Patterns" />
    <Card title="Gatekeeper Limits">
      <NumberInput label="Max Iterationen" value={s.max_iterations} onChange={v => set("security.max_iterations", v)} min={1} max={50} desc="Endlosschleifen-Schutz" />
    </Card>
    <Card title="Gatekeeper Dateisystem">
      <ListInput label="Erlaubte Pfade" value={s.allowed_paths} onChange={v => set("security.allowed_paths", v)} placeholder="~/.jarvis/" />
    </Card>
    <Card title="Gatekeeper Blockierte Befehle (Regex)">
      <ListInput label="Patterns" value={s.blocked_commands} onChange={v => set("security.blocked_commands", v)} placeholder="rm\\s+-rf\\s+/" />
    </Card>
    <Card title="Gatekeeper Credential-Erkennung (Regex)">
      <ListInput label="Patterns" value={s.credential_patterns} onChange={v => set("security.credential_patterns", v)} placeholder="sk-[a-zA-Z0-9]{20,}" />
    </Card>
  </>);
}

// ── Web Tools ──────────────────────────────────────────────────────────
function WebPage({ cfg, set }) {
  const w = cfg.web || {};
  return (<>
    <Section title="Web-Tools" desc="Such-Backends: SearXNG → Brave → DuckDuckGo" />
    <Card title="SearXNG (Self-hosted)">
      <TextInput label="SearXNG URL" value={w.searxng_url} onChange={v => set("web.searxng_url", v)} mono placeholder="http://localhost:8888" desc="Höchste Priorität" />
    </Card>
    <Card title="Brave Search">
      <TextInput label="API Key" value={w.brave_api_key} onChange={v => set("web.brave_api_key", v)} type="password" desc="2000 Anfragen/Monat kostenlos" />
    </Card>
    <Card title="DuckDuckGo">
      <Toggle label="DuckDuckGo aktiviert" value={w.duckduckgo_enabled} onChange={v => set("web.duckduckgo_enabled", v)} desc="Kostenloser Fallback" />
    </Card>
  </>);
}

// ── MCP & A2A ──────────────────────────────────────────────────────────
function McpPage({ cfg, set, mcpServers, setMcpServers, a2a, setA2a, setValidationErrors }) {
  const extServers = mcpServers.external_servers || {};
  const extServerNames = Object.keys(extServers);

  const addExtServer = () => {
    const name = `server_${extServerNames.length + 1}`;
    setMcpServers({
      ...mcpServers,
      external_servers: {
        ...extServers,
        [name]: { command: "", args: [], env: {}, disabled: false, always_allow: [] }
      }
    });
  };

  const updateExtServer = (oldName, newName, data) => {
    const newExt = { ...extServers };
    if (oldName !== newName) {
      delete newExt[oldName];
    }
    newExt[newName] = data;
    setMcpServers({ ...mcpServers, external_servers: newExt });
  };

  const removeExtServer = (name) => {
    const newExt = { ...extServers };
    delete newExt[name];
    setMcpServers({ ...mcpServers, external_servers: newExt });
  };

  return (<>
    <Section title="MCP & A2A" desc="Model Context Protocol Server + Agent-zu-Agent Kommunikation" />
    <Card title="MCP Server-Modus">
      <SelectInput label="Modus" value={mcpServers.mode || "disabled"} onChange={v => setMcpServers({...mcpServers, mode: v})} options={["disabled","stdio","http","both"]} desc="Jarvis als MCP-Server exponieren" />
      {mcpServers.mode !== "disabled" && <>
        <TextInput label="HTTP Host" value={mcpServers.http_host || "127.0.0.1"} onChange={v => setMcpServers({...mcpServers, http_host: v})} mono />
        <NumberInput label="HTTP Port" value={mcpServers.http_port || 3001} onChange={v => setMcpServers({...mcpServers, http_port: v})} min={1024} max={65535} />
        <TextInput label="Server-Name" value={mcpServers.server_name || "jarvis"} onChange={v => setMcpServers({...mcpServers, server_name: v})} />
        <Toggle label="Auth erforderlich" value={mcpServers.require_auth} onChange={v => setMcpServers({...mcpServers, require_auth: v})} />
        <TextInput label="Auth Token" value={mcpServers.auth_token || ""} onChange={v => setMcpServers({...mcpServers, auth_token: v})} type="password" />
        <Toggle label="Tools exponieren" value={mcpServers.expose_tools !== false} onChange={v => setMcpServers({...mcpServers, expose_tools: v})} />
        <Toggle label="Resources exponieren" value={mcpServers.expose_resources !== false} onChange={v => setMcpServers({...mcpServers, expose_resources: v})} />
        <Toggle label="Prompts exponieren" value={mcpServers.expose_prompts !== false} onChange={v => setMcpServers({...mcpServers, expose_prompts: v})} />
        <Toggle label="Sampling aktiviert" value={mcpServers.enable_sampling} onChange={v => setMcpServers({...mcpServers, enable_sampling: v})} />
      </>}
    </Card>
    <Card title="Externe MCP Server">
      <div className="cc-desc" style={{marginBottom:16}}>Verbundene externe MCP-Server (Tools & Resources)</div>
      {extServerNames.map(name => {
        const srv = extServers[name];
        return (
          <div key={name} style={{ border: "1px solid var(--border)", padding: 12, borderRadius: 6, marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <div style={{ flex: 1 }}>
                <TextInput label="Server Name" value={name} onChange={v => updateExtServer(name, v, srv)} mono />
              </div>
              <button className="cc-btn cc-btn-danger" onClick={() => removeExtServer(name)} style={{ marginTop: 22 }}>Löschen</button>
            </div>
            <TextInput label="Command" value={srv.command || ""} onChange={v => updateExtServer(name, name, { ...srv, command: v })} mono placeholder="z.B. npx, uvx, python" />
            <ListInput label="Arguments" value={srv.args || []} onChange={v => updateExtServer(name, name, { ...srv, args: v })} placeholder="z.B. -y, @modelcontextprotocol/server-sqlite, --db, test.db" />
            <JsonEditor 
              label="Environment Variables (JSON)" 
              value={srv.env || {}} 
              onChange={v => updateExtServer(name, name, { ...srv, env: v })} 
              onValidationError={err => setValidationErrors(prev => ({ ...prev, [`mcp_env_${name}`]: err }))}
              rows={2} 
            />
            <ListInput label="Always Allow (Tools)" value={srv.always_allow || []} onChange={v => updateExtServer(name, name, { ...srv, always_allow: v })} placeholder="z.B. read_query" />
            <Toggle label="Deaktiviert" value={srv.disabled || false} onChange={v => updateExtServer(name, name, { ...srv, disabled: v })} />
          </div>
        );
      })}
      <button className="cc-btn" onClick={addExtServer}>+ Server hinzufügen</button>
    </Card>
    <Card title="A2A Protokoll">
      <Toggle label="A2A aktiviert" value={a2a.enabled} onChange={v => setA2a({...a2a, enabled: v})} desc="Agent-zu-Agent Kommunikation (Linux Foundation A2A RC v1.0)" />
      {a2a.enabled && <>
        <TextInput label="Host" value={a2a.host || "127.0.0.1"} onChange={v => setA2a({...a2a, host: v})} mono />
        <NumberInput label="Port" value={a2a.port || 3002} onChange={v => setA2a({...a2a, port: v})} min={1024} max={65535} />
        <TextInput label="Agent Name" value={a2a.agent_name || "Jarvis"} onChange={v => setA2a({...a2a, agent_name: v})} />
        <TextInput label="Beschreibung" value={a2a.agent_description || ""} onChange={v => setA2a({...a2a, agent_description: v})} />
        <Toggle label="Auth erforderlich" value={a2a.require_auth} onChange={v => setA2a({...a2a, require_auth: v})} />
        <TextInput label="Auth Token" value={a2a.auth_token || ""} onChange={v => setA2a({...a2a, auth_token: v})} type="password" />
        <NumberInput label="Max Tasks" value={a2a.max_tasks || 100} onChange={v => setA2a({...a2a, max_tasks: v})} min={1} max={1000} />
        <NumberInput label="Task Timeout (Sek.)" value={a2a.task_timeout_seconds || 3600} onChange={v => setA2a({...a2a, task_timeout_seconds: v})} min={60} max={86400} />
        <Toggle label="Streaming" value={a2a.enable_streaming} onChange={v => setA2a({...a2a, enable_streaming: v})} />
        <Toggle label="Push Notifications" value={a2a.enable_push} onChange={v => setA2a({...a2a, enable_push: v})} />
        <JsonEditor 
          label="Remotes (JSON Array)" 
          value={a2a.remotes || []} 
          onChange={v => setA2a({...a2a, remotes: v})} 
          onValidationError={err => setValidationErrors(prev => ({ ...prev, a2a_remotes: err }))}
          rows={3} 
          desc='Liste von Remote-Agenten: [{"endpoint": "http://...", "auth_token": ""}]'
        />
      </>}
    </Card>
  </>);
}

// ── Cron & Heartbeat ───────────────────────────────────────────────────
// Fix #10: cron human-readable preview
function CronPage({ cfg, set, cronJobs, setCronJobs }) {
  // B6: Generic cron job updater (like agents/bindings)
  const updCron = useCallback((i, field, v) => {
    setCronJobs(prev => prev.map((j, idx) => idx === i ? { ...j, [field]: v } : j));
  }, [setCronJobs]);
  return (<>
    <Section title="Cron & Heartbeat" desc="Periodische Aufgaben, Heartbeat-Checkliste, Plugins" />
    <Card title="Heartbeat">
      <Toggle label="Heartbeat aktiviert" value={cfg.heartbeat?.enabled} onChange={v => set("heartbeat.enabled", v)} desc="Periodische System-Checks" />
      <NumberInput label="Intervall (Min.)" value={cfg.heartbeat?.interval_minutes} onChange={v => set("heartbeat.interval_minutes", v)} min={1} max={1440} />
      <TextInput label="Checklisten-Datei" value={cfg.heartbeat?.checklist_file} onChange={v => set("heartbeat.checklist_file", v)} mono desc="Relativ zu jarvis_home" />
      <SelectInput label="Channel" value={cfg.heartbeat?.channel} onChange={v => set("heartbeat.channel", v)} options={["cli","telegram","webui","slack","discord","whatsapp","mattermost","google_chat"]} />
      <TextInput label="Modell" value={cfg.heartbeat?.model} onChange={v => set("heartbeat.model", v)} mono />
    </Card>
    <Card title="Cron-Jobs">
      {cronJobs.map((job, i) => (
        <div key={i} className="cc-cron-job">
          <div className="cc-field-row" style={{marginBottom:8}}>
            <strong className="cc-label" style={{margin:0}}>{job.name}</strong>
            <div style={{display:"flex",alignItems:"center",gap:8}}>
              <Toggle label="" value={job.enabled} onChange={v => updCron(i, "enabled", v)} />
              <button className="cc-btn-icon" onClick={() => setCronJobs(prev => prev.filter((_,j) => j!==i))} type="button" title="Job löschen">{I.trash}</button>
            </div>
          </div>
          <TextInput label="Name" value={job.name} onChange={v => updCron(i, "name", v)} mono />
          <TextInput label="Schedule (Cron)" value={job.schedule} onChange={v => updCron(i, "schedule", v)} mono placeholder="0 7 * * 1-5" />
          {job.schedule && <div className="cc-cron-preview">⏰ {cronToHuman(job.schedule)}</div>}
          <TextArea label="Prompt" value={job.prompt} onChange={v => updCron(i, "prompt", v)} rows={4} />
          <SelectInput label="Channel" value={job.channel} onChange={v => updCron(i, "channel", v)} options={["telegram","cli","webui","slack","discord","whatsapp","mattermost","google_chat"]} />
          <TextInput label="Modell" value={job.model} onChange={v => updCron(i, "model", v)} mono />
          <TextInput label="Agent" value={job.agent || ""} onChange={v => updCron(i, "agent", v)} placeholder="(leer = normales Routing)" />
        </div>
      ))}
      <button className="cc-btn" onClick={() => setCronJobs([...cronJobs, { name: `job_${cronJobs.length+1}`, schedule: "0 9 * * *", prompt: "", channel: "cli", model: "qwen3:8b", enabled: false, agent: "" }])} type="button">{I.plus} Neuen Job hinzufügen</button>
    </Card>
    <Card title="Plugins">
      <TextInput label="Skills-Verzeichnis" value={cfg.plugins?.skills_dir} onChange={v => set("plugins.skills_dir", v)} mono desc="Relativ zu jarvis_home" />
      <Toggle label="Auto-Update" value={cfg.plugins?.auto_update} onChange={v => set("plugins.auto_update", v)} desc="Automatische Plugin-Updates beim Start" />
    </Card>
  </>);
}

// ── Logging ────────────────────────────────────────────────────────────
function LoggingPage({ cfg, set }) {
  const l = cfg.logging || {};
  return (<>
    <Section title="Logging" desc="Protokollierung und Debugging" />
    <Card title="Allgemein">
      <SelectInput label="Log-Level" value={l.level} onChange={v => set("logging.level", v)} options={["DEBUG","INFO","WARNING","ERROR","CRITICAL"]} />
      <Toggle label="JSON-Logs" value={l.json_logs} onChange={v => set("logging.json_logs", v)} desc="Strukturierte JSON-Logs für externe Tools" />
      <Toggle label="Console Output" value={l.console} onChange={v => set("logging.console", v)} desc="Logs in der Konsole ausgeben" />
    </Card>
  </>);
}

// ── Database ───────────────────────────────────────────────────────────
function DatabasePage({ cfg, set }) {
  const d = cfg.database || {};
  return (<>
    <Section title="Datenbank" desc="Speicher-Backend für Memory, Logs und State" />
    <Card title="Verbindung">
      <SelectInput label="Backend" value={d.backend} onChange={v => set("database.backend", v)} options={[{value:"sqlite",label:"SQLite (Standard)"},{value:"postgresql",label:"PostgreSQL"}]} />
      {d.backend === "postgresql" && (
        <>
          <TextInput label="Host" value={d.pg_host} onChange={v => set("database.pg_host", v)} mono />
          <NumberInput label="Port" value={d.pg_port} onChange={v => set("database.pg_port", v)} min={1} max={65535} />
          <TextInput label="Datenbank" value={d.pg_dbname} onChange={v => set("database.pg_dbname", v)} mono />
          <TextInput label="Benutzer" value={d.pg_user} onChange={v => set("database.pg_user", v)} mono />
          <TextInput label="Passwort" value={d.pg_password} onChange={v => set("database.pg_password", v)} type="password" />
        </>
      )}
    </Card>
    {d.backend === "postgresql" && (
      <Card title="Connection Pool">
        <NumberInput label="Pool Min" value={d.pg_pool_min} onChange={v => set("database.pg_pool_min", v)} min={1} max={50} />
        <NumberInput label="Pool Max" value={d.pg_pool_max} onChange={v => set("database.pg_pool_max", v)} min={1} max={100} />
      </Card>
    )}
  </>);
}

// ── Prompts & Policies ─────────────────────────────────────────────────
// Fix #9: Reset button per textarea
function PromptsPage({ prompts, setPrompts, defaultPromptsRef }) {
  return (<>
    <Section title="Prompts & Policies" desc="System-Prompts einsehen und anpassen, Gatekeeper-Policies, CORE.md, HEARTBEAT.md" />
    <Card title="CORE.md — System-Persönlichkeit">
      <TextArea label="Core Memory (Markdown)" value={prompts.coreMd} onChange={v => setPrompts({...prompts, coreMd: v})} rows={16} desc="Die Persönlichkeit, Regeln und Präferenzen von Jarvis" onReset={() => setPrompts({...prompts, coreMd: defaultPromptsRef.current.coreMd})} resetLabel="CORE.md zurücksetzen" />
    </Card>
    <Card title="Planner System-Prompt" open={false}>
      <TextArea label="SYSTEM_PROMPT" value={prompts.plannerSystem} onChange={v => setPrompts({...prompts, plannerSystem: v})} rows={20} mono desc="Haupt-Prompt — Variablen: {owner_name}, {tools_section}, {context_section}, {current_datetime}" onReset={() => setPrompts({...prompts, plannerSystem: defaultPromptsRef.current.plannerSystem})} />
    </Card>
    <Card title="Replan-Prompt" open={false}>
      <TextArea label="REPLAN_PROMPT" value={prompts.replanPrompt} onChange={v => setPrompts({...prompts, replanPrompt: v})} rows={12} mono desc="Prompt nach der Tool-Ausführung — Variablen: {results_section}, {original_goal}" onReset={() => setPrompts({...prompts, replanPrompt: defaultPromptsRef.current.replanPrompt})} />
    </Card>
    <Card title="Eskalations-Prompt" open={false}>
      <TextArea label="ESCALATION_PROMPT" value={prompts.escalationPrompt} onChange={v => setPrompts({...prompts, escalationPrompt: v})} rows={6} mono desc="Wenn ein Tool vom Gatekeeper blockiert wird — Variablen: {tool}, {reason}" onReset={() => setPrompts({...prompts, escalationPrompt: defaultPromptsRef.current.escalationPrompt})} />
    </Card>
    <Card title="Gatekeeper-Policies (YAML)" open={false}>
      <TextArea label="default.yaml" value={prompts.policyYaml} onChange={v => setPrompts({...prompts, policyYaml: v})} rows={20} mono desc="Regeln für Tool-Ausführung: ALLOW, INFORM, APPROVE, MASK, BLOCK" onReset={() => setPrompts({...prompts, policyYaml: defaultPromptsRef.current.policyYaml})} />
    </Card>
    <Card title="HEARTBEAT.md — Heartbeat-Checkliste" open={false}>
      <TextArea label="Heartbeat Checkliste" value={prompts.heartbeatMd} onChange={v => setPrompts({...prompts, heartbeatMd: v})} rows={10} desc="Periodisch ausgeführte Aufgaben" onReset={() => setPrompts({...prompts, heartbeatMd: defaultPromptsRef.current.heartbeatMd})} />
    </Card>
  </>);
}

// ── Agents ──────────────────────────────────────────────────────────────
// Fix #6: Generic updateAgent helper
function AgentsPage({ agents, setAgents }) {
  const upd = useCallback((i, field, v) => {
    setAgents(prev => prev.map((a, j) => j === i ? { ...a, [field]: v } : a));
  }, [setAgents]);
  const add = () => setAgents(prev => [...prev, {
    name: `agent_${prev.length+1}`, display_name: "", description: "",
    system_prompt: "", language: "de", trigger_patterns: [], trigger_keywords: [],
    priority: 0, allowed_tools: null, blocked_tools: [], preferred_model: "",
    temperature: null, enabled: true,
  }]);
  return (<>
    <Section title="Agenten" desc="Multi-Agent Profile verwalten — Routing, Modelle, Berechtigungen" />
    {agents.map((a, i) => (
      <Card key={i} title={a.display_name || a.name} badge={a.enabled ? "aktiv" : "inaktiv"}>
        <TextInput label="Name (ID)" value={a.name} onChange={v => upd(i, "name", v)} mono error={!a.name ? "Pflichtfeld" : null} />
        <TextInput label="Anzeigename" value={a.display_name} onChange={v => upd(i, "display_name", v)} />
        <TextInput label="Beschreibung" value={a.description} onChange={v => upd(i, "description", v)} />
        <TextArea label="System-Prompt" value={a.system_prompt} onChange={v => upd(i, "system_prompt", v)} rows={6} />
        <SelectInput label="Sprache" value={a.language} onChange={v => upd(i, "language", v)} options={["de","en","fr","es","it"]} />
        <NumberInput label="Priorität" value={a.priority} onChange={v => upd(i, "priority", v)} min={0} max={100} />
        <TextInput label="Bevorzugtes Modell" value={a.preferred_model} onChange={v => upd(i, "preferred_model", v)} mono />
        <SliderInput label="Temperature" value={a.temperature} onChange={v => upd(i, "temperature", v)} min={0} max={2} step={0.1} />
        <ListInput label="Trigger Keywords" value={a.trigger_keywords} onChange={v => upd(i, "trigger_keywords", v)} />
        <ListInput label="Blockierte Tools" value={a.blocked_tools} onChange={v => upd(i, "blocked_tools", v)} />
        <Toggle label="Aktiviert" value={a.enabled} onChange={v => upd(i, "enabled", v)} />
        {a.name !== "jarvis" && <button className="cc-btn cc-btn-danger" onClick={() => setAgents(prev => prev.filter((_,j) => j!==i))} type="button">{I.trash} Agent löschen</button>}
      </Card>
    ))}
    <button className="cc-btn" onClick={add} type="button">{I.plus} Neuen Agenten anlegen</button>
  </>);
}

// ── Bindings ────────────────────────────────────────────────────────────
// Fix #6: Generic updateBinding helper
function BindingsPage({ bindings, setBindings, agents }) {
  const upd = useCallback((i, field, v) => {
    setBindings(prev => prev.map((b, j) => j === i ? { ...b, [field]: v } : b));
  }, [setBindings]);
  const add = () => setBindings(prev => [...prev, {
    name: `binding_${prev.length+1}`, target_agent: "jarvis", priority: 100,
    description: "", channels: [], command_prefixes: [], message_patterns: [], enabled: true,
  }]);
  return (<>
    <Section title="Bindings" desc="Routing-Regeln: Welche Nachricht geht an welchen Agenten?" />
    {bindings.map((b, i) => (
      <Card key={i} title={b.name}>
        <TextInput label="Name" value={b.name} onChange={v => upd(i, "name", v)} mono error={!b.name ? "Pflichtfeld" : null} />
        <SelectInput label="Ziel-Agent" value={b.target_agent} onChange={v => upd(i, "target_agent", v)} options={agents.map(a => a.name)} />
        <NumberInput label="Priorität" value={b.priority} onChange={v => upd(i, "priority", v)} min={0} max={1000} />
        <TextInput label="Beschreibung" value={b.description} onChange={v => upd(i, "description", v)} />
        <ListInput label="Channels" value={b.channels || []} onChange={v => upd(i, "channels", v)} placeholder="telegram, cli..." />
        <ListInput label="Command Prefixes" value={b.command_prefixes || []} onChange={v => upd(i, "command_prefixes", v)} placeholder="/briefing, /code..." />
        <ListInput label="Message Patterns (Regex)" value={b.message_patterns || []} onChange={v => upd(i, "message_patterns", v)} />
        <Toggle label="Aktiviert" value={b.enabled} onChange={v => upd(i, "enabled", v)} />
        <button className="cc-btn cc-btn-danger" onClick={() => setBindings(prev => prev.filter((_,j) => j!==i))} type="button">{I.trash} Binding löschen</button>
      </Card>
    ))}
    <button className="cc-btn" onClick={add} type="button">{I.plus} Neues Binding</button>
  </>);
}

// ── System ──────────────────────────────────────────────────────────────
// Fix #14: Preset details expandable + Fix #16: Import button
function SystemPage({ cfg, onRestart, onExport, onImport, restartState, presets, onApplyPreset }) {
  const [expandedPreset, setExpandedPreset] = useState(null);
  const fileRef = useRef(null);
  const PRESET_DETAILS = {
    minimal: "CLI aktiv, kleine Modelle (qwen3:8b), kein Dashboard, kein Heartbeat, SQLite",
    standard: "CLI + WebUI (Port 8080), Heartbeat alle 30min, Dashboard aktiv, Kosten-Tracking",
    full: "Alle Channels (Telegram, Slack, Discord, WhatsApp), Heartbeat, Dashboard, Plugins, A2A, PostgreSQL",
  };
  return (<>
    <Section title="System" desc="Neustart, Export/Import, Presets" />
    <Card title="⚡ Jarvis neu starten">
      <p className="cc-desc">Fährt Jarvis sauber herunter (Sessions speichern, Channels stoppen, Memory flushen) und startet mit der aktuellen Konfiguration neu.</p>
      <button
        className={`cc-btn cc-btn-restart ${restartState === "restarting" ? "pulsing" : ""}`}
        onClick={onRestart}
        disabled={restartState === "restarting"}
        type="button"
      >
        {restartState === "restarting" ? "⏳ Neustart läuft..." : restartState === "done" ? "✅ Neu gestartet!" : <>{I.restart} Jarvis neu starten</>}
      </button>
    </Card>
    <Card title="Konfigurations-Presets">
      <div className="cc-presets">
        {presets.map(p => (
          <div key={p.name} className={`cc-preset-card ${expandedPreset === p.name ? "expanded" : ""}`} onClick={() => setExpandedPreset(expandedPreset === p.name ? null : p.name)}>
            <strong>{p.name === "minimal" ? "🔹 Minimal" : p.name === "standard" ? "🔸 Standard" : "🔷 Vollausbau"}</strong>
            <span className="cc-desc">{p.description}</span>
            {expandedPreset === p.name && (
              <div className="cc-preset-details">
                <div className="cc-preset-info">{PRESET_DETAILS[p.name]}</div>
                <button className="cc-btn cc-btn-sm-full" onClick={(e) => { e.stopPropagation(); onApplyPreset(p.name); }} type="button">Preset anwenden</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
    <Card title="Export & Import">
      <div className="cc-export-row">
        <button className="cc-btn" onClick={onExport} type="button">{I.save} Konfiguration exportieren</button>
        <button className="cc-btn" onClick={() => fileRef.current?.click()} type="button">{I.upload} Konfiguration importieren</button>
        <input ref={fileRef} type="file" accept=".json" style={{display:"none"}} onChange={e => { if (e.target.files[0]) onImport(e.target.files[0]); e.target.value = ""; }} />
      </div>
    </Card>
    <Card title="System-Info">
      <div className="cc-info-grid">
        <div className="cc-info-item"><span className="cc-info-label">Version</span><span className="cc-info-val">{cfg.version}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">Backend</span><span className="cc-info-val">{cfg.llm_backend_type}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">Besitzer</span><span className="cc-info-val">{cfg.owner_name}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">Modus</span><span className="cc-info-val">{cfg.operation_mode}</span></div>
      </div>
    </Card>
  </>);
}

// ═══════════════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════════════

export default function App() {
  const [page, setPage] = useState("general");
  const [cfg, setCfg] = useState(defaults());
  const [saveState, setSaveState] = useState("idle");
  const [restartState, setRestartState] = useState("idle");
  const [appStatus, setAppStatus] = useState("stopped"); // "running", "stopped", "starting", "stopping"
  const [menuOpen, setMenuOpen] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});
  const { toasts, add: toast, dismiss: dismissToast } = useToast();

  // ALL useState declarations MUST come before useMemo that references them
  const [cronJobs, setCronJobs] = useState([
    { name: "morning_briefing", schedule: "0 7 * * 1-5", prompt: "Erstelle mein Morning Briefing:\n1. Heutige Termine\n2. Ungelesene E-Mails (Zusammenfassung)\n3. Offene Aufgaben aus gestern\n4. Wetter für Nürnberg", channel: "telegram", model: "qwen3:8b", enabled: false, agent: "" },
    { name: "weekly_review", schedule: "0 18 * * 5", prompt: "Wochenrückblick:\n- Was wurde diese Woche erledigt?\n- Welche neuen Prozeduren wurden gelernt?\n- Was ist noch offen?", channel: "telegram", model: "qwen3:32b", enabled: false, agent: "" },
  ]);
  const [mcpServers, setMcpServers] = useState({ mode: "disabled", external_servers: {} });
  const [a2a, setA2a] = useState({ enabled: false, host: "127.0.0.1", port: 3002, agent_name: "Jarvis" });
  const [prompts, setPrompts] = useState(defaultPrompts());
  const defaultPromptsRef = useRef(defaultPrompts());
  const [agents, setAgents] = useState([{ name: "jarvis", display_name: "Jarvis", description: "Haupt-Agent (Default)", system_prompt: "", language: "de", trigger_patterns: [], trigger_keywords: [], priority: 0, allowed_tools: null, blocked_tools: [], preferred_model: "", temperature: 0.7, enabled: true }]);
  const [bindings, setBindings] = useState([]);
  const [presets] = useState([
    { name: "minimal", description: "CLI-only, kleine Modelle" },
    { name: "standard", description: "CLI + WebUI, Heartbeat, Dashboard" },
    { name: "full", description: "Alle Channels, Heartbeat, Dashboard, Plugins" },
  ]);

  // Dirty state tracking — AFTER all useState declarations
  const [savedSnapshot, setSavedSnapshot] = useState("");
  const currentSnapshot = useMemo(() => JSON.stringify({ cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }), [cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts]);
  const hasChanges = useMemo(() => {
    if (!savedSnapshot) return false;
    return currentSnapshot !== savedSnapshot;
  }, [currentSnapshot, savedSnapshot]);

  // App Status Polling
  useEffect(() => {
    const checkStatus = async () => {
      // Dummy-Annahme: Das API "/system/status" gibt { status: "running" | "stopped" } zurück.
      const res = await api("GET", "/system/status");
      if (res && res.status) setAppStatus(res.status);
    };
    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const toggleAppStatus = async () => {
    if (appStatus === "running") {
      setAppStatus("stopping");
      const res = await api("POST", "/system/stop");
      if (!res.error) {
        setAppStatus("stopped");
        toast("Cognithor/Jarvis wurde beendet.", "info");
      } else {
        setAppStatus("running");
        toast(`Fehler beim Beenden: ${res.error}`, "error");
      }
    } else {
      setAppStatus("starting");
      const res = await api("POST", "/system/start");
      if (!res.error) {
        setAppStatus("running");
        toast("Cognithor/Jarvis wurde gestartet.", "success");
      } else {
        setAppStatus("stopped");
        toast(`Fehler beim Starten: ${res.error}`, "error");
      }
    }
  };

  // Fix #4: Structured deep-setter (no JSON roundtrip)
  const set = useCallback((path, value) => {
    setCfg(prev => {
      const parts = path.split(".");
      const next = { ...prev };
      let current = next;
      for (let i = 0; i < parts.length - 1; i++) {
        const part = parts[i];
        current[part] = { ...(current[part] || {}) };
        current = current[part];
      }
      current[parts[parts.length - 1]] = value;
      return next;
    });
  }, []);

  // Fix #5: Load config from API with loading state
  useEffect(() => {
    (async () => {
      const data = await api("GET", "/config");
      if (data && !data.error) {
        setCfg(prev => ({ ...prev, ...data }));
      }
      setLoaded(true);
      const agentData = await api("GET", "/agents");
      if (agentData?.agents?.length) setAgents(agentData.agents);
      const bindData = await api("GET", "/bindings");
      if (bindData?.bindings?.length) setBindings(bindData.bindings);
      // B9+B10: Load prompts, cronJobs, mcpServers, a2a from API
      const promptData = await api("GET", "/prompts");
      if (promptData && !promptData.error) { setPrompts(promptData); defaultPromptsRef.current = { ...defaultPrompts(), ...promptData }; }
      const cronData = await api("GET", "/cron-jobs");
      if (cronData?.jobs?.length) setCronJobs(cronData.jobs);
      const mcpData = await api("GET", "/mcp-servers");
      if (mcpData && !mcpData.error) setMcpServers(mcpData);
      const a2aData = await api("GET", "/a2a");
      if (a2aData && !a2aData.error) setA2a(a2aData);
    })();
  }, []);

  // B4: Set initial snapshot AFTER all state is loaded (one-time)
  useEffect(() => {
    if (loaded && !savedSnapshot) {
      setSavedSnapshot(JSON.stringify({ cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }));
    }
  }, [loaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fix #2: Parallel save with error tracking + Fix #15: includes prompts
  const save = useCallback(async () => {
    const activeErrors = Object.values(validationErrors).filter(Boolean);
    if (activeErrors.length > 0 || document.querySelector('.cc-error')) {
      toast(`Bitte beheben Sie zuerst alle rot markierten Validierungsfehler.`, "error");
      return;
    }

    setSaveState("saving");
    const errors = [];
    const sections = ["ollama","models","gatekeeper","planner","memory","channels","sandbox","logging","security","heartbeat","plugins","dashboard","model_overrides","web","database"];
    const sectionPromises = sections.map(async (s) => {
      if (cfg[s]) {
        const r = await api("PATCH", `/config/${s}`, cfg[s]);
        if (r?.error) errors.push(`${s}: ${r.error}`);
      }
    });
    const topLevel = api("PATCH", "/config", {
      owner_name: cfg.owner_name, llm_backend_type: cfg.llm_backend_type,
      operation_mode: cfg.operation_mode, cost_tracking_enabled: cfg.cost_tracking_enabled,
      daily_budget_usd: cfg.daily_budget_usd, monthly_budget_usd: cfg.monthly_budget_usd,
      vision_model: cfg.vision_model, vision_model_detail: cfg.vision_model_detail,
      openai_api_key: cfg.openai_api_key, openai_base_url: cfg.openai_base_url,
      anthropic_api_key: cfg.anthropic_api_key, anthropic_max_tokens: cfg.anthropic_max_tokens,
      gemini_api_key: cfg.gemini_api_key, groq_api_key: cfg.groq_api_key,
      deepseek_api_key: cfg.deepseek_api_key, mistral_api_key: cfg.mistral_api_key,
      together_api_key: cfg.together_api_key, openrouter_api_key: cfg.openrouter_api_key,
      xai_api_key: cfg.xai_api_key, cerebras_api_key: cfg.cerebras_api_key,
      github_api_key: cfg.github_api_key, bedrock_api_key: cfg.bedrock_api_key,
      huggingface_api_key: cfg.huggingface_api_key, moonshot_api_key: cfg.moonshot_api_key,
    }).then(r => {
      if (r?.error) errors.push(`top-level: ${r.error}`);
    });
    const agentPromises = agents.map(a => api("POST", `/agents/${a.name}`, a).then(r => { if (r?.error) errors.push(`agent ${a.name}: ${r.error}`); }));
    const bindingPromises = bindings.map(b => api("POST", `/bindings/${b.name}`, b).then(r => { if (r?.error) errors.push(`binding ${b.name}: ${r.error}`); }));

    // B3: Save cronJobs, mcpServers, a2a, prompts
    const extraSaves = [
      api("PUT", "/cron-jobs", { jobs: cronJobs }).then(r => { if (r?.error) errors.push(`cron: ${r.error}`); }),
      api("PUT", "/mcp-servers", mcpServers).then(r => { if (r?.error) errors.push(`mcp: ${r.error}`); }),
      api("PUT", "/a2a", a2a).then(r => { if (r?.error) errors.push(`a2a: ${r.error}`); }),
      api("PUT", "/prompts", prompts).then(r => { if (r?.error) errors.push(`prompts: ${r.error}`); }),
    ];

    await Promise.all([...sectionPromises, topLevel, ...agentPromises, ...bindingPromises, ...extraSaves]);

    if (errors.length > 0) {
      toast(`${errors.length} Fehler beim Speichern: ${errors.slice(0, 3).join("; ")}`, "error");
      setSaveState("idle");
      // Reload config to ensure UI reflects the actual backend state after partial failure
      const data = await api("GET", "/config");
      if (data && !data.error) setCfg(prev => ({ ...prev, ...data }));
    } else {
      setSavedSnapshot(currentSnapshot);
      toast("Konfiguration gespeichert", "success");
      setSaveState("saved");
      setTimeout(() => setSaveState("idle"), 2000);
    }
  }, [cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts, currentSnapshot, validationErrors, toast]);

  // Restart
  const restart = useCallback(async () => {
    if (!confirm("Jarvis wirklich neu starten? Alle laufenden Tasks werden beendet.")) return;
    setRestartState("restarting");
    await save();
    await api("POST", "/config/reload");
    try { await fetch(`${API}/shutdown`, { method: "POST" }); } catch {}
    setTimeout(() => { setRestartState("done"); toast("Jarvis wurde neu gestartet", "success"); setTimeout(() => setRestartState("idle"), 3000); }, 3000);
  }, [save, toast]);

  // Fix #15: Export includes prompts
  const onExport = () => {
    const blob = new Blob([JSON.stringify({ config: cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cognithor-config-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast("Konfiguration exportiert", "success");
  };

  // Fix #16: Import
  const onImport = async (file) => {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (!confirm("Konfiguration aus Datei importieren? Aktuelle Einstellungen werden überschrieben.")) return;
      if (data.config) setCfg(prev => ({ ...prev, ...data.config }));
      if (data.agents?.length) setAgents(data.agents);
      if (data.bindings) setBindings(data.bindings);
      if (data.cronJobs) setCronJobs(data.cronJobs);
      if (data.mcpServers) setMcpServers(data.mcpServers);
      if (data.a2a) setA2a(data.a2a);
      if (data.prompts) setPrompts(data.prompts);
      toast("Konfiguration importiert — bitte speichern!", "success");
    } catch (e) {
      toast(`Import fehlgeschlagen: ${e.message}`, "error");
    }
  };

  // Apply preset
  const onApplyPreset = async (name) => {
    if (!confirm(`Preset „${name}" anwenden? Aktuelle Einstellungen werden überschrieben.`)) return;
    const r = await api("POST", `/config/presets/${name}`);
    if (!r.error) {
      const data = await api("GET", "/config");
      if (data && !data.error) {
        setCfg(prev => ({ ...prev, ...data }));
        toast(`Preset „${name}" angewendet`, "success");
      }
    } else {
      toast(`Preset-Fehler: ${r.error}`, "error");
    }
  };

  // Fix #23: Ctrl+S shortcut + Fix #17: keyboard nav
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        save();
      }
      // Ctrl+1..0 for pages
      if ((e.ctrlKey || e.metaKey) && e.key >= "1" && e.key <= "9") {
        const idx = parseInt(e.key) - 1;
        if (PAGES[idx]) { e.preventDefault(); setPage(PAGES[idx].id); }
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "0") {
        e.preventDefault();
        setPage("cron");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [save]);

  // B5: Warn before closing tab with unsaved changes
  useEffect(() => {
    const handler = (e) => {
      if (hasChanges) { e.preventDefault(); e.returnValue = ""; }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasChanges]);

  // Fix #1: Warn on page change with unsaved changes
  const changePage = useCallback((newPage) => {
    if (hasChanges && !confirm("Es gibt ungespeicherte Änderungen. Wirklich die Seite wechseln?")) return;
    setPage(newPage);
    setMenuOpen(false);
  }, [hasChanges]);

  // Render current page
  const renderPage = () => {
    switch (page) {
      case "general": return <GeneralPage cfg={cfg} set={set} />;
      case "providers": return <ProvidersPage cfg={cfg} set={set} />;
      case "models": return <ModelsPage cfg={cfg} set={set} setValidationErrors={setValidationErrors} />;
      case "planner": return <PlannerPage cfg={cfg} set={set} setValidationErrors={setValidationErrors} />;
      case "memory": return <MemoryPage cfg={cfg} set={set} />;
      case "channels": return <ChannelsPage cfg={cfg} set={set} />;
      case "security": return <SecurityPage cfg={cfg} set={set} />;
      case "web": return <WebPage cfg={cfg} set={set} />;
      case "mcp": return <McpPage cfg={cfg} set={set} mcpServers={mcpServers} setMcpServers={setMcpServers} a2a={a2a} setA2a={setA2a} setValidationErrors={setValidationErrors} />;
      case "cron": return <CronPage cfg={cfg} set={set} cronJobs={cronJobs} setCronJobs={setCronJobs} />;
      case "database": return <DatabasePage cfg={cfg} set={set} />;
      case "logging": return <LoggingPage cfg={cfg} set={set} />;
      case "prompts": return <PromptsPage prompts={prompts} setPrompts={setPrompts} defaultPromptsRef={defaultPromptsRef} />;
      case "agents": return <AgentsPage agents={agents} setAgents={setAgents} />;
      case "bindings": return <BindingsPage bindings={bindings} setBindings={setBindings} agents={agents} />;
      case "system": return <SystemPage cfg={cfg} onRestart={restart} onExport={onExport} onImport={onImport} restartState={restartState} presets={presets} onApplyPreset={onApplyPreset} />;
      default: return null;
    }
  };

  return (
    <div className="cc-root">
      {/* Fix #22: font-display: swap via preconnect + manual @font-face fallback */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        .cc-root {
          font-family: 'DM Sans', -apple-system, sans-serif;
          background: #08080f;
          color: #e0e0e8;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          --accent: #00d4ff;
          --accent2: #6c63ff;
          --bg: #08080f;
          --bg2: #10101a;
          --bg3: #181825;
          --border: #1e1e30;
          --text: #e0e0e8;
          --text2: #8888a0;
          --danger: #ff4466;
          --success: #00e676;
          --warn: #ffab40;
          --radius: 10px;
        }

        /* ── Header ──────────────────────────────── */
        .cc-header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; backdrop-filter: blur(20px); }
        .cc-header-left { display: flex; align-items: center; gap: 12px; }
        .cc-logo { font-size: 18px; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
        .cc-header-actions { display: flex; gap: 8px; align-items: center; }
        .cc-menu-btn { display: none; background: none; border: none; color: var(--text); font-size: 24px; cursor: pointer; }
        @media (max-width: 768px) { .cc-menu-btn { display: block; } }

        /* ── Layout ──────────────────────────────── */
        .cc-layout { display: flex; flex: 1; }
        .cc-sidebar { width: 220px; background: var(--bg2); border-right: 1px solid var(--border); padding: 12px 8px; overflow-y: auto; position: sticky; top: 52px; height: calc(100vh - 52px); flex-shrink: 0; }
        @media (max-width: 768px) { .cc-sidebar { display: none; } }
        .cc-sidebar.mobile-open { display: flex; flex-direction: column; position: fixed; top: 52px; left: 0; bottom: 0; z-index: 99; width: 260px; box-shadow: 4px 0 20px rgba(0,0,0,0.5); }
        .cc-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 98; }
        .cc-overlay.visible { display: block; }
        .cc-nav-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 8px; cursor: pointer; color: var(--text2); font-size: 13px; font-weight: 500; transition: all 0.15s; border: none; background: none; width: 100%; text-align: left; }
        .cc-nav-item:hover { background: var(--bg3); color: var(--text); }
        .cc-nav-item.active { background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(108,99,255,0.08)); color: var(--accent); border-left: 2px solid var(--accent); }
        .cc-nav-key { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--text2); opacity: 0.4; margin-left: auto; }
        .cc-main { flex: 1; padding: 20px; max-width: 800px; margin: 0 auto; width: 100%; }
        @media (max-width: 768px) { .cc-main { padding: 16px; } }

        /* ── Section ─────────────────────────────── */
        .cc-section-head { margin-bottom: 20px; }
        .cc-section-title { font-size: 22px; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .cc-section-desc { color: var(--text2); font-size: 13px; margin-top: 4px; }

        /* ── Card ────────────────────────────────── */
        .cc-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 12px; overflow: hidden; }
        .cc-card-head { padding: 14px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
        .cc-card-title { font-size: 14px; font-weight: 600; }
        .cc-card-right { display: flex; align-items: center; gap: 8px; }
        .cc-chevron { color: var(--text2); transition: transform 0.2s; font-size: 14px; }
        .cc-chevron.open { transform: rotate(180deg); }
        .cc-card-body { padding: 0 16px 16px; }
        .cc-badge { font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: 600; text-transform: uppercase; }
        .cc-badge.aktiv { background: rgba(0,230,118,0.15); color: var(--success); }
        .cc-badge.inaktiv { background: rgba(136,136,160,0.15); color: var(--text2); }
        .cc-badge.green { background: rgba(0,230,118,0.15); color: var(--success); }
        .cc-badge.yellow { background: rgba(255,171,64,0.15); color: var(--warn); }
        .cc-badge.orange { background: rgba(255,111,0,0.15); color: #ff6f00; }
        .cc-badge.red { background: rgba(255,68,102,0.15); color: var(--danger); }

        /* ── Fields ──────────────────────────────── */
        .cc-field { margin-bottom: 14px; }
        .cc-field-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .cc-label { font-size: 13px; font-weight: 500; color: var(--text); margin-bottom: 4px; }
        .cc-desc { font-size: 11px; color: var(--text2); margin-bottom: 4px; }
        .cc-input-wrap { position: relative; display: flex; align-items: center; }
        .cc-input { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; color: var(--text); font-size: 13px; font-family: inherit; outline: none; transition: border 0.15s; }
        .cc-input:focus { border-color: var(--accent); }
        .cc-input.mono, .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
        .cc-input-error { border-color: var(--danger) !important; background: rgba(255,68,102,0.05); }
        .cc-input-disabled { opacity: 0.5; cursor: not-allowed; }
        .cc-field-error { font-size: 11px; color: var(--danger); margin-top: 4px; }
        .cc-eye-btn { position: absolute; right: 8px; background: none; border: none; color: var(--text2); cursor: pointer; padding: 4px; }
        .cc-select { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 30px 8px 10px; color: var(--text); font-size: 13px; font-family: inherit; outline: none; appearance: none; cursor: pointer; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238888a0' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; }
        .cc-select:focus { border-color: var(--accent); }
        .cc-textarea { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; color: var(--text); font-size: 13px; font-family: inherit; outline: none; resize: vertical; min-height: 80px; transition: border 0.15s; }
        .cc-textarea:focus { border-color: var(--accent); }
        .cc-textarea.mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.5; }

        /* Fix #3: Read-only field */
        .cc-readonly { padding: 8px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text2); font-size: 13px; font-family: 'JetBrains Mono', monospace; opacity: 0.7; }

        /* ── Toggle ──────────────────────────────── */
        .cc-toggle { width: 40px; height: 22px; border-radius: 11px; background: var(--bg); border: 1px solid var(--border); position: relative; cursor: pointer; transition: all 0.2s; flex-shrink: 0; }
        .cc-toggle.on { background: var(--accent); border-color: var(--accent); }
        .cc-toggle-dot { width: 16px; height: 16px; border-radius: 50%; background: #fff; position: absolute; top: 2px; left: 2px; transition: transform 0.2s; }
        .cc-toggle.on .cc-toggle-dot { transform: translateX(18px); }

        /* ── Slider ──────────────────────────────── */
        .cc-slider { width: 100%; appearance: none; height: 4px; background: var(--border); border-radius: 2px; outline: none; margin-top: 4px; }
        .cc-slider::-webkit-slider-thumb { appearance: none; width: 16px; height: 16px; border-radius: 50%; background: var(--accent); cursor: pointer; }
        .cc-slider::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: var(--accent); cursor: pointer; border: none; }
        .cc-slider::-moz-range-track { height: 4px; background: var(--border); border-radius: 2px; }
        .cc-slider-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--accent); font-weight: 500; cursor: pointer; padding: 2px 6px; border-radius: 4px; transition: background 0.15s; }
        .cc-slider-val:hover { background: rgba(0,212,255,0.1); }
        .cc-slider-edit { width: 70px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--accent); background: var(--bg); border: 1px solid var(--accent); border-radius: 4px; padding: 2px 6px; outline: none; text-align: right; }

        /* ── List ────────────────────────────────── */
        .cc-list-items { display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px; }
        .cc-list-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: var(--bg); border-radius: 6px; font-size: 12px; }
        .cc-list-add { display: flex; gap: 6px; }
        .cc-list-add .cc-input { flex: 1; }
        .cc-btn-icon { background: none; border: none; color: var(--text2); cursor: pointer; padding: 4px; border-radius: 4px; }
        .cc-btn-icon:hover { color: var(--danger); background: rgba(255,68,102,0.1); }

        /* Fix #9: Reset button */
        .cc-btn-reset { display: inline-flex; align-items: center; gap: 4px; background: none; border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 11px; color: var(--text2); cursor: pointer; transition: all 0.15s; font-family: inherit; flex-shrink: 0; }
        .cc-btn-reset:hover { border-color: var(--warn); color: var(--warn); }

        /* ── Buttons ─────────────────────────────── */
        .cc-btn { display: inline-flex; align-items: center; gap: 6px; padding: 10px 18px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg3); color: var(--text); font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.15s; font-family: inherit; }
        .cc-btn:hover { border-color: var(--accent); color: var(--accent); }
        .cc-btn-sm { display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg3); color: var(--text); cursor: pointer; }
        .cc-btn-sm:hover { border-color: var(--accent); color: var(--accent); }
        .cc-btn-sm-full { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid var(--accent); background: rgba(0,212,255,0.1); color: var(--accent); font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 8px; transition: all 0.15s; font-family: inherit; }
        .cc-btn-sm-full:hover { background: rgba(0,212,255,0.2); }
        .cc-btn-danger { border-color: rgba(255,68,102,0.3); color: var(--danger); }
        .cc-btn-danger:hover { background: rgba(255,68,102,0.1); border-color: var(--danger); }
        .cc-btn-restart { background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(108,99,255,0.1)); border-color: var(--accent); color: var(--accent); font-size: 15px; padding: 14px 28px; width: 100%; justify-content: center; }
        .cc-btn-restart:hover { background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(108,99,255,0.2)); }
        .cc-btn-restart.pulsing { animation: pulse 1.5s infinite; }

        /* Fix #18: Save bar with safe-area */
        .cc-save-bar { position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg2); border-top: 1px solid var(--border); padding: 10px 16px; padding-bottom: calc(10px + env(safe-area-inset-bottom, 0px)); display: flex; justify-content: center; align-items: center; gap: 12px; z-index: 100; backdrop-filter: blur(20px); }
        .cc-save-btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 32px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; transition: all 0.2s; }
        .cc-save-btn.primary { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #000; }
        .cc-save-btn.primary:hover { opacity: 0.9; transform: translateY(-1px); }
        .cc-save-btn.saved { background: var(--success); color: #000; }
        /* Fix #1: dirty indicator */
        .cc-dirty-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); animation: pulse 2s infinite; }
        .cc-save-hint { font-size: 11px; color: var(--text2); }

        /* Fix #13: Search bar */
        .cc-search-bar { position: relative; margin-bottom: 12px; }
        .cc-search-icon { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--text2); }
        .cc-search-input { width: 100%; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 10px 10px 36px; color: var(--text); font-size: 13px; font-family: inherit; outline: none; }
        .cc-search-input:focus { border-color: var(--accent); }
        .cc-search-clear { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: none; border: none; color: var(--text2); cursor: pointer; }
        .cc-empty { text-align: center; padding: 20px; color: var(--text2); font-size: 13px; }

        /* Fix #10: Cron preview */
        .cc-cron-preview { font-size: 11px; color: var(--accent); background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.15); border-radius: 6px; padding: 6px 10px; margin-bottom: 10px; }

        /* Fix #11: Weight sum display */
        .cc-weight-sum { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
        .cc-weight-sum-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 15px; }
        .cc-weight-sum-hint { font-size: 11px; color: var(--text2); margin-left: auto; }

        /* Fix #14: Preset details */
        .cc-preset-details { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
        .cc-preset-info { font-size: 11px; color: var(--text); line-height: 1.5; }
        .cc-preset-card.expanded { border-color: var(--accent); }

        /* Fix #16: Export row */
        .cc-export-row { display: flex; gap: 8px; flex-wrap: wrap; }

        /* Fix #20: Toasts */
        .cc-toast-container { position: fixed; top: 60px; right: 16px; z-index: 200; display: flex; flex-direction: column; gap: 8px; max-width: 380px; }
        .cc-toast { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 8px; font-size: 13px; color: var(--text); backdrop-filter: blur(20px); animation: slideIn 0.25s ease-out; }
        .cc-toast-success { background: rgba(0,230,118,0.15); border: 1px solid rgba(0,230,118,0.3); }
        .cc-toast-error { background: rgba(255,68,102,0.15); border: 1px solid rgba(255,68,102,0.3); }
        .cc-toast-info { background: rgba(0,212,255,0.15); border: 1px solid rgba(0,212,255,0.3); }
        .cc-toast-msg { flex: 1; }
        .cc-toast-close { background: none; border: none; color: var(--text2); cursor: pointer; padding: 2px; }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

        /* Fix #5: Spinner */
        .cc-spinner-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; gap: 16px; }
        .cc-spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
        .cc-spinner-text { color: var(--text2); font-size: 14px; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Misc ────────────────────────────────── */
        .cc-warn { background: rgba(255,171,64,0.1); border: 1px solid rgba(255,171,64,0.3); border-radius: 6px; padding: 8px 12px; font-size: 12px; color: var(--warn); margin-bottom: 12px; }
        .cc-cron-job { background: var(--bg); border-radius: 8px; padding: 12px; margin-bottom: 12px; border: 1px solid var(--border); }
        .cc-presets { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
        .cc-preset-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; cursor: pointer; transition: all 0.15s; display: flex; flex-direction: column; gap: 4px; }
        .cc-preset-card:hover { border-color: var(--accent); }
        .cc-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .cc-info-item { background: var(--bg); border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 2px; }
        .cc-info-label { font-size: 11px; color: var(--text2); }
        .cc-info-val { font-size: 14px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
      `}</style>

      {/* Fix #20: Toast container */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Header */}
      <div className="cc-header">
        <div className="cc-header-left">
          <button className="cc-menu-btn" onClick={() => setMenuOpen(!menuOpen)} type="button">☰</button>
          <span className="cc-logo">⚡ Cognithor</span>
        </div>
        <div className="cc-header-actions">
          {/* Prominenter Start/Stop Button */}
          <button 
            className={`cc-btn-sm ${appStatus === "running" ? "cc-btn-danger" : "cc-btn-success"}`} 
            style={{ 
              width: "auto", 
              padding: "0 12px", 
              height: "28px", 
              marginRight: "8px", 
              backgroundColor: appStatus === "running" ? "rgba(255,68,102,0.1)" : "rgba(0,230,118,0.1)",
              borderColor: appStatus === "running" ? "rgba(255,68,102,0.4)" : "rgba(0,230,118,0.4)",
              color: appStatus === "running" ? "var(--danger)" : "var(--success)"
            }}
            onClick={toggleAppStatus}
            disabled={appStatus === "starting" || appStatus === "stopping"}
            title={appStatus === "running" ? "Cognithor stoppen" : "Cognithor starten"}
          >
            {appStatus === "running" ? (
              <>{I.stop} <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>Power Off</span></>
            ) : appStatus === "starting" || appStatus === "stopping" ? (
              <span style={{fontSize: "12px", fontWeight: "600"}}>⏳ Bitte warten...</span>
            ) : (
              <>{I.play} <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>Power On</span></>
            )}
          </button>
          
          {hasChanges && <div className="cc-dirty-dot" title="Ungespeicherte Änderungen" />}
          <span style={{ fontSize: 11, color: "var(--text2)" }}>Control Center v{cfg.version}</span>
        </div>
      </div>

      {/* Layout */}
      <div className="cc-layout">
        <div className={`cc-overlay ${menuOpen ? "visible" : ""}`} onClick={() => setMenuOpen(false)} />

        {/* Fix #17: Sidebar with keyboard hints */}
        <nav className={`cc-sidebar ${menuOpen ? "mobile-open" : ""}`}>
          {PAGES.map(p => (
            <button key={p.id} className={`cc-nav-item ${page === p.id ? "active" : ""}`} onClick={() => changePage(p.id)}>
              {p.icon} {p.label}
              {p.key && <span className="cc-nav-key">⌘{p.key}</span>}
            </button>
          ))}
        </nav>

        {/* Main */}
        <main className="cc-main">
          {/* Fix #5: Loading spinner */}
          {!loaded ? <Spinner /> : renderPage()}
          <div style={{ height: 80 }} />
        </main>
      </div>

      {/* Fix #1 + #18: Save bar with dirty state + safe area */}
      <div className="cc-save-bar">
        {hasChanges && <span className="cc-save-hint">Ungespeicherte Änderungen</span>}
        <button
          className={`cc-save-btn ${saveState === "saved" ? "saved" : "primary"}`}
          onClick={save}
          disabled={saveState === "saving"}
          type="button"
        >
          {saveState === "saving" ? "⏳ Speichern..." : saveState === "saved" ? <>{I.check} Gespeichert!</> : <>{I.save} {hasChanges ? "Änderungen speichern" : "Speichern"}</>}
        </button>
        <span className="cc-save-hint" style={{opacity: 0.4}}>⌘S</span>
      </div>
    </div>
  );
}
