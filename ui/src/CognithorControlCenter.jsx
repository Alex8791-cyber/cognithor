import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { GlobalSearch } from "./components/GlobalSearch";
import { ThemeToggle, useTheme } from "./components/ThemeToggle";
import { ConfirmModal } from "./components/ConfirmModal";
import ChatPage from "./pages/ChatPage";
import WorkflowGraphPage from "./pages/WorkflowGraphPage";
import KnowledgeGraphPage from "./pages/KnowledgeGraphPage";
import { t, tLocale, setLocale as setI18nLocale, onLocaleChange } from "./utils/i18n";

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

// ── Auth token (fetched once from bootstrap endpoint) ──────────────────
let _ccToken = null;
let _ccTokenPromise = null;
async function getCCToken() {
  if (_ccToken) return _ccToken;
  if (!_ccTokenPromise) {
    _ccTokenPromise = fetch(`${API}/bootstrap`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { _ccToken = d?.token || null; return _ccToken; })
      .catch(() => null);
  }
  return _ccTokenPromise;
}

// ── API Helper ─────────────────────────────────────────────────────────
const HTTP_ERRORS = {
  400: "Bad request — please check your inputs",
  401: "Unauthorized — please log in again",
  403: "Access denied",
  404: "Endpoint not found — check backend version",
  409: "Conflict — data was modified elsewhere",
  422: "Validation error — check your inputs",
  429: "Too many requests — please wait and try again",
  500: "Internal server error — check backend logs",
  502: "Backend unreachable",
  503: "Backend overloaded — please wait",
  504: "Backend timeout — request took too long",
};

async function _ccFetch(method, path, body) {
  const token = await getCCToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  return fetch(`${API}${path}`, opts);
}

async function api(method, path, body) {
  try {
    let r = await _ccFetch(method, path, body);
    // On 401, token may be stale — re-fetch and retry once
    if (r.status === 401) {
      _ccToken = null;
      _ccTokenPromise = null;
      r = await _ccFetch(method, path, body);
    }
    if (!r.ok) {
      const friendly = HTTP_ERRORS[r.status] || `HTTP ${r.status}`;
      return { error: friendly, status: r.status };
    }
    const text = await r.text();
    if (!text) return {};
    // Fix #19: Prevent precision loss for large integers (like Discord IDs)
    const safeText = text.replace(/:\s*([0-9]{16,})\b/g, ':"$1"');
    return JSON.parse(safeText);
  } catch (e) {
    console.error(`API ${method} ${path}:`, e);
    if (e.name === "TypeError" && e.message.includes("fetch")) {
      return { error: "Backend unreachable — is Cognithor running?" };
    }
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
  chat: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>,
  workflow: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="5" cy="6" r="3"/><circle cx="19" cy="6" r="3"/><circle cx="12" cy="18" r="3"/><path d="M7.5 7.5L10.5 16M16.5 7.5L13.5 16"/></svg>,
  graph: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><path d="M9 6h6M6 9v6M18 9v6M9 18h6M8.5 8.5l7 7"/></svg>,
  language: <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M5 8l4 12M3 8h8M7 8l1-4M13 2l5.5 14M16.5 16H21M14.5 11h5"/><path d="M11.5 22L16 11l4.5 11"/></svg>,
};

// ── Navigation ─────────────────────────────────────────────────────────
// Labels use i18n keys — resolved at render time via t()
const PAGE_DEFS = [
  { id: "chat", i18n: "nav.chat", icon: I.chat, key: "1" },
  { id: "general", i18n: "nav.general", icon: I.home, key: "2" },
  { id: "language", i18n: "nav.language", icon: I.language, key: "3" },
  { id: "providers", i18n: "nav.providers", icon: I.llm, key: "4" },
  { id: "models", i18n: "nav.models", icon: I.model, key: "5" },
  { id: "planner", i18n: "nav.planner", icon: I.brain, key: "6" },
  { id: "executor", i18n: "nav.executor", icon: I.terminal, key: "7" },
  { id: "memory", i18n: "nav.memory", icon: I.mem, key: "8" },
  { id: "channels", i18n: "nav.channels", icon: I.ch, key: "9" },
  { id: "security", i18n: "nav.security", icon: I.shield, key: "0" },
  { id: "web", i18n: "nav.web", icon: I.web, key: null },
  { id: "mcp", i18n: "nav.mcp", icon: I.plug, key: null },
  { id: "cron", i18n: "nav.cron", icon: I.clock, key: null },
  { id: "database", i18n: "nav.database", icon: I.db, key: null },
  { id: "logging", i18n: "nav.logging", icon: I.terminal, key: null },
  { id: "prompts", i18n: "nav.prompts", icon: I.file, key: null },
  { id: "agents", i18n: "nav.agents", icon: I.bot, key: null },
  { id: "bindings", i18n: "nav.bindings", icon: I.link, key: null },
  { id: "workflows", i18n: "nav.workflows", icon: I.workflow, key: null },
  { id: "knowledge-graph", i18n: "nav.knowledge_graph", icon: I.graph, key: null },
  { id: "system", i18n: "nav.system", icon: I.gear, key: null },
];

// ── Default Config ─────────────────────────────────────────────────────
const defaults = () => ({
  owner_name: "User", version: "",
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
    planner: { name: "qwen3:32b", context_window: 32768, vram_gb: 20, strengths: ["reasoning","planning","reflection","german"], speed: "medium", backend: "" },
    executor: { name: "qwen3:8b", context_window: 32768, vram_gb: 6, strengths: ["tool-calling","simple-tasks"], speed: "fast", backend: "" },
    coder: { name: "qwen3-coder:30b", context_window: 32768, vram_gb: 20, strengths: ["code-generation","debugging","testing"], speed: "medium", backend: "" },
    embedding: { name: "qwen3-embedding:0.6b", context_window: 8192, vram_gb: 0.5, strengths: ["semantic-search","multilingual"], speed: "fast", backend: "" },
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
  security: { max_iterations: 10, allowed_paths: ["~/.jarvis/","/tmp/jarvis/"], blocked_commands: ["rm\\s+-rf\\s+/","mkfs\\b","dd\\s+if=/dev"], credential_patterns: ["sk-[a-zA-Z0-9]{20,}","token_[a-zA-Z0-9]+"], max_sub_agent_depth: 3 },
  executor: { default_timeout_seconds: 30, max_output_chars: 10000, max_retries: 3, backoff_base_delay_seconds: 1.0, max_parallel_tools: 4, media_analyze_image_timeout: 180, media_transcribe_audio_timeout: 120, media_extract_text_timeout: 120, media_tts_timeout: 120, run_python_timeout: 120 },
  web: { searxng_url: "", brave_api_key: "", google_cse_api_key: "", google_cse_cx: "", jina_api_key: "", duckduckgo_enabled: true, domain_blocklist: [], domain_allowlist: [], max_fetch_bytes: 500000, max_text_chars: 20000, fetch_timeout_seconds: 15, search_timeout_seconds: 10, max_search_results: 10, ddg_min_delay_seconds: 2.0, ddg_ratelimit_wait_seconds: 30, ddg_cache_ttl_seconds: 3600, search_and_read_max_chars: 5000, http_request_max_body_bytes: 1048576, http_request_timeout_seconds: 30, http_request_rate_limit_seconds: 1.0 },
  logging: { level: "INFO", json_logs: false, console: true },
  database: { backend: "sqlite", encryption_enabled: false, pg_host: "localhost", pg_port: 5432, pg_dbname: "jarvis", pg_user: "jarvis", pg_password: "", pg_pool_min: 2, pg_pool_max: 10 },
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
  if (parts.length < 5) return "Invalid expression";
  const [min, hour, dom, mon, dow] = parts;
  const dayNames = { 0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun" };
  const monNames = { 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec" };
  let time = "";
  if (hour !== "*" && min !== "*") time = `at ${hour.padStart(2, "0")}:${min.padStart(2, "0")}`;
  else if (hour !== "*") time = `at hour ${hour}`;
  else time = "every minute";
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
  if (dom !== "*") domStr = `on the ${dom}.`;
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

// Fix #7: validation prop for inputs + tooltip support
function TextInput({ label, value, onChange, desc, placeholder, type = "text", mono, error, disabled, tooltip }) {
  const [show, setShow] = useState(false);
  const isSecret = type === "password";
  // Backend masks saved secrets as "***" — these aren't the real value.
  // When masked, show a placeholder and clear on focus so user can type a new key.
  const isMasked = isSecret && value === "***";
  return (
    <div className="cc-field">
      <div className="cc-label">{label} {tooltip && <span className="cc-tooltip-trigger" title={tooltip}>{I.help}</span>}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-input-wrap">
        <input
          className={`cc-input ${mono ? "mono" : ""} ${error ? "cc-error" : ""} ${disabled ? "cc-input-disabled" : ""}`}
          type={isSecret && !show ? "password" : "text"}
          value={isMasked ? "" : (value || "")}
          onChange={e => !disabled && onChange(e.target.value)}
          placeholder={isMasked ? "(saved — enter new key to change)" : (placeholder || "")}
          readOnly={disabled}
          tabIndex={disabled ? -1 : 0}
          aria-label={label}
          aria-invalid={!!error}
        />
        {isSecret && !isMasked && (
          <button className="cc-eye-btn" onClick={() => setShow(!show)} type="button" aria-label={show ? "Hide" : "Show"}>{show ? I.eyeOff : I.eye}</button>
        )}
        {isMasked && <span className="cc-secret-badge">Saved</span>}
      </div>
      {error && <div className="cc-field-error" role="alert">{error}</div>}
    </div>
  );
}

function NumberInput({ label, value, onChange, desc, min, max, step = 1, error }) {
  const localErr = (value !== undefined && value !== null) && ((min !== undefined && value < min) || (max !== undefined && value > max));
  const displayErr = error || (localErr ? `Value must be between ${min} and ${max}.` : null);
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
            title="Click to edit manually"
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
        <input className="cc-input" value={draft} onChange={e => setDraft(e.target.value)} placeholder={placeholder || "Add..."} onKeyDown={e => e.key === "Enter" && add()} />
        <button className="cc-btn-sm" onClick={add} type="button">{I.plus}</button>
      </div>
    </div>
  );
}

// Domain-validated ListInput: only accepts valid hostnames (no scheme, no paths, no wildcards)
const _DOMAIN_RE = /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$/i;
function DomainListInput({ label, value = [], onChange, desc, placeholder }) {
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const validate = (d) => {
    if (!d) return "Domain must not be empty";
    if (d.includes("://")) return "No scheme (http/https) — hostname only";
    if (d.includes("/")) return "No paths allowed — hostname only (e.g. example.com)";
    if (d.includes("*")) return "No wildcards allowed — exact domains only";
    if (d.includes(" ")) return "No spaces allowed";
    if (!_DOMAIN_RE.test(d)) return "Invalid domain format (e.g. example.com)";
    if (value.some(v => v.toLowerCase() === d.toLowerCase())) return "Domain already exists";
    return "";
  };
  const add = () => {
    const d = draft.trim().toLowerCase();
    const err = validate(d);
    if (err) { setError(err); return; }
    onChange([...value, d]);
    setDraft("");
    setError("");
  };
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
        <input
          className={`cc-input${error ? " cc-input-error" : ""}`}
          value={draft}
          onChange={e => { setDraft(e.target.value); if (error) setError(""); }}
          placeholder={placeholder || "example.com"}
          onKeyDown={e => e.key === "Enter" && add()}
        />
        <button className="cc-btn-sm" onClick={add} type="button">{I.plus}</button>
      </div>
      {error && <div className="cc-error">{error}</div>}
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
          <button className="cc-btn-reset" onClick={onReset} title={resetLabel || "Reset to default"} type="button">
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
      const errorMsg = `JSON error: ${e.message.replace(/^JSON\.parse: /, "")}`;
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
      <span className="cc-spinner-text">{t("common.loading_config")}</span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Pages
// ═══════════════════════════════════════════════════════════════════════

// ── Prompt-Evolution Card ──────────────────────────────────────────────
function PromptEvolutionCard() {
  const [stats, setStats] = useState(null);
  const [evolving, setEvolving] = useState(false);
  const [evolveResult, setEvolveResult] = useState(null);

  const refresh = useCallback(async () => {
    const r = await api("GET", "/prompt-evolution/stats");
    if (!r.error) setStats(r);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleToggle = async (val) => {
    await api("POST", "/prompt-evolution/toggle", { enabled: val });
    await refresh();
  };

  const handleEvolve = async () => {
    setEvolving(true);
    setEvolveResult(null);
    const r = await api("POST", "/prompt-evolution/evolve");
    setEvolveResult(r);
    setEvolving(false);
    await refresh();
  };

  const enabled = stats?.enabled ?? false;

  return (
    <Card title={t("evo.title")} open={enabled}>
      <Toggle label={t("evo.enabled")} value={enabled} onChange={handleToggle} />
      {enabled && stats && (<>
        <ReadOnly label={t("evo.active_version")} value={stats.active_version_id || "–"} />
        <ReadOnly label={t("evo.versions")} value={String(stats.version_count ?? 0)} />
        <ReadOnly label={t("evo.total_sessions")} value={String(stats.total_sessions ?? 0)} />
        <ReadOnly label={t("evo.running_tests")} value={String(stats.running_tests ?? 0)} />
        <ReadOnly label={t("evo.completed_tests")} value={String(stats.completed_tests ?? 0)} />
        <div style={{ marginTop: 8 }}>
          <button className="cc-btn" onClick={handleEvolve} disabled={evolving} style={{ whiteSpace: "nowrap" }}>
            {evolving ? t("evo.evolving") : t("evo.evolve_now")}
          </button>
          {evolveResult && (
            <span style={{ marginLeft: 8, fontSize: 13 }}>
              {evolveResult.error
                ? `Error: ${evolveResult.error}`
                : evolveResult.evolved
                  ? `New version: ${evolveResult.version_id}`
                  : t("evo.no_data")}
            </span>
          )}
        </div>
      </>)}
    </Card>
  );
}

// ── Language Settings ──────────────────────────────────────────────────
function LanguagePage({ cfg, set }) {
  const [locales, setLocales] = useState([]);
  const [presetLocales, setPresetLocales] = useState([]);
  const [activeLocale, setActiveLocale] = useState(cfg.language || "de");
  const [method, setMethod] = useState("preset");
  const [translating, setTranslating] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showPromptTranslation, setShowPromptTranslation] = useState(false);
  const pendingLocaleRef = useRef(null);

  // Fetch available locales on mount
  useEffect(() => {
    (async () => {
      const data = await api("GET", "/locales");
      if (data && !data.error) {
        setLocales(data.locales || []);
        // Prefer cfg.language (already loaded from backend config) over
        // data.active to avoid stale backend locale state (#33)
        setActiveLocale(cfg.language || data.active || "de");
        setPresetLocales(data.preset_locales || []);
      }
    })();
  }, [cfg.language]);

  const handleLocaleChange = useCallback((newLocale) => {
    setActiveLocale(newLocale);
    set("language", newLocale);
    // NOTE: Do NOT call setI18nLocale() here — global UI language only changes
    // on save, not on click (#32). The popup uses local t_preview() instead.
    setError(null);
    setSuccess(null);
    setPreview(null);
    if (newLocale !== (cfg.language || "de")) {
      pendingLocaleRef.current = newLocale;
      setShowPromptTranslation(true);
    } else {
      setShowPromptTranslation(false);
    }
  }, [cfg.language, set]);

  const handleTranslate = useCallback(async () => {
    const targetLocale = pendingLocaleRef.current || activeLocale;
    setTranslating(true);
    setError(null);
    setPreview(null);

    try {
      if (method === "preset") {
        const data = await api("POST", "/translate-prompts", {
          target_locale: targetLocale,
          method: "preset",
        });
        if (data.error) {
          setError(data.error);
        } else {
          setPreview(data.translations);
        }
      } else {
        // First fetch current prompts
        const prompts = await api("GET", "/prompts");
        if (prompts.error) { setError("Failed to load current prompts"); return; }

        const data = await api("POST", "/translate-prompts", {
          target_locale: targetLocale,
          method: "ollama",
          prompts: {
            plannerSystem: prompts.plannerSystem || "",
            replanPrompt: prompts.replanPrompt || "",
            escalationPrompt: prompts.escalationPrompt || "",
          },
        });
        if (data.error) {
          setError(data.error);
        } else {
          setPreview(data.translations);
          if (data.warnings?.length) {
            setError("Warnings: " + data.warnings.join("; "));
          }
        }
      }
    } catch (e) {
      setError(e.message || "Translation failed");
    } finally {
      setTranslating(false);
    }
  }, [method, activeLocale]);

  const handleApply = useCallback(async () => {
    if (!preview) return;
    setError(null);
    try {
      const res = await api("PUT", "/prompts", preview);
      if (res.error) {
        setError("Failed to apply prompts: " + res.error);
      } else {
        setSuccess("Prompts updated successfully!");
        setPreview(null);
        setShowPromptTranslation(false);
      }
    } catch (e) {
      setError(e.message || "Failed to apply");
    }
  }, [preview]);

  const handleDiscard = useCallback(() => {
    setPreview(null);
    setError(null);
  }, []);

  const LOCALE_LABELS = {
    de: "Deutsch (German)",
    en: "English",
    fr: "Français (French)",
    es: "Español (Spanish)",
    it: "Italiano (Italian)",
    pt: "Português (Portuguese)",
    zh: "中文 (Chinese)",
    ja: "日本語 (Japanese)",
    ko: "한국어 (Korean)",
    ru: "Русский (Russian)",
  };

  // Preview helper: shows language page in the *selected* language,
  // without changing the global UI locale (#32)
  const tl = (key, params) => tLocale(activeLocale, key, params);

  return (<>
    <Section title={tl("lang.title")} desc={tl("lang.subtitle")} />

    {/* Card 1: Active Language */}
    <Card title={tl("lang.active")}>
      <SelectInput
        label={tl("lang.system")}
        value={activeLocale}
        onChange={handleLocaleChange}
        options={locales.map(l => ({ value: l, label: LOCALE_LABELS[l] || l.toUpperCase() }))}
        desc={tl("lang.system_desc")}
      />
      <div className="cc-language-info">
        <small>
          {tl("lang.packs_info", { packs: locales.length > 0 ? locales.map(l => l.toUpperCase()).join(", ") : "...", presets: presetLocales.map(l => l.toUpperCase()).join(", ") })}
        </small>
      </div>
    </Card>

    {/* Card 2: Prompt Translation — shown when locale changes */}
    {showPromptTranslation && (
      <Card title={tl("lang.translate_title")} open>
        <div className="cc-language-translation-intro">
          {tl("lang.translate_desc", { lang: LOCALE_LABELS[activeLocale] || activeLocale.toUpperCase() })}
          {" "}{tl("lang.translate_question")}
        </div>

        <SelectInput
          label={tl("lang.method_label")}
          value={method}
          onChange={setMethod}
          options={[
            { value: "preset", label: tl("lang.method_preset") + (presetLocales.includes(activeLocale) ? "" : " — N/A") },
            { value: "ollama", label: tl("lang.method_ollama") },
          ]}
          desc={method === "preset"
            ? tl("lang.desc_preset") + " " + presetLocales.map(l => l.toUpperCase()).join(", ")
            : tl("lang.desc_ollama")
          }
        />

        <div className="cc-language-actions">
          <button
            className="cc-btn cc-btn-primary"
            onClick={handleTranslate}
            disabled={translating || (method === "preset" && !presetLocales.includes(activeLocale))}
          >
            {translating ? tl("lang.translating") : tl("lang.translate_btn")}
          </button>
          <button className="cc-btn" onClick={() => { setShowPromptTranslation(false); setPreview(null); }}>
            {tl("lang.skip")}
          </button>
        </div>

        {error && <div className="cc-toast error" style={{ marginTop: 12 }}>{error}</div>}
        {success && <div className="cc-toast success" style={{ marginTop: 12 }}>{success}</div>}
      </Card>
    )}

    {/* Card 3: Translation Preview */}
    {preview && (
      <Card title={tl("lang.preview_title")} open>
        <div className="cc-language-preview-intro">
          {tl("lang.preview_desc")}
        </div>

        {Object.entries(preview).map(([key, text]) => (
          <div key={key} className="cc-language-preview-block">
            <label className="cc-label">{
              key === "plannerSystem" ? tl("lang.planner_prompt") :
              key === "replanPrompt" ? tl("lang.replan_prompt") :
              key === "escalationPrompt" ? tl("lang.escalation_prompt") : key
            }</label>
            <pre className="cc-language-preview-text">{text.slice(0, 800)}{text.length > 800 ? "\n…" : ""}</pre>
          </div>
        ))}

        <div className="cc-language-actions" style={{ marginTop: 16 }}>
          <button className="cc-btn cc-btn-primary" onClick={handleApply}>
            {I.check} {tl("lang.accept")}
          </button>
          <button className="cc-btn" onClick={handleDiscard}>
            {tl("lang.discard")}
          </button>
        </div>
      </Card>
    )}

    {!showPromptTranslation && !preview && success && (
      <div className="cc-toast success">{success}</div>
    )}
  </>);
}

// ── General ────────────────────────────────────────────────────────────
function GeneralPage({ cfg, set }) {
  return (<>
    <Section title={t("gen.title")} desc={t("gen.desc")} />
    <Card title={t("gen.identity")}>
      <TextInput label={t("gen.owner")} value={cfg.owner_name} onChange={v => set("owner_name", v)} desc={t("gen.owner_desc")} error={!cfg.owner_name ? t("gen.owner_required") : null} />
      <SelectInput label={t("gen.mode")} value={cfg.operation_mode} onChange={v => set("operation_mode", v)} options={[{value:"auto",label:"Auto-Detect"},{value:"offline",label:"Offline (local only)"},{value:"online",label:"Online (Cloud APIs)"},{value:"hybrid",label:"Hybrid"}]} desc={t("gen.mode_desc")} />
      <ReadOnly label={t("gen.version")} value={cfg.version} desc={t("gen.version_desc")} />
    </Card>
    <Card title={t("gen.cost")}>
      <Toggle label={t("gen.cost_enabled")} value={cfg.cost_tracking_enabled} onChange={v => set("cost_tracking_enabled", v)} />
      <NumberInput label={t("gen.daily_limit")} value={cfg.daily_budget_usd} onChange={v => set("daily_budget_usd", v)} min={0} step={0.5} desc={t("gen.no_limit")} />
      <NumberInput label={t("gen.monthly_limit")} value={cfg.monthly_budget_usd} onChange={v => set("monthly_budget_usd", v)} min={0} step={1} desc={t("gen.no_limit")} />
    </Card>
    <PromptEvolutionCard />
  </>);
}

// ── Providers ──────────────────────────────────────────────────────────
const PROVIDERS = [
  { id: "ollama", name: "Ollama (Local)", keyField: null },
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
  const filterRef = useRef(null);
  // Kill browser autofill: Chrome ignores autoComplete="off", so we
  // force-clear the DOM value on mount + on any autofill event.
  useEffect(() => {
    const el = filterRef.current;
    if (!el) return;
    const kill = () => { if (el.value && !filter) { el.value = ""; setFilter(""); } };
    // Chrome fires 'animationstart' on autofill via its internal :-webkit-autofill CSS
    el.addEventListener("animationstart", kill);
    // Also force-clear after a short delay (autofill races React hydration)
    const t = setTimeout(kill, 100);
    return () => { el.removeEventListener("animationstart", kill); clearTimeout(t); };
  }, []);
  const filtered = PROVIDERS.filter(p =>
    !filter || p.name.toLowerCase().includes(filter.toLowerCase()) || p.id.includes(filter.toLowerCase())
  );
  return (<>
    <Section title={t("prov.title")} desc={t("prov.desc")} />
    <Card title={t("prov.active")}>
      <SelectInput label={t("prov.backend")} value={cfg.llm_backend_type} onChange={v => set("llm_backend_type", v)} options={PROVIDERS.map(p => ({ value: p.id, label: p.name }))} desc={t("prov.backend_desc")} />
    </Card>
    <form autoComplete="off" onSubmit={e => e.preventDefault()} className="cc-search-bar">
      <span className="cc-search-icon">{I.search}</span>
      <input ref={filterRef} className="cc-search-input" placeholder={t("prov.filter")} value={filter} onChange={e => setFilter(e.target.value)} autoComplete="new-password" role="presentation" name={"q_" + Date.now()} />
      {filter && <button type="button" className="cc-search-clear" onClick={() => setFilter("")}>{I.x}</button>}
    </form>
    {filtered.map(p => {
      if (p.id === "ollama") return (
        <Card key="ollama" title="Ollama (Local)" open={cfg.llm_backend_type === "ollama"}>
          <TextInput label={t("prov.base_url")} value={cfg.ollama?.base_url} onChange={v => set("ollama.base_url", v)} mono />
          <NumberInput label={t("prov.timeout")} value={cfg.ollama?.timeout_seconds} onChange={v => set("ollama.timeout_seconds", v)} min={10} max={600} />
          <TextInput label={t("prov.keep_alive")} value={cfg.ollama?.keep_alive} onChange={v => set("ollama.keep_alive", v)} desc={t("prov.keep_alive_desc")} />
        </Card>
      );
      return (
        <Card key={p.id} title={p.name} open={cfg.llm_backend_type === p.id}>
          <TextInput label={t("prov.api_key")} value={cfg[p.keyField]} onChange={v => set(p.keyField, v)} type="password" placeholder={`${p.id}-api-key...`} />
          {p.extra?.includes("openai_base_url") && <TextInput label={t("prov.base_url")} value={cfg.openai_base_url} onChange={v => set("openai_base_url", v)} mono desc={t("prov.openai_url_desc")} />}
          {p.extra?.includes("anthropic_max_tokens") && <NumberInput label={t("prov.max_tokens")} value={cfg.anthropic_max_tokens} onChange={v => set("anthropic_max_tokens", v)} min={256} max={32768} />}
        </Card>
      );
    })}
    {filtered.length === 0 && <div className="cc-empty">{t("prov.no_match")} &ldquo;{filter}&rdquo;</div>}
  </>);
}

// ── Models ─────────────────────────────────────────────────────────────
const ROLES = ["planner","executor","coder","embedding"];
const BACKEND_OPTIONS = [
  { value: "", label: "Global (Standard)" },
  { value: "ollama", label: "Ollama (Local)" },
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "gemini", label: "Google Gemini" },
  { value: "groq", label: "Groq" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "mistral", label: "Mistral" },
  { value: "together", label: "Together AI" },
  { value: "openrouter", label: "OpenRouter" },
  { value: "xai", label: "xAI (Grok)" },
  { value: "lmstudio", label: "LM Studio" },
];

function ModelsPage({ cfg, set, setValidationErrors }) {
  return (<>
    <Section title={t("models.title")} desc={t("models.desc")} />
    {ROLES.map(r => (
      <Card key={r} title={`${r.charAt(0).toUpperCase() + r.slice(1)} Model`}>
        <TextInput label={t("models.name")} value={cfg.models?.[r]?.name} onChange={v => set(`models.${r}.name`, v)} mono />
        <SelectInput label={t("models.backend")} value={cfg.models?.[r]?.backend || ""} onChange={v => set(`models.${r}.backend`, v)} options={BACKEND_OPTIONS} desc={t("models.backend_desc")} />
        <NumberInput label={t("models.context")} value={cfg.models?.[r]?.context_window} onChange={v => set(`models.${r}.context_window`, v)} min={1024} max={2000000} step={1024} />
        <NumberInput label={t("models.vram")} value={cfg.models?.[r]?.vram_gb} onChange={v => set(`models.${r}.vram_gb`, v)} min={0} max={80} step={0.5} />
        <SliderInput label={t("models.temp")} value={cfg.models?.[r]?.temperature} onChange={v => set(`models.${r}.temperature`, v)} min={0} max={2} step={0.05} desc={t("models.temp_desc")} />
        <SliderInput label={t("models.top_p")} value={cfg.models?.[r]?.top_p} onChange={v => set(`models.${r}.top_p`, v)} min={0} max={1} step={0.05} desc={t("models.top_p_desc")} />
        <ListInput label={t("models.strengths")} value={cfg.models?.[r]?.strengths} onChange={v => set(`models.${r}.strengths`, v)} placeholder="e.g. reasoning" />
      </Card>
    ))}
    <Card title={t("models.vision")}>
      <TextInput label={t("models.vision_standard")} value={cfg.vision_model} onChange={v => set("vision_model", v)} mono desc={t("models.vision_standard_desc")} />
      <TextInput label={t("models.vision_detail")} value={cfg.vision_model_detail} onChange={v => set("vision_model_detail", v)} mono desc={t("models.vision_detail_desc")} />
    </Card>
    <Card title={t("models.skill_overrides")} open={false}>
      <div className="cc-desc" style={{marginBottom:8}}>{t("models.skill_overrides_desc")}</div>
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
    <Section title={t("pge.title")} desc={t("pge.desc")} />
    <Card title={t("pge.planner")}>
      <NumberInput label={t("pge.max_iter")} value={cfg.planner?.max_iterations} onChange={v => set("planner.max_iterations", v)} min={1} max={50} desc={t("pge.max_iter_desc")} />
      <NumberInput label={t("pge.escalation")} value={cfg.planner?.escalation_after} onChange={v => set("planner.escalation_after", v)} min={1} max={10} desc={t("pge.escalation_desc")} />
      <SliderInput label={t("pge.temp")} value={cfg.planner?.temperature} onChange={v => set("planner.temperature", v)} min={0} max={2} step={0.05} desc={t("pge.temp_desc")} />
      <NumberInput label={t("pge.token_budget")} value={cfg.planner?.response_token_budget} onChange={v => set("planner.response_token_budget", v)} min={500} max={8000} />
    </Card>
    <Card title={t("pge.gatekeeper")}>
      <TextInput label={t("pge.policies_dir")} value={cfg.gatekeeper?.policies_dir} onChange={v => set("gatekeeper.policies_dir", v)} mono desc={t("pge.policies_desc")} />
      <SelectInput label={t("pge.risk_level")} value={cfg.gatekeeper?.default_risk_level} onChange={v => set("gatekeeper.default_risk_level", v)} options={[{value:"green",label:"🟢 Green"},{value:"yellow",label:"🟡 Yellow"},{value:"orange",label:"🟠 Orange"},{value:"red",label:"🔴 Red"}]} />
      <NumberInput label={t("pge.max_retries")} value={cfg.gatekeeper?.max_blocked_retries} onChange={v => set("gatekeeper.max_blocked_retries", v)} min={1} max={10} />
    </Card>
    <Card title={t("pge.sandbox")}>
      <SelectInput label={t("pge.sandbox_level")} value={cfg.sandbox?.level} onChange={v => set("sandbox.level", v)} options={["process","namespace","container","jobobject"]} desc={t("pge.sandbox_level_desc")} />
      <NumberInput label={t("pge.sandbox_timeout")} value={cfg.sandbox?.timeout_seconds} onChange={v => set("sandbox.timeout_seconds", v)} min={1} max={600} />
      <NumberInput label={t("pge.sandbox_memory")} value={cfg.sandbox?.max_memory_mb} onChange={v => set("sandbox.max_memory_mb", v)} min={64} max={8192} />
      <NumberInput label={t("pge.sandbox_cpu")} value={cfg.sandbox?.max_cpu_seconds} onChange={v => set("sandbox.max_cpu_seconds", v)} min={1} max={300} />
      <Toggle label={t("pge.network")} value={cfg.sandbox?.network_access} onChange={v => set("sandbox.network_access", v)} />
      <ListInput label={t("pge.allowed_paths")} value={cfg.sandbox?.allowed_paths} onChange={v => set("sandbox.allowed_paths", v)} placeholder="/path/..." />
      <JsonEditor
        label={t("pge.env_vars")} 
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
    <Section title={t("mem.title")} desc={t("mem.desc")} />
    <Card title={t("mem.indexing")}>
      <NumberInput label={t("mem.chunk_size")} value={m.chunk_size_tokens} onChange={v => set("memory.chunk_size_tokens", v)} min={100} max={2000} />
      <NumberInput label={t("mem.chunk_overlap")} value={m.chunk_overlap_tokens} onChange={v => set("memory.chunk_overlap_tokens", v)} min={0} max={500} />
      <NumberInput label={t("mem.top_k")} value={m.search_top_k} onChange={v => set("memory.search_top_k", v)} min={1} max={20} />
    </Card>
    <Card title={t("mem.search")}>
      <div className="cc-weight-sum" style={{ borderColor: sumColor }}>
        <span>{t("mem.sum")}</span>
        <span className="cc-weight-sum-val" style={{ color: sumColor }}>{sum.toFixed(2)}</span>
        <span className="cc-weight-sum-hint">{Math.abs(sum - 1.0) < 0.01 ? "✓ " + t("mem.perfect") : sum > 1.01 ? "⚠ " + t("mem.normalized") : "⚠ " + t("mem.below")}</span>
      </div>
      <SliderInput label={t("mem.vector")} value={m.weight_vector} onChange={v => set("memory.weight_vector", v)} />
      <SliderInput label={t("mem.bm25")} value={m.weight_bm25} onChange={v => set("memory.weight_bm25", v)} />
      <SliderInput label={t("mem.graph")} value={m.weight_graph} onChange={v => set("memory.weight_graph", v)} />
      <Toggle label={t("mem.dynamic")} value={m.dynamic_weighting} onChange={v => set("memory.dynamic_weighting", v)} desc={t("mem.dynamic_desc")} />
    </Card>
    <Card title={t("mem.recency")}>
      <NumberInput label={t("mem.half_life")} value={m.recency_half_life_days} onChange={v => set("memory.recency_half_life_days", v)} min={1} max={365} />
      <SliderInput label={t("mem.compact_thresh")} value={m.compaction_threshold} onChange={v => set("memory.compaction_threshold", v)} min={0.5} max={0.95} />
      <NumberInput label={t("mem.keep_last")} value={m.compaction_keep_last_n} onChange={v => set("memory.compaction_keep_last_n", v)} min={2} max={20} />
      <NumberInput label={t("mem.retention")} value={m.episodic_retention_days} onChange={v => set("memory.episodic_retention_days", v)} min={1} max={3650} desc={t("mem.retention_desc")} />
    </Card>
  </>);
}

// ── Channels ───────────────────────────────────────────────────────────
const CHANNEL_DEFS = [
  { id: "cli", label: "CLI (Terminal)", fields: [] },
  { id: "telegram", label: "Telegram", fields: [{ k: "telegram_whitelist", l: "Whitelist (User IDs)", t: "list" }] },
  { id: "slack", label: "Slack", fields: [{ k: "slack_default_channel", l: "Default Channel", t: "text" }] },
  { id: "discord", label: "Discord", fields: [{ k: "discord_channel_id", l: "Channel ID", t: "num" }] },
  { id: "whatsapp", label: "WhatsApp", fields: [
    { k: "whatsapp_default_chat", l: "Default Chat", t: "text" },
    { k: "whatsapp_phone_number_id", l: "Phone Number ID", t: "text" },
    { k: "whatsapp_webhook_port", l: "Webhook Port", t: "num", min: 1024, max: 65535 },
    { k: "whatsapp_verify_token", l: "Verify Token", t: "secret" },
    { k: "whatsapp_allowed_numbers", l: "Allowed numbers", t: "list" },
  ]},
  { id: "signal", label: "Signal", fields: [{ k: "signal_default_user", l: "Default User", t: "text" }] },
  { id: "matrix", label: "Matrix", fields: [{ k: "matrix_homeserver", l: "Homeserver URL", t: "text" }, { k: "matrix_user_id", l: "User ID", t: "text" }] },
  { id: "teams", label: "Microsoft Teams", fields: [{ k: "teams_default_channel", l: "Default Channel", t: "text" }] },
  { id: "imessage", label: "iMessage", fields: [{ k: "imessage_device_id", l: "Device ID", t: "text" }] },
  { id: "google_chat", label: "Google Chat", fields: [
    { k: "google_chat_credentials_path", l: "Credentials path", t: "text" },
    { k: "google_chat_allowed_spaces", l: "Allowed spaces", t: "list" },
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
    { k: "twitch_allowed_users", l: "Allowed users", t: "list" },
  ]},
  { id: "voice", label: "Voice (TTS/STT)", fields: [] },
];

// Fix #12: channels auto-open when enabled
function ChannelsPage({ cfg, set }) {
  const ch = cfg.channels || {};
  return (<>
    <Section title={t("ch.title")} desc={t("ch.desc")} />
    {CHANNEL_DEFS.map(cd => (
      <Card key={cd.id} title={cd.label} open={!!ch[`${cd.id}_enabled`]} forceOpen={!!ch[`${cd.id}_enabled`]}>
        <Toggle label={`Enable ${cd.label}`} value={ch[`${cd.id}_enabled`]} onChange={v => set(`channels.${cd.id}_enabled`, v)} />
        {cd.fields.map(f => {
          if (f.t === "text") return <TextInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} />;
          if (f.t === "secret") return <TextInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} type="password" />;
          if (f.t === "num") return <NumberInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} min={f.min} max={f.max} />;
          if (f.t === "list") return <ListInput key={f.k} label={f.l} value={ch[f.k]} onChange={v => set(`channels.${f.k}`, v)} />;
          return null;
        })}
      </Card>
    ))}
    <Card title={t("ch.voice_config")} open={!!ch.voice_enabled} forceOpen={!!ch.voice_enabled}>
      <Toggle label={t("ch.voice_enabled")} value={ch.voice_enabled} onChange={v => set("channels.voice_enabled", v)} desc={t("ch.voice_enabled_desc")} />
      <SelectInput label={t("ch.tts_backend")} value={ch.voice_config?.tts_backend} onChange={v => set("channels.voice_config.tts_backend", v)} options={["piper","espeak","elevenlabs"]} desc={t("ch.tts_backend_desc")} />
      {(ch.voice_config?.tts_backend || "piper") === "piper" && <>
        <SelectInput label={t("ch.piper_voice")} value={ch.voice_config?.piper_voice || "de_DE-pavoque-low"} onChange={v => set("channels.voice_config.piper_voice", v)} options={["de_DE-pavoque-low","de_DE-karlsson-low","de_DE-thorsten-high","de_DE-thorsten-medium","de_DE-thorsten_emotional-medium","de_DE-kerstin-low","de_DE-ramona-low","de_DE-eva_k-x_low"]} desc={t("ch.piper_voice_desc")} />
        <SliderInput label={t("ch.speech_rate")} value={ch.voice_config?.piper_length_scale ?? 1.0} onChange={v => set("channels.voice_config.piper_length_scale", v)} min={0.5} max={2.0} step={0.1} desc={t("ch.speech_rate_desc")} />
      </>}
      {(ch.voice_config?.tts_backend) === "elevenlabs" && <>
        <TextInput label={t("ch.elevenlabs_key")} value={ch.voice_config?.elevenlabs_api_key} onChange={v => set("channels.voice_config.elevenlabs_api_key", v)} type="password" />
        <TextInput label={t("ch.elevenlabs_voice")} value={ch.voice_config?.elevenlabs_voice_id} onChange={v => set("channels.voice_config.elevenlabs_voice_id", v)} mono />
        <TextInput label={t("ch.elevenlabs_model")} value={ch.voice_config?.elevenlabs_model} onChange={v => set("channels.voice_config.elevenlabs_model", v)} mono />
      </>}
      <Toggle label={t("ch.wake_word_enabled")} value={ch.voice_config?.wake_word_enabled} onChange={v => set("channels.voice_config.wake_word_enabled", v)} desc={t("ch.wake_word_enabled_desc")} />
      <TextInput label={t("ch.wake_word")} value={ch.voice_config?.wake_word || "jarvis"} onChange={v => set("channels.voice_config.wake_word", v)} desc={t("ch.wake_word_desc")} />
      <SelectInput label={t("ch.wake_word_backend")} value={ch.voice_config?.wake_word_backend} onChange={v => set("channels.voice_config.wake_word_backend", v)} options={["browser","vosk","porcupine"]} desc={t("ch.wake_word_backend_desc")} />
      <Toggle label={t("ch.talk_mode")} value={ch.voice_config?.talk_mode_enabled} onChange={v => set("channels.voice_config.talk_mode_enabled", v)} desc={t("ch.talk_mode_desc")} />
      <Toggle label={t("ch.auto_listen")} value={ch.voice_config?.talk_mode_auto_listen} onChange={v => set("channels.voice_config.talk_mode_auto_listen", v)} desc={t("ch.auto_listen_desc")} />
    </Card>
  </>);
}

// ── Security ───────────────────────────────────────────────────────────
function SecurityPage({ cfg, set }) {
  const s = cfg.security || {};
  return (<>
    <Section title={t("sec.title")} desc={t("sec.desc")} />
    <Card title={t("sec.limits")}>
      <NumberInput label={t("sec.max_iter")} value={s.max_iterations} onChange={v => set("security.max_iterations", v)} min={1} max={50} desc={t("sec.max_iter_desc")} />
      <NumberInput label={t("sec.max_depth")} value={s.max_sub_agent_depth} onChange={v => set("security.max_sub_agent_depth", v)} min={1} max={10} desc={t("sec.max_depth_desc")} />
    </Card>
    <Card title={t("sec.filesystem")}>
      <ListInput label={t("sec.allowed_paths")} value={s.allowed_paths} onChange={v => set("security.allowed_paths", v)} placeholder="~/.jarvis/" />
    </Card>
    <Card title={t("sec.blocked")}>
      <ListInput label={t("sec.blocked_patterns")} value={s.blocked_commands} onChange={v => set("security.blocked_commands", v)} placeholder="rm\\s+-rf\\s+/" />
    </Card>
    <Card title={t("sec.credentials")}>
      <ListInput label={t("sec.cred_patterns")} value={s.credential_patterns} onChange={v => set("security.credential_patterns", v)} placeholder="sk-[a-zA-Z0-9]{20,}" />
    </Card>
  </>);
}

// ── Executor ──────────────────────────────────────────────────────────
function ExecutorPage({ cfg, set }) {
  const e = cfg.executor || {};
  return (<>
    <Section title={t("exec.title")} desc={t("exec.desc")} />
    <Card title={t("exec.general")}>
      <NumberInput label={t("exec.llm_timeout")} value={cfg.ollama?.timeout_seconds} onChange={v => set("ollama.timeout_seconds", v)} min={10} max={600} desc={t("exec.llm_timeout_desc")} />
      <NumberInput label={t("exec.default_timeout")} value={e.default_timeout_seconds} onChange={v => set("executor.default_timeout_seconds", v)} min={5} max={600} desc={t("exec.default_timeout_desc")} />
      <NumberInput label={t("exec.max_output")} value={e.max_output_chars} onChange={v => set("executor.max_output_chars", v)} min={1000} max={100000} desc={t("exec.max_output_desc")} />
      <NumberInput label={t("exec.max_retries")} value={e.max_retries} onChange={v => set("executor.max_retries", v)} min={0} max={10} desc={t("exec.max_retries_desc")} />
      <SliderInput label={t("exec.backoff")} value={e.backoff_base_delay_seconds} onChange={v => set("executor.backoff_base_delay_seconds", v)} min={0.1} max={30} step={0.1} desc={t("exec.backoff_desc")} />
    </Card>
    <Card title={t("exec.dag")}>
      <NumberInput label={t("exec.max_parallel")} value={e.max_parallel_tools} onChange={v => set("executor.max_parallel_tools", v)} min={1} max={16} desc={t("exec.max_parallel_desc")} />
    </Card>
    <Card title={t("exec.tool_timeouts")}>
      <NumberInput label={t("exec.image_timeout")} value={e.media_analyze_image_timeout} onChange={v => set("executor.media_analyze_image_timeout", v)} min={30} max={600} desc={t("exec.image_timeout_desc")} />
      <NumberInput label={t("exec.audio_timeout")} value={e.media_transcribe_audio_timeout} onChange={v => set("executor.media_transcribe_audio_timeout", v)} min={30} max={600} />
      <NumberInput label={t("exec.text_timeout")} value={e.media_extract_text_timeout} onChange={v => set("executor.media_extract_text_timeout", v)} min={30} max={600} />
      <NumberInput label={t("exec.tts_timeout")} value={e.media_tts_timeout} onChange={v => set("executor.media_tts_timeout", v)} min={30} max={600} />
      <NumberInput label={t("exec.python_timeout")} value={e.run_python_timeout} onChange={v => set("executor.run_python_timeout", v)} min={30} max={600} />
    </Card>
  </>);
}

// ── Web Tools ──────────────────────────────────────────────────────────
function WebPage({ cfg, set }) {
  const w = cfg.web || {};
  return (<>
    <Section title={t("web.title")} desc={t("web.desc")} />
    <Card title={t("web.searxng")}>
      <TextInput label={t("web.searxng_url")} value={w.searxng_url} onChange={v => set("web.searxng_url", v)} mono placeholder="http://localhost:8888" desc={t("web.searxng_desc")} />
    </Card>
    <Card title={t("web.brave")}>
      <TextInput label={t("web.brave_key")} value={w.brave_api_key} onChange={v => set("web.brave_api_key", v)} type="password" desc={t("web.brave_desc")} />
    </Card>
    <Card title={t("web.google")}>
      <TextInput label={t("web.google_key")} value={w.google_cse_api_key} onChange={v => set("web.google_cse_api_key", v)} type="password" desc={t("web.google_desc")} />
      <TextInput label={t("web.google_cx")} value={w.google_cse_cx} onChange={v => set("web.google_cse_cx", v)} mono placeholder="a1b2c3d4e5f6g7h8i" />
    </Card>
    <Card title={t("web.jina")}>
      <TextInput label={t("web.jina_key")} value={w.jina_api_key} onChange={v => set("web.jina_api_key", v)} type="password" desc={t("web.jina_desc")} />
    </Card>
    <Card title={t("web.ddg")}>
      <Toggle label={t("web.ddg_enabled")} value={w.duckduckgo_enabled} onChange={v => set("web.duckduckgo_enabled", v)} desc={t("web.ddg_enabled_desc")} />
      <SliderInput label={t("web.ddg_delay")} value={w.ddg_min_delay_seconds} onChange={v => set("web.ddg_min_delay_seconds", v)} min={0.5} max={10} step={0.5} desc={t("web.ddg_delay_desc")} />
      <NumberInput label={t("web.ddg_wait")} value={w.ddg_ratelimit_wait_seconds} onChange={v => set("web.ddg_ratelimit_wait_seconds", v)} min={5} max={120} desc={t("web.ddg_wait_desc")} />
      <NumberInput label={t("web.ddg_cache")} value={w.ddg_cache_ttl_seconds} onChange={v => set("web.ddg_cache_ttl_seconds", v)} min={60} max={86400} desc={t("web.ddg_cache_desc")} />
    </Card>
    <Card title={t("web.fetch_limits")}>
      <NumberInput label={t("web.max_fetch")} value={w.max_fetch_bytes} onChange={v => set("web.max_fetch_bytes", v)} min={10000} max={10000000} desc={t("web.max_fetch_desc")} />
      <NumberInput label={t("web.max_text")} value={w.max_text_chars} onChange={v => set("web.max_text_chars", v)} min={1000} max={200000} desc={t("web.max_text_desc")} />
      <NumberInput label={t("web.fetch_timeout")} value={w.fetch_timeout_seconds} onChange={v => set("web.fetch_timeout_seconds", v)} min={5} max={120} />
      <NumberInput label={t("web.search_timeout")} value={w.search_timeout_seconds} onChange={v => set("web.search_timeout_seconds", v)} min={5} max={60} />
      <NumberInput label={t("web.max_results")} value={w.max_search_results} onChange={v => set("web.max_search_results", v)} min={1} max={50} />
      <NumberInput label={t("web.sar_chars")} value={w.search_and_read_max_chars} onChange={v => set("web.search_and_read_max_chars", v)} min={1000} max={50000} />
    </Card>
    <Card title={t("web.http")}>
      <NumberInput label={t("web.http_body")} value={w.http_request_max_body_bytes} onChange={v => set("web.http_request_max_body_bytes", v)} min={1024} max={10485760} desc={t("web.http_body_desc")} />
      <NumberInput label={t("web.http_timeout")} value={w.http_request_timeout_seconds} onChange={v => set("web.http_request_timeout_seconds", v)} min={1} max={120} />
      <SliderInput label={t("web.http_rate")} value={w.http_request_rate_limit_seconds} onChange={v => set("web.http_request_rate_limit_seconds", v)} min={0} max={30} step={0.5} desc={t("web.http_rate_desc")} />
    </Card>
    <Card title={t("web.domain_filter")}>
      <DomainListInput label={t("web.blocklist")} value={w.domain_blocklist} onChange={v => set("web.domain_blocklist", v)} placeholder="example.com" desc={t("web.blocklist_desc")} />
      <DomainListInput label={t("web.allowlist")} value={w.domain_allowlist} onChange={v => set("web.domain_allowlist", v)} placeholder="trusted.com" desc={t("web.allowlist_desc")} />
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
    <Section title={t("mcp.title")} desc={t("mcp.desc")} />
    <Card title={t("mcp.server_mode")}>
      <SelectInput label={t("mcp.mode")} value={mcpServers.mode || "disabled"} onChange={v => setMcpServers({...mcpServers, mode: v})} options={["disabled","stdio","http","both"]} desc={t("mcp.mode_desc")} />
      {mcpServers.mode !== "disabled" && <>
        <TextInput label={t("mcp.http_host")} value={mcpServers.http_host || "127.0.0.1"} onChange={v => setMcpServers({...mcpServers, http_host: v})} mono />
        <NumberInput label={t("mcp.http_port")} value={mcpServers.http_port || 3001} onChange={v => setMcpServers({...mcpServers, http_port: v})} min={1024} max={65535} />
        <TextInput label={t("mcp.server_name")} value={mcpServers.server_name || "jarvis"} onChange={v => setMcpServers({...mcpServers, server_name: v})} />
        <Toggle label={t("mcp.auth_required")} value={mcpServers.require_auth} onChange={v => setMcpServers({...mcpServers, require_auth: v})} />
        <TextInput label={t("mcp.auth_token")} value={mcpServers.auth_token || ""} onChange={v => setMcpServers({...mcpServers, auth_token: v})} type="password" />
        <Toggle label={t("mcp.expose_tools")} value={mcpServers.expose_tools !== false} onChange={v => setMcpServers({...mcpServers, expose_tools: v})} />
        <Toggle label={t("mcp.expose_resources")} value={mcpServers.expose_resources !== false} onChange={v => setMcpServers({...mcpServers, expose_resources: v})} />
        <Toggle label={t("mcp.expose_prompts")} value={mcpServers.expose_prompts !== false} onChange={v => setMcpServers({...mcpServers, expose_prompts: v})} />
        <Toggle label={t("mcp.sampling")} value={mcpServers.enable_sampling} onChange={v => setMcpServers({...mcpServers, enable_sampling: v})} />
      </>}
    </Card>
    <Card title={t("mcp.external")}>
      <div className="cc-desc" style={{marginBottom:16}}>{t("mcp.external_desc")}</div>
      {extServerNames.map(name => {
        const srv = extServers[name];
        return (
          <div key={name} style={{ border: "1px solid var(--border)", padding: 12, borderRadius: 6, marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <div style={{ flex: 1 }}>
                <TextInput label={t("mcp.srv_name")} value={name} onChange={v => updateExtServer(name, v, srv)} mono />
              </div>
              <button className="cc-btn cc-btn-danger" onClick={() => removeExtServer(name)} style={{ marginTop: 22 }}>{t("mcp.delete")}</button>
            </div>
            <TextInput label={t("mcp.command")} value={srv.command || ""} onChange={v => updateExtServer(name, name, { ...srv, command: v })} mono placeholder="e.g. npx, uvx, python" />
            <ListInput label={t("mcp.arguments")} value={srv.args || []} onChange={v => updateExtServer(name, name, { ...srv, args: v })} placeholder="e.g. -y, @modelcontextprotocol/server-sqlite, --db, test.db" />
            <JsonEditor
              label={t("mcp.env_vars")}
              value={srv.env || {}}
              onChange={v => updateExtServer(name, name, { ...srv, env: v })}
              onValidationError={err => setValidationErrors(prev => ({ ...prev, [`mcp_env_${name}`]: err }))}
              rows={2}
            />
            <ListInput label={t("mcp.always_allow")} value={srv.always_allow || []} onChange={v => updateExtServer(name, name, { ...srv, always_allow: v })} placeholder="e.g. read_query" />
            <Toggle label={t("mcp.disabled")} value={srv.disabled || false} onChange={v => updateExtServer(name, name, { ...srv, disabled: v })} />
          </div>
        );
      })}
      <button className="cc-btn" onClick={addExtServer}>{t("mcp.add_server")}</button>
    </Card>
    <Card title={t("mcp.a2a")}>
      <Toggle label={t("mcp.a2a_enabled")} value={a2a.enabled} onChange={v => setA2a({...a2a, enabled: v})} desc={t("mcp.a2a_desc")} />
      {a2a.enabled && <>
        <TextInput label={t("mcp.a2a_host")} value={a2a.host || "127.0.0.1"} onChange={v => setA2a({...a2a, host: v})} mono />
        <NumberInput label={t("mcp.a2a_port")} value={a2a.port || 3002} onChange={v => setA2a({...a2a, port: v})} min={1024} max={65535} />
        <TextInput label={t("mcp.a2a_name")} value={a2a.agent_name || "Jarvis"} onChange={v => setA2a({...a2a, agent_name: v})} />
        <TextInput label={t("mcp.a2a_description")} value={a2a.agent_description || ""} onChange={v => setA2a({...a2a, agent_description: v})} />
        <Toggle label={t("mcp.auth_required")} value={a2a.require_auth} onChange={v => setA2a({...a2a, require_auth: v})} />
        <TextInput label={t("mcp.auth_token")} value={a2a.auth_token || ""} onChange={v => setA2a({...a2a, auth_token: v})} type="password" />
        <NumberInput label={t("mcp.a2a_max_tasks")} value={a2a.max_tasks || 100} onChange={v => setA2a({...a2a, max_tasks: v})} min={1} max={1000} />
        <NumberInput label={t("mcp.a2a_task_timeout")} value={a2a.task_timeout_seconds || 3600} onChange={v => setA2a({...a2a, task_timeout_seconds: v})} min={60} max={86400} />
        <Toggle label={t("mcp.a2a_streaming")} value={a2a.enable_streaming} onChange={v => setA2a({...a2a, enable_streaming: v})} />
        <Toggle label={t("mcp.a2a_push")} value={a2a.enable_push} onChange={v => setA2a({...a2a, enable_push: v})} />
        <JsonEditor
          label={t("mcp.a2a_remotes")}
          value={a2a.remotes || []}
          onChange={v => setA2a({...a2a, remotes: v})}
          onValidationError={err => setValidationErrors(prev => ({ ...prev, a2a_remotes: err }))}
          rows={3}
          desc='List of remote agents: [{"endpoint": "http://...", "auth_token": ""}]'
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
    <Section title={t("cron.title")} desc={t("cron.desc")} />
    <Card title={t("cron.heartbeat")}>
      <Toggle label={t("cron.hb_enabled")} value={cfg.heartbeat?.enabled} onChange={v => set("heartbeat.enabled", v)} desc={t("cron.hb_enabled_desc")} />
      <NumberInput label={t("cron.hb_interval")} value={cfg.heartbeat?.interval_minutes} onChange={v => set("heartbeat.interval_minutes", v)} min={1} max={1440} />
      <TextInput label={t("cron.hb_file")} value={cfg.heartbeat?.checklist_file} onChange={v => set("heartbeat.checklist_file", v)} mono desc={t("cron.hb_file_desc")} />
      <SelectInput label={t("cron.hb_channel")} value={cfg.heartbeat?.channel} onChange={v => set("heartbeat.channel", v)} options={["cli","telegram","webui","slack","discord","whatsapp","mattermost","google_chat"]} />
      <TextInput label={t("cron.hb_model")} value={cfg.heartbeat?.model} onChange={v => set("heartbeat.model", v)} mono />
    </Card>
    <Card title={t("cron.jobs")}>
      {cronJobs.map((job, i) => (
        <div key={i} className="cc-cron-job">
          <div className="cc-field-row" style={{marginBottom:8}}>
            <strong className="cc-label" style={{margin:0}}>{job.name}</strong>
            <div style={{display:"flex",alignItems:"center",gap:8}}>
              <Toggle label="" value={job.enabled} onChange={v => updCron(i, "enabled", v)} />
              <button className="cc-btn-icon" onClick={() => setCronJobs(prev => prev.filter((_,j) => j!==i))} type="button" title={t("cron.delete_job")}>{I.trash}</button>
            </div>
          </div>
          <TextInput label={t("cron.name")} value={job.name} onChange={v => updCron(i, "name", v)} mono />
          <TextInput label={t("cron.schedule")} value={job.schedule} onChange={v => updCron(i, "schedule", v)} mono placeholder="0 7 * * 1-5" />
          {job.schedule && <div className="cc-cron-preview">⏰ {cronToHuman(job.schedule)}</div>}
          <TextArea label={t("cron.prompt")} value={job.prompt} onChange={v => updCron(i, "prompt", v)} rows={4} />
          <SelectInput label={t("cron.channel")} value={job.channel} onChange={v => updCron(i, "channel", v)} options={["telegram","cli","webui","slack","discord","whatsapp","mattermost","google_chat"]} />
          <TextInput label={t("cron.model")} value={job.model} onChange={v => updCron(i, "model", v)} mono />
          <TextInput label={t("cron.agent")} value={job.agent || ""} onChange={v => updCron(i, "agent", v)} placeholder={t("cron.agent_placeholder")} />
        </div>
      ))}
      <button className="cc-btn" onClick={() => setCronJobs([...cronJobs, { name: `job_${cronJobs.length+1}`, schedule: "0 9 * * *", prompt: "", channel: "cli", model: "qwen3:8b", enabled: false, agent: "" }])} type="button">{I.plus} {t("cron.add_job")}</button>
    </Card>
    <Card title={t("cron.plugins")}>
      <TextInput label={t("cron.skills_dir")} value={cfg.plugins?.skills_dir} onChange={v => set("plugins.skills_dir", v)} mono desc={t("cron.skills_desc")} />
      <Toggle label={t("cron.auto_update")} value={cfg.plugins?.auto_update} onChange={v => set("plugins.auto_update", v)} desc={t("cron.auto_update_desc")} />
    </Card>
  </>);
}

// ── Logging ────────────────────────────────────────────────────────────
function LoggingPage({ cfg, set }) {
  const l = cfg.logging || {};
  return (<>
    <Section title={t("log.title")} desc={t("log.desc")} />
    <Card title={t("log.general")}>
      <SelectInput label={t("log.level")} value={l.level} onChange={v => set("logging.level", v)} options={["DEBUG","INFO","WARNING","ERROR","CRITICAL"]} />
      <Toggle label={t("log.json")} value={l.json_logs} onChange={v => set("logging.json_logs", v)} desc={t("log.json_desc")} />
      <Toggle label={t("log.console")} value={l.console} onChange={v => set("logging.console", v)} desc={t("log.console_desc")} />
    </Card>
  </>);
}

// ── Database ───────────────────────────────────────────────────────────
function DatabasePage({ cfg, set }) {
  const d = cfg.database || {};
  return (<>
    <Section title={t("db.title")} desc={t("db.desc")} />
    <Card title={t("db.connection")}>
      <SelectInput label={t("db.backend")} value={d.backend} onChange={v => set("database.backend", v)} options={[{value:"sqlite",label:"SQLite (Standard)"},{value:"postgresql",label:"PostgreSQL"}]} />
      {d.backend === "postgresql" && (
        <>
          <TextInput label={t("db.host")} value={d.pg_host} onChange={v => set("database.pg_host", v)} mono />
          <NumberInput label={t("db.port")} value={d.pg_port} onChange={v => set("database.pg_port", v)} min={1} max={65535} />
          <TextInput label={t("db.database")} value={d.pg_dbname} onChange={v => set("database.pg_dbname", v)} mono />
          <TextInput label={t("db.user")} value={d.pg_user} onChange={v => set("database.pg_user", v)} mono />
          <TextInput label={t("db.password")} value={d.pg_password} onChange={v => set("database.pg_password", v)} type="password" />
        </>
      )}
    </Card>
    {d.backend === "postgresql" && (
      <Card title={t("db.pool")}>
        <NumberInput label={t("db.pool_min")} value={d.pg_pool_min} onChange={v => set("database.pg_pool_min", v)} min={1} max={50} />
        <NumberInput label={t("db.pool_max")} value={d.pg_pool_max} onChange={v => set("database.pg_pool_max", v)} min={1} max={100} />
      </Card>
    )}
    {d.backend !== "postgresql" && (
      <Card title={t("db.encryption")} open={false}>
        <Toggle label={t("db.encryption_enabled")} value={d.encryption_enabled} onChange={v => set("database.encryption_enabled", v)} desc={t("db.encryption_desc")} />
        {d.encryption_enabled && (
          <div className="cc-desc" style={{marginTop:8,lineHeight:1.5}}>
            {t("db.encryption_info")}
            <br/><br/>
            <strong>{t("db.encryption_prereq")}</strong> <code style={{fontSize:12,background:"var(--bg3)",padding:"2px 6px",borderRadius:4}}>pip install cognithor[encryption]</code>
          </div>
        )}
      </Card>
    )}
  </>);
}

// ── Prompts & Policies ─────────────────────────────────────────────────
// Fix #9: Reset button per textarea
function PromptsPage({ prompts, setPrompts, defaultPromptsRef }) {
  return (<>
    <Section title={t("prompts.title")} desc={t("prompts.subtitle")} />
    <Card title={t("prompts.core_title")}>
      <TextArea label={t("prompts.core_desc")} value={prompts.coreMd} onChange={v => setPrompts({...prompts, coreMd: v})} rows={16} desc={t("prompts.core_hint")} onReset={() => setPrompts({...prompts, coreMd: defaultPromptsRef.current.coreMd})} resetLabel={t("prompts.reset")} />
    </Card>
    <Card title={t("prompts.planner")} open={false}>
      <TextArea label="SYSTEM_PROMPT" value={prompts.plannerSystem} onChange={v => setPrompts({...prompts, plannerSystem: v})} rows={20} mono desc="Main prompt — variables: {owner_name}, {tools_section}, {context_section}, {current_datetime}" onReset={() => setPrompts({...prompts, plannerSystem: defaultPromptsRef.current.plannerSystem})} />
    </Card>
    <Card title={t("prompts.replan")} open={false}>
      <TextArea label="REPLAN_PROMPT" value={prompts.replanPrompt} onChange={v => setPrompts({...prompts, replanPrompt: v})} rows={12} mono desc="Prompt after tool execution — variables: {results_section}, {original_goal}" onReset={() => setPrompts({...prompts, replanPrompt: defaultPromptsRef.current.replanPrompt})} />
    </Card>
    <Card title={t("prompts.escalation")} open={false}>
      <TextArea label="ESCALATION_PROMPT" value={prompts.escalationPrompt} onChange={v => setPrompts({...prompts, escalationPrompt: v})} rows={6} mono desc="When a tool is blocked by the gatekeeper — variables: {tool}, {reason}" onReset={() => setPrompts({...prompts, escalationPrompt: defaultPromptsRef.current.escalationPrompt})} />
    </Card>
    <Card title={t("prompts.gatekeeper")} open={false}>
      <TextArea label="default.yaml" value={prompts.policyYaml} onChange={v => setPrompts({...prompts, policyYaml: v})} rows={20} mono desc="Rules for tool execution: ALLOW, INFORM, APPROVE, MASK, BLOCK" onReset={() => setPrompts({...prompts, policyYaml: defaultPromptsRef.current.policyYaml})} />
    </Card>
    <Card title={t("prompts.heartbeat")} open={false}>
      <TextArea label={t("prompts.heartbeat")} value={prompts.heartbeatMd} onChange={v => setPrompts({...prompts, heartbeatMd: v})} rows={10} desc="Periodically executed tasks" onReset={() => setPrompts({...prompts, heartbeatMd: defaultPromptsRef.current.heartbeatMd})} />
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
    <Section title={t("agents.title")} desc={t("agents.desc")} />
    {agents.map((a, i) => (
      <Card key={i} title={a.display_name || a.name} badge={a.enabled ? "active" : "inactive"}>
        <TextInput label={t("agents.name_id")} value={a.name} onChange={v => upd(i, "name", v)} mono error={!a.name ? t("agents.required") : null} />
        <TextInput label={t("agents.display_name")} value={a.display_name} onChange={v => upd(i, "display_name", v)} />
        <TextInput label={t("agents.description")} value={a.description} onChange={v => upd(i, "description", v)} />
        <TextArea label={t("agents.system_prompt")} value={a.system_prompt} onChange={v => upd(i, "system_prompt", v)} rows={6} />
        <SelectInput label={t("agents.language")} value={a.language} onChange={v => upd(i, "language", v)} options={["de","en","fr","es","it"]} />
        <NumberInput label={t("agents.priority")} value={a.priority} onChange={v => upd(i, "priority", v)} min={0} max={100} />
        <TextInput label={t("agents.preferred_model")} value={a.preferred_model} onChange={v => upd(i, "preferred_model", v)} mono />
        <SliderInput label={t("agents.temperature")} value={a.temperature} onChange={v => upd(i, "temperature", v)} min={0} max={2} step={0.1} />
        <ListInput label={t("agents.trigger_keywords")} value={a.trigger_keywords} onChange={v => upd(i, "trigger_keywords", v)} />
        <ListInput label={t("agents.blocked_tools")} value={a.blocked_tools} onChange={v => upd(i, "blocked_tools", v)} />
        <Toggle label={t("agents.enabled")} value={a.enabled} onChange={v => upd(i, "enabled", v)} />
        {a.name !== "jarvis" && <button className="cc-btn cc-btn-danger" onClick={() => setAgents(prev => prev.filter((_,j) => j!==i))} type="button">{I.trash} {t("agents.delete")}</button>}
      </Card>
    ))}
    <button className="cc-btn" onClick={add} type="button">{I.plus} {t("agents.create")}</button>
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
    <Section title={t("bind.title")} desc={t("bind.desc")} />
    {bindings.map((b, i) => (
      <Card key={i} title={b.name}>
        <TextInput label={t("bind.name")} value={b.name} onChange={v => upd(i, "name", v)} mono error={!b.name ? t("bind.required") : null} />
        <SelectInput label={t("bind.target")} value={b.target_agent} onChange={v => upd(i, "target_agent", v)} options={agents.map(a => a.name)} />
        <NumberInput label={t("bind.priority")} value={b.priority} onChange={v => upd(i, "priority", v)} min={0} max={1000} />
        <TextInput label={t("bind.description")} value={b.description} onChange={v => upd(i, "description", v)} />
        <ListInput label={t("bind.channels")} value={b.channels || []} onChange={v => upd(i, "channels", v)} placeholder="telegram, cli..." />
        <ListInput label={t("bind.prefixes")} value={b.command_prefixes || []} onChange={v => upd(i, "command_prefixes", v)} placeholder="/briefing, /code..." />
        <ListInput label={t("bind.patterns")} value={b.message_patterns || []} onChange={v => upd(i, "message_patterns", v)} />
        <Toggle label={t("bind.enabled")} value={b.enabled} onChange={v => upd(i, "enabled", v)} />
        <button className="cc-btn cc-btn-danger" onClick={() => setBindings(prev => prev.filter((_,j) => j!==i))} type="button">{I.trash} {t("bind.delete")}</button>
      </Card>
    ))}
    <button className="cc-btn" onClick={add} type="button">{I.plus} {t("bind.create")}</button>
  </>);
}

// ── System ──────────────────────────────────────────────────────────────
// Fix #14: Preset details expandable + Fix #16: Import button
function SystemPage({ cfg, onRestart, onExport, onImport, restartState, onResetDefaults }) {
  const fileRef = useRef(null);
  return (<>
    <Section title={t("sys.title")} desc={t("sys.desc")} />
    <Card title={t("sys.restart")}>
      <p className="cc-desc">{t("sys.restart_desc")}</p>
      <button
        className={`cc-btn cc-btn-restart ${restartState === "restarting" ? "pulsing" : ""}`}
        onClick={onRestart}
        disabled={restartState === "restarting"}
        type="button"
      >
        {restartState === "restarting" ? t("sys.restarting") : restartState === "done" ? t("sys.restarted") : <>{I.restart} {t("sys.restart_btn")}</>}
      </button>
    </Card>
    <Card title={t("sys.export_import")}>
      <div className="cc-export-row">
        <button className="cc-btn" onClick={onExport} type="button">{I.save} {t("sys.export")}</button>
        <button className="cc-btn" onClick={() => fileRef.current?.click()} type="button">{I.upload} {t("sys.import")}</button>
        <input ref={fileRef} type="file" accept=".json" style={{display:"none"}} onChange={e => { if (e.target.files[0]) onImport(e.target.files[0]); e.target.value = ""; }} />
      </div>
    </Card>
    <Card title={t("sys.factory_reset")}>
      <p className="cc-desc">{t("sys.factory_desc")}</p>
      <button className="cc-btn cc-btn-danger" onClick={onResetDefaults} type="button">{I.reset} {t("sys.factory_btn")}</button>
    </Card>
    <Card title={t("sys.info")}>
      <div className="cc-info-grid">
        <div className="cc-info-item"><span className="cc-info-label">{t("sys.info_version")}</span><span className="cc-info-val">{cfg.version}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">{t("sys.info_backend")}</span><span className="cc-info-val">{cfg.llm_backend_type}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">{t("sys.info_owner")}</span><span className="cc-info-val">{cfg.owner_name}</span></div>
        <div className="cc-info-item"><span className="cc-info-label">{t("sys.info_mode")}</span><span className="cc-info-val">{cfg.operation_mode}</span></div>
      </div>
    </Card>
  </>);
}

// ═══════════════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════════════

export default function App() {
  const [page, setPage] = useState("chat");
  const [cfg, setCfg] = useState(defaults());
  const [saveState, setSaveState] = useState("idle");
  const [restartState, setRestartState] = useState("idle");
  const [appStatus, setAppStatus] = useState("stopped"); // "running", "stopped", "starting", "stopping"
  const [menuOpen, setMenuOpen] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const configLoadedRef = useRef(false);
  const prevStatusRef = useRef("stopped");
  const [validationErrors, setValidationErrors] = useState({});
  const { toasts, add: toast, dismiss: dismissToast } = useToast();
  const { theme, toggle: toggleTheme } = useTheme();
  // i18n: re-render when locale changes
  const [, setI18nTick] = useState(0);
  useEffect(() => onLocaleChange(() => setI18nTick(n => n + 1)), []);

  // Styled ConfirmModal state (replaces native confirm())
  const [confirmState, setConfirmState] = useState({ open: false, title: "", message: "", danger: false, resolve: null });
  const styledConfirm = useCallback(({ title, message, danger = false }) => {
    return new Promise((resolve) => {
      setConfirmState({ open: true, title, message, danger, resolve });
    });
  }, []);
  const handleConfirmYes = useCallback(() => { confirmState.resolve?.(true); setConfirmState(s => ({ ...s, open: false })); }, [confirmState]);
  const handleConfirmNo = useCallback(() => { confirmState.resolve?.(false); setConfirmState(s => ({ ...s, open: false })); }, [confirmState]);

  // ALL useState declarations MUST come before useMemo that references them
  const [cronJobs, setCronJobs] = useState([
    { name: "morning_briefing", schedule: "0 7 * * 1-5", prompt: "Create my Morning Briefing:\n1. Today's appointments\n2. Unread emails (summary)\n3. Open tasks from yesterday\n4. Weather forecast", channel: "telegram", model: "qwen3:8b", enabled: false, agent: "" },
    { name: "weekly_review", schedule: "0 18 * * 5", prompt: "Weekly review:\n- What was completed this week?\n- What new procedures were learned?\n- What is still open?", channel: "telegram", model: "qwen3:32b", enabled: false, agent: "" },
  ]);
  const [mcpServers, setMcpServers] = useState({ mode: "disabled", external_servers: {} });
  const [a2a, setA2a] = useState({ enabled: false, host: "127.0.0.1", port: 3002, agent_name: "Jarvis" });
  const [prompts, setPrompts] = useState(defaultPrompts());
  const [promptsLoaded, setPromptsLoaded] = useState(false);
  const defaultPromptsRef = useRef(defaultPrompts());
  const [agents, setAgents] = useState([{ name: "jarvis", display_name: "Jarvis", description: "Main agent (default)", system_prompt: "", language: "de", trigger_patterns: [], trigger_keywords: [], priority: 0, allowed_tools: null, blocked_tools: [], preferred_model: "", temperature: 0.7, enabled: true }]);
  const [bindings, setBindings] = useState([]);

  // Dirty state tracking — AFTER all useState declarations
  const [savedSnapshot, setSavedSnapshot] = useState("");
  const currentSnapshot = useMemo(() => JSON.stringify({ cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }), [cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts]);
  const hasChanges = useMemo(() => {
    if (!savedSnapshot) return false;
    return currentSnapshot !== savedSnapshot;
  }, [currentSnapshot, savedSnapshot]);

  // Fix #25: Reusable config loader — called on mount AND when backend starts.
  // Only sets loaded=true when the backend is reachable and config is fetched.
  // This prevents default/empty values from being saved to disk.
  const loadAllConfig = useCallback(async () => {
    const data = await api("GET", "/config");
    if (!data || data.error) return false;
    const mergedCfg = { ...defaults(), ...data };
    setCfg(mergedCfg);
    const agentData = await api("GET", "/agents");
    const loadedAgents = agentData?.agents?.length ? agentData.agents : [];
    if (loadedAgents.length) setAgents(loadedAgents);
    const bindData = await api("GET", "/bindings");
    const loadedBindings = bindData?.bindings?.length ? bindData.bindings : [];
    if (loadedBindings.length) setBindings(loadedBindings);
    const promptData = await api("GET", "/prompts");
    let loadedPrompts = {};
    if (promptData && !promptData.error) {
      loadedPrompts = promptData;
      setPrompts(promptData);
      setPromptsLoaded(true);
      defaultPromptsRef.current = { ...defaultPrompts(), ...promptData };
    }
    const cronData = await api("GET", "/cron-jobs");
    const loadedCron = cronData?.jobs?.length ? cronData.jobs : [];
    if (loadedCron.length) setCronJobs(loadedCron);
    const mcpData = await api("GET", "/mcp-servers");
    const loadedMcp = (mcpData && !mcpData.error) ? mcpData : {};
    if (mcpData && !mcpData.error) setMcpServers(mcpData);
    const a2aData = await api("GET", "/a2a");
    const loadedA2a = (a2aData && !a2aData.error) ? a2aData : {};
    if (a2aData && !a2aData.error) setA2a(a2aData);
    configLoadedRef.current = true;
    // Capture snapshot directly from fetched data — eliminates race with React batching
    setSavedSnapshot(JSON.stringify({
      cfg: mergedCfg,
      agents: loadedAgents.length ? loadedAgents : [],
      bindings: loadedBindings.length ? loadedBindings : [],
      cronJobs: loadedCron.length ? loadedCron : [],
      mcpServers: loadedMcp,
      a2a: loadedA2a,
      prompts: loadedPrompts,
    }));
    setLoaded(true);
    // Sync UI language from backend config
    if (mergedCfg.language) setI18nLocale(mergedCfg.language);
    return true;
  }, []); // All setters are stable React refs — no deps needed

  // Track appStatus in a ref so polling can read it without re-creating the interval
  const appStatusRef = useRef(appStatus);
  useEffect(() => { appStatusRef.current = appStatus; }, [appStatus]);

  // App Status Polling — auto-reload config when backend transitions to "running"
  useEffect(() => {
    const checkStatus = async () => {
      const res = await api("GET", "/system/status");
      if (res && res.status) {
        const prev = prevStatusRef.current;
        const current = appStatusRef.current;
        prevStatusRef.current = res.status;

        // Respect transitional states: don't let poll revert "starting" → "stopped"
        // or "stopping" → "running" — only allow confirmed transitions
        if (current === "starting") {
          if (res.status === "running") {
            if (transitionTimerRef.current) { clearTimeout(transitionTimerRef.current); transitionTimerRef.current = null; }
            setAppStatus("running");
            toast("Cognithor started.", "success");
            await loadAllConfig();
          }
          // else stay in "starting" — backend not ready yet
          return;
        }
        if (current === "stopping") {
          if (res.status !== "running") {
            if (transitionTimerRef.current) { clearTimeout(transitionTimerRef.current); transitionTimerRef.current = null; }
            setAppStatus("stopped");
            toast("Cognithor stopped.", "info");
          }
          // else stay in "stopping" — backend still shutting down
          return;
        }

        setAppStatus(res.status);
        // Backend just came up — reload config from the now-running backend
        if (res.status === "running" && prev !== "running" && !configLoadedRef.current) {
          await loadAllConfig();
        }
        // Backend went down — mark config as stale so saves are blocked
        // and config will auto-reload when backend comes back
        if (res.status !== "running" && configLoadedRef.current) {
          configLoadedRef.current = false;
        }
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, [loadAllConfig, toast]);

  const toggleLockRef = useRef(false);
  const transitionTimerRef = useRef(null);
  const toggleAppStatus = useCallback(async () => {
    if (toggleLockRef.current) return;
    toggleLockRef.current = true;
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    try {
      if (appStatus === "running") {
        setAppStatus("stopping");
        const res = await api("POST", "/system/stop");
        if (res.error) {
          setAppStatus("running");
          toast(`Error stopping: ${res.error}`, "error");
        } else {
          // Safety timeout: revert after 60s if polling never confirms
          transitionTimerRef.current = setTimeout(() => {
            setAppStatus(s => s === "stopping" ? "stopped" : s);
          }, 60000);
        }
      } else {
        setAppStatus("starting");
        const res = await api("POST", "/system/start");
        if (res.error) {
          setAppStatus("stopped");
          toast(`Error starting: ${res.error}`, "error");
        } else {
          // Safety timeout: revert after 90s if polling never confirms
          transitionTimerRef.current = setTimeout(() => {
            setAppStatus(s => {
              if (s === "starting") {
                toast("Backend did not start in time — check logs.", "error");
                return "stopped";
              }
              return s;
            });
          }, 90000);
        }
      }
    } finally {
      setTimeout(() => { toggleLockRef.current = false; }, 2000);
    }
  }, [appStatus, toast]);

  // Revert all changes to last saved state
  const revertChanges = useCallback(async () => {
    if (!savedSnapshot) return;
    try {
      const snap = JSON.parse(savedSnapshot);
      if (snap.cfg) setCfg(snap.cfg);
      if (snap.agents) setAgents(snap.agents);
      if (snap.bindings) setBindings(snap.bindings);
      if (snap.cronJobs) setCronJobs(snap.cronJobs);
      if (snap.mcpServers) setMcpServers(snap.mcpServers);
      if (snap.a2a) setA2a(snap.a2a);
      if (snap.prompts) setPrompts(snap.prompts);
      toast("Changes discarded", "info");
    } catch {}
  }, [savedSnapshot, toast]);

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

  // Fix #25: Load config from API — only marks loaded when backend responds.
  // If backend is not running, loaded stays false → UI shows waiting message
  // and saves are blocked until real config has been fetched.
  useEffect(() => { loadAllConfig(); }, [loadAllConfig]);

  // Snapshot is now set directly in loadAllConfig() from fetched data,
  // eliminating the React batching race condition that caused false-positive dirty state.

  // Fix #2: Parallel save with error tracking + Fix #15: includes prompts
  // Fix #23: Block save before config loads + only send changed API keys
  const saveLockRef = useRef(false);
  const save = useCallback(async () => {
    // Debounce: prevent double-save from rapid clicks
    if (saveLockRef.current) return;
    saveLockRef.current = true;
    setTimeout(() => { saveLockRef.current = false; }, 2000);

    // Guard: don't save until config has been successfully loaded from backend.
    // This prevents default/empty values from overwriting real config on disk.
    if (!loaded || !configLoadedRef.current) {
      toast("Configuration not yet loaded from backend — please start Cognithor and wait.", "warn");
      return;
    }

    const activeErrors = Object.values(validationErrors).filter(Boolean);
    if (activeErrors.length > 0 || document.querySelector('.cc-error')) {
      toast("Please fix all red-highlighted validation errors first.", "error");
      return;
    }

    setSaveState("saving");
    const errors = [];
    const sections = ["ollama","models","gatekeeper","planner","memory","channels","sandbox","logging","security","heartbeat","plugins","dashboard","model_overrides","web","database","executor"];
    const warnings = [];
    const sectionPromises = sections.map(async (s) => {
      if (cfg[s]) {
        const r = await api("PATCH", `/config/${s}`, cfg[s]);
        if (r?.error) errors.push(`${s}: ${r.error}`);
        if (r?.warnings) warnings.push(...r.warnings);
      }
    });

    // Build top-level payload — only include API keys that were actually
    // changed by the user.  Masked values ("***") mean "not touched" and
    // are skipped so the backend keeps the existing secret.  Empty strings
    // mean "user explicitly cleared" and ARE sent.
    const API_KEY_FIELDS = [
      "openai_api_key","anthropic_api_key","gemini_api_key","groq_api_key",
      "deepseek_api_key","mistral_api_key","together_api_key","openrouter_api_key",
      "xai_api_key","cerebras_api_key","github_api_key","bedrock_api_key",
      "huggingface_api_key","moonshot_api_key",
    ];
    const topPayload = {
      owner_name: cfg.owner_name, llm_backend_type: cfg.llm_backend_type,
      operation_mode: cfg.operation_mode, cost_tracking_enabled: cfg.cost_tracking_enabled,
      daily_budget_usd: cfg.daily_budget_usd, monthly_budget_usd: cfg.monthly_budget_usd,
      vision_model: cfg.vision_model, vision_model_detail: cfg.vision_model_detail,
      openai_base_url: cfg.openai_base_url, anthropic_max_tokens: cfg.anthropic_max_tokens,
      language: cfg.language,
    };
    for (const k of API_KEY_FIELDS) {
      if (cfg[k] !== "***") topPayload[k] = cfg[k];
    }

    const topLevel = api("PATCH", "/config", topPayload).then(r => {
      if (r?.error) errors.push(`top-level: ${r.error}`);
    });
    const agentPromises = agents.map(a => api("POST", `/agents/${a.name}`, a).then(r => { if (r?.error) errors.push(`agent ${a.name}: ${r.error}`); }));
    const bindingPromises = bindings.map(b => api("POST", `/bindings/${b.name}`, b).then(r => { if (r?.error) errors.push(`binding ${b.name}: ${r.error}`); }));

    // B3: Save cronJobs, mcpServers, a2a, prompts
    // Only save prompts if they were successfully loaded from the backend.
    // Otherwise empty defaults would overwrite the real files on disk.
    const extraSaves = [
      api("PUT", "/cron-jobs", { jobs: cronJobs }).then(r => { if (r?.error) errors.push(`cron: ${r.error}`); }),
      api("PUT", "/mcp-servers", mcpServers).then(r => { if (r?.error) errors.push(`mcp: ${r.error}`); }),
      api("PUT", "/a2a", a2a).then(r => { if (r?.error) errors.push(`a2a: ${r.error}`); }),
      ...(promptsLoaded
        ? [api("PUT", "/prompts", prompts).then(r => { if (r?.error) errors.push(`prompts: ${r.error}`); })]
        : []),
    ];

    await Promise.all([...sectionPromises, topLevel, ...agentPromises, ...bindingPromises, ...extraSaves]);

    if (errors.length > 0) {
      toast(`${errors.length} error(s) saving: ${errors.slice(0, 3).join("; ")}`, "error");
      setSaveState("idle");
      // Reload config to ensure UI reflects the actual backend state after partial failure
      const data = await api("GET", "/config");
      if (data && !data.error) setCfg(prev => ({ ...prev, ...data }));
    } else {
      // Reload config from backend to reflect effective values
      // (e.g. model_post_init auto-remapping after backend change)
      const data = await api("GET", "/config");
      if (data && !data.error) setCfg(prev => ({ ...prev, ...data }));
      setSavedSnapshot(JSON.stringify({ cfg: data || cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }));
      // Apply UI locale after successful save if language changed
      const effectiveLang = (data && data.language) || cfg.language;
      if (effectiveLang) setI18nLocale(effectiveLang);
      if (warnings.length > 0) {
        toast(`Saved with warnings: ${warnings.join("; ")}`, "warn");
      } else {
        toast("Configuration saved", "success");
      }
      setSaveState("saved");
      setTimeout(() => setSaveState("idle"), 2000);
    }
  }, [cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts, promptsLoaded, loaded, currentSnapshot, validationErrors, toast]);

  // Restart
  const restart = useCallback(async () => {
    if (!await styledConfirm({ title: "Restart", message: "Really restart Cognithor? All running tasks will be terminated.", danger: true })) return;
    setRestartState("restarting");
    await save();
    await api("POST", "/config/reload");
    try { const _t = await getCCToken(); await fetch(`${API}/shutdown`, { method: "POST", headers: _t ? { Authorization: `Bearer ${_t}` } : {} }); } catch {}
    setTimeout(() => { setRestartState("done"); toast("Cognithor restarted", "success"); setTimeout(() => setRestartState("idle"), 3000); }, 3000);
  }, [save, toast, styledConfirm]);

  // Fix #15: Export includes prompts
  const onExport = () => {
    const blob = new Blob([JSON.stringify({ config: cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cognithor-config-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast("Configuration exported", "success");
  };

  // Fix #16: Import
  const onImport = async (file) => {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (!await styledConfirm({ title: "Import", message: "Import configuration from file? Current settings will be overwritten." })) return;
      if (data.config) setCfg(prev => ({ ...prev, ...data.config }));
      if (data.agents?.length) setAgents(data.agents);
      if (data.bindings) setBindings(data.bindings);
      if (data.cronJobs) setCronJobs(data.cronJobs);
      if (data.mcpServers) setMcpServers(data.mcpServers);
      if (data.a2a) setA2a(data.a2a);
      if (data.prompts) setPrompts(data.prompts);
      toast("Configuration imported — please save!", "success");
    } catch (e) {
      toast(`Import failed: ${e.message}`, "error");
    }
  };

  // Reset all settings to defaults
  const resetToDefaults = useCallback(async () => {
    if (!await styledConfirm({ title: "Factory Reset", message: "Reset ALL settings to defaults? API keys and stored data will be preserved.", danger: true })) return;
    if (!await styledConfirm({ title: "Are you sure?", message: "This cannot be undone. Are you sure?", danger: true })) return;
    setCfg(defaults());
    toast("Settings reset to defaults — please save!", "info");
  }, [toast, styledConfirm]);

  // Apply preset
  const onApplyPreset = async (name) => {
    if (!await styledConfirm({ title: "Apply Preset", message: `Apply preset "${name}"? Current settings will be overwritten.` })) return;
    const r = await api("POST", `/config/presets/${name}`);
    if (!r.error) {
      // Full reload to pick up all preset changes
      await loadAllConfig();
      toast(`Preset "${name}" applied — settings updated`, "success");
    } else {
      toast(`Preset error: ${r.error}`, "error");
    }
  };

  // Fix #23: Ctrl+S shortcut + Fix #17: keyboard nav
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        save();
      }
      // Ctrl/Cmd+1..9,0 for pages (matched by key field, not index)
      if ((e.ctrlKey || e.metaKey) && e.key >= "0" && e.key <= "9") {
        const target = PAGE_DEFS.find(p => p.key === e.key);
        if (target) { e.preventDefault(); setPage(target.id); }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [save]);

  // Auto-save drafts to localStorage (recovery after accidental close)
  useEffect(() => {
    if (!loaded) return;
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("cc-draft", JSON.stringify({ cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts, ts: Date.now() }));
      } catch {}
    }, 2000);
    return () => clearTimeout(timer);
  }, [cfg, agents, bindings, cronJobs, mcpServers, a2a, prompts, loaded]);

  // Restore draft on load if backend unavailable
  useEffect(() => {
    if (loaded) return;
    try {
      const draft = localStorage.getItem("cc-draft");
      if (draft) {
        const data = JSON.parse(draft);
        if (data.ts && Date.now() - data.ts < 86400000) { // Max 24h old
          // Don't auto-restore, but inform user
          toast("Unsaved draft found (will be restored after backend starts)", "info");
        }
      }
    } catch {}
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // B5: Warn before closing tab with unsaved changes
  useEffect(() => {
    const handler = (e) => {
      if (hasChanges) { e.preventDefault(); e.returnValue = ""; }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasChanges]);

  // Page switch is purely a render change — state persists across pages,
  // so no "unsaved changes" warning is needed on navigation.
  const changePage = useCallback((newPage) => {
    setPage(newPage);
    setMenuOpen(false);
  }, []);

  // Render current page
  const renderPage = () => {
    switch (page) {
      case "chat": return <ChatPage />;
      case "general": return <GeneralPage cfg={cfg} set={set} />;
      case "language": return <LanguagePage cfg={cfg} set={set} />;
      case "providers": return <ProvidersPage cfg={cfg} set={set} />;
      case "models": return <ModelsPage cfg={cfg} set={set} setValidationErrors={setValidationErrors} />;
      case "planner": return <PlannerPage cfg={cfg} set={set} setValidationErrors={setValidationErrors} />;
      case "executor": return <ExecutorPage cfg={cfg} set={set} />;
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
      case "workflows": return <WorkflowGraphPage />;
      case "knowledge-graph": return <KnowledgeGraphPage />;
      case "system": return <SystemPage cfg={cfg} onRestart={restart} onExport={onExport} onImport={onImport} restartState={restartState} onResetDefaults={resetToDefaults} />;
      default: return null;
    }
  };

  return (
    <div className="cc-root">
      {/* Fix #22: font-display: swap via preconnect + manual @font-face fallback */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { scrollbar-gutter: stable; }
        .cc-root {
          font-family: 'DM Sans', -apple-system, sans-serif;
          background: var(--bg);
          color: var(--text);
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
        .cc-badge.active { background: rgba(0,230,118,0.15); color: var(--success); }
        .cc-badge.inactive { background: rgba(136,136,160,0.15); color: var(--text2); }
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
        .cc-secret-badge { position: absolute; right: 8px; font-size: 11px; color: #00e676; background: rgba(0,230,118,0.1); padding: 2px 8px; border-radius: 4px; pointer-events: none; }
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
        .cc-btn-success { border-color: rgba(0,230,118,0.3); color: var(--success); }
        .cc-btn-success:hover { background: rgba(0,230,118,0.1); border-color: var(--success); }
        .cc-btn-restart { background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(108,99,255,0.1)); border-color: var(--accent); color: var(--accent); font-size: 15px; padding: 14px 28px; width: 100%; justify-content: center; }
        .cc-btn-restart:hover { background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(108,99,255,0.2)); }
        .cc-btn-restart.pulsing { animation: pulse 1.5s infinite; }

        /* Fix #18: Save bar with safe-area */
        .cc-save-bar { position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg2); border-top: 1px solid var(--border); padding: 10px 16px; padding-bottom: calc(10px + env(safe-area-inset-bottom, 0px)); display: flex; justify-content: center; align-items: center; gap: 12px; z-index: 100; backdrop-filter: blur(20px); transform: translateY(100%); opacity: 0; pointer-events: none; transition: transform 0.25s ease, opacity 0.25s ease; }
        .cc-save-bar-visible { transform: translateY(0); opacity: 1; pointer-events: auto; }
        .cc-save-btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 32px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; transition: all 0.2s; }
        .cc-save-btn.primary { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #000; }
        .cc-save-btn.primary:hover { opacity: 0.9; transform: translateY(-1px); }
        .cc-save-btn.saved { background: var(--success); color: #000; }
        .cc-save-btn-revert { background: var(--bg3); color: var(--text2); border: 1px solid var(--border); padding: 8px 18px; font-size: 13px; }
        .cc-save-btn-revert:hover { border-color: var(--warn); color: var(--warn); }
        /* Fix #1: dirty indicator */
        .cc-dirty-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); animation: pulse 2s infinite; }
        .cc-save-hint { font-size: 11px; color: var(--text2); }

        /* Fix #13: Search bar */
        .cc-search-bar { position: relative; margin-bottom: 12px; }
        .cc-search-icon { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--text2); }
        .cc-search-input { width: 100%; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 10px 10px 36px; color: var(--text); font-size: 13px; font-family: inherit; outline: none; }
        .cc-search-input:focus { border-color: var(--accent); }
        .cc-search-input:-webkit-autofill, .cc-search-input:-webkit-autofill:hover, .cc-search-input:-webkit-autofill:focus, .cc-global-search-input:-webkit-autofill, .cc-global-search-input:-webkit-autofill:hover, .cc-global-search-input:-webkit-autofill:focus { -webkit-box-shadow: 0 0 0 1000px var(--bg2) inset !important; -webkit-text-fill-color: var(--text) !important; caret-color: var(--text); transition: background-color 5000s ease-in-out 0s; }
        @keyframes cc-autofill-hack { from {} }
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
        .cc-toast-warn { background: rgba(255,215,64,0.15); border: 1px solid rgba(255,215,64,0.3); color: #ffd740; }
        .cc-toast-msg { flex: 1; }
        .cc-toast-close { background: none; border: none; color: var(--text2); cursor: pointer; padding: 2px; }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

        /* Fix #5: Spinner */
        .cc-spinner-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; gap: 16px; }
        .cc-spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
        .cc-spinner-text { color: var(--text2); font-size: 14px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .cc-btn-spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; opacity: 0.8; }

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

        /* ── Confirm Modal ──────────────────────────── */
        .cc-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 300; animation: fadeIn 0.15s ease; padding: 16px; }
        .cc-modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; max-width: 420px; width: 100%; box-shadow: 0 8px 32px rgba(0,0,0,0.5); animation: modalIn 0.2s ease; outline: none; }
        .cc-modal-title { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
        .cc-modal-message { font-size: 13px; color: var(--text2); line-height: 1.5; margin-bottom: 20px; }
        .cc-modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
        .cc-modal-btn { padding: 8px 20px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; border: 1px solid var(--border); font-family: inherit; transition: all 0.15s; }
        .cc-modal-btn-cancel { background: var(--bg3); color: var(--text); }
        .cc-modal-btn-cancel:hover { border-color: var(--text2); }
        .cc-modal-btn-confirm { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #000; border-color: transparent; }
        .cc-modal-btn-confirm:hover { opacity: 0.9; }
        .cc-modal-btn-danger { background: var(--danger); color: #fff; border-color: transparent; }
        .cc-modal-btn-danger:hover { opacity: 0.9; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes modalIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }

        /* ── Global Search ──────────────────────────── */
        .cc-global-search-trigger { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 12px; cursor: pointer; transition: all 0.15s; font-family: inherit; }
        .cc-global-search-trigger:hover { border-color: var(--accent); color: var(--accent); }
        .cc-global-search-hint { color: var(--text2); }
        .cc-global-search-kbd { font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 1px 4px; background: var(--bg); border: 1px solid var(--border); border-radius: 3px; margin-left: 4px; }
        .cc-global-search-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.18); z-index: 250; display: flex; align-items: flex-start; justify-content: center; padding-top: 15vh; animation: modalIn 0.15s ease-out; cursor: pointer; }
        [data-theme="light"] .cc-global-search-overlay { background: rgba(0,0,0,0.12); }
        .cc-global-search-dialog { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; width: 90%; max-width: 520px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.05); cursor: default; }
        [data-theme="light"] .cc-global-search-dialog { box-shadow: 0 8px 32px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06); }
        .cc-global-search-input-wrap { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text2); }
        .cc-global-search-input { flex: 1; background: transparent; border: none; color: var(--text); font-size: 15px; font-family: inherit; outline: none; }
        .cc-global-search-esc { font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 2px 6px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text2); }
        .cc-global-search-results { max-height: 320px; overflow-y: auto; }
        .cc-global-search-result { display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 12px 16px; border: none; background: transparent; color: var(--text); font-family: inherit; cursor: pointer; transition: background 0.1s; text-align: left; font-size: 13px; }
        .cc-global-search-result:hover { background: var(--bg); }
        .cc-global-search-result-label { font-weight: 600; }
        .cc-global-search-result-terms { font-size: 11px; color: var(--text2); }
        .cc-global-search-empty { padding: 20px; text-align: center; color: var(--text2); font-size: 13px; }

        /* ── Theme Toggle ───────────────────────────── */
        .cc-theme-toggle { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; color: var(--text2); cursor: pointer; transition: all 0.15s; }
        .cc-theme-toggle:hover { border-color: var(--accent); color: var(--accent); }

        /* ── Tooltip ─────────────────────────────────── */
        .cc-tooltip-trigger { display: inline-flex; align-items: center; margin-left: 4px; color: var(--text2); cursor: help; vertical-align: middle; opacity: 0.5; transition: opacity 0.15s; }
        .cc-tooltip-trigger:hover { opacity: 1; color: var(--accent); }

        /* ── Focus Styles ────────────────────────────── */
        *:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

        /* ── Reduced Motion ──────────────────────────── */
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
        }

        /* ══════════════════════════════════════════════ */
        /* Chat Page                                      */
        /* ══════════════════════════════════════════════ */
        .cc-main-chat { max-width: none !important; padding: 0 !important; margin: 0 !important; overflow: hidden; }

        /* Layout */
        .cc-chat-layout { display: flex; height: calc(100vh - 52px); width: 100%; }
        .cc-chat-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
        .cc-canvas-panel { flex: 1; border-left: 1px solid var(--border); min-width: 300px; display: flex; flex-direction: column; }
        @media (max-width: 768px) { .cc-canvas-panel { display: none; } }

        /* Chat Header */
        .cc-chat-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
        .cc-chat-header-left { display: flex; align-items: center; gap: 8px; }
        .cc-chat-header-right { display: flex; align-items: center; gap: 8px; }
        .cc-chat-title { font-size: 14px; font-weight: 600; }
        .cc-chat-status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); flex-shrink: 0; }
        .cc-chat-status-dot.cc-connected { background: var(--success); }
        .cc-chat-header-btn { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg3); color: var(--text2); font-size: 12px; cursor: pointer; font-family: inherit; transition: all 0.15s; }
        .cc-chat-header-btn:hover { border-color: var(--accent); color: var(--accent); }

        /* Message List */
        .cc-msg-list { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px; }
        .cc-msg-row { display: flex; }
        .cc-msg-row-user { justify-content: flex-end; }
        .cc-msg-row-assistant, .cc-msg-row-system { justify-content: flex-start; }
        .cc-msg-bubble { max-width: 75%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; word-break: break-word; }
        .cc-msg-user { background: var(--accent); color: #000; border-bottom-right-radius: 4px; }
        .cc-msg-assistant { background: var(--bg3); color: var(--text); border-bottom-left-radius: 4px; }
        .cc-msg-system { background: rgba(255,68,102,0.1); color: var(--danger); border: 1px solid rgba(255,68,102,0.2); font-size: 13px; }
        .cc-msg-time { font-size: 10px; color: var(--text2); margin-top: 4px; opacity: 0.7; }
        .cc-msg-user .cc-msg-time { color: rgba(0,0,0,0.5); }
        .cc-msg-streaming::after { content: '\u258B'; animation: blink 1s step-end infinite; color: var(--accent); }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        .cc-msg-content p { margin: 0 0 8px; }
        .cc-msg-content p:last-child { margin-bottom: 0; }
        .cc-msg-codeblock { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin: 8px 0; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; }
        .cc-msg-inline-code { background: var(--bg); padding: 1px 5px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }

        /* Empty State */
        .cc-msg-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; min-height: 300px; gap: 12px; color: var(--text2); }
        .cc-msg-empty-text { font-size: 16px; font-weight: 500; }

        /* Chat Input */
        .cc-chat-input { padding: 12px 16px; border-top: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
        .cc-chat-input-row { display: flex; align-items: center; gap: 8px; }
        .cc-chat-textarea { flex: 1; resize: none; background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; color: var(--text); font-size: 14px; font-family: inherit; outline: none; transition: border 0.15s; min-height: 42px; max-height: 130px; line-height: 22px; }
        .cc-chat-textarea:focus { border-color: var(--accent); }
        .cc-chat-textarea::placeholder { color: var(--text2); }
        .cc-chat-textarea:disabled { opacity: 0.5; }
        .cc-chat-input-actions { display: flex; gap: 4px; flex-shrink: 0; }
        .cc-chat-input-btn { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg3); color: var(--text2); cursor: pointer; transition: all 0.15s; }
        .cc-chat-input-btn:hover { border-color: var(--accent); color: var(--accent); }
        .cc-chat-input-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .cc-chat-send-btn { background: var(--accent); color: #000; border-color: var(--accent); }
        .cc-chat-send-btn:hover { opacity: 0.85; }
        .cc-chat-send-btn:disabled { background: var(--bg3); color: var(--text2); border-color: var(--border); opacity: 0.4; }
        .cc-recording { border-color: var(--danger) !important; color: var(--danger) !important; animation: pulse 1s infinite; }
        .cc-voice-active { border-color: var(--success) !important; color: var(--success) !important; background: rgba(0,230,118,0.1) !important; }

        /* Tool Indicator */
        .cc-tool-bar { display: flex; align-items: center; gap: 8px; padding: 6px 16px; color: var(--accent); font-size: 13px; border-top: 1px solid var(--border); background: rgba(0,212,255,0.04); flex-shrink: 0; }
        .cc-tool-spinner { width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }
        .cc-tool-label { opacity: 0.9; }

        /* Pipeline Canvas */
        .cc-pipeline { border-top: 1px solid var(--border); background: rgba(0,212,255,0.03); flex-shrink: 0; overflow: hidden; }
        .cc-pipeline-header { display: flex; justify-content: space-between; align-items: center; padding: 6px 16px; cursor: pointer; user-select: none; }
        .cc-pipeline-title { font-size: 12px; font-weight: 600; color: var(--text2); letter-spacing: 0.5px; text-transform: uppercase; }
        .cc-pipeline-chevron { font-size: 10px; color: var(--text2); }
        .cc-pipeline-svg { display: block; margin: 0 auto; padding-bottom: 8px; }

        /* Approval Banner */
        .cc-approval { padding: 10px 16px; background: rgba(255,171,64,0.08); border-top: 1px solid rgba(255,171,64,0.2); flex-shrink: 0; }
        .cc-approval-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
        .cc-approval-info { display: flex; align-items: center; gap: 6px; font-size: 13px; }
        .cc-approval-icon { color: var(--warn); display: flex; }
        .cc-approval-tool { font-weight: 600; color: var(--warn); }
        .cc-approval-reason { color: var(--text2); }
        .cc-approval-actions { display: flex; align-items: center; gap: 6px; }
        .cc-approval-toggle { background: none; border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px; font-size: 11px; color: var(--text2); cursor: pointer; font-family: inherit; }
        .cc-approval-toggle:hover { border-color: var(--accent); color: var(--accent); }
        .cc-approval-btn { padding: 5px 14px; border-radius: 6px; border: none; font-size: 12px; font-weight: 600; cursor: pointer; font-family: inherit; transition: all 0.15s; }
        .cc-approval-allow { background: var(--success); color: #000; }
        .cc-approval-allow:hover { opacity: 0.85; }
        .cc-approval-deny { background: var(--danger); color: #fff; }
        .cc-approval-deny:hover { opacity: 0.85; }
        .cc-approval-params { margin-top: 8px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text2); overflow-x: auto; white-space: pre-wrap; }

        /* Canvas */
        .cc-canvas-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
        .cc-canvas-title { font-size: 13px; font-weight: 600; }
        .cc-canvas-close { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg3); color: var(--text2); cursor: pointer; transition: all 0.15s; }
        .cc-canvas-close:hover { border-color: var(--danger); color: var(--danger); }
        .cc-canvas-frame { width: 100%; flex: 1; border: none; background: var(--bg3); }

        /* Voice Mode */
        .cc-voice-active { border-color: var(--success) !important; color: var(--success) !important; background: rgba(0,230,118,0.1) !important; }
        .cc-voice-bar { display: flex; align-items: center; gap: 8px; padding: 6px 16px; font-size: 13px; border-bottom: 1px solid var(--border); background: rgba(0,230,118,0.04); flex-shrink: 0; }
        .cc-voice-indicator { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
        .cc-voice-pulse { background: var(--success); animation: pulse 2s infinite; }
        .cc-voice-flash { background: var(--accent); animation: blink 0.5s 3; }
        .cc-voice-record { background: var(--danger); animation: pulse 0.8s infinite; }
        .cc-voice-spin { width: 10px; height: 10px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
        .cc-voice-speak { background: var(--accent); animation: pulse 1s infinite; }
        .cc-voice-label { color: var(--text2); }
        .cc-voice-transcript { color: var(--text); font-style: italic; margin-left: auto; max-width: 50%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        /* Language Settings page */
        .cc-btn { padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--text); cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }
        .cc-btn:hover { background: var(--hover); }
        .cc-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .cc-btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
        .cc-btn-primary:hover { filter: brightness(1.1); background: var(--accent); }
        .cc-language-info { margin-top: 8px; color: var(--text2); }
        .cc-language-info small { font-size: 12px; }
        .cc-language-translation-intro { color: var(--text); font-size: 14px; margin-bottom: 16px; line-height: 1.5; }
        .cc-language-actions { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
        .cc-language-preview-intro { color: var(--text2); font-size: 13px; margin-bottom: 16px; line-height: 1.5; }
        .cc-language-preview-block { margin-bottom: 16px; }
        .cc-language-preview-block .cc-label { font-weight: 600; color: var(--text); margin-bottom: 6px; display: block; font-size: 13px; }
        .cc-language-preview-text { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: var(--text2); white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; line-height: 1.5; }
        .cc-toast.error { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 10px 14px; border-radius: 8px; font-size: 13px; }
        .cc-toast.success { background: #f0fdf4; border: 1px solid #bbf7d0; color: #16a34a; padding: 10px 14px; border-radius: 8px; font-size: 13px; }
        [data-theme="dark"] .cc-toast.error { background: #451a1a; border-color: #7f1d1d; color: #fca5a5; }
        [data-theme="dark"] .cc-toast.success { background: #14532d; border-color: #166534; color: #86efac; }
      `}</style>

      {/* Toast container */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Styled Confirm Modal */}
      <ConfirmModal
        open={confirmState.open}
        title={confirmState.title}
        message={confirmState.message}
        danger={confirmState.danger}
        onConfirm={handleConfirmYes}
        onCancel={handleConfirmNo}
      />

      {/* Header */}
      <div className="cc-header" role="banner">
        <div className="cc-header-left">
          <button className="cc-menu-btn" onClick={() => setMenuOpen(!menuOpen)} type="button" aria-label="Open menu">☰</button>
          <span className="cc-logo">⚡ Cognithor</span>
        </div>
        <div className="cc-header-actions">
          <GlobalSearch onNavigate={(pageId) => { setPage(pageId); setMenuOpen(false); }} />
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          {/* Start/Stop Button */}
          <button
            className={`cc-btn-sm ${appStatus === "running" ? "cc-btn-danger" : appStatus === "starting" || appStatus === "stopping" ? "cc-btn-transitioning" : "cc-btn-success"}`}
            style={{
              width: "auto",
              padding: "0 12px",
              height: "28px",
              marginRight: "8px",
              backgroundColor: appStatus === "starting" || appStatus === "stopping" ? "rgba(255,183,77,0.1)" : appStatus === "running" ? "rgba(255,68,102,0.1)" : "rgba(0,230,118,0.1)",
              borderColor: appStatus === "starting" || appStatus === "stopping" ? "rgba(255,183,77,0.4)" : appStatus === "running" ? "rgba(255,68,102,0.4)" : "rgba(0,230,118,0.4)",
              color: appStatus === "starting" || appStatus === "stopping" ? "#ffb74d" : appStatus === "running" ? "var(--danger)" : "var(--success)"
            }}
            onClick={toggleAppStatus}
            disabled={appStatus === "starting" || appStatus === "stopping"}
            title={appStatus === "running" ? "Stop Cognithor" : "Start Cognithor"}
          >
            {appStatus === "running" ? (
              <>{I.stop} <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>{t("header.power_off")}</span></>
            ) : appStatus === "starting" ? (
              <><span className="cc-btn-spinner" /> <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>{t("header.starting")}</span></>
            ) : appStatus === "stopping" ? (
              <><span className="cc-btn-spinner" /> <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>{t("header.stopping")}</span></>
            ) : (
              <>{I.play} <span style={{marginLeft: "6px", fontSize: "12px", fontWeight: "600"}}>{t("header.power_on")}</span></>
            )}
          </button>
          
          {hasChanges && <div className="cc-dirty-dot" title={t("save.unsaved")} />}
          <span style={{ fontSize: 11, color: "var(--text2)" }}>Control Center v{cfg.version}</span>
        </div>
      </div>

      {/* Layout */}
      <div className="cc-layout">
        <div className={`cc-overlay ${menuOpen ? "visible" : ""}`} onClick={() => setMenuOpen(false)} />

        {/* Sidebar with keyboard hints */}
        <nav className={`cc-sidebar ${menuOpen ? "mobile-open" : ""}`} role="navigation" aria-label="Main navigation">
          {PAGE_DEFS.map(p => (
            <button key={p.id} className={`cc-nav-item ${page === p.id ? "active" : ""}`} onClick={() => changePage(p.id)} aria-current={page === p.id ? "page" : undefined}>
              {p.icon} {t(p.i18n)}
              {p.key && <span className="cc-nav-key">⌘{p.key}</span>}
            </button>
          ))}
        </nav>

        {/* Main */}
        <main className={`cc-main ${page === "chat" ? "cc-main-chat" : ""}`} role="main">
          {page === "chat" ? (
            renderPage()
          ) : !loaded ? (
            appStatus === "running" || appStatus === "starting" ? <Spinner /> : (
              <div className="cc-spinner-wrap">
                <span className="cc-spinner-text" style={{textAlign:"center",lineHeight:"1.6"}}>
                  {t("header.backend_stopped")}
                </span>
              </div>
            )
          ) : renderPage()}
          {page !== "chat" && <div style={{ height: 80 }} />}
        </main>
      </div>

      {/* Fix #1 + #18: Save bar — only visible when there are changes */}
      {page !== "chat" && page !== "workflows" && page !== "knowledge-graph" && (
        <div className={`cc-save-bar ${hasChanges ? "cc-save-bar-visible" : ""}`}>
          <span className="cc-save-hint">{t("save.unsaved")}</span>
          <button
            className="cc-save-btn cc-save-btn-revert"
            onClick={revertChanges}
            type="button"
          >
            {I.reset} {t("save.discard")}
          </button>
          <button
            className={`cc-save-btn ${saveState === "saved" ? "saved" : "primary"}`}
            onClick={save}
            disabled={saveState === "saving"}
            type="button"
          >
            {saveState === "saving" ? t("save.saving") : saveState === "saved" ? <>{I.check} {t("save.saved")}</> : <>{I.save} {t("save.save")}</>}
          </button>
          <span className="cc-save-hint" style={{opacity: 0.4}}>Ctrl+S</span>
        </div>
      )}
    </div>
  );
}
