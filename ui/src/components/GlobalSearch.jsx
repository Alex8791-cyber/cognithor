import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { I } from "../utils/icons";
import { t } from "../utils/i18n";

/**
 * PAGES and FIELD_INDEX define the searchable content.
 * Each entry maps a search term to a page ID.
 */
const FIELD_INDEX = [
  // General
  { page: "general", terms: ["owner", "name", "operation", "mode", "version", "cost", "budget"] },
  // Language
  { page: "language", terms: ["language", "locale", "i18n", "translation", "translate", "prompt translation", "ollama translate", "preset", "german", "english", "french", "spanish"] },
  // Providers
  { page: "providers", terms: ["provider", "backend", "ollama", "openai", "anthropic", "claude", "gemini", "groq", "deepseek", "mistral", "together", "openrouter", "xai", "grok", "cerebras", "github", "bedrock", "huggingface", "moonshot", "api key", "base url"] },
  // Models
  { page: "models", terms: ["model", "planner", "executor", "coder", "embedding", "context window", "vram", "temperature", "top p", "speed", "vision", "skill override"] },
  // PGE Trinity
  { page: "planner", terms: ["pge", "trinity", "planner", "gatekeeper", "sandbox", "iteration", "escalation", "risk", "policies", "timeout", "memory", "cpu", "network"] },
  // Executor
  { page: "executor", terms: ["executor", "timeout", "retry", "retries", "backoff", "parallel", "dag", "tool", "image analysis", "audio", "transcription", "tts", "python", "llm timeout"] },
  // Memory
  { page: "memory", terms: ["memory", "chunk", "overlap", "search", "top-k", "vector", "bm25", "graph", "weight", "recency", "compaction", "episodic", "retention"] },
  // Channels
  { page: "channels", terms: ["channel", "cli", "terminal", "telegram", "slack", "discord", "whatsapp", "signal", "matrix", "teams", "imessage", "google chat", "mattermost", "feishu", "lark", "irc", "twitch", "voice", "tts", "stt", "wake word", "talk mode", "elevenlabs", "piper"] },
  // Security
  { page: "security", terms: ["security", "iteration", "path", "blocked", "command", "credential", "pattern", "regex"] },
  // Web
  { page: "web", terms: ["web", "search", "searxng", "brave", "duckduckgo", "ddg", "fetch", "http", "request"] },
  // MCP
  { page: "mcp", terms: ["mcp", "a2a", "agent", "protocol", "server", "stdio", "http", "auth", "token", "tool", "resource", "prompt", "sampling", "remote"] },
  // Cron
  { page: "cron", terms: ["cron", "heartbeat", "job", "schedule", "plugin", "skill", "auto update"] },
  // Database
  { page: "database", terms: ["database", "sqlite", "postgresql", "postgres", "host", "port", "pool", "connection"] },
  // Logging
  { page: "logging", terms: ["logging", "log", "debug", "info", "warning", "error", "json", "console"] },
  // Prompts
  { page: "prompts", terms: ["prompt", "system prompt", "replan", "escalation", "policy", "yaml", "core.md", "heartbeat.md", "personality"] },
  // Agents
  { page: "agents", terms: ["agent", "multi-agent", "routing", "trigger", "keyword", "pattern", "priority", "model", "language"] },
  // Bindings
  { page: "bindings", terms: ["binding", "routing", "rule", "command", "prefix", "pattern", "target"] },
  // Workflows
  { page: "workflows", terms: ["workflow", "dag", "graph", "process", "automation"] },
  // Knowledge Graph
  { page: "knowledge-graph", terms: ["knowledge", "graph", "entity", "relation", "ontology"] },
  // System
  { page: "system", terms: ["system", "restart", "export", "import", "info", "version", "factory reset"] },
];

const PAGE_NAV_KEYS = {
  general: "nav.general",
  language: "nav.language",
  providers: "nav.providers",
  models: "nav.models",
  planner: "nav.planner",
  executor: "nav.executor",
  memory: "nav.memory",
  channels: "nav.channels",
  security: "nav.security",
  web: "nav.web",
  mcp: "nav.mcp",
  cron: "nav.cron",
  database: "nav.database",
  logging: "nav.logging",
  prompts: "nav.prompts",
  agents: "nav.agents",
  bindings: "nav.bindings",
  workflows: "nav.workflows",
  "knowledge-graph": "nav.knowledge_graph",
  system: "nav.system",
};

export function GlobalSearch({ onNavigate }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  // Kill browser autofill: Chrome ignores autoComplete="off", so we
  // force-clear the DOM value on mount + on any autofill event.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.value = "";
    const onAnim = () => { el.value = ""; setQuery(""); };
    el.addEventListener("animationstart", onAnim);
    const t = setTimeout(() => { if (el.value && !query) { el.value = ""; } }, 100);
    return () => { el.removeEventListener("animationstart", onAnim); clearTimeout(t); };
  }, [open]); // re-run when dialog opens

  // Ctrl+K shortcut
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setOpen(prev => {
          if (!prev) setTimeout(() => inputRef.current?.focus(), 50);
          else setQuery("");
          return !prev;
        });
      }
      if (e.key === "Escape") {
        setOpen(false);
        setQuery("");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    const matches = new Map();
    for (const entry of FIELD_INDEX) {
      for (const term of entry.terms) {
        if (term.includes(q) || q.includes(term)) {
          if (!matches.has(entry.page)) {
            const navKey = PAGE_NAV_KEYS[entry.page];
            matches.set(entry.page, {
              page: entry.page,
              label: navKey ? t(navKey) : entry.page,
              matchedTerms: [],
            });
          }
          matches.get(entry.page).matchedTerms.push(term);
        }
      }
    }
    return Array.from(matches.values()).slice(0, 8);
  }, [query]);

  const handleSelect = useCallback((pageId) => {
    onNavigate(pageId);
    setOpen(false);
    setQuery("");
  }, [onNavigate]);

  if (!open) {
    return (
      <button
        className="cc-global-search-trigger"
        onClick={() => { setOpen(true); setTimeout(() => inputRef.current?.focus(), 50); }}
        title="Search (Ctrl+K)"
        type="button"
      >
        {I.search}
        <span className="cc-global-search-hint">{t("search.hint")}</span>
        <kbd className="cc-global-search-kbd">⌘K</kbd>
      </button>
    );
  }

  return (
    <div className="cc-global-search-overlay" onClick={() => { setOpen(false); setQuery(""); }}>
      <div className="cc-global-search-dialog" onClick={(e) => e.stopPropagation()} role="search">
        <div className="cc-global-search-input-wrap">
          {I.search}
          <input
            ref={inputRef}
            className="cc-global-search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("search.placeholder")}
            autoComplete="new-password"
            role="presentation"
            name={"gs_" + Date.now()}
            autoFocus
            aria-label="Global search"
          />
          <kbd className="cc-global-search-esc">Esc</kbd>
        </div>
        {results.length > 0 && (
          <div className="cc-global-search-results">
            {results.map((r) => (
              <button
                key={r.page}
                className="cc-global-search-result"
                onClick={() => handleSelect(r.page)}
              >
                <span className="cc-global-search-result-label">{r.label}</span>
                <span className="cc-global-search-result-terms">
                  {r.matchedTerms.slice(0, 3).join(", ")}
                </span>
              </button>
            ))}
          </div>
        )}
        {query && results.length === 0 && (
          <div className="cc-global-search-empty">
            {t("search.no_results", { query })}
          </div>
        )}
      </div>
    </div>
  );
}
