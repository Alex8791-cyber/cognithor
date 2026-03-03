import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { I } from "../utils/icons";

/**
 * PAGES and FIELD_INDEX define the searchable content.
 * Each entry maps a search term to a page ID.
 */
const FIELD_INDEX = [
  // General
  { page: "general", terms: ["besitzer", "owner", "name", "betriebsmodus", "operation", "mode", "version", "kosten", "cost", "budget", "dashboard", "port"] },
  // Providers
  { page: "providers", terms: ["provider", "backend", "ollama", "openai", "anthropic", "claude", "gemini", "groq", "deepseek", "mistral", "together", "openrouter", "xai", "grok", "cerebras", "github", "bedrock", "huggingface", "moonshot", "api key", "base url"] },
  // Models
  { page: "models", terms: ["modell", "model", "planner", "executor", "coder", "embedding", "context window", "vram", "temperature", "top p", "speed", "vision", "skill override"] },
  // PGE Trinity
  { page: "planner", terms: ["pge", "trinity", "planner", "gatekeeper", "executor", "sandbox", "iteration", "eskalation", "escalation", "risk", "risiko", "policies", "timeout", "memory", "cpu", "netzwerk", "network"] },
  // Memory
  { page: "memory", terms: ["memory", "chunk", "overlap", "search", "top-k", "vector", "bm25", "graph", "gewichtung", "weight", "recency", "compaction", "episodic", "retention"] },
  // Channels
  { page: "channels", terms: ["channel", "kanal", "cli", "terminal", "webui", "telegram", "slack", "discord", "whatsapp", "signal", "matrix", "teams", "imessage", "google chat", "mattermost", "feishu", "lark", "irc", "twitch", "voice", "tts", "stt", "wake word", "talk mode", "elevenlabs", "piper"] },
  // Security
  { page: "security", terms: ["sicherheit", "security", "iteration", "pfad", "path", "blockiert", "blocked", "command", "befehl", "credential", "pattern", "regex"] },
  // Web
  { page: "web", terms: ["web", "suche", "search", "searxng", "brave", "duckduckgo", "ddg"] },
  // MCP
  { page: "mcp", terms: ["mcp", "a2a", "agent", "protocol", "server", "stdio", "http", "auth", "token", "tool", "resource", "prompt", "sampling", "remote"] },
  // Cron
  { page: "cron", terms: ["cron", "heartbeat", "job", "schedule", "zeitplan", "plugin", "skill", "auto update"] },
  // Database
  { page: "database", terms: ["datenbank", "database", "sqlite", "postgresql", "postgres", "host", "port", "pool", "verbindung", "connection"] },
  // Logging
  { page: "logging", terms: ["logging", "log", "debug", "info", "warning", "error", "json", "konsole", "console"] },
  // Prompts
  { page: "prompts", terms: ["prompt", "system prompt", "replan", "eskalation", "escalation", "policy", "yaml", "core.md", "heartbeat.md", "persönlichkeit", "personality"] },
  // Agents
  { page: "agents", terms: ["agent", "multi-agent", "routing", "trigger", "keyword", "pattern", "priorität", "priority", "modell", "model", "sprache", "language"] },
  // Bindings
  { page: "bindings", terms: ["binding", "routing", "regel", "rule", "command", "prefix", "pattern", "ziel", "target"] },
  // System
  { page: "system", terms: ["system", "neustart", "restart", "export", "import", "preset", "minimal", "standard", "vollausbau", "full", "info", "version"] },
];

const PAGE_LABELS = {
  general: "Allgemein",
  providers: "LLM Provider",
  models: "Modelle",
  planner: "PGE Trinity",
  memory: "Memory",
  channels: "Channels",
  security: "Sicherheit",
  web: "Web-Tools",
  mcp: "MCP & A2A",
  cron: "Cron & Heartbeat",
  database: "Datenbank",
  logging: "Logging",
  prompts: "Prompts & Policies",
  agents: "Agenten",
  bindings: "Bindings",
  system: "System",
};

export function GlobalSearch({ onNavigate }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  // Ctrl+K shortcut
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setOpen(true);
        setTimeout(() => inputRef.current?.focus(), 50);
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
            matches.set(entry.page, {
              page: entry.page,
              label: PAGE_LABELS[entry.page],
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
        title="Suche (Ctrl+K)"
        type="button"
      >
        {I.search}
        <span className="cc-global-search-hint">Suche...</span>
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
            placeholder="Einstellung suchen..."
            autoFocus
            aria-label="Globale Suche"
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
            Keine Ergebnisse für "{query}"
          </div>
        )}
      </div>
    </div>
  );
}
