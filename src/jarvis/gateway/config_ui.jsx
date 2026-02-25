import { useState, useEffect, useCallback } from "react";

const API = "/api/v1/config";

// Simple fetch wrapper
const api = {
  get: (path) => fetch(`${API}${path}`).then((r) => r.json()),
  put: (path, data) =>
    fetch(`${API}${path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then((r) => r.json()),
  post: (path, data) =>
    fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then((r) => r.json()),
  del: (path) => fetch(`${API}${path}`, { method: "DELETE" }).then((r) => r.json()),
};

// ============================================================
// Tab Components
// ============================================================

function OverviewTab({ overview }) {
  if (!overview) return <div className="text-gray-400">Laden...</div>;
  const items = [
    ["Version", overview.version],
    ["Besitzer", overview.owner_name],
    ["LLM-Backend", overview.llm_backend],
    ["Heartbeat", overview.heartbeat_enabled ? `✅ Aktiv (alle ${overview.heartbeat_interval} Min)` : "❌ Inaktiv"],
    ["Agenten", overview.agent_count],
    ["Bindings", overview.binding_count],
    ["Sandbox", overview.sandbox_enabled ? "✅ Aktiv" : "❌ Inaktiv"],
    ["Channels", overview.channels_active?.join(", ") || "cli"],
  ];
  return (
    <div className="space-y-3">
      <h2 className="text-xl font-semibold text-white">Systemübersicht</h2>
      <div className="bg-gray-800 rounded-lg p-4 space-y-2">
        {items.map(([label, value]) => (
          <div key={label} className="flex justify-between border-b border-gray-700 pb-1">
            <span className="text-gray-400">{label}</span>
            <span className="text-white font-medium">{String(value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HeartbeatTab({ heartbeat, onSave }) {
  const [form, setForm] = useState(heartbeat || {});
  useEffect(() => { if (heartbeat) setForm(heartbeat); }, [heartbeat]);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-white">Heartbeat</h2>
      <div className="bg-gray-800 rounded-lg p-4 space-y-3">
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={form.enabled || false} onChange={(e) => set("enabled", e.target.checked)} className="w-5 h-5 rounded" />
          <span className="text-white">Heartbeat aktiviert</span>
        </label>
        <div>
          <label className="text-gray-400 text-sm">Intervall (Minuten)</label>
          <input type="number" min={1} max={1440} value={form.interval_minutes || 30} onChange={(e) => set("interval_minutes", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" />
        </div>
        <div>
          <label className="text-gray-400 text-sm">Channel</label>
          <select value={form.channel || "cli"} onChange={(e) => set("channel", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1">
            {["cli", "telegram", "slack", "discord", "webui"].map((ch) => <option key={ch} value={ch}>{ch}</option>)}
          </select>
        </div>
        <div>
          <label className="text-gray-400 text-sm">Modell</label>
          <input type="text" value={form.model || ""} onChange={(e) => set("model", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" />
        </div>
        <button onClick={() => onSave(form)} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition-colors">Speichern</button>
      </div>
    </div>
  );
}

function AgentsTab({ agents, onSave, onDelete }) {
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({});
  const empty = { name: "", display_name: "", description: "", trigger_keywords: [], priority: 0, sandbox_network: "allow", credential_scope: "", enabled: true };

  const startEdit = (agent) => { setForm(agent ? { ...agent } : { ...empty }); setEditing(agent?.name || "__new__"); };
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-white">Agenten ({agents.length})</h2>
        <button onClick={() => startEdit(null)} className="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-sm transition-colors">+ Neuer Agent</button>
      </div>
      {agents.map((a) => (
        <div key={a.name} className="bg-gray-800 rounded-lg p-3 flex justify-between items-center">
          <div>
            <span className="text-white font-medium">{a.display_name || a.name}</span>
            <span className="text-gray-500 text-sm ml-2">({a.name})</span>
            {a.credential_scope && <span className="text-yellow-400 text-xs ml-2">🔑 {a.credential_scope}</span>}
            {!a.enabled && <span className="text-red-400 text-xs ml-2">deaktiviert</span>}
          </div>
          <div className="flex gap-2">
            <button onClick={() => startEdit(a)} className="text-blue-400 hover:text-blue-300 text-sm">Bearbeiten</button>
            {a.name !== "jarvis" && <button onClick={() => onDelete(a.name)} className="text-red-400 hover:text-red-300 text-sm">Löschen</button>}
          </div>
        </div>
      ))}
      {editing && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-3 border border-blue-500">
          <h3 className="text-white font-medium">{editing === "__new__" ? "Neuer Agent" : `Agent: ${editing}`}</h3>
          {editing === "__new__" && (
            <div><label className="text-gray-400 text-sm">Name (eindeutig)</label>
              <input type="text" value={form.name || ""} onChange={(e) => set("name", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          )}
          <div><label className="text-gray-400 text-sm">Anzeigename</label>
            <input type="text" value={form.display_name || ""} onChange={(e) => set("display_name", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          <div><label className="text-gray-400 text-sm">Beschreibung</label>
            <input type="text" value={form.description || ""} onChange={(e) => set("description", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          <div><label className="text-gray-400 text-sm">Keywords (kommagetrennt)</label>
            <input type="text" value={(form.trigger_keywords || []).join(", ")} onChange={(e) => set("trigger_keywords", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          <div className="grid grid-cols-2 gap-3">
            <div><label className="text-gray-400 text-sm">Priorität</label>
              <input type="number" value={form.priority || 0} onChange={(e) => set("priority", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
            <div><label className="text-gray-400 text-sm">Netzwerk</label>
              <select value={form.sandbox_network || "allow"} onChange={(e) => set("sandbox_network", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1">
                <option value="allow">Erlaubt</option><option value="block">Blockiert</option>
              </select></div>
          </div>
          <div><label className="text-gray-400 text-sm">Credential-Scope (leer = global)</label>
            <input type="text" value={form.credential_scope || ""} onChange={(e) => set("credential_scope", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          <div className="flex gap-2">
            <button onClick={() => { onSave(form); setEditing(null); }} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition-colors">Speichern</button>
            <button onClick={() => setEditing(null)} className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded transition-colors">Abbrechen</button>
          </div>
        </div>
      )}
    </div>
  );
}

function BindingsTab({ bindings, agents, onSave, onDelete }) {
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({});
  const empty = { name: "", target_agent: "jarvis", priority: 100, command_prefixes: [], channels: null, message_patterns: null, enabled: true };

  const startEdit = (b) => { setForm(b ? { ...b } : { ...empty }); setEditing(b?.name || "__new__"); };
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-white">Bindings ({bindings.length})</h2>
        <button onClick={() => startEdit(null)} className="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-sm transition-colors">+ Neue Regel</button>
      </div>
      {bindings.map((b) => (
        <div key={b.name} className="bg-gray-800 rounded-lg p-3 flex justify-between items-center">
          <div>
            <span className="text-white font-medium">{b.name}</span>
            <span className="text-gray-400 text-sm ml-2">→ {b.target_agent}</span>
            {b.command_prefixes?.length > 0 && <span className="text-green-400 text-xs ml-2">{b.command_prefixes.join(" ")}</span>}
            {!b.enabled && <span className="text-red-400 text-xs ml-2">deaktiviert</span>}
          </div>
          <div className="flex gap-2">
            <button onClick={() => startEdit(b)} className="text-blue-400 hover:text-blue-300 text-sm">Bearbeiten</button>
            <button onClick={() => onDelete(b.name)} className="text-red-400 hover:text-red-300 text-sm">Löschen</button>
          </div>
        </div>
      ))}
      {editing && (
        <div className="bg-gray-800 rounded-lg p-4 space-y-3 border border-blue-500">
          {editing === "__new__" && (
            <div><label className="text-gray-400 text-sm">Name</label>
              <input type="text" value={form.name || ""} onChange={(e) => set("name", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          )}
          <div><label className="text-gray-400 text-sm">Ziel-Agent</label>
            <select value={form.target_agent || "jarvis"} onChange={(e) => set("target_agent", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1">
              {agents.map((a) => <option key={a.name} value={a.name}>{a.display_name || a.name}</option>)}
              {!agents.find((a) => a.name === form.target_agent) && <option value={form.target_agent}>{form.target_agent}</option>}
            </select></div>
          <div><label className="text-gray-400 text-sm">Slash-Commands (kommagetrennt)</label>
            <input type="text" value={(form.command_prefixes || []).join(", ")} onChange={(e) => set("command_prefixes", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" placeholder="/code, /debug" /></div>
          <div><label className="text-gray-400 text-sm">Channels (kommagetrennt, leer = alle)</label>
            <input type="text" value={(form.channels || []).join(", ")} onChange={(e) => { const v = e.target.value.trim(); set("channels", v ? v.split(",").map((s) => s.trim()) : null); }} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" placeholder="telegram, cli" /></div>
          <div className="grid grid-cols-2 gap-3">
            <div><label className="text-gray-400 text-sm">Priorität</label>
              <input type="number" value={form.priority || 100} onChange={(e) => set("priority", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
            <label className="flex items-center gap-2 mt-5"><input type="checkbox" checked={form.enabled !== false} onChange={(e) => set("enabled", e.target.checked)} /><span className="text-white text-sm">Aktiv</span></label>
          </div>
          <div className="flex gap-2">
            <button onClick={() => { onSave(form); setEditing(null); }} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition-colors">Speichern</button>
            <button onClick={() => setEditing(null)} className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded transition-colors">Abbrechen</button>
          </div>
        </div>
      )}
    </div>
  );
}

function SandboxTab({ sandbox, onSave }) {
  const [form, setForm] = useState(sandbox || {});
  useEffect(() => { if (sandbox) setForm(sandbox); }, [sandbox]);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-white">Sandbox</h2>
      <div className="bg-gray-800 rounded-lg p-4 space-y-3">
        <label className="flex items-center gap-3 cursor-pointer">
          <input type="checkbox" checked={form.enabled !== false} onChange={(e) => set("enabled", e.target.checked)} className="w-5 h-5 rounded" />
          <span className="text-white">Sandbox aktiviert</span>
        </label>
        <div><label className="text-gray-400 text-sm">Netzwerk</label>
          <select value={form.network || "allow"} onChange={(e) => set("network", e.target.value)} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1">
            <option value="allow">Erlaubt</option><option value="block">Blockiert</option>
          </select></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className="text-gray-400 text-sm">Max Memory (MB)</label>
            <input type="number" min={64} max={8192} value={form.max_memory_mb || 512} onChange={(e) => set("max_memory_mb", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
          <div><label className="text-gray-400 text-sm">Max Prozesse</label>
            <input type="number" min={1} max={512} value={form.max_processes || 64} onChange={(e) => set("max_processes", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
        </div>
        <div><label className="text-gray-400 text-sm">Timeout (Sekunden)</label>
          <input type="number" min={5} max={600} value={form.timeout_seconds || 30} onChange={(e) => set("timeout_seconds", parseInt(e.target.value))} className="w-full bg-gray-700 text-white rounded px-3 py-2 mt-1" /></div>
        <button onClick={() => onSave(form)} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition-colors">Speichern</button>
      </div>
    </div>
  );
}

function PresetsTab({ presets, onApply }) {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-white">Presets</h2>
      <p className="text-gray-400 text-sm">Vorkonfigurierte Profile für verschiedene Einsatzzwecke. Presets ergänzen die bestehende Konfiguration.</p>
      <div className="grid gap-3">
        {presets.map((p) => (
          <div key={p.name} className="bg-gray-800 rounded-lg p-4 flex justify-between items-start">
            <div>
              <div className="text-white font-medium">{p.description}</div>
              <div className="text-gray-400 text-sm mt-1">Agenten: {p.agents.join(", ") || "Standard"} | Heartbeat: {p.heartbeat_enabled ? "✅" : "❌"}</div>
            </div>
            <button onClick={() => onApply(p.name)} className="bg-purple-600 hover:bg-purple-500 text-white px-3 py-1 rounded text-sm transition-colors whitespace-nowrap ml-3">Anwenden</button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================
// Main App
// ============================================================

const TABS = ["Übersicht", "Heartbeat", "Agenten", "Bindings", "Sandbox", "Presets"];

export default function ConfigUI() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [heartbeat, setHeartbeat] = useState(null);
  const [agents, setAgents] = useState([]);
  const [bindings, setBindings] = useState([]);
  const [sandbox, setSandbox] = useState(null);
  const [presets, setPresets] = useState([]);
  const [toast, setToast] = useState("");

  // Demo-Daten (da kein Backend verfügbar)
  useEffect(() => {
    setOverview({
      version: "0.9.0", owner_name: "Alexander", llm_backend: "ollama",
      heartbeat_enabled: false, heartbeat_interval: 30, agent_count: 1,
      binding_count: 0, sandbox_enabled: true, channels_active: ["cli", "telegram"],
    });
    setHeartbeat({ enabled: false, interval_minutes: 30, channel: "cli", model: "qwen3:8b", checklist_file: "HEARTBEAT.md" });
    setAgents([{ name: "jarvis", display_name: "Jarvis", description: "Allgemeiner Assistent", trigger_keywords: [], priority: 0, sandbox_network: "allow", credential_scope: "", enabled: true }]);
    setSandbox({ enabled: true, network: "allow", max_memory_mb: 512, max_processes: 64, timeout_seconds: 30 });
    setPresets([
      { name: "office", description: "Büro-Assistent", agents: ["jarvis"], heartbeat_enabled: true },
      { name: "developer", description: "Developer", agents: ["jarvis", "coder", "researcher"], heartbeat_enabled: false },
      { name: "family", description: "Familien-Planer", agents: ["jarvis"], heartbeat_enabled: true },
    ]);
  }, []);

  const notify = (msg) => { setToast(msg); setTimeout(() => setToast(""), 2000); };

  const saveHeartbeat = (data) => { setHeartbeat(data); setOverview((o) => o ? { ...o, heartbeat_enabled: data.enabled, heartbeat_interval: data.interval_minutes } : o); notify("Heartbeat gespeichert"); };
  const saveAgent = (data) => { setAgents((a) => { const idx = a.findIndex((x) => x.name === data.name); if (idx >= 0) { const n = [...a]; n[idx] = data; return n; } return [...a, data]; }); setOverview((o) => o ? { ...o, agent_count: agents.length + (agents.find((a) => a.name === data.name) ? 0 : 1) } : o); notify(`Agent "${data.name}" gespeichert`); };
  const deleteAgent = (name) => { setAgents((a) => a.filter((x) => x.name !== name)); notify(`Agent "${name}" gelöscht`); };
  const saveBinding = (data) => { setBindings((b) => { const idx = b.findIndex((x) => x.name === data.name); if (idx >= 0) { const n = [...b]; n[idx] = data; return n; } return [...b, data]; }); notify(`Binding "${data.name}" gespeichert`); };
  const deleteBinding = (name) => { setBindings((b) => b.filter((x) => x.name !== name)); notify(`Binding "${name}" gelöscht`); };
  const saveSandbox = (data) => { setSandbox(data); notify("Sandbox gespeichert"); };
  const applyPreset = (name) => {
    const preset = { office: { hb: { enabled: true, interval_minutes: 30, channel: "telegram" }, agents: [{ name: "jarvis", display_name: "Jarvis", description: "Allgemeiner Assistent", trigger_keywords: [], priority: 0, sandbox_network: "allow", credential_scope: "", enabled: true }] }, developer: { hb: { enabled: false }, agents: [{ name: "jarvis", display_name: "Jarvis", description: "Allgemeiner Assistent", trigger_keywords: [], priority: 0, sandbox_network: "allow", credential_scope: "", enabled: true }, { name: "coder", display_name: "Coder", description: "Code-Spezialist", trigger_keywords: ["code", "python", "debug"], priority: 10, sandbox_network: "block", credential_scope: "coder", enabled: true }, { name: "researcher", display_name: "Researcher", description: "Web-Recherche", trigger_keywords: ["recherche", "suche"], priority: 5, sandbox_network: "allow", credential_scope: "", enabled: true }] }, family: { hb: { enabled: true, interval_minutes: 60, channel: "telegram" }, agents: [{ name: "jarvis", display_name: "Jarvis", description: "Familien-Assistent", trigger_keywords: [], priority: 0, sandbox_network: "allow", credential_scope: "", enabled: true }] } }[name];
    if (preset) {
      if (preset.hb) setHeartbeat((h) => ({ ...h, ...preset.hb }));
      setAgents(preset.agents);
      notify(`Preset "${name}" angewandt`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-3xl mx-auto p-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center text-xl">⚙</div>
          <div>
            <h1 className="text-2xl font-bold">Jarvis Konfiguration</h1>
            <p className="text-gray-400 text-sm">Heartbeat, Agenten, Bindings & Sandbox verwalten</p>
          </div>
        </div>

        <div className="flex gap-1 mb-6 overflow-x-auto">
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} className={`px-3 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${tab === i ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>{t}</button>
          ))}
        </div>

        {tab === 0 && <OverviewTab overview={overview} />}
        {tab === 1 && <HeartbeatTab heartbeat={heartbeat} onSave={saveHeartbeat} />}
        {tab === 2 && <AgentsTab agents={agents} onSave={saveAgent} onDelete={deleteAgent} />}
        {tab === 3 && <BindingsTab bindings={bindings} agents={agents} onSave={saveBinding} onDelete={deleteBinding} />}
        {tab === 4 && <SandboxTab sandbox={sandbox} onSave={saveSandbox} />}
        {tab === 5 && <PresetsTab presets={presets} onApply={applyPreset} />}

        {toast && (
          <div className="fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg animate-pulse">{toast}</div>
        )}
      </div>
    </div>
  );
}
