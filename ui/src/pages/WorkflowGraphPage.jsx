/**
 * WorkflowGraphPage — DAG-based workflow execution graph visualization.
 *
 * Shows workflow templates, running instances, and interactive DAG graphs
 * with real-time node status rendering.
 */
import { useState } from "react";
import { useWorkflowGraph } from "../hooks/useWorkflowGraph";
import { t } from "../utils/i18n";

// ── Status colors ─────────────────────────────────────────────────────
const STATUS_COLORS = {
  pending:  { bg: "#2a2a3e", border: "#555570", text: "#8888a0" },
  waiting:  { bg: "#2a2a3e", border: "#555570", text: "#8888a0" },
  running:  { bg: "#1a2a4a", border: "#00d4ff", text: "#00d4ff" },
  success:  { bg: "#1a3a2a", border: "#00e676", text: "#00e676" },
  failure:  { bg: "#3a1a1a", border: "#ff5252", text: "#ff5252" },
  skipped:  { bg: "#2a2a2a", border: "#666", text: "#888" },
};

const NODE_W = 180;
const NODE_H = 56;
const LAYER_GAP_X = 240;
const NODE_GAP_Y = 80;

// ── DAG Layout (Kahn's topological sort into layers) ──────────────────
function layoutDag(nodeResults, nodes) {
  if (!nodes || !nodes.length) return { positioned: [], edges: [] };

  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  // Build adjacency + in-degree
  const inDeg = {};
  const children = {};
  nodes.forEach(n => {
    inDeg[n.id] = 0;
    children[n.id] = [];
  });
  nodes.forEach(n => {
    (n.depends_on || []).forEach(dep => {
      if (children[dep]) {
        children[dep].push(n.id);
        inDeg[n.id] = (inDeg[n.id] || 0) + 1;
      }
    });
  });

  // Kahn's algorithm — group by layers
  const layers = [];
  let queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
  const visited = new Set();

  while (queue.length > 0) {
    layers.push([...queue]);
    queue.forEach(id => visited.add(id));
    const next = [];
    queue.forEach(id => {
      (children[id] || []).forEach(child => {
        inDeg[child]--;
        if (inDeg[child] === 0 && !visited.has(child)) next.push(child);
      });
    });
    queue = next;
  }

  // Position nodes
  const positioned = [];
  const posMap = {};
  layers.forEach((layer, li) => {
    const x = 40 + li * LAYER_GAP_X;
    const totalH = layer.length * NODE_H + (layer.length - 1) * (NODE_GAP_Y - NODE_H);
    const startY = Math.max(20, 150 - totalH / 2);
    layer.forEach((id, ni) => {
      const y = startY + ni * NODE_GAP_Y;
      const result = nodeResults?.[id];
      const status = result?.status || "pending";
      const node = nodeMap[id] || {};
      posMap[id] = { x, y };
      positioned.push({ id, x, y, status, node, result });
    });
  });

  // Build edges
  const edges = [];
  nodes.forEach(n => {
    (n.depends_on || []).forEach(dep => {
      if (posMap[dep] && posMap[n.id]) {
        edges.push({
          from: posMap[dep],
          to: posMap[n.id],
          fromId: dep,
          toId: n.id,
          status: nodeResults?.[n.id]?.status || "pending",
        });
      }
    });
  });

  return { positioned, edges };
}

// ── SVG DAG Renderer ──────────────────────────────────────────────────
function DagGraph({ nodeResults, nodes, onNodeClick }) {
  const { positioned, edges } = layoutDag(nodeResults, nodes);
  if (!positioned.length) return <div className="cc-wf-empty">{t("wf.no_nodes")}</div>;

  const maxX = Math.max(...positioned.map(n => n.x)) + NODE_W + 40;
  const maxY = Math.max(...positioned.map(n => n.y)) + NODE_H + 40;

  return (
    <svg width={maxX} height={maxY} className="cc-wf-svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="8" markerHeight="8" orient="auto-start-reverse"
          fill="var(--text2)">
          <path d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="8" markerHeight="8" orient="auto-start-reverse"
          fill="#00d4ff">
          <path d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
      </defs>

      {/* Edges */}
      {edges.map((e, i) => {
        const x1 = e.from.x + NODE_W;
        const y1 = e.from.y + NODE_H / 2;
        const x2 = e.to.x;
        const y2 = e.to.y + NODE_H / 2;
        const mx = (x1 + x2) / 2;
        const isActive = e.status === "running";
        return (
          <path
            key={i}
            d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
            fill="none"
            stroke={isActive ? "#00d4ff" : "var(--border)"}
            strokeWidth={isActive ? 2 : 1.5}
            markerEnd={isActive ? "url(#arrow-active)" : "url(#arrow)"}
            opacity={e.status === "skipped" ? 0.3 : 0.8}
          />
        );
      })}

      {/* Nodes */}
      {positioned.map(n => {
        const c = STATUS_COLORS[n.status] || STATUS_COLORS.pending;
        const typeLabel = { tool: t("wf.type_tool"), llm: t("wf.type_llm"), condition: t("wf.type_condition"), human_approval: t("wf.type_approval") }[n.node.type] || "?";
        return (
          <g key={n.id} onClick={() => onNodeClick?.(n)}
            style={{ cursor: "pointer" }}>
            <rect x={n.x} y={n.y} width={NODE_W} height={NODE_H} rx={8}
              fill={c.bg} stroke={c.border} strokeWidth={n.status === "running" ? 2 : 1.5} />
            {n.status === "running" && (
              <rect x={n.x} y={n.y} width={NODE_W} height={NODE_H} rx={8}
                fill="none" stroke={c.border} strokeWidth={2}
                strokeDasharray="8 4" opacity={0.5}>
                <animate attributeName="stroke-dashoffset" from="0" to="-24" dur="1s" repeatCount="indefinite" />
              </rect>
            )}
            <text x={n.x + 10} y={n.y + 20} fill={c.text}
              fontSize={13} fontWeight={600} fontFamily="DM Sans, sans-serif">
              {n.node.name || n.id}
            </text>
            <text x={n.x + 10} y={n.y + 40} fill="var(--text2)"
              fontSize={11} fontFamily="DM Sans, sans-serif">
              {typeLabel}
              {n.result?.duration_ms ? ` · ${n.result.duration_ms}ms` : ""}
            </text>
            {/* Status badge */}
            <circle cx={n.x + NODE_W - 16} cy={n.y + NODE_H / 2} r={6}
              fill={c.border} opacity={0.9} />
          </g>
        );
      })}
    </svg>
  );
}

// ── Node Detail Panel ─────────────────────────────────────────────────
function NodeDetail({ node, onClose }) {
  if (!node) return null;
  const c = STATUS_COLORS[node.status] || STATUS_COLORS.pending;
  return (
    <div className="cc-wf-detail">
      <div className="cc-wf-detail-header">
        <span style={{ color: c.text, fontWeight: 600 }}>{node.node.name || node.id}</span>
        <button className="cc-wf-close" onClick={onClose}>&times;</button>
      </div>
      <div className="cc-wf-detail-row"><b>ID:</b> {node.id}</div>
      <div className="cc-wf-detail-row"><b>{t("wf.detail_type")}</b> {node.node.type}</div>
      <div className="cc-wf-detail-row"><b>{t("wf.detail_status")}</b> <span style={{ color: c.text }}>{node.status}</span></div>
      {node.result?.output && (
        <div className="cc-wf-detail-row"><b>{t("wf.detail_output")}</b>
          <pre className="cc-wf-output">{node.result.output}</pre>
        </div>
      )}
      {node.result?.error && (
        <div className="cc-wf-detail-row"><b>{t("wf.detail_error")}</b>
          <pre className="cc-wf-output" style={{ color: "#ff5252" }}>{node.result.error}</pre>
        </div>
      )}
      {node.result?.duration_ms > 0 && (
        <div className="cc-wf-detail-row"><b>{t("wf.detail_duration")}</b> {node.result.duration_ms}ms</div>
      )}
      {node.result?.retry_count > 0 && (
        <div className="cc-wf-detail-row"><b>{t("wf.detail_retries")}</b> {node.result.retry_count}</div>
      )}
      {node.node.depends_on?.length > 0 && (
        <div className="cc-wf-detail-row"><b>{t("wf.detail_depends")}</b> {node.node.depends_on.join(", ")}</div>
      )}
    </div>
  );
}

// ── Template Card ─────────────────────────────────────────────────────
function TemplateCard({ template, onStart }) {
  return (
    <div className="cc-wf-card">
      <div className="cc-wf-card-header">
        <span className="cc-wf-card-icon">{template.icon || "📋"}</span>
        <div>
          <div className="cc-wf-card-title">{template.name}</div>
          <div className="cc-wf-card-desc">{template.description}</div>
        </div>
      </div>
      <div className="cc-wf-card-footer">
        <span className="cc-wf-card-meta">
          {template.step_count} {t("wf.steps")} · {template.category}
          {template.estimated_minutes > 0 && ` · ~${template.estimated_minutes}${t("wf.min")}`}
        </span>
        <button className="cc-wf-start-btn" onClick={() => onStart(template.template_id)}>
          {t("wf.start")}
        </button>
      </div>
    </div>
  );
}

// ── Instance Row ──────────────────────────────────────────────────────
function InstanceRow({ inst }) {
  const statusStyles = {
    running: { color: "#00d4ff" },
    completed: { color: "#00e676" },
    failed: { color: "#ff5252" },
    paused: { color: "#ffd740" },
    cancelled: { color: "#888" },
    pending: { color: "#8888a0" },
  };
  return (
    <div className="cc-wf-instance">
      <span className="cc-wf-inst-name">{inst.template_name}</span>
      <span className="cc-wf-inst-progress">{inst.progress}</span>
      <span style={statusStyles[inst.status] || {}}>{inst.status}</span>
      <span className="cc-wf-inst-time">{inst.started_at?.slice(0, 16).replace("T", " ")}</span>
    </div>
  );
}

// ── DAG Run Row ───────────────────────────────────────────────────────
function DagRunRow({ run, onClick }) {
  const statusStyles = {
    success: { color: "#00e676" },
    failure: { color: "#ff5252" },
    running: { color: "#00d4ff" },
    pending: { color: "#8888a0" },
  };
  return (
    <div className="cc-wf-instance cc-wf-clickable" onClick={() => onClick(run.id)}>
      <span className="cc-wf-inst-name">{run.workflow_name || run.id}</span>
      <span className="cc-wf-inst-progress">{run.node_count} {t("wf.nodes")}</span>
      <span style={statusStyles[run.status] || {}}>{run.status}</span>
      <span className="cc-wf-inst-time">{run.started_at?.slice(0, 16).replace("T", " ")}</span>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function WorkflowGraphPage() {
  const {
    templates, instances, stats, dagRuns, selectedRun,
    loading, error, startWorkflow, fetchDagRun, setSelectedRun,
  } = useWorkflowGraph();

  const [selectedNode, setSelectedNode] = useState(null);
  const [tab, setTab] = useState("templates");

  if (loading) {
    return <div className="cc-wf-loading">{t("wf.loading")}</div>;
  }

  return (
    <div className="cc-wf-page">
      <style>{WF_STYLES}</style>

      <div className="cc-wf-header">
        <h2 className="cc-wf-title">{t("wf.title")}</h2>
        {stats && (
          <div className="cc-wf-stats">
            <span>{stats.templates || 0} {t("wf.templates")}</span>
            <span>{stats.simple?.running || 0} {t("wf.active")}</span>
            <span>{stats.simple?.completed || 0} {t("wf.completed")}</span>
            <span>{stats.dag_runs || 0} {t("wf.dag_runs")}</span>
          </div>
        )}
      </div>

      {error && <div className="cc-wf-error">{error}</div>}

      {/* Tab Navigation */}
      <div className="cc-wf-tabs">
        {[
          { id: "templates", label: t("wf.templates") },
          { id: "instances", label: `${t("wf.instances")} (${instances.length})` },
          { id: "dag", label: `${t("wf.dag_runs")} (${dagRuns.length})` },
        ].map(t => (
          <button key={t.id}
            className={`cc-wf-tab ${tab === t.id ? "active" : ""}`}
            onClick={() => { setTab(t.id); setSelectedRun(null); setSelectedNode(null); }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Templates Tab */}
      {tab === "templates" && (
        <div className="cc-wf-grid">
          {templates.length === 0 ? (
            <div className="cc-wf-empty">{t("wf.no_templates")}</div>
          ) : templates.map(t => (
            <TemplateCard key={t.template_id} template={t} onStart={startWorkflow} />
          ))}
        </div>
      )}

      {/* Instances Tab */}
      {tab === "instances" && (
        <div className="cc-wf-list">
          {instances.length === 0 ? (
            <div className="cc-wf-empty">{t("wf.no_instances")}</div>
          ) : instances.map(inst => (
            <InstanceRow key={inst.instance_id} inst={inst} />
          ))}
        </div>
      )}

      {/* DAG Runs Tab */}
      {tab === "dag" && !selectedRun && (
        <div className="cc-wf-list">
          {dagRuns.length === 0 ? (
            <div className="cc-wf-empty">{t("wf.no_dag_runs")}</div>
          ) : dagRuns.map(run => (
            <DagRunRow key={run.id} run={run} onClick={fetchDagRun} />
          ))}
        </div>
      )}

      {/* DAG Graph View */}
      {tab === "dag" && selectedRun && (
        <div className="cc-wf-graph-container">
          <div className="cc-wf-graph-toolbar">
            <button className="cc-wf-back-btn" onClick={() => { setSelectedRun(null); setSelectedNode(null); }}>
              {t("wf.back")}
            </button>
            <span className="cc-wf-graph-title">
              {selectedRun.workflow_name || selectedRun.id}
              <span style={{ color: "var(--text2)", marginLeft: 8 }}>
                ({selectedRun.status})
              </span>
            </span>
          </div>
          <div className="cc-wf-graph-scroll">
            <DagGraph
              nodeResults={selectedRun.node_results}
              nodes={selectedRun.nodes || []}
              onNodeClick={setSelectedNode}
            />
          </div>
          <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
        </div>
      )}
    </div>
  );
}

// ── Scoped CSS ────────────────────────────────────────────────────────
const WF_STYLES = `
.cc-wf-page { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
.cc-wf-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
.cc-wf-title { font-size: 22px; font-weight: 700; color: var(--text); margin: 0; }
.cc-wf-stats { display: flex; gap: 16px; font-size: 13px; color: var(--text2); }
.cc-wf-stats span { background: var(--bg2); padding: 4px 12px; border-radius: 12px; border: 1px solid var(--border); }
.cc-wf-error { background: rgba(255,82,82,0.1); border: 1px solid #ff5252; color: #ff5252; padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; }
.cc-wf-loading { text-align: center; padding: 60px 0; color: var(--text2); font-size: 15px; }

.cc-wf-tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 0; }
.cc-wf-tab { background: none; border: none; color: var(--text2); padding: 10px 18px; font-size: 14px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s; font-family: inherit; }
.cc-wf-tab:hover { color: var(--text); }
.cc-wf-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

.cc-wf-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.cc-wf-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; transition: border-color 0.15s; }
.cc-wf-card:hover { border-color: var(--accent); }
.cc-wf-card-header { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
.cc-wf-card-icon { font-size: 28px; line-height: 1; }
.cc-wf-card-title { font-size: 15px; font-weight: 600; color: var(--text); }
.cc-wf-card-desc { font-size: 12px; color: var(--text2); margin-top: 4px; line-height: 1.4; }
.cc-wf-card-footer { display: flex; justify-content: space-between; align-items: center; }
.cc-wf-card-meta { font-size: 12px; color: var(--text2); }
.cc-wf-start-btn { background: var(--accent); color: #000; border: none; padding: 6px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; font-family: inherit; }
.cc-wf-start-btn:hover { opacity: 0.85; }

.cc-wf-list { display: flex; flex-direction: column; gap: 2px; }
.cc-wf-instance { display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr; gap: 12px; padding: 12px 16px; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; align-items: center; font-size: 13px; }
.cc-wf-clickable { cursor: pointer; transition: border-color 0.15s; }
.cc-wf-clickable:hover { border-color: var(--accent); }
.cc-wf-inst-name { color: var(--text); font-weight: 500; }
.cc-wf-inst-progress { color: var(--text2); }
.cc-wf-inst-time { color: var(--text2); font-size: 12px; }
.cc-wf-empty { text-align: center; padding: 40px 0; color: var(--text2); font-size: 14px; }

.cc-wf-graph-container { position: relative; }
.cc-wf-graph-toolbar { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
.cc-wf-back-btn { background: var(--bg2); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-family: inherit; }
.cc-wf-back-btn:hover { border-color: var(--accent); }
.cc-wf-graph-title { font-size: 16px; font-weight: 600; color: var(--text); }
.cc-wf-graph-scroll { overflow: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 8px; min-height: 300px; }
.cc-wf-svg { display: block; }

.cc-wf-detail { position: absolute; top: 60px; right: 16px; width: 320px; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); z-index: 10; }
.cc-wf-detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 15px; }
.cc-wf-close { background: none; border: none; color: var(--text2); font-size: 20px; cursor: pointer; padding: 0 4px; }
.cc-wf-detail-row { font-size: 13px; color: var(--text); margin-bottom: 8px; line-height: 1.4; }
.cc-wf-detail-row b { color: var(--text2); margin-right: 6px; }
.cc-wf-output { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-size: 12px; max-height: 120px; overflow: auto; margin-top: 4px; white-space: pre-wrap; word-break: break-all; font-family: 'JetBrains Mono', monospace; }

@media (max-width: 768px) {
  .cc-wf-page { padding: 16px; }
  .cc-wf-grid { grid-template-columns: 1fr; }
  .cc-wf-instance { grid-template-columns: 1fr 1fr; }
  .cc-wf-detail { position: fixed; top: auto; bottom: 0; left: 0; right: 0; width: 100%; border-radius: 12px 12px 0 0; }
}
`;
