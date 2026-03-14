/**
 * KnowledgeGraphPage — Interactive force-directed knowledge graph visualization.
 *
 * Displays entities and relations from the semantic memory tier.
 * Supports entity search, type filtering, and detail panel.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "../utils/api";
import { t } from "../utils/i18n";

// ── Entity type colors ────────────────────────────────────────────────
const TYPE_COLORS = {
  person:       { fill: "#4fc3f7", stroke: "#0288d1" },
  organization: { fill: "#ba68c8", stroke: "#7b1fa2" },
  location:     { fill: "#81c784", stroke: "#388e3c" },
  product:      { fill: "#ffb74d", stroke: "#f57c00" },
  concept:      { fill: "#90a4ae", stroke: "#546e7a" },
  unknown:      { fill: "#78909c", stroke: "#455a64" },
};

const NODE_R = 24;

// ── Force simulation (simplified spring-electric) ─────────────────────
function layoutForceDirected(entities, relations, width, height) {
  if (!entities.length) return { nodes: [], links: [] };

  const nodeMap = {};
  const nodes = entities.map((e, i) => {
    const angle = (2 * Math.PI * i) / entities.length;
    const r = Math.min(width, height) * 0.35;
    const node = {
      id: e.id,
      name: e.name,
      type: e.type || "unknown",
      confidence: e.confidence || 0.5,
      x: width / 2 + r * Math.cos(angle) + (Math.random() - 0.5) * 40,
      y: height / 2 + r * Math.sin(angle) + (Math.random() - 0.5) * 40,
      vx: 0, vy: 0,
      entity: e,
    };
    nodeMap[e.id] = node;
    return node;
  });

  const links = relations
    .filter(r => nodeMap[r.source_entity] && nodeMap[r.target_entity])
    .map(r => ({
      source: nodeMap[r.source_entity],
      target: nodeMap[r.target_entity],
      type: r.relation_type,
      confidence: r.confidence || 0.5,
      relation: r,
    }));

  // Simple force simulation (50 iterations)
  const REPULSION = 5000;
  const ATTRACTION = 0.005;
  const DAMPING = 0.9;
  const CENTER_PULL = 0.01;

  for (let iter = 0; iter < 50; iter++) {
    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = REPULSION / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        nodes[i].vx -= fx;
        nodes[i].vy -= fy;
        nodes[j].vx += fx;
        nodes[j].vy += fy;
      }
    }

    // Attraction along links
    for (const link of links) {
      const dx = link.target.x - link.source.x;
      const dy = link.target.y - link.source.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const force = ATTRACTION * dist;
      const fx = dx * force;
      const fy = dy * force;
      link.source.vx += fx;
      link.source.vy += fy;
      link.target.vx -= fx;
      link.target.vy -= fy;
    }

    // Center pull
    for (const node of nodes) {
      node.vx += (width / 2 - node.x) * CENTER_PULL;
      node.vy += (height / 2 - node.y) * CENTER_PULL;
    }

    // Apply velocities
    for (const node of nodes) {
      node.vx *= DAMPING;
      node.vy *= DAMPING;
      node.x += node.vx;
      node.y += node.vy;
      node.x = Math.max(NODE_R + 10, Math.min(width - NODE_R - 10, node.x));
      node.y = Math.max(NODE_R + 10, Math.min(height - NODE_R - 10, node.y));
    }
  }

  return { nodes, links };
}

// ── Graph SVG Renderer ────────────────────────────────────────────────
function GraphRenderer({ entities, relations, width, height, onNodeClick, selectedId }) {
  const { nodes, links } = layoutForceDirected(entities, relations, width, height);

  if (!nodes.length) {
    return (
      <div className="cc-kg-empty">
        <svg width="40" height="40" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" style={{ opacity: 0.3, marginBottom: 12 }}>
          <circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/>
          <path d="M6 7l4 4M14 13l4-6M6 17l4-4"/>
        </svg>
        <div>{t("kg.empty_title")}</div>
        <div style={{ fontSize: 12, marginTop: 6, opacity: 0.6 }}>
          {t("kg.empty_desc")}
        </div>
      </div>
    );
  }

  return (
    <svg width={width} height={height} className="cc-kg-svg">
      <defs>
        <marker id="kg-arrow" viewBox="0 0 10 10" refX="28" refY="5"
          markerWidth="6" markerHeight="6" orient="auto" fill="var(--text2)">
          <path d="M 0 0 L 10 5 L 0 10 z" />
        </marker>
      </defs>

      {/* Links */}
      {links.map((link, i) => (
        <g key={i}>
          <line
            x1={link.source.x} y1={link.source.y}
            x2={link.target.x} y2={link.target.y}
            stroke="var(--border)" strokeWidth={1 + link.confidence}
            markerEnd="url(#kg-arrow)"
            opacity={0.6}
          />
          <text
            x={(link.source.x + link.target.x) / 2}
            y={(link.source.y + link.target.y) / 2 - 6}
            fill="var(--text2)" fontSize={9} textAnchor="middle"
            fontFamily="DM Sans, sans-serif"
          >
            {link.type}
          </text>
        </g>
      ))}

      {/* Nodes */}
      {nodes.map(node => {
        const colors = TYPE_COLORS[node.type] || TYPE_COLORS.unknown;
        const isSelected = selectedId === node.id;
        return (
          <g key={node.id} onClick={() => onNodeClick?.(node)}
            style={{ cursor: "pointer" }}>
            <circle
              cx={node.x} cy={node.y} r={NODE_R}
              fill={colors.fill} stroke={isSelected ? "#fff" : colors.stroke}
              strokeWidth={isSelected ? 3 : 1.5}
              opacity={0.7 + node.confidence * 0.3}
            />
            <text
              x={node.x} y={node.y + NODE_R + 14}
              fill="var(--text)" fontSize={11} textAnchor="middle"
              fontWeight={500} fontFamily="DM Sans, sans-serif"
            >
              {node.name.length > 16 ? node.name.slice(0, 14) + "…" : node.name}
            </text>
            <text
              x={node.x} y={node.y + 4}
              fill="#fff" fontSize={10} textAnchor="middle"
              fontWeight={600} fontFamily="DM Sans, sans-serif"
            >
              {node.type.slice(0, 3).toUpperCase()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Entity Detail Panel ───────────────────────────────────────────────
function EntityDetail({ entity, relations, onClose }) {
  if (!entity) return null;
  const colors = TYPE_COLORS[entity.type] || TYPE_COLORS.unknown;
  return (
    <div className="cc-kg-detail">
      <div className="cc-kg-detail-header">
        <span style={{ color: colors.fill, fontWeight: 600 }}>{entity.name}</span>
        <button className="cc-kg-close" onClick={onClose}>&times;</button>
      </div>
      <div className="cc-kg-detail-row"><b>{t("kg.id_label")}</b> {entity.id?.slice(0, 12)}</div>
      <div className="cc-kg-detail-row"><b>{t("kg.type_label")}</b> {entity.type}</div>
      <div className="cc-kg-detail-row">
        <b>{t("kg.confidence_label")}</b> {Math.round((entity.confidence || 0) * 100)}%
      </div>
      {entity.attributes && Object.keys(entity.attributes).length > 0 && (
        <div className="cc-kg-detail-row">
          <b>{t("kg.attributes_label")}</b>
          <pre className="cc-kg-output">{JSON.stringify(entity.attributes, null, 2)}</pre>
        </div>
      )}
      {relations.length > 0 && (
        <div className="cc-kg-detail-row">
          <b>{t("kg.relationships_label", { count: relations.length })}</b>
          <div className="cc-kg-rel-list">
            {relations.map((r, i) => (
              <div key={i} className="cc-kg-rel-item">
                <span style={{ color: "var(--accent)" }}>{r.relation_type}</span>
                {" → "}{r.target_name || r.target_entity?.slice(0, 8)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function KnowledgeGraphPage() {
  const [entities, setEntities] = useState([]);
  const [relations, setRelations] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [selectedRelations, setSelectedRelations] = useState([]);
  const containerRef = useRef(null);

  const fetchData = useCallback(async () => {
    setError(null);
    try {
      const [sRes, eRes] = await Promise.all([
        api("GET", "/memory/graph/stats"),
        api("GET", "/memory/graph/entities"),
      ]);
      if (sRes?.error) {
        setError(`Failed to load stats (${sRes.error})`);
      } else if (sRes) {
        setStats(sRes);
      }
      if (eRes?.error) {
        setError(`Failed to load entities (${eRes.error})`);
      } else if (eRes?.entities) {
        setEntities(eRes.entities);
        setRelations(eRes.relations || []);
      }
    } catch (e) {
      setError(`Connection error: ${e.message || "Backend not reachable"}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleNodeClick = useCallback(async (node) => {
    setSelectedEntity(node.entity);
    try {
      const res = await api("GET", `/memory/graph/entities/${node.id}/relations`);
      setSelectedRelations(res?.relations || []);
    } catch {
      setSelectedRelations([]);
    }
  }, []);

  // Filter entities
  const filtered = entities.filter(e => {
    if (search && !e.name.toLowerCase().includes(search.toLowerCase())) return false;
    if (typeFilter && e.type !== typeFilter) return false;
    return true;
  });
  const filteredIds = new Set(filtered.map(e => e.id));
  const filteredRelations = relations.filter(
    r => filteredIds.has(r.source_entity) && filteredIds.has(r.target_entity)
  );

  const entityTypes = [...new Set(entities.map(e => e.type))].sort();
  const graphW = 800;
  const graphH = 500;

  if (loading) return <div className="cc-kg-loading">{t("kg.loading")}</div>;

  return (
    <div className="cc-kg-page">
      <style>{KG_STYLES}</style>

      <div className="cc-kg-header">
        <h2 className="cc-kg-title">{t("kg.title")}</h2>
        <div className="cc-kg-stats">
          <span>{stats.entities || 0} {t("kg.entities")}</span>
          <span>{stats.relations || 0} {t("kg.relationships")}</span>
          <button className="cc-kg-refresh" onClick={fetchData} title="Refresh">
            <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M23 4v6h-6M1 20v-6h6"/>
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
            </svg>
          </button>
        </div>
      </div>

      {error && (
        <div className="cc-kg-error">
          <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          {error}
        </div>
      )}

      {/* Filters */}
      <div className="cc-kg-filters">
        <input
          className="cc-kg-search"
          type="text"
          placeholder={t("kg.search_placeholder")}
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <select className="cc-kg-select" value={typeFilter} onChange={e => setTypeFilter(e.target.value)}>
          <option value="">{t("kg.all_types")}</option>
          {entityTypes.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <span className="cc-kg-count">{t("kg.visible", { count: filtered.length })}</span>
      </div>

      {/* Graph */}
      <div className="cc-kg-graph-container" ref={containerRef}>
        <GraphRenderer
          entities={filtered}
          relations={filteredRelations}
          width={graphW}
          height={graphH}
          onNodeClick={handleNodeClick}
          selectedId={selectedEntity?.id}
        />
        <EntityDetail
          entity={selectedEntity}
          relations={selectedRelations}
          onClose={() => { setSelectedEntity(null); setSelectedRelations([]); }}
        />
      </div>

      {/* Legend */}
      <div className="cc-kg-legend">
        {Object.entries(TYPE_COLORS).map(([type, colors]) => (
          <span key={type} className="cc-kg-legend-item">
            <span className="cc-kg-legend-dot" style={{ background: colors.fill }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Scoped CSS ────────────────────────────────────────────────────────
const KG_STYLES = `
.cc-kg-page { padding: 24px 32px; max-width: 1200px; margin: 0 auto; }
.cc-kg-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }
.cc-kg-title { font-size: 22px; font-weight: 700; color: var(--text); margin: 0; }
.cc-kg-stats { display: flex; gap: 12px; font-size: 13px; color: var(--text2); }
.cc-kg-stats span { background: var(--bg2); padding: 4px 12px; border-radius: 12px; border: 1px solid var(--border); }
.cc-kg-loading { text-align: center; padding: 60px 0; color: var(--text2); }
.cc-kg-empty { text-align: center; padding: 40px 0; color: var(--text2); font-size: 14px; display: flex; flex-direction: column; align-items: center; }
.cc-kg-error { background: rgba(255,82,82,0.1); border: 1px solid rgba(255,82,82,0.3); color: #ff5252; padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.cc-kg-refresh { background: none; border: 1px solid var(--border); color: var(--text2); padding: 4px 8px; border-radius: 6px; cursor: pointer; display: inline-flex; align-items: center; transition: all 0.15s; }
.cc-kg-refresh:hover { border-color: var(--accent); color: var(--accent); }

.cc-kg-filters { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.cc-kg-search { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 14px; border-radius: 8px; font-size: 13px; width: 220px; font-family: inherit; box-sizing: border-box; }
.cc-kg-search:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,212,255,0.15); }
.cc-kg-select { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 14px; border-radius: 8px; font-size: 13px; font-family: inherit; box-sizing: border-box; cursor: pointer; }
.cc-kg-select:focus { outline: none; border-color: var(--accent); }
.cc-kg-count { font-size: 12px; color: var(--text2); }

.cc-kg-graph-container { position: relative; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: auto; min-height: 400px; box-shadow: inset 0 2px 8px rgba(0,0,0,0.05); }
.cc-kg-svg { display: block; }

.cc-kg-detail { position: absolute; top: 16px; right: 16px; width: 300px; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); z-index: 10; }
.cc-kg-detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 15px; }
.cc-kg-close { background: none; border: none; color: var(--text2); font-size: 20px; cursor: pointer; }
.cc-kg-detail-row { font-size: 13px; color: var(--text); margin-bottom: 8px; line-height: 1.4; }
.cc-kg-detail-row b { color: var(--text2); margin-right: 6px; }
.cc-kg-output { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-size: 12px; max-height: 100px; overflow: auto; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.cc-kg-rel-list { margin-top: 4px; }
.cc-kg-rel-item { font-size: 12px; padding: 2px 0; color: var(--text2); }

.cc-kg-legend { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.cc-kg-legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }
.cc-kg-legend-dot { width: 10px; height: 10px; border-radius: 50%; }

@media (max-width: 768px) {
  .cc-kg-page { padding: 16px; }
  .cc-kg-detail { position: fixed; top: auto; bottom: 0; left: 0; right: 0; width: 100%; border-radius: 12px 12px 0 0; }
}
`;
