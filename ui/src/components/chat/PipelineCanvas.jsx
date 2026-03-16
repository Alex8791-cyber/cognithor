/**
 * PipelineCanvas — Live SVG visualization of the PGE agent pipeline.
 *
 * Shows Plan → Gatekeeper → Execute → Replan phases with real-time
 * status updates, sub-tool tracking, and duration display.
 */
import { useState, useEffect, useRef } from "react";

// ── Constants ────────────────────────────────────────────────────────────────

const PHASE_ORDER = ["plan", "gate", "execute", "replan"];

const PHASE_LABELS = {
  plan: { icon: "\u{1F9E0}", en: "Plan", de: "Planen", zh: "\u8BA1\u5212" },
  gate: { icon: "\u{1F6E1}\uFE0F", en: "Gatekeeper", de: "Gatekeeper", zh: "\u5B88\u95E8\u4EBA" },
  execute: { icon: "\u2699\uFE0F", en: "Execute", de: "Ausf\u00FChren", zh: "\u6267\u884C" },
  replan: { icon: "\u{1F4DD}", en: "Replan", de: "Replanen", zh: "\u91CD\u65B0\u8BA1\u5212" },
};

const STATUS_COLORS = {
  pending: { bg: "#2a2a3e", border: "#555570", text: "#8888a0", dot: "#555570" },
  running: { bg: "#1a2a4a", border: "#00d4ff", text: "#00d4ff", dot: "#00d4ff" },
  done:    { bg: "#1a3a2a", border: "#00e676", text: "#00e676", dot: "#00e676" },
  error:   { bg: "#3a1a1a", border: "#ff5252", text: "#ff5252", dot: "#ff5252" },
};

const NODE_W = 200;
const NODE_H = 40;
const NODE_GAP = 12;
const TOOL_H = 28;
const TOOL_W = 180;
const TOOL_GAP = 6;
const PAD_X = 16;
const PAD_Y = 12;

// ── Elapsed Timer Hook ──────────────────────────────────────────────────────

function useElapsed(startMs, active) {
  const [elapsed, setElapsed] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    if (!active || !startMs) { setElapsed(0); return; }
    const tick = () => setElapsed(Date.now() - startMs);
    tick();
    ref.current = setInterval(tick, 100);
    return () => clearInterval(ref.current);
  }, [startMs, active]);
  return elapsed;
}

// ── Sub-components ──────────────────────────────────────────────────────────

function PhaseNode({ phase, x, y, status, durationMs, startMs, label, icon }) {
  const c = STATUS_COLORS[status] || STATUS_COLORS.pending;
  const isRunning = status === "running";
  const elapsed = useElapsed(startMs, isRunning);
  const timeStr = isRunning
    ? `${(elapsed / 1000).toFixed(1)}s`
    : durationMs != null
      ? `${(durationMs / 1000).toFixed(1)}s`
      : "";

  return (
    <g>
      <rect x={x} y={y} width={NODE_W} height={NODE_H} rx={8}
        fill={c.bg} stroke={c.border} strokeWidth={isRunning ? 2 : 1.2} />
      {isRunning && (
        <rect x={x} y={y} width={NODE_W} height={NODE_H} rx={8}
          fill="none" stroke={c.border} strokeWidth={2}
          strokeDasharray="8 4" opacity={0.5}>
          <animate attributeName="stroke-dashoffset" from="0" to="-24" dur="1s" repeatCount="indefinite" />
        </rect>
      )}
      <text x={x + 12} y={y + NODE_H / 2 + 1} fill={c.text}
        fontSize={13} fontWeight={600} fontFamily="DM Sans, sans-serif"
        dominantBaseline="middle">
        {icon} {label}
      </text>
      {timeStr && (
        <text x={x + NODE_W - 12} y={y + NODE_H / 2 + 1} fill={c.text}
          fontSize={11} fontFamily="DM Sans, sans-serif"
          dominantBaseline="middle" textAnchor="end">
          {timeStr}
        </text>
      )}
      {/* Status dot */}
      <circle cx={x + NODE_W - 32 - (timeStr ? timeStr.length * 6 : 0)} cy={y + NODE_H / 2}
        r={4} fill={c.dot} opacity={0.9} />
    </g>
  );
}

function ToolNode({ name, status, durationMs, startMs, x, y }) {
  const c = STATUS_COLORS[status] || STATUS_COLORS.pending;
  const isRunning = status === "running";
  const elapsed = useElapsed(startMs, isRunning);
  const timeStr = isRunning
    ? `${(elapsed / 1000).toFixed(1)}s`
    : durationMs != null
      ? `${(durationMs / 1000).toFixed(1)}s`
      : "";

  return (
    <g>
      <rect x={x} y={y} width={TOOL_W} height={TOOL_H} rx={5}
        fill={c.bg} stroke={c.border} strokeWidth={1} opacity={0.8} />
      <text x={x + 8} y={y + TOOL_H / 2 + 1} fill={c.text}
        fontSize={11} fontFamily="DM Sans, monospace"
        dominantBaseline="middle">
        {name}
      </text>
      {timeStr && (
        <text x={x + TOOL_W - 8} y={y + TOOL_H / 2 + 1} fill={c.text}
          fontSize={10} fontFamily="DM Sans, sans-serif"
          dominantBaseline="middle" textAnchor="end">
          {timeStr}
        </text>
      )}
    </g>
  );
}

function Connector({ x, y1, y2 }) {
  const mx = x;
  return (
    <line x1={mx} y1={y1} x2={mx} y2={y2}
      stroke="#555570" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.5} />
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function PipelineCanvas({ pipeline, collapsed, onToggle }) {
  if (!pipeline || !pipeline.iterations || pipeline.iterations.length === 0) {
    return null;
  }

  const lastIter = pipeline.iterations[pipeline.iterations.length - 1];
  const phases = lastIter.phases || {};
  const tools = lastIter.tools || [];
  const iterNum = lastIter.number || pipeline.iterations.length;

  // Calculate SVG height
  let y = PAD_Y;
  const phasePositions = [];
  for (const phaseKey of PHASE_ORDER) {
    phasePositions.push({ key: phaseKey, y });
    y += NODE_H;
    // After execute, insert tool sub-items
    if (phaseKey === "execute" && tools.length > 0) {
      for (let i = 0; i < tools.length; i++) {
        y += TOOL_GAP + TOOL_H;
      }
    }
    y += NODE_GAP;
  }
  const svgH = y + PAD_Y;
  const svgW = PAD_X * 2 + NODE_W;

  const headerText = pipeline.active
    ? `Pipeline \u00B7 Iteration ${iterNum}`
    : `Pipeline \u00B7 ${pipeline.iterations.length} Iteration${pipeline.iterations.length > 1 ? "en" : ""}`;

  return (
    <div className="cc-pipeline">
      <div className="cc-pipeline-header" onClick={onToggle} role="button" tabIndex={0}>
        <span className="cc-pipeline-title">{headerText}</span>
        <span className="cc-pipeline-chevron">{collapsed ? "\u25B6" : "\u25BC"}</span>
      </div>
      {!collapsed && (
        <svg width={svgW} height={svgH} className="cc-pipeline-svg">
          {phasePositions.map((pp, i) => {
            const phaseData = phases[pp.key] || { status: "pending" };
            const meta = PHASE_LABELS[pp.key] || {};
            const label = meta.en || pp.key;
            const icon = meta.icon || "";

            const elements = [
              <PhaseNode
                key={pp.key}
                phase={pp.key}
                x={PAD_X}
                y={pp.y}
                status={phaseData.status}
                durationMs={phaseData.durationMs}
                startMs={phaseData.startMs}
                label={label}
                icon={icon}
              />,
            ];

            // Connector to next phase
            if (i < phasePositions.length - 1) {
              let connEndY = phasePositions[i + 1].y;
              // If this is execute with tools, connector goes from last tool to next phase
              if (pp.key === "execute" && tools.length > 0) {
                const toolBlockEnd = pp.y + NODE_H + tools.length * (TOOL_H + TOOL_GAP);
                elements.push(
                  <Connector key={`conn-${pp.key}`}
                    x={PAD_X + NODE_W / 2}
                    y1={pp.y + NODE_H}
                    y2={connEndY} />
                );
              } else {
                elements.push(
                  <Connector key={`conn-${pp.key}`}
                    x={PAD_X + NODE_W / 2}
                    y1={pp.y + NODE_H}
                    y2={connEndY} />
                );
              }
            }

            // Tool sub-items after execute node
            if (pp.key === "execute" && tools.length > 0) {
              let toolY = pp.y + NODE_H + TOOL_GAP;
              for (const tool of tools) {
                elements.push(
                  <ToolNode
                    key={`tool-${tool.name}-${toolY}`}
                    name={tool.name}
                    status={tool.status}
                    durationMs={tool.durationMs}
                    startMs={tool.startMs}
                    x={PAD_X + 10}
                    y={toolY}
                  />
                );
                toolY += TOOL_H + TOOL_GAP;
              }
            }

            return elements;
          })}
        </svg>
      )}
    </div>
  );
}
