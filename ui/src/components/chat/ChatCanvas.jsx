import { useMemo } from "react";

const CANVAS_DARK_CSS = `
  body {
    background: #181825;
    color: #e0e0e8;
    font-family: 'DM Sans', -apple-system, sans-serif;
    margin: 16px;
    line-height: 1.6;
  }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #1e1e30; padding: 8px 12px; text-align: left; }
  th { background: #10101a; font-weight: 600; }
  a { color: #00d4ff; }
  code { background: #10101a; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
  pre { background: #10101a; padding: 12px; border-radius: 8px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  img { max-width: 100%; height: auto; }
`;

export function ChatCanvas({ html, title, onClose }) {
  const srcdoc = useMemo(() => {
    if (!html) return "";
    return `<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>${CANVAS_DARK_CSS}</style>
</head><body>${html}</body></html>`;
  }, [html]);

  return (
    <div className="cc-canvas-panel">
      <div className="cc-canvas-header">
        <span className="cc-canvas-title">{title || "Canvas"}</span>
        <button className="cc-canvas-close" onClick={onClose} type="button" title="Canvas schliessen">
          <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      </div>
      <iframe
        className="cc-canvas-frame"
        sandbox=""
        srcDoc={srcdoc}
        title="Jarvis Canvas"
      />
    </div>
  );
}
