/**
 * API Helper — REST calls to Jarvis backend.
 */

const API = "/api/v1";

export async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(`${API}${path}`, opts);
    if (!r.ok) return { error: `HTTP ${r.status}`, status: r.status };
    const text = await r.text();
    if (!text) return {};
    // Prevent precision loss for large integers (like Discord IDs)
    const safeText = text.replace(/:\s*([0-9]{16,})\b/g, ':"$1"');
    return JSON.parse(safeText);
  } catch (e) {
    console.error(`API ${method} ${path}:`, e);
    return { error: e.message };
  }
}
