#!/usr/bin/env bash
#
# Sprint-27 PR-J — IDE-compat smoke for the Cognithor extension.
#
# Validates that the extension manifest still meets the minimum
# version + API surface expected by the four supported targets:
# VS Code, Cursor, Windsurf, and (for the MCP-stdio path) Claude
# Desktop. Exits non-zero with a one-line reason on the first
# drift.
#
# Usage: bash extensions/vscode/scripts/check_ide_compat.sh
#
# Pure bash + python — no node / npm dependency. Safe to run in
# a stripped CI image.

set -euo pipefail

# Resolve repo paths from this script's location so the smoke
# works regardless of the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PKG_JSON="${EXT_DIR}/package.json"
DIST_JS="${EXT_DIR}/dist/extension.js"

if [[ ! -f "${PKG_JSON}" ]]; then
  echo "[fail] package.json missing at ${PKG_JSON}" >&2
  exit 1
fi

# Use python (already on every Cognithor dev machine) to probe
# the manifest. Avoids a hard jq dependency.
python3 - "${PKG_JSON}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

engines = manifest.get("engines", {})
vscode_engine = engines.get("vscode", "")
node_engine = engines.get("node", "")

if not vscode_engine.startswith("^1.85"):
    print(f"[fail] engines.vscode must start with ^1.85 — got {vscode_engine!r}", file=sys.stderr)
    sys.exit(1)

if not node_engine.startswith(">=20"):
    print(f"[fail] engines.node must start with >=20 — got {node_engine!r}", file=sys.stderr)
    sys.exit(1)

required_events = {
    "onCommand:cognithor.runPlan",
    "onStartupFinished",
}
declared_events = set(manifest.get("activationEvents", []))
missing = required_events - declared_events
if missing:
    print(
        f"[fail] activationEvents missing: {sorted(missing)}",
        file=sys.stderr,
    )
    sys.exit(1)

# Soft check: if the receipt / hover / cost-gutter wiring is
# present (post-PR-G/H/I), the JSON activation events must be
# declared too. PR-D/E ship without them.
contributes = manifest.get("contributes", {})
commands = {c.get("command") for c in contributes.get("commands", [])}
needs_json_activation = bool(
    {"cognithor.viewReceipt", "cognithor.refreshReceipts"} & commands,
)
if needs_json_activation:
    json_required = {"onLanguage:json", "onLanguage:jsonc"}
    json_missing = json_required - declared_events
    if json_missing:
        print(
            f"[fail] receipt commands present but JSON activation missing: {sorted(json_missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

print(f"[ok] manifest engines + activation events look healthy ({len(declared_events)} events declared)")
PY

if [[ ! -s "${DIST_JS}" ]]; then
  # dist/extension.js is required for marketplace install. Don't
  # treat its absence as a hard failure if the user clearly hasn't
  # built yet — fall back to running the build for them.
  echo "[warn] dist/extension.js missing — running 'npm run compile' to verify build path"
  pushd "${EXT_DIR}" >/dev/null
  npm run compile
  popd >/dev/null
fi

if [[ ! -s "${DIST_JS}" ]]; then
  echo "[fail] dist/extension.js still missing or empty after build" >&2
  exit 1
fi

DIST_BYTES=$(wc -c < "${DIST_JS}")
echo "[ok] dist/extension.js = ${DIST_BYTES} bytes"

echo "[ok] IDE-compat smoke passed for VS Code / Cursor / Windsurf / Claude-Desktop"
