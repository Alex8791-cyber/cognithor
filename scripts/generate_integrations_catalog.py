#!/usr/bin/env python3
"""Scan src/cognithor/ for MCP tool definitions and emit catalog.json.

Tool discovery:
  * Any module under src/cognithor/mcp/ containing a function decorated with
    @mcp_tool (or class registering to the tool_registry).
  * Skill modules that register MCP-compatible tools.

Output JSON shape:
  {
    "generated_at": "<iso8601>",
    "tool_count": N,
    "tools": [
      {"name": "...", "module": "cognithor.mcp.foo", "category": "...",
       "description": "...", "dach_specific": false}, ...
    ]
  }
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = REPO_ROOT / "src" / "cognithor" / "mcp"

DACH_MARKERS = {"datev", "lexware", "sevdesk", "elster", "schufa"}

# Modules that carry @mcp_tool-decorated functions but are NOT yet wired into
# the live MCP server (no register_tool calls, module not imported by the
# server boot path). Excluded from the public catalog so the cognithor.ai
# /integrations page doesn't over-promise capability.
#
# When wiring a module into the live server, remove its prefix from this set
# AND add the appropriate register_tool calls in the module's __init__.
NOT_YET_REGISTERED_PREFIXES: set[str] = {
    "cognithor.mcp.sevdesk",
}


def extract_tools(py_file: Path) -> list[dict]:
    """Parse a Python file and return any @mcp_tool-decorated function metadata."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    results: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for dec in node.decorator_list:
            dec_name = _decorator_name(dec)
            if dec_name in {"mcp_tool", "cognithor_tool", "tool"}:
                docstring = ast.get_docstring(node) or ""
                module = (
                    py_file.relative_to(REPO_ROOT / "src")
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                )
                category = _infer_category(py_file, docstring)
                name_lower = node.name.lower()
                dach = any(
                    marker in name_lower or marker in docstring.lower() for marker in DACH_MARKERS
                )
                results.append(
                    {
                        "name": node.name,
                        "module": module,
                        "category": category,
                        "description": docstring.split("\n")[0][:200],
                        "dach_specific": dach,
                    }
                )
                break
    return results


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return ""


def _infer_category(py_file: Path, docstring: str) -> str:
    parts = py_file.parts
    for marker in (
        "filesystem",
        "web",
        "shell",
        "memory",
        "vault",
        "browser",
        "documents",
        "kanban",
        "identity",
        "reddit",
        "sevdesk",
    ):
        if marker in parts:
            return marker
    low = docstring.lower()
    if "http" in low or "url" in low or "web" in low:
        return "web"
    if "file" in low or "pdf" in low:
        return "filesystem"
    return "misc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    tools: list[dict] = []
    for py in MCP_DIR.rglob("*.py"):
        tools.extend(extract_tools(py))

    # Filter out tools from modules that aren't wired into the live server yet.
    # See NOT_YET_REGISTERED_PREFIXES at top of file.
    tools = [
        t for t in tools if not any(t["module"].startswith(p) for p in NOT_YET_REGISTERED_PREFIXES)
    ]

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for t in tools:
        key = (t["module"], t["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    deduped.sort(key=lambda t: (t["category"], t["name"]))

    catalog = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tool_count": len(deduped),
        "tools": deduped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(deduped)} tools to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
