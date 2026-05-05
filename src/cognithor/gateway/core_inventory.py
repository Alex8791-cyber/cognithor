"""CORE.md inventory sync — extracted from Gateway.

Updates the auto-managed INVENTORY section of CORE.md with the current set
of registered MCP tools, installed skills, and learned procedures, using
``ToolRegistryDB`` when available and falling back to a static schema-
only listing otherwise.

Lives outside ``Gateway`` itself so the gateway module stays focused on
orchestration; the gateway exposes thin ``self`` wrappers for back-compat
and so existing callers / tests keep working unchanged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway

log = get_logger(__name__)


def sync_core_inventory(gw: Gateway) -> None:
    """Refresh the INVENTORY section of CORE.md from current state.

    Uses ``ToolRegistryDB`` for DB-backed, localized, role-scoped tool
    sections; falls back to the legacy static method when the DB is
    unavailable.
    """
    if not gw._memory_manager or not hasattr(gw._memory_manager, "_core"):
        return
    core = gw._memory_manager._core
    content = core.content
    if not content:
        return
    language = getattr(gw._config, "language", "de")

    # Try DB-backed generation
    tool_count = 0
    try:
        from cognithor.mcp.tool_registry_db import (
            _SECTION_HEADERS,
            ToolRegistryDB,
            _ProcedureEntry,
            deduplicate_procedures,
        )

        db_path = gw._config.cognithor_home / "tool_registry.db"
        registry_db = ToolRegistryDB(db_path)

        # Tools aus MCP-Client synchronisieren
        if gw._mcp_client:
            registry_db.sync_from_mcp(gw._mcp_client)

        tool_count = registry_db.tool_count()
        registry_db.close()
    except Exception:
        log.debug("tool_registry_db_failed_falling_back", exc_info=True)
        # Fallback: legacy method just to validate MCP is alive
        if sync_core_inventory_legacy(gw) is None:
            return
        tool_count = 0

    # Compile skill list
    skill_lines: list[str] = []
    if hasattr(gw, "_skill_registry") and gw._skill_registry:
        try:
            for slug, skill in gw._skill_registry._skills.items():
                status = "active" if skill.enabled else "inactive"
                skill_lines.append(f"- **{skill.name}** (`{slug}`) -- {status}")
        except Exception:
            log.debug("core_inventory_skills_failed", exc_info=True)
    if not skill_lines:
        skill_lines = ["- (no skills registered)"]

    # Procedure list with deduplication
    proc_lines: list[str] = []
    if gw._memory_manager:
        try:
            from cognithor.mcp.tool_registry_db import (
                _ProcedureEntry,
                deduplicate_procedures,
            )

            procedural = gw._memory_manager.procedural
            raw_procs = [
                _ProcedureEntry(
                    name=meta.name,
                    total_uses=meta.total_uses,
                    trigger_keywords=list(meta.trigger_keywords),
                )
                for meta in procedural.list_procedures()
            ]
            proc_lines = deduplicate_procedures(
                raw_procs,
                language=language,
            )
        except Exception:
            log.debug("core_inventory_procedures_dedup_failed", exc_info=True)
            # Fallback: simple list
            try:
                procedural = gw._memory_manager.procedural
                for meta in procedural.list_procedures():
                    uses = f"{meta.total_uses}x" if meta.total_uses else "0x"
                    kw = ", ".join(meta.trigger_keywords[:3]) if meta.trigger_keywords else ""
                    suffix = f" [{kw}]" if kw else ""
                    proc_lines.append(f"- `{meta.name}` ({uses} used){suffix}")
            except Exception:
                log.debug("core_inventory_procedures_failed", exc_info=True)

    if not proc_lines:
        proc_lines = ["- (no procedures stored)"]

    # Lokalisierte Header
    try:
        from cognithor.mcp.tool_registry_db import _SECTION_HEADERS

        headers = _SECTION_HEADERS.get(language, _SECTION_HEADERS["en"])
    except Exception:
        headers = {
            "inventory_title": "INVENTORY (auto-updated)",
            "skills_title": "Installed Skills ({count})",
            "procedures_title": "Learned Procedures ({count})",
        }

    inv_title = headers["inventory_title"]
    skills_title = headers["skills_title"].format(count=len(skill_lines))
    procs_title = headers["procedures_title"].format(count=len(proc_lines))

    # Tool descriptions are injected directly into the Planner prompt
    # via {tools_section} — no need to duplicate them in CORE.md
    tool_ref = f"*{tool_count} Tools registriert (werden direkt in den Planner-Prompt injiziert)*"

    inventory = (
        f"## {inv_title}\n\n"
        + tool_ref
        + "\n\n"
        + f"### {skills_title}\n"
        + "\n".join(skill_lines)
        + "\n\n"
        + f"### {procs_title}\n"
        + "\n".join(proc_lines)
    )

    # Bestehenden INVENTAR/INVENTORY-Abschnitt ersetzen oder am Ende anhaengen
    marker_candidates = [
        "## INVENTAR (auto-aktualisiert)",
        "## INVENTAR (automatisch aktualisiert)",
        "## INVENTORY (auto-updated)",
        f"## {inv_title}",
    ]
    marker_start = None
    for marker in marker_candidates:
        if marker in content:
            marker_start = marker
            break

    if marker_start:
        pattern = re.escape(marker_start) + r".*?(?=\n## (?!INVENT|清单)|\Z)"
        content = re.sub(pattern, inventory, content, flags=re.DOTALL)
    else:
        content = content.rstrip() + "\n\n---\n\n" + inventory + "\n"

    core.save(content)
    log.info(
        "core_inventory_synced",
        tools=tool_count,
        skills=len(skill_lines),
        procedures=len(proc_lines),
    )


def sync_core_inventory_legacy(gw: Gateway) -> str | None:
    """Schema-only fallback when ``ToolRegistryDB`` is unavailable.

    Returns the formatted tools section, or ``None`` if no schemas are
    registered (which the caller treats as "MCP is not alive yet").
    """
    tool_schemas = gw._mcp_client.get_tool_schemas() if gw._mcp_client else {}
    if not tool_schemas:
        return None

    tool_lines: list[str] = []
    for name in sorted(tool_schemas):
        schema = tool_schemas[name]
        desc = schema.get("description", "")
        props = schema.get("inputSchema", {}).get("properties", {})
        required = set(schema.get("inputSchema", {}).get("required", []))
        if props:
            parts = []
            for k, v in props.items():
                typ = v.get("type", "?")
                req = " *" if k in required else ""
                parts.append(f"{k}: {typ}{req}")
            param_str = ", ".join(parts)
            tool_lines.append(f"- `{name}({param_str})` -- {desc}")
        else:
            tool_lines.append(f"- `{name}()` -- {desc}")

    tool_count = len(tool_schemas)
    return (
        f"### Registered Tools ({tool_count})\n"
        + "Parameters marked with * are required.\n\n"
        + "\n".join(tool_lines)
    )
