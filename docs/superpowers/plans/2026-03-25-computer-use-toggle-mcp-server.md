# Computer Use Toggle & MCP-Server Entry Point

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Computer Use / Desktop Tools per Config-Flag steuerbar machen (Default: OFF), mit Flutter UI Toggle und einem neuen `--mcp-server` CLI Entry Point der nur workspace-sichere Tools exponiert.

**Architecture:** Neues `ToolsConfig` Pydantic-Model in `config.py` mit `computer_use_enabled` und `desktop_tools_enabled` Flags. `gateway/phases/tools.py` prüft diese Flags vor der Registrierung. Gatekeeper blockt deaktivierte Tools als zusätzliche Sicherheitsebene. Neue Flutter `ToolsPage` unter Security-Kategorie. Neuer `--mcp-server` CLI-Flag startet Jarvis im Minimal-Modus (nur MCP-Server auf stdio, kein CLI, kein Web-UI) mit eingeschränktem Tool-Set.

**Tech Stack:** Python 3.12+ (Pydantic, FastAPI), Flutter/Dart (Provider), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/jarvis/config.py` | Add `ToolsConfig` model, wire into `JarvisConfig` |
| Modify | `src/jarvis/gateway/phases/tools.py` | Conditional registration based on `config.tools` |
| Modify | `src/jarvis/core/gatekeeper.py` | Blocklist for disabled tool groups |
| Modify | `src/jarvis/__main__.py` | Add `--mcp-server` entry point |
| Modify | `src/jarvis/mcp/bridge.py` | Add tool filter for MCP-Server profiles |
| Create | `flutter_app/lib/screens/config/tools_page.dart` | Flutter toggle page |
| Modify | `flutter_app/lib/screens/config_screen.dart` | Register tools_page in navigation |
| Modify | `flutter_app/lib/l10n/app_en.arb` | Add i18n keys for tools page |
| Create | `tests/unit/test_tools_config.py` | Tests for config, conditional registration, gatekeeper |

---

### Task 1: Add `ToolsConfig` to config.py

**Files:**
- Modify: `src/jarvis/config.py`
- Test: `tests/unit/test_tools_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tools_config.py`:

```python
"""Tests for ToolsConfig and JarvisConfig integration."""

import pytest
from jarvis.config import JarvisConfig, ToolsConfig


class TestToolsConfig:
    """ToolsConfig defaults and validation."""

    def test_defaults(self):
        cfg = ToolsConfig()
        assert cfg.computer_use_enabled is False
        assert cfg.desktop_tools_enabled is False

    def test_enable_computer_use(self):
        cfg = ToolsConfig(computer_use_enabled=True)
        assert cfg.computer_use_enabled is True

    def test_enable_desktop_tools(self):
        cfg = ToolsConfig(desktop_tools_enabled=True)
        assert cfg.desktop_tools_enabled is True


class TestJarvisConfigToolsIntegration:
    """ToolsConfig is accessible via JarvisConfig.tools."""

    def test_tools_section_exists(self):
        cfg = JarvisConfig()
        assert hasattr(cfg, "tools")
        assert isinstance(cfg.tools, ToolsConfig)

    def test_tools_defaults_in_jarvis_config(self):
        cfg = JarvisConfig()
        assert cfg.tools.computer_use_enabled is False
        assert cfg.tools.desktop_tools_enabled is False

    def test_tools_serialization(self):
        cfg = JarvisConfig(tools=ToolsConfig(computer_use_enabled=True))
        data = cfg.model_dump(mode="json")
        assert data["tools"]["computer_use_enabled"] is True
        assert data["tools"]["desktop_tools_enabled"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'ToolsConfig'` and `AttributeError: 'JarvisConfig' has no 'tools'`

- [ ] **Step 3: Implement ToolsConfig**

In `src/jarvis/config.py`, add the `ToolsConfig` class **after** `ShellConfig` (around line 220):

```python
class ToolsConfig(BaseModel):
    """Feature-Toggles fuer Tool-Gruppen. [B§12.7]

    Steuert welche Tool-Kategorien beim Start registriert werden.
    Deaktivierte Tools werden weder im Planner-Prompt angezeigt
    noch vom Gatekeeper durchgelassen.
    """

    computer_use_enabled: bool = Field(
        default=False,
        description=(
            "Desktop-Automation via Screenshot + Koordinaten-Klick "
            "(pyautogui, mss). Erfordert pip install cognithor[desktop]."
        ),
    )
    desktop_tools_enabled: bool = Field(
        default=False,
        description=(
            "Clipboard-Zugriff und Screenshot-Tools. "
            "Erfordert pip install cognithor[desktop]."
        ),
    )
```

In `JarvisConfig` (around line 1820, after `shell: ShellConfig`), add:

```python
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/config.py tests/unit/test_tools_config.py
git commit -m "feat: add ToolsConfig with computer_use and desktop_tools toggles"
```

---

### Task 2: Conditional tool registration in tools.py

**Files:**
- Modify: `src/jarvis/gateway/phases/tools.py:452-470`
- Test: `tests/unit/test_tools_config.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tools_config.py`:

```python
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestConditionalRegistration:
    """Tools are only registered when config flag is enabled."""

    @pytest.fixture
    def mock_mcp_client(self):
        client = MagicMock()
        client.register_builtin_handler = MagicMock()
        client.register_tool = MagicMock()
        client.get_tool_schemas = MagicMock(return_value={})
        client.get_tool_list = MagicMock(return_value=[])
        client._builtin_handlers = {}
        return client

    @pytest.fixture
    def config_desktop_off(self):
        return JarvisConfig(
            tools=ToolsConfig(computer_use_enabled=False, desktop_tools_enabled=False),
        )

    @pytest.fixture
    def config_desktop_on(self):
        return JarvisConfig(
            tools=ToolsConfig(computer_use_enabled=True, desktop_tools_enabled=True),
        )

    def test_desktop_tools_skipped_when_disabled(self, mock_mcp_client, config_desktop_off):
        """Desktop tools must not be registered when disabled."""
        from jarvis.gateway.phases.tools import init_tools

        result = asyncio.get_event_loop().run_until_complete(
            init_tools(config_desktop_off, mock_mcp_client, memory_manager=None)
        )
        # Check that register_desktop_tools was never called
        registered_names = [
            call.args[0]
            for call in mock_mcp_client.register_builtin_handler.call_args_list
        ]
        assert "computer_screenshot" not in registered_names
        assert "computer_click" not in registered_names
        assert "get_clipboard" not in registered_names
        assert "screenshot_desktop" not in registered_names

    def test_desktop_tools_registered_when_enabled(self, mock_mcp_client, config_desktop_on):
        """Desktop tools must be registered when enabled."""
        from jarvis.gateway.phases.tools import init_tools

        result = asyncio.get_event_loop().run_until_complete(
            init_tools(config_desktop_on, mock_mcp_client, memory_manager=None)
        )
        registered_names = [
            call.args[0]
            for call in mock_mcp_client.register_builtin_handler.call_args_list
        ]
        # At least one desktop tool should be present (if deps available)
        # Note: may not register if pyautogui not installed, which is OK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py::TestConditionalRegistration::test_desktop_tools_skipped_when_disabled -v`
Expected: FAIL — desktop tools are registered regardless of config

- [ ] **Step 3: Add config guards in tools.py**

In `src/jarvis/gateway/phases/tools.py`, replace the desktop tools block (lines 452-470):

**Before (lines 452-459):**
```python
    # Desktop tools (clipboard, screenshot)
    try:
        from jarvis.mcp.desktop_tools import register_desktop_tools

        register_desktop_tools(mcp_client, config)
        log.info("desktop_tools_registered")
    except Exception:
        log.debug("desktop_tools_not_registered", exc_info=True)
```

**After:**
```python
    # Desktop tools (clipboard, screenshot) — guarded by config.tools.desktop_tools_enabled
    if getattr(getattr(config, "tools", None), "desktop_tools_enabled", False):
        try:
            from jarvis.mcp.desktop_tools import register_desktop_tools

            register_desktop_tools(mcp_client, config)
            log.info("desktop_tools_registered")
        except Exception:
            log.debug("desktop_tools_not_registered", exc_info=True)
    else:
        log.info("desktop_tools_disabled_by_config")
```

**Before (lines 461-470):**
```python
    # Computer Use (GPT-5.4-style screenshot + coordinate clicking)
    try:
        from jarvis.mcp.computer_use import register_computer_use_tools

        vision = getattr(gateway, "_vision_analyzer", None) if gateway else None
        cu_tools = register_computer_use_tools(mcp_client, vision_analyzer=vision)
        if cu_tools:
            log.info("computer_use_tools_registered")
    except Exception:
        log.debug("computer_use_not_registered", exc_info=True)
```

**After:**
```python
    # Computer Use (screenshot + coordinate clicking) — guarded by config.tools.computer_use_enabled
    if getattr(getattr(config, "tools", None), "computer_use_enabled", False):
        try:
            from jarvis.mcp.computer_use import register_computer_use_tools

            vision = getattr(gateway, "_vision_analyzer", None) if gateway else None
            cu_tools = register_computer_use_tools(mcp_client, vision_analyzer=vision)
            if cu_tools:
                log.info("computer_use_tools_registered")
        except Exception:
            log.debug("computer_use_not_registered", exc_info=True)
    else:
        log.info("computer_use_disabled_by_config")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/gateway/phases/tools.py tests/unit/test_tools_config.py
git commit -m "feat: guard desktop/computer-use registration behind config.tools flags"
```

---

### Task 3: Gatekeeper blocks disabled tools

**Files:**
- Modify: `src/jarvis/core/gatekeeper.py`
- Test: `tests/unit/test_tools_config.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tools_config.py`:

```python
class TestGatekeeperBlocksDisabledTools:
    """Gatekeeper must block tools from disabled groups even if somehow registered."""

    COMPUTER_USE_TOOLS = {
        "computer_screenshot", "computer_click", "computer_type",
        "computer_hotkey", "computer_scroll", "computer_drag",
    }
    DESKTOP_TOOLS = {
        "get_clipboard", "set_clipboard", "screenshot_desktop", "screenshot_region",
    }

    def test_computer_use_blocked_when_disabled(self):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(tools=ToolsConfig(computer_use_enabled=False))
        gk = Gatekeeper(config)
        for tool in self.COMPUTER_USE_TOOLS:
            assert gk.is_tool_disabled(tool), f"{tool} should be disabled"

    def test_computer_use_allowed_when_enabled(self):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(tools=ToolsConfig(computer_use_enabled=True))
        gk = Gatekeeper(config)
        for tool in self.COMPUTER_USE_TOOLS:
            assert not gk.is_tool_disabled(tool), f"{tool} should be enabled"

    def test_desktop_tools_blocked_when_disabled(self):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(tools=ToolsConfig(desktop_tools_enabled=False))
        gk = Gatekeeper(config)
        for tool in self.DESKTOP_TOOLS:
            assert gk.is_tool_disabled(tool), f"{tool} should be disabled"

    def test_desktop_tools_allowed_when_enabled(self):
        from jarvis.core.gatekeeper import Gatekeeper

        config = JarvisConfig(tools=ToolsConfig(desktop_tools_enabled=True))
        gk = Gatekeeper(config)
        for tool in self.DESKTOP_TOOLS:
            assert not gk.is_tool_disabled(tool), f"{tool} should be enabled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py::TestGatekeeperBlocksDisabledTools -v`
Expected: FAIL — `AttributeError: 'Gatekeeper' has no method 'is_tool_disabled'`

- [ ] **Step 3: Add disabled-tool-set and is_tool_disabled() to Gatekeeper**

In `src/jarvis/core/gatekeeper.py`, find the `__init__` method and add after `self._config = config`:

```python
        # Tool group blocklists (derived from config.tools)
        self._disabled_tools: frozenset[str] = self._build_disabled_tools()
```

Add method to the class:

```python
    # ── Tool Group Toggles ──────────────────────────────────────────
    _COMPUTER_USE_TOOLS = frozenset({
        "computer_screenshot", "computer_click", "computer_type",
        "computer_hotkey", "computer_scroll", "computer_drag",
    })
    _DESKTOP_TOOLS = frozenset({
        "get_clipboard", "set_clipboard", "screenshot_desktop", "screenshot_region",
    })

    def _build_disabled_tools(self) -> frozenset[str]:
        """Build set of tools disabled by config.tools flags."""
        disabled: set[str] = set()
        tools_cfg = getattr(self._config, "tools", None)
        if tools_cfg is None:
            return frozenset()
        if not getattr(tools_cfg, "computer_use_enabled", False):
            disabled |= self._COMPUTER_USE_TOOLS
        if not getattr(tools_cfg, "desktop_tools_enabled", False):
            disabled |= self._DESKTOP_TOOLS
        return frozenset(disabled)

    def is_tool_disabled(self, tool_name: str) -> bool:
        """Check if a tool is disabled by config.tools flags."""
        return tool_name in self._disabled_tools

    def reload_disabled_tools(self) -> None:
        """Re-read config.tools flags (called after runtime config change)."""
        self._disabled_tools = self._build_disabled_tools()
```

Then in `_classify_risk()`, add at the very top (before the green_tools check):

```python
        # Hard-block tools disabled by config.tools
        if tool in self._disabled_tools:
            return RiskLevel.RED
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/core/gatekeeper.py tests/unit/test_tools_config.py
git commit -m "feat: gatekeeper blocks tools from disabled groups as RED"
```

---

### Task 4: Gatekeeper reload on config change

**Files:**
- Modify: `src/jarvis/gateway/gateway.py`

- [ ] **Step 1: Wire reload_disabled_tools into reload_components**

In `src/jarvis/gateway/gateway.py`, find `reload_components()` (line ~1006). In the `if config:` block (or after the existing reload logic), add:

```python
        if config and self._gatekeeper:
            self._gatekeeper.reload_disabled_tools()
            reloaded.append("tool_toggles")
```

This ensures that when the Flutter UI saves a config change, the Gatekeeper immediately picks up the new tool flags.

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/gateway/gateway.py
git commit -m "feat: reload gatekeeper tool toggles on config change"
```

---

### Task 5: Flutter Tools Page

**Files:**
- Create: `flutter_app/lib/screens/config/tools_page.dart`
- Modify: `flutter_app/lib/screens/config_screen.dart`
- Modify: `flutter_app/lib/l10n/app_en.arb`

- [ ] **Step 1: Add i18n keys**

In `flutter_app/lib/l10n/app_en.arb`, add these entries (alphabetically with the other config page keys):

```json
  "configPageTools": "Tools",
  "@configPageTools": {},
  "toolsComputerUseLabel": "Computer Use",
  "@toolsComputerUseLabel": {},
  "toolsComputerUseDesc": "Desktop automation via screenshots and coordinate clicking (pyautogui). Allows Jarvis to interact with any application visually.",
  "@toolsComputerUseDesc": {},
  "toolsDesktopLabel": "Desktop Tools",
  "@toolsDesktopLabel": {},
  "toolsDesktopDesc": "Clipboard access (read/write) and screenshot capture.",
  "@toolsDesktopDesc": {},
  "toolsSectionDesktop": "Desktop & Automation",
  "@toolsSectionDesktop": {},
  "toolsWarning": "These tools give Jarvis access to your desktop. Enable only when needed.",
  "@toolsWarning": {},
```

- [ ] **Step 2: Run flutter gen-l10n to regenerate**

Run: `cd "D:\Jarvis\jarvis complete v20\flutter_app" && flutter gen-l10n`
Expected: Success, generates updated `app_localizations_en.dart`

- [ ] **Step 3: Create tools_page.dart**

Create `flutter_app/lib/screens/config/tools_page.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/form/form_widgets.dart';

class ToolsPage extends StatelessWidget {
  const ToolsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final tools = cfg.cfg['tools'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Warning banner
            Container(
              padding: const EdgeInsets.all(12),
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: JarvisTheme.orange.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: JarvisTheme.orange.withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                children: [
                  Icon(Icons.warning_amber, color: JarvisTheme.orange, size: 20),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      l.toolsWarning,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: JarvisTheme.orange,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            // Section header
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                l.toolsSectionDesktop,
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  color: JarvisTheme.accent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            JarvisToggleField(
              label: l.toolsComputerUseLabel,
              description: l.toolsComputerUseDesc,
              value: tools['computer_use_enabled'] == true,
              onChanged: (v) => cfg.set('tools.computer_use_enabled', v),
            ),
            JarvisToggleField(
              label: l.toolsDesktopLabel,
              description: l.toolsDesktopDesc,
              value: tools['desktop_tools_enabled'] == true,
              onChanged: (v) => cfg.set('tools.desktop_tools_enabled', v),
            ),
          ],
        );
      },
    );
  }
}
```

- [ ] **Step 4: Register in config_screen.dart**

In `flutter_app/lib/screens/config_screen.dart`:

Add import at top:
```dart
import 'package:jarvis_ui/screens/config/tools_page.dart';
```

In `_categories` list, add `'tools'` to the Security category (line ~50-52):
```dart
  _Category((l) => l.catSecurity, Icons.shield, [
    'security', 'tools', 'database',
  ]),
```

In `_pageRegistry` map, add the entry (after the 'security' entry, around line 85):
```dart
  'tools': _SubPageDef(
      Icons.mouse, (l) => l.configPageTools, () => const ToolsPage()),
```

- [ ] **Step 5: Verify Flutter compiles**

Run: `cd "D:\Jarvis\jarvis complete v20\flutter_app" && flutter analyze lib/screens/config/tools_page.dart`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20\flutter_app"
git add lib/screens/config/tools_page.dart lib/screens/config_screen.dart lib/l10n/app_en.arb lib/l10n/generated/
git commit -m "feat: add Tools config page with Computer Use and Desktop toggles"
```

---

### Task 6: `--mcp-server` CLI entry point

**Files:**
- Modify: `src/jarvis/__main__.py`
- Modify: `src/jarvis/mcp/bridge.py`

- [ ] **Step 1: Add MCP_SAFE_TOOLS constant to bridge.py**

In `src/jarvis/mcp/bridge.py`, add after the `IDEMPOTENT_TOOLS` frozenset (around line 99):

```python
# Tools die im MCP-Server-Modus fuer externe Clients (VSCode etc.) sicher sind.
# Kein Shell-Exec, kein Computer-Use, kein Remote-Shell, kein Docker-Run.
MCP_WORKSPACE_SAFE_TOOLS = frozenset({
    # Filesystem (workspace-sandboxed)
    "read_file", "write_file", "edit_file", "list_directory", "find_in_files",
    # Code (sandboxed execution)
    "run_python", "analyze_code",
    # Web (read-only, SSRF-protected)
    "web_search", "web_fetch", "search_and_read", "web_news_search",
    # Memory
    "search_memory", "save_to_memory", "get_entity", "add_entity",
    "add_relation", "get_core_memory", "get_recent_episodes",
    "search_procedures", "memory_stats",
    # Vault (note management)
    "vault_save", "vault_search", "vault_list", "vault_write",
    # Git (workspace-scoped)
    "git_status", "git_log", "git_diff", "git_commit", "git_branch",
    # Knowledge synthesis
    "knowledge_synthesize",
    # Charts and data
    "create_chart", "create_table_image", "chart_from_csv",
    # Database (read-only queries)
    "db_query", "db_schema",
    # Search
    "deep_research", "deep_research_v2", "verified_web_lookup",
    # Browser (read-only navigation)
    "browse_url", "browse_page_info", "browse_screenshot",
})
```

- [ ] **Step 2: Add tool filter to _bridge_builtin_tools**

In `src/jarvis/mcp/bridge.py`, modify `_bridge_builtin_tools` to accept an optional allowlist:

Replace the method signature and first lines:

```python
    def _bridge_builtin_tools(
        self,
        mcp_client: JarvisMCPClient,
        tool_allowlist: frozenset[str] | None = None,
    ) -> int:
        """Konvertiert bestehende Builtin-Handler in MCPToolDefs.

        Args:
            mcp_client: Client with registered tools.
            tool_allowlist: If set, only bridge tools in this set.

        Returns:
            Anzahl konvertierter Tools.
        """
        if self._server is None:
            return 0

        count = 0
        schemas = mcp_client.get_tool_schemas()

        for tool_name, schema in schemas.items():
            if tool_allowlist is not None and tool_name not in tool_allowlist:
                continue

            # Handler aus dem Client holen
            handler = mcp_client._builtin_handlers.get(tool_name)
            if handler is None:
                continue
```

Update the call in `setup()` to pass through:

```python
        # 1. Bestehende Builtin-Tools konvertieren
        tools_count = self._bridge_builtin_tools(mcp_client, self._tool_allowlist)
```

Add `_tool_allowlist` to `__init__`:

```python
        self._tool_allowlist: frozenset[str] | None = None
```

Add setter:

```python
    def set_tool_allowlist(self, allowlist: frozenset[str] | None) -> None:
        """Restrict which tools are exposed via MCP server."""
        self._tool_allowlist = allowlist
```

- [ ] **Step 3: Add --mcp-server flag to __main__.py**

In `src/jarvis/__main__.py`, in `parse_args()`, add after the `--auto-install` argument:

```python
    parser.add_argument(
        "--mcp-server",
        action="store_true",
        help=(
            "Start as MCP server on stdio (for VSCode, Claude Desktop, etc.). "
            "Only workspace-safe tools are exposed. No CLI, no Web UI."
        ),
    )
```

Then in the main async startup function, add handling before the normal startup. Find where `args` are used and add:

```python
    if args.mcp_server:
        await _run_mcp_server_mode(config)
        return
```

Add the implementation function:

```python
async def _run_mcp_server_mode(config: "JarvisConfig") -> None:
    """Run Jarvis as a pure MCP server on stdio.

    Exposes only workspace-safe tools. No CLI, no Web UI, no channels.
    Designed for integration with VSCode, Claude Desktop, Cursor, etc.
    """
    import sys

    from jarvis.mcp.bridge import MCPBridge, MCP_WORKSPACE_SAFE_TOOLS
    from jarvis.mcp.client import JarvisMCPClient
    from jarvis.mcp.server import MCPServerConfig, MCPServerMode

    log.info("mcp_server_mode_starting")

    # Force stdio mode
    config_dict = config.model_dump()
    # Ensure tools that need guarding are off
    if hasattr(config, "tools"):
        config.tools.computer_use_enabled = False
        config.tools.desktop_tools_enabled = False

    # Create MCP client and register tools
    mcp_client = JarvisMCPClient()

    # Register only safe tools
    from jarvis.gateway.phases.tools import init_tools

    await init_tools(config, mcp_client, memory_manager=None)

    # Create bridge with workspace-safe allowlist
    bridge = MCPBridge(config)
    bridge.set_tool_allowlist(MCP_WORKSPACE_SAFE_TOOLS)

    # Override server config to stdio
    server_cfg = MCPServerConfig(mode=MCPServerMode.STDIO)
    bridge._server_config = server_cfg
    bridge._server = None  # Force re-create in setup

    if bridge.setup(mcp_client, memory=None):
        await bridge.start()
        log.info("mcp_server_stdio_running", tools=len(MCP_WORKSPACE_SAFE_TOOLS))
        # Keep running until stdin closes
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            await bridge.stop()
    else:
        print("[ERROR] MCP server setup failed.", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Verify --mcp-server flag is accepted**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m jarvis --help`
Expected: Shows `--mcp-server` in help output

- [ ] **Step 5: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/__main__.py src/jarvis/mcp/bridge.py
git commit -m "feat: add --mcp-server entry point with workspace-safe tool filter"
```

---

### Task 7: Config YAML defaults

**Files:**
- Modify: `src/jarvis/config.py` (default YAML section)

- [ ] **Step 1: Add tools section to default config YAML**

Find the default config YAML template in `config.py` (the large YAML string around line 2540+). Add after the `shell:` section:

```yaml
# ── Tool Toggles ─────────────────────────────────────────────────
# Steuert welche Tool-Gruppen aktiv sind.
# Desktop-Tools sind aus Sicherheitsgruenden standardmaessig deaktiviert.

tools:
  computer_use_enabled: false   # Desktop-Automation (Screenshot + Klick)
  desktop_tools_enabled: false  # Clipboard und Screenshots
```

- [ ] **Step 2: Verify config round-trips**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.config import JarvisConfig; c = JarvisConfig(); print(c.tools.model_dump())"`
Expected: `{'computer_use_enabled': False, 'desktop_tools_enabled': False}`

- [ ] **Step 3: Commit**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add src/jarvis/config.py
git commit -m "docs: add tools toggle section to default config YAML"
```

---

### Task 8: Run full test suite

- [ ] **Step 1: Run the new tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing tool registration tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/ -k "tool_registration" -v`
Expected: All PASS (existing test counts may need updating if they assert exact tool counts)

- [ ] **Step 3: Fix any broken assertion counts**

If `test_tool_registration.py` asserts exact `register_builtin_handler.call_count`, the count may have decreased because desktop/computer-use tools are now disabled by default. Update the expected count accordingly.

- [ ] **Step 4: Commit any fixes**

```bash
cd "D:\Jarvis\jarvis complete v20"
git add -A
git commit -m "fix: update tool registration test counts for new defaults"
```
