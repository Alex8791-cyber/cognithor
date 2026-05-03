"""Non-interactive config commands: get, set, list."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

_SENTINEL = object()

_DISPLAY_FIELDS: list[tuple[str, str]] = [
    ("LLM Backend", "llm_backend_type"),
    ("Planner Model", "models.planner.name"),
    ("Executor Model", "models.executor.name"),
    ("API Port", "api_port"),
    ("Language", "language"),
    ("Operation Mode", "operation_mode"),
    ("Owner Name", "owner_name"),
]

_DEFAULT_CONFIG: Path = Path.home() / ".cognithor" / "config.yaml"

console = Console()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """Load YAML config from *config_path* and return as dict."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save_config(data: dict[str, Any], config_path: Path = _DEFAULT_CONFIG) -> None:
    """Write *data* to *config_path* atomically via a temp file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        dir=config_path.parent,
        delete=False,
        encoding="utf-8",
    )
    try:
        yaml.safe_dump(data, tmp_fd, default_flow_style=False, allow_unicode=True)
        tmp_fd.close()
        Path(tmp_fd.name).replace(config_path)
    except BaseException:
        tmp_fd.close()
        Path(tmp_fd.name).unlink(missing_ok=True)
        raise


def _get_nested(data: dict[str, Any], dot_path: str) -> Any:
    """Traverse *data* by dot-separated *dot_path*. Return ``_SENTINEL`` on miss."""
    parts = dot_path.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return _SENTINEL
        current = current[part]
    return current


def _set_nested(data: dict[str, Any], dot_path: str, value: Any) -> None:
    """Set *value* at *dot_path* inside *data*, creating intermediate dicts.

    Numeric strings are auto-converted to ``int``.
    """
    if isinstance(value, str) and value.isdigit():
        value = int(value)

    parts = dot_path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


# ------------------------------------------------------------------
# Public commands
# ------------------------------------------------------------------


def cmd_get(key: str, config_path: Path = _DEFAULT_CONFIG) -> int:
    """Print the value for *key* (dot-path). Returns 0 on success, 1 if missing."""
    data = _load_config(config_path)
    result = _get_nested(data, key)
    if result is _SENTINEL:
        console.print(f"[red]Key not found:[/red] {key}")
        return 1
    console.print(result)
    return 0


def cmd_set(key: str, value: str, config_path: Path = _DEFAULT_CONFIG) -> int:
    """Set *key* to *value* and save. Returns 0."""
    data = _load_config(config_path)
    _set_nested(data, key, value)
    _save_config(data, config_path)
    console.print(f"[green]Set[/green] {key} = {_get_nested(data, key)}")
    return 0


def cmd_list(config_path: Path = _DEFAULT_CONFIG) -> int:
    """Print a rich table of all display fields. Returns 0."""
    data = _load_config(config_path)

    table = Table(title="Cognithor Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Key", style="dim")
    table.add_column("Value", style="green")

    for label, dot_path in _DISPLAY_FIELDS:
        val = _get_nested(data, dot_path)
        display = str(val) if val is not _SENTINEL else "[dim]-[/dim]"
        table.add_row(label, dot_path, display)

    console.print(table)
    return 0
