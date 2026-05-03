"""Interactive config TUI using rich + prompt_toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from rich.console import Console
from rich.table import Table

from cognithor.cli.config_cmd import (
    _DISPLAY_FIELDS,
    _SENTINEL,
    _get_nested,
    _load_config,
    _save_config,
    _set_nested,
)
from cognithor.cli.model_registry import ModelRegistry

console = Console()

_SELECTION_OPTIONS: dict[str, list[str]] = {
    "llm_backend_type": ["ollama", "openai", "anthropic", "gemini", "lmstudio", "claude-code"],
    "language": ["de", "en", "zh", "ar"],
    "operation_mode": ["offline", "hybrid", "online"],
}

_MODEL_FIELDS = {"models.planner.name", "models.executor.name"}

_PORT_FIELDS = {"api_port"}


def _show_table(data: dict[str, Any]) -> None:
    """Print the numbered config table."""
    table = Table(title="Cognithor Configuration")
    table.add_column("#", style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Key", style="dim")
    table.add_column("Value", style="green")

    for idx, (label, dot_path) in enumerate(_DISPLAY_FIELDS, 1):
        val = _get_nested(data, dot_path)
        display = str(val) if val is not _SENTINEL else "[dim]-[/dim]"
        table.add_row(str(idx), label, dot_path, display)

    console.print(table)


def _prompt_selection(label: str, options: list[str], current: Any) -> str | None:
    """Show a numbered list and let the user pick. Returns None on cancel."""
    console.print(f"\n[bold]{label}[/bold]  (current: {current})")
    for i, opt in enumerate(options, 1):
        console.print(f"  {i}) {opt}")
    console.print("  0) Cancel")
    while True:
        raw = pt_prompt(f"Pick [0-{len(options)}]: ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print("[red]Invalid choice.[/red]")


def _prompt_port(current: Any) -> int | None:
    """Prompt for a port number (1024-65535). Returns None on cancel."""
    console.print(f"\n[bold]API Port[/bold]  (current: {current})")
    console.print("  Enter port (1024-65535) or 0 to cancel.")
    while True:
        raw = pt_prompt("Port: ").strip()
        if raw == "0":
            return None
        if raw.isdigit():
            val = int(raw)
            if 1024 <= val <= 65535:
                return val
        console.print("[red]Must be a number between 1024 and 65535.[/red]")


def _prompt_text(label: str, current: Any) -> str | None:
    """Free-text prompt with default. Returns None on empty (cancel)."""
    default = str(current) if current is not _SENTINEL else ""
    console.print(f"\n[bold]{label}[/bold]  (current: {current})")
    raw = pt_prompt("Value (Enter=keep): ", default=default).strip()
    if raw == default:
        return None
    return raw if raw else None


def _edit_field(idx: int, data: dict[str, Any]) -> bool:
    """Edit the field at 1-based *idx*. Returns True if changed."""
    if idx < 1 or idx > len(_DISPLAY_FIELDS):
        console.print("[red]Invalid field number.[/red]")
        return False

    label, dot_path = _DISPLAY_FIELDS[idx - 1]
    current = _get_nested(data, dot_path)
    current_display = current if current is not _SENTINEL else "-"

    # Static selection
    if dot_path in _SELECTION_OPTIONS:
        new_val = _prompt_selection(label, _SELECTION_OPTIONS[dot_path], current_display)
        if new_val is not None:
            _set_nested(data, dot_path, new_val)
            return True
        return False

    # Dynamic model list
    if dot_path in _MODEL_FIELDS:
        backend = _get_nested(data, "llm_backend_type")
        if backend is _SENTINEL:
            backend = "ollama"
        console.print(f"\n[dim]Discovering models for {backend} ...[/dim]")
        models = ModelRegistry().discover_models_sync(str(backend))
        new_val = _prompt_selection(label, models, current_display)
        if new_val == "[ Custom... ]":
            custom = pt_prompt("Enter custom model name: ").strip()
            if custom:
                _set_nested(data, dot_path, custom)
                return True
            return False
        if new_val is not None:
            _set_nested(data, dot_path, new_val)
            return True
        return False

    # Port
    if dot_path in _PORT_FIELDS:
        new_val = _prompt_port(current_display)
        if new_val is not None:
            _set_nested(data, dot_path, new_val)
            return True
        return False

    # Text (owner_name or fallback)
    new_val = _prompt_text(label, current_display)
    if new_val is not None:
        _set_nested(data, dot_path, new_val)
        return True
    return False


def launch(config_path: Path | None = None) -> None:
    """Launch the interactive config TUI."""
    if config_path is None:
        config_path = Path.home() / ".cognithor" / "config.yaml"

    # Ensure directory and file exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("{}\n", encoding="utf-8")

    data = _load_config(config_path)
    original = _deep_copy_dict(data)

    _show_table(data)

    while True:
        try:
            raw = pt_prompt("\nEdit [1-7, q=quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        if raw == "q":
            break

        if raw.isdigit():
            changed = _edit_field(int(raw), data)
            if changed:
                console.print()
                _show_table(data)
        else:
            console.print("[red]Enter a number 1-7 or q to quit.[/red]")

    # Check if anything changed
    if data != original:
        try:
            answer = pt_prompt("Save changes? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("", "y", "yes"):
            _save_config(data, config_path)
            console.print("[green]Configuration saved.[/green]")
        else:
            console.print("[yellow]Changes discarded.[/yellow]")
    else:
        console.print("[dim]No changes made.[/dim]")


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Simple deep copy for nested dicts of primitives."""
    import copy

    return copy.deepcopy(d)
