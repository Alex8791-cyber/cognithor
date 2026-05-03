"""Chart- und Visualisierungs-Tools fuer Cognithor -- Diagramme als MCP-Tools.

Tools:
  - create_chart: Erstellt ein Diagramm aus Daten (bar/line/pie/scatter/hbar)
  - create_table_image: Rendert Daten als formatierte Tabelle als Bild
  - chart_from_csv: Erstellt ein Diagramm direkt aus einer CSV-Datei

Factory: register_chart_tools(mcp_client, config) -> ChartTools

Bibel-Referenz: $5.3 (MCP-Tools)
"""

from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_MAX_DATA_POINTS = 10_000
_DEFAULT_WIDTH = 10
_DEFAULT_HEIGHT = 6
_DPI = 150

# Cognithor dark theme
_BG_COLOR = "#0d1117"
_TEXT_COLOR = "#e6edf3"
_ACCENT_COLOR = "#58a6ff"
_GRID_COLOR = "#30363d"
_DEFAULT_COLORS = [
    "#58a6ff",  # blue
    "#3fb950",  # green
    "#d29922",  # yellow
    "#f85149",  # red
    "#bc8cff",  # purple
    "#39d2c0",  # teal
    "#f778ba",  # pink
    "#e3b341",  # gold
    "#79c0ff",  # light blue
    "#7ee787",  # light green
]

# ASCII chart block characters
_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

__all__ = [
    "ChartError",
    "ChartTools",
    "register_chart_tools",
]


class ChartError(Exception):
    """Fehler bei Chart-Operationen."""


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:60] if slug else "chart"


def _parse_data(data: Any) -> list[dict[str, Any]]:
    """Parse data from list-of-dicts or CSV string.

    Returns a list of dicts with string keys.
    """
    if isinstance(data, list):
        if not data:
            raise ChartError("Leere Datenliste.")
        if isinstance(data[0], dict):
            return data
        raise ChartError("Daten muessen eine Liste von Dicts sein oder ein CSV-String.")

    if isinstance(data, str):
        # Try parsing as CSV
        data = data.strip()
        if not data:
            raise ChartError("Leerer CSV-String.")
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader)
        if not rows:
            raise ChartError("CSV enthielt keine Datenzeilen.")
        return rows

    raise ChartError(f"Unbekanntes Datenformat: {type(data).__name__}")


def _to_float(value: Any) -> float:
    """Convert a value to float, raise ChartError on failure."""
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        raise ChartError(f"Wert '{value}' ist nicht numerisch.") from exc


def _matplotlib_available() -> bool:
    """Check if matplotlib is importable."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# ASCII fallback chart
# --------------------------------------------------------------------------- #


def _ascii_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    width: int = 60,
) -> str:
    """Generate a simple ASCII bar chart as fallback."""
    if not values:
        return "(keine Daten)"

    max_val = max(abs(v) for v in values) if values else 1
    if max_val == 0:
        max_val = 1

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    max_label_len = max((len(str(lbl)) for lbl in labels), default=10)
    max_label_len = min(max_label_len, 25)

    for label, value in zip(labels, values, strict=False):
        label_str = str(label)[:max_label_len].ljust(max_label_len)
        bar_len = int(abs(value) / max_val * width)
        # Use Unicode block characters for the bar
        bar = _BLOCKS[-1] * bar_len
        lines.append(f"{label_str} | {bar} {value:.2f}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ChartTools class
# --------------------------------------------------------------------------- #


class ChartTools:
    """Erstellt Diagramme und Tabellen-Bilder.

    Primaer matplotlib; ASCII-Fallback wenn nicht verfuegbar.
    """

    def __init__(self, config: CognithorConfig) -> None:
        self._config = config
        self._workspace: Path = config.workspace_dir
        self._charts_dir: Path = self._workspace / "charts"
        self._charts_dir.mkdir(parents=True, exist_ok=True)
        self._allowed_roots: list[Path] = [
            Path(p).expanduser().resolve() for p in config.security.allowed_paths
        ]
        # Allow workspace
        ws_resolved = self._workspace.expanduser().resolve()
        if ws_resolved not in self._allowed_roots:
            self._allowed_roots.append(ws_resolved)

    def _validate_path(self, path_str: str) -> Path:
        """Validate a file path is within allowed directories."""
        try:
            path = Path(path_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise ChartError(f"Ungueltiger Pfad: {path_str}") from exc

        for root in self._allowed_roots:
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue

        raise ChartError(f"Zugriff verweigert: {path_str} liegt ausserhalb erlaubter Verzeichnisse")

    def _output_path(self, title: str) -> Path:
        """Generate an output path for a chart image."""
        slug = _slugify(title) if title else "chart"
        timestamp = int(time.time())
        filename = f"{slug}_{timestamp}.png"
        return self._charts_dir / filename

    # ------------------------------------------------------------------ #
    # create_chart
    # ------------------------------------------------------------------ #

    async def create_chart(
        self,
        data: Any,
        chart_type: str = "bar",
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        x_key: str = "",
        y_key: str = "",
        colors: list[str] | None = None,
        width: float = _DEFAULT_WIDTH,
        height: float = _DEFAULT_HEIGHT,
    ) -> str:
        """Create a chart from data.

        Args:
            data: List of dicts or CSV string.
            chart_type: bar, line, pie, scatter, hbar.
            title: Chart title.
            x_label: X axis label.
            y_label: Y axis label.
            x_key: Column name for X axis.
            y_key: Column name for Y axis.
            colors: Optional list of color hex codes.
            width: Figure width in inches (default 10).
            height: Figure height in inches (default 6).

        Returns:
            File path and dimensions.
        """
        parsed = _parse_data(data)

        if len(parsed) > _MAX_DATA_POINTS:
            raise ChartError(f"Zu viele Datenpunkte: {len(parsed)} (max: {_MAX_DATA_POINTS})")

        valid_types = {"bar", "line", "pie", "scatter", "hbar"}
        if chart_type not in valid_types:
            raise ChartError(
                f"Unbekannter Chart-Typ: {chart_type}. Erlaubt: {', '.join(sorted(valid_types))}"
            )

        # Auto-detect keys if not given
        if not x_key or not y_key:
            keys = list(parsed[0].keys())
            if len(keys) < 2 and chart_type != "pie":
                raise ChartError("Mindestens 2 Spalten benoetigt (x_key und y_key).")
            if not x_key:
                x_key = keys[0]
            if not y_key:
                y_key = keys[1] if len(keys) > 1 else keys[0]

        # Extract data
        x_values = [row.get(x_key, "") for row in parsed]
        y_values = [_to_float(row.get(y_key, 0)) for row in parsed]

        if not _matplotlib_available():
            # ASCII fallback
            labels = [str(v) for v in x_values]
            return _ascii_bar_chart(labels, y_values, title or "Chart")

        # Matplotlib rendering
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width, height), dpi=_DPI)

        # Apply dark theme
        fig.set_facecolor(_BG_COLOR)
        ax.set_facecolor(_BG_COLOR)
        ax.tick_params(colors=_TEXT_COLOR)
        ax.xaxis.label.set_color(_TEXT_COLOR)
        ax.yaxis.label.set_color(_TEXT_COLOR)
        ax.title.set_color(_TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(_GRID_COLOR)
        ax.grid(True, color=_GRID_COLOR, linewidth=0.5, alpha=0.5)

        chart_colors = colors or _DEFAULT_COLORS
        # Ensure enough colors by cycling
        while len(chart_colors) < len(x_values):
            chart_colors = chart_colors + _DEFAULT_COLORS

        if chart_type == "bar":
            ax.bar(
                range(len(x_values)),
                y_values,
                color=chart_colors[: len(x_values)],
            )
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(
                [str(v) for v in x_values], rotation=45, ha="right", color=_TEXT_COLOR
            )
        elif chart_type == "hbar":
            ax.barh(
                range(len(x_values)),
                y_values,
                color=chart_colors[: len(x_values)],
            )
            ax.set_yticks(range(len(x_values)))
            ax.set_yticklabels([str(v) for v in x_values], color=_TEXT_COLOR)
        elif chart_type == "line":
            ax.plot(
                range(len(x_values)),
                y_values,
                color=_ACCENT_COLOR,
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(
                [str(v) for v in x_values], rotation=45, ha="right", color=_TEXT_COLOR
            )
        elif chart_type == "pie":
            # For pie, use a separate axis without background grid
            ax.grid(False)
            wedges, texts, autotexts = ax.pie(
                y_values,
                labels=[str(v) for v in x_values],
                colors=chart_colors[: len(x_values)],
                autopct="%1.1f%%",
                textprops={"color": _TEXT_COLOR},
            )
            for text in autotexts:
                text.set_color(_TEXT_COLOR)
        elif chart_type == "scatter":
            # For scatter, try to convert x values to floats too
            try:
                x_float = [float(v) for v in x_values]
            except (ValueError, TypeError):
                x_float = list(range(len(x_values)))
            ax.scatter(
                x_float,
                y_values,
                color=_ACCENT_COLOR,
                s=50,
                alpha=0.8,
            )

        if title:
            ax.set_title(title, color=_TEXT_COLOR, fontsize=14, fontweight="bold")
        if x_label:
            ax.set_xlabel(x_label, color=_TEXT_COLOR)
        if y_label:
            ax.set_ylabel(y_label, color=_TEXT_COLOR)

        plt.tight_layout()

        out_path = self._output_path(title or chart_type)
        fig.savefig(
            str(out_path),
            dpi=_DPI,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)

        # Get actual image dimensions
        img_w = int(width * _DPI)
        img_h = int(height * _DPI)

        log.info(
            "chart_created",
            path=str(out_path),
            chart_type=chart_type,
            data_points=len(parsed),
        )

        return t(
            "tools.chart_created",
            path=str(out_path),
            chart_type=chart_type,
            count=len(parsed),
            width=img_w,
            height=img_h,
        )

    # ------------------------------------------------------------------ #
    # create_table_image
    # ------------------------------------------------------------------ #

    async def create_table_image(
        self,
        data: Any,
        title: str = "",
        highlight_max: bool = False,
    ) -> str:
        """Render data as a formatted table image.

        Args:
            data: List of dicts.
            title: Optional title.
            highlight_max: Highlight maximum values in each numeric column.

        Returns:
            File path.
        """
        parsed = _parse_data(data)

        if len(parsed) > _MAX_DATA_POINTS:
            raise ChartError(f"Zu viele Datenpunkte: {len(parsed)} (max: {_MAX_DATA_POINTS})")

        if not _matplotlib_available():
            # ASCII fallback: simple text table
            if not parsed:
                return "(keine Daten)"
            columns = list(parsed[0].keys())
            widths = [len(c) for c in columns]
            for row in parsed:
                for i, col in enumerate(columns):
                    widths[i] = max(widths[i], len(str(row.get(col, ""))))
            header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
            sep = "-+-".join("-" * w for w in widths)
            lines = [header, sep]
            for row in parsed:
                line = " | ".join(
                    str(row.get(col, "")).ljust(widths[i]) for i, col in enumerate(columns)
                )
                lines.append(line)
            if title:
                lines.insert(0, title)
                lines.insert(1, "=" * len(title))
            return "\n".join(lines)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        columns = list(parsed[0].keys())
        cell_text = [[str(row.get(col, "")) for col in columns] for row in parsed]

        # Find max values per column (for highlight)
        max_indices: dict[int, int] = {}
        if highlight_max:
            for col_idx, col in enumerate(columns):
                max_val = None
                max_row = -1
                for row_idx, row in enumerate(parsed):
                    try:
                        val = float(row.get(col, ""))
                        if max_val is None or val > max_val:
                            max_val = val
                            max_row = row_idx
                    except (ValueError, TypeError):
                        continue
                if max_row >= 0:
                    max_indices[col_idx] = max_row

        # Calculate figure size
        n_rows = len(parsed)
        n_cols = len(columns)
        fig_width = max(6, n_cols * 2.0)
        fig_height = max(2, (n_rows + 1) * 0.4 + (1.0 if title else 0.2))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=_DPI)
        fig.set_facecolor(_BG_COLOR)
        ax.set_facecolor(_BG_COLOR)
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Style the table
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor(_GRID_COLOR)
            if row_idx == 0:
                # Header row
                cell.set_facecolor("#21262d")
                cell.set_text_props(color=_ACCENT_COLOR, fontweight="bold")
            else:
                cell.set_facecolor(_BG_COLOR)
                cell.set_text_props(color=_TEXT_COLOR)

                # Highlight max values
                if highlight_max and col_idx in max_indices and max_indices[col_idx] == row_idx - 1:
                    cell.set_facecolor("#1a3a2a")
                    cell.set_text_props(color="#3fb950", fontweight="bold")

        if title:
            ax.set_title(title, color=_TEXT_COLOR, fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()

        out_path = self._output_path(title or "table")
        fig.savefig(
            str(out_path),
            dpi=_DPI,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)

        log.info("table_image_created", path=str(out_path), rows=n_rows, cols=n_cols)

        return t("tools.table_created", path=str(out_path), rows=n_rows, cols=n_cols)

    # ------------------------------------------------------------------ #
    # chart_from_csv
    # ------------------------------------------------------------------ #

    async def chart_from_csv(
        self,
        file_path: str,
        chart_type: str = "bar",
        x_column: str = "",
        y_column: str = "",
        title: str = "",
    ) -> str:
        """Generate chart from a CSV file.

        Args:
            file_path: Path to CSV file within workspace.
            chart_type: Chart type (bar/line/pie/scatter/hbar).
            x_column: Column name for X axis.
            y_column: Column name for Y axis.
            title: Chart title.

        Returns:
            File path and dimensions.
        """
        validated = self._validate_path(file_path)

        if not validated.exists():
            raise ChartError(t("fs.file_not_found", path=str(file_path)))

        if not validated.suffix.lower() == ".csv":
            raise ChartError(f"Keine CSV-Datei: {file_path}")

        try:
            content = validated.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = validated.read_text(encoding="latin-1")
            except Exception as exc:
                raise ChartError(f"Datei nicht lesbar: {exc}") from exc

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            raise ChartError("CSV-Datei enthaelt keine Datenzeilen.")

        if len(rows) > _MAX_DATA_POINTS:
            raise ChartError(f"Zu viele Datenpunkte: {len(rows)} (max: {_MAX_DATA_POINTS})")

        if not title:
            title = validated.stem.replace("_", " ").replace("-", " ").title()

        return await self.create_chart(
            data=rows,
            chart_type=chart_type,
            title=title,
            x_key=x_column,
            y_key=y_column,
        )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def register_chart_tools(
    mcp_client: Any,
    config: CognithorConfig,
) -> ChartTools:
    """Registriert Chart-Tools beim MCP-Client.

    Returns:
        ChartTools-Instanz.
    """
    tools = ChartTools(config)

    mcp_client.register_builtin_handler(
        "create_chart",
        tools.create_chart,
        description=(
            "Erstellt ein Diagramm aus Daten (bar, line, pie, scatter, hbar). "
            "Akzeptiert Daten als Liste von Dicts oder CSV-String. "
            "Speichert als PNG im Workspace. Dark-Theme."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": (
                        "Daten als Liste von Dicts oder CSV-String. "
                        "Jedes Dict ist eine Zeile mit Spaltenname -> Wert."
                    ),
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "scatter", "hbar"],
                    "description": "Chart-Typ",
                    "default": "bar",
                },
                "title": {
                    "type": "string",
                    "description": "Diagramm-Titel",
                    "default": "",
                },
                "x_label": {
                    "type": "string",
                    "description": "Beschriftung der X-Achse",
                    "default": "",
                },
                "y_label": {
                    "type": "string",
                    "description": "Beschriftung der Y-Achse",
                    "default": "",
                },
                "x_key": {
                    "type": "string",
                    "description": "Spaltenname fuer X-Achse (auto-detect wenn leer)",
                    "default": "",
                },
                "y_key": {
                    "type": "string",
                    "description": "Spaltenname fuer Y-Achse (auto-detect wenn leer)",
                    "default": "",
                },
                "colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optionale Farbliste (Hex-Codes)",
                    "default": None,
                },
                "width": {
                    "type": "number",
                    "description": "Breite in Zoll (Default: 10)",
                    "default": _DEFAULT_WIDTH,
                },
                "height": {
                    "type": "number",
                    "description": "Hoehe in Zoll (Default: 6)",
                    "default": _DEFAULT_HEIGHT,
                },
            },
            "required": ["data"],
        },
    )

    mcp_client.register_builtin_handler(
        "create_table_image",
        tools.create_table_image,
        description=(
            "Rendert Daten als formatierte Tabelle in einem Bild (PNG). "
            "Optionales Highlighting der Maximalwerte. Dark-Theme."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": "Daten als Liste von Dicts (jedes Dict = eine Zeile).",
                },
                "title": {
                    "type": "string",
                    "description": "Optionaler Titel",
                    "default": "",
                },
                "highlight_max": {
                    "type": "boolean",
                    "description": "Maximalwerte pro Spalte hervorheben",
                    "default": False,
                },
            },
            "required": ["data"],
        },
    )

    mcp_client.register_builtin_handler(
        "chart_from_csv",
        tools.chart_from_csv,
        description=(
            "Erstellt ein Diagramm direkt aus einer CSV-Datei. "
            "Liest die Datei, erkennt Spalten automatisch, erstellt PNG."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Pfad zur CSV-Datei (im Workspace)",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "scatter", "hbar"],
                    "description": "Chart-Typ",
                    "default": "bar",
                },
                "x_column": {
                    "type": "string",
                    "description": "Spaltenname fuer X-Achse",
                    "default": "",
                },
                "y_column": {
                    "type": "string",
                    "description": "Spaltenname fuer Y-Achse",
                    "default": "",
                },
                "title": {
                    "type": "string",
                    "description": "Diagramm-Titel (Default: Dateiname)",
                    "default": "",
                },
            },
            "required": ["file_path"],
        },
    )

    log.info(
        "chart_tools_registered",
        tools=["create_chart", "create_table_image", "chart_from_csv"],
    )
    return tools
