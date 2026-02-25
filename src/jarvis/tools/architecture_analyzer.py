"""Import-Graph-basierte Architektur-Analyse.

Erkennt:
  - Zirkulaere Imports (DFS-Zyklen)
  - Layer-Verletzungen (z.B. core importiert gateway)
  - Afferent/Efferent Coupling und Instabilitaet
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from jarvis.models import ArchitectureFinding
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Default-Layer-Definition (nach unten erlaubt)
# channels → gateway → core → memory → models
DEFAULT_LAYERS: dict[str, int] = {
    "channels": 0,
    "gateway": 1,
    "core": 2,
    "memory": 3,
    "models": 4,
    "utils": 5,  # Utilities duerfen von ueberall importiert werden
}


class ArchitectureAnalyzer:
    """Import-Graph-Aufbau und Architektur-Analyse."""

    def __init__(
        self,
        layer_config: dict[str, int] | None = None,
        base_package: str = "jarvis",
    ) -> None:
        self._layers = layer_config or DEFAULT_LAYERS
        self._base_package = base_package
        # Import-Graph: module -> set of imported modules
        self._import_graph: dict[str, set[str]] = defaultdict(set)
        self._file_modules: dict[str, str] = {}  # file_path -> module_name

    def build_import_graph(self, src_dir: str | Path) -> int:
        """Baut den Import-Graph aus einem Quellverzeichnis auf.

        Returns:
            Anzahl analysierter Module.
        """
        src_dir = Path(src_dir)
        self._import_graph.clear()
        self._file_modules.clear()

        count = 0
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            module_name = self._path_to_module(py_file, src_dir)
            if not module_name:
                continue

            self._file_modules[str(py_file)] = module_name
            imports = self._extract_imports(py_file)

            # Nur interne Imports tracken
            internal_imports = {
                imp for imp in imports
                if imp.startswith(self._base_package + ".")
            }
            self._import_graph[module_name] = internal_imports
            count += 1

        return count

    def detect_circular_imports(self) -> list[ArchitectureFinding]:
        """Erkennt zirkulaere Imports via DFS."""
        findings: list[ArchitectureFinding] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        found_cycles: set[frozenset[str]] = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._import_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Zyklus gefunden
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycle_key = frozenset(cycle)
                    if cycle_key not in found_cycles:
                        found_cycles.add(cycle_key)
                        findings.append(
                            ArchitectureFinding(
                                finding_type="circular_import",
                                severity="error",
                                modules=cycle,
                                message=f"Zirkulaerer Import: {' -> '.join(cycle)} -> {neighbor}",
                            )
                        )

            path.pop()
            rec_stack.discard(node)

        for module in self._import_graph:
            if module not in visited:
                dfs(module, [])

        return findings

    def detect_layer_violations(self) -> list[ArchitectureFinding]:
        """Erkennt Layer-Verletzungen basierend auf Layer-Konfiguration.

        Eine Verletzung liegt vor wenn ein tieferes Layer ein hoeheres importiert.
        (z.B. core importiert gateway)
        """
        findings: list[ArchitectureFinding] = []

        for module, imports in self._import_graph.items():
            src_layer = self._get_layer(module)
            if src_layer is None:
                continue

            for imp in imports:
                dst_layer = self._get_layer(imp)
                if dst_layer is None:
                    continue

                # Utils darf ueberall importiert werden
                dst_layer_name = self._get_layer_name(imp)
                if dst_layer_name == "utils":
                    continue

                # Verletzung: Import von einer hoeheren Schicht (niedrigerer Layer-Wert)
                if dst_layer < src_layer:
                    findings.append(
                        ArchitectureFinding(
                            finding_type="layer_violation",
                            severity="warning",
                            modules=[module, imp],
                            message=(
                                f"Layer-Verletzung: {module} (Layer {self._get_layer_name(module)}) "
                                f"importiert {imp} (Layer {self._get_layer_name(imp)})"
                            ),
                        )
                    )

        return findings

    def get_dependency_metrics(self) -> dict[str, dict[str, Any]]:
        """Berechnet Afferent/Efferent Coupling und Instabilitaet pro Modul.

        - Ca (Afferent Coupling): Wie viele Module importieren dieses Modul
        - Ce (Efferent Coupling): Wie viele Module importiert dieses Modul
        - Instability = Ce / (Ca + Ce)  [0=stabil, 1=instabil]
        """
        # Afferent: wer importiert mich?
        afferent: dict[str, int] = defaultdict(int)
        # Efferent: wen importiere ich?
        efferent: dict[str, int] = {}

        for module, imports in self._import_graph.items():
            efferent[module] = len(imports)
            for imp in imports:
                afferent[imp] += 1

        metrics: dict[str, dict[str, Any]] = {}
        all_modules = set(self._import_graph.keys()) | set(afferent.keys())

        for module in all_modules:
            ca = afferent.get(module, 0)
            ce = efferent.get(module, 0)
            instability = ce / (ca + ce) if (ca + ce) > 0 else 0.0
            metrics[module] = {
                "afferent_coupling": ca,
                "efferent_coupling": ce,
                "instability": round(instability, 3),
            }

        return metrics

    def _get_layer(self, module: str) -> int | None:
        """Ermittelt die Layer-Nummer eines Moduls."""
        # Extract layer name from module path
        # e.g. "jarvis.core.planner" -> "core"
        parts = module.split(".")
        for part in parts:
            if part in self._layers:
                return self._layers[part]
        return None

    def _get_layer_name(self, module: str) -> str:
        """Ermittelt den Layer-Namen eines Moduls."""
        parts = module.split(".")
        for part in parts:
            if part in self._layers:
                return part
        return "unknown"

    def _path_to_module(self, file_path: Path, src_dir: Path) -> str:
        """Konvertiert einen Dateipfad in einen Modulnamen."""
        try:
            rel = file_path.relative_to(src_dir)
        except ValueError:
            return ""

        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        return ".".join(parts) if parts else ""

    def _extract_imports(self, file_path: Path) -> set[str]:
        """Extrahiert alle Import-Pfade aus einer Python-Datei."""
        imports: set[str] = set()

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (OSError, SyntaxError):
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return imports

    @property
    def module_count(self) -> int:
        return len(self._import_graph)

    @property
    def import_graph(self) -> dict[str, set[str]]:
        return dict(self._import_graph)
