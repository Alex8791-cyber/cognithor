"""AST-basierte Code-Analyse fuer Code-Smell-Erkennung.

Detektiert:
  - Lange Funktionen (>50 Zeilen)
  - Tiefe Verschachtelung (>4 Level)
  - Zu viele Parameter (>6)
  - God-Classes (>15 Methoden)
  - Duplikate (Jaccard-Similarity auf Funktions-Bodies)
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from jarvis.models import CodeSmell
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Schwellenwerte
MAX_FUNCTION_LINES = 50
MAX_NESTING_DEPTH = 4
MAX_PARAMETERS = 6
MAX_CLASS_METHODS = 15
DUPLICATE_THRESHOLD = 0.7  # Jaccard Similarity


def _count_lines(node: ast.AST) -> int:
    """Zaehlt die Zeilen eines AST-Knotens."""
    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
        return (node.end_lineno or node.lineno) - node.lineno + 1
    return 0


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Berechnet die maximale Verschachtelungstiefe."""
    max_d = depth
    nesting_nodes = (
        ast.If, ast.For, ast.While, ast.With,
        ast.Try, ast.ExceptHandler,
        ast.AsyncFor, ast.AsyncWith,
    )
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_nodes):
            max_d = max(max_d, _max_nesting(child, depth + 1))
        else:
            max_d = max(max_d, _max_nesting(child, depth))
    return max_d


def _function_body_tokens(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Extrahiert Token-Set aus einem Funktions-Body fuer Duplikaterkennung."""
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.add(child.id)
        elif isinstance(child, ast.Attribute):
            tokens.add(child.attr)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            tokens.add(f"call:{child.func.id}")
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            tokens.add(f"str:{child.value[:20]}")
    return tokens


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard-Similarity zweier Sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class CodeSmellDetector:
    """AST-basierte Code-Smell-Erkennung."""

    def __init__(
        self,
        max_function_lines: int = MAX_FUNCTION_LINES,
        max_nesting_depth: int = MAX_NESTING_DEPTH,
        max_parameters: int = MAX_PARAMETERS,
        max_class_methods: int = MAX_CLASS_METHODS,
        duplicate_threshold: float = DUPLICATE_THRESHOLD,
    ) -> None:
        self._max_function_lines = max_function_lines
        self._max_nesting_depth = max_nesting_depth
        self._max_parameters = max_parameters
        self._max_class_methods = max_class_methods
        self._duplicate_threshold = duplicate_threshold

    def analyze_file(self, path: str | Path) -> list[CodeSmell]:
        """Analysiert eine einzelne Python-Datei."""
        path = Path(path)
        if not path.exists() or not path.suffix == ".py":
            return []

        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            log.warning("code_analyzer_read_error", path=str(path), error=str(e))
            return []

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return [
                CodeSmell(
                    file_path=str(path),
                    line=e.lineno or 0,
                    smell_type="syntax_error",
                    severity="error",
                    message=f"Syntaxfehler: {e.msg}",
                )
            ]

        smells: list[CodeSmell] = []
        file_str = str(path)

        # Sammle alle Funktionen fuer Duplikaterkennung
        func_tokens: list[tuple[str, int, set[str]]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                smells.extend(self._check_function(node, file_str))
                tokens = _function_body_tokens(node)
                if len(tokens) >= 5:  # Nur Funktionen mit genug Inhalt
                    func_tokens.append((node.name, node.lineno, tokens))

            elif isinstance(node, ast.ClassDef):
                smells.extend(self._check_class(node, file_str))

        # Duplikaterkennung
        smells.extend(self._check_duplicates(func_tokens, file_str))

        return smells

    def analyze_directory(
        self, path: str | Path, recursive: bool = True,
    ) -> list[CodeSmell]:
        """Analysiert ein Verzeichnis."""
        path = Path(path)
        if not path.is_dir():
            return []

        all_smells: list[CodeSmell] = []
        glob_pattern = "**/*.py" if recursive else "*.py"

        for py_file in path.glob(glob_pattern):
            # __pycache__ ueberspringen
            if "__pycache__" in str(py_file):
                continue
            all_smells.extend(self.analyze_file(py_file))

        return all_smells

    def _check_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str,
    ) -> list[CodeSmell]:
        """Prueft eine Funktion auf Smells."""
        smells: list[CodeSmell] = []

        # Lange Funktionen
        lines = _count_lines(node)
        if lines > self._max_function_lines:
            smells.append(
                CodeSmell(
                    file_path=file_path,
                    line=node.lineno,
                    smell_type="long_function",
                    severity="warning",
                    message=f"Funktion '{node.name}' hat {lines} Zeilen (max: {self._max_function_lines})",
                    suggestion=f"Funktion '{node.name}' in kleinere Funktionen aufteilen",
                )
            )

        # Tiefe Verschachtelung
        depth = _max_nesting(node)
        if depth > self._max_nesting_depth:
            smells.append(
                CodeSmell(
                    file_path=file_path,
                    line=node.lineno,
                    smell_type="deep_nesting",
                    severity="warning",
                    message=f"Funktion '{node.name}' hat Verschachtelungstiefe {depth} (max: {self._max_nesting_depth})",
                    suggestion="Early returns oder Guard-Clauses verwenden",
                )
            )

        # Zu viele Parameter
        params = node.args
        param_count = (
            len(params.args)
            + len(params.posonlyargs)
            + len(params.kwonlyargs)
        )
        # self/cls abziehen
        if params.args and params.args[0].arg in ("self", "cls"):
            param_count -= 1

        if param_count > self._max_parameters:
            smells.append(
                CodeSmell(
                    file_path=file_path,
                    line=node.lineno,
                    smell_type="too_many_params",
                    severity="warning",
                    message=f"Funktion '{node.name}' hat {param_count} Parameter (max: {self._max_parameters})",
                    suggestion="Parameter-Objekt oder Builder-Pattern verwenden",
                )
            )

        return smells

    def _check_class(self, node: ast.ClassDef, file_path: str) -> list[CodeSmell]:
        """Prueft eine Klasse auf Smells."""
        smells: list[CodeSmell] = []

        # God-Class: Zu viele Methoden
        methods = [
            n for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(methods) > self._max_class_methods:
            smells.append(
                CodeSmell(
                    file_path=file_path,
                    line=node.lineno,
                    smell_type="god_class",
                    severity="warning",
                    message=f"Klasse '{node.name}' hat {len(methods)} Methoden (max: {self._max_class_methods})",
                    suggestion="Klasse in kleinere Klassen mit Single Responsibility aufteilen",
                )
            )

        return smells

    def _check_duplicates(
        self,
        func_tokens: list[tuple[str, int, set[str]]],
        file_path: str,
    ) -> list[CodeSmell]:
        """Erkennt moegliche Duplikate ueber Jaccard-Similarity."""
        smells: list[CodeSmell] = []
        reported: set[tuple[str, str]] = set()

        for i in range(len(func_tokens)):
            name_a, line_a, tokens_a = func_tokens[i]
            for j in range(i + 1, len(func_tokens)):
                name_b, line_b, tokens_b = func_tokens[j]

                pair_key = (min(name_a, name_b), max(name_a, name_b))
                if pair_key in reported:
                    continue

                similarity = _jaccard_similarity(tokens_a, tokens_b)
                if similarity >= self._duplicate_threshold:
                    reported.add(pair_key)
                    smells.append(
                        CodeSmell(
                            file_path=file_path,
                            line=line_a,
                            smell_type="duplicate",
                            severity="info",
                            message=(
                                f"Funktionen '{name_a}' (Zeile {line_a}) und "
                                f"'{name_b}' (Zeile {line_b}) sind zu "
                                f"{similarity:.0%} aehnlich"
                            ),
                            suggestion="Gemeinsame Logik in eine Hilfsfunktion extrahieren",
                        )
                    )

        return smells
