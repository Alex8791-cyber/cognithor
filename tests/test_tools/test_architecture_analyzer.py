"""Tests fuer ArchitectureAnalyzer."""

import pytest
import tempfile
from pathlib import Path

from jarvis.tools.architecture_analyzer import ArchitectureAnalyzer


class TestArchitectureAnalyzer:
    def setup_method(self):
        self.analyzer = ArchitectureAnalyzer(base_package="pkg")
        self.tmpdir = tempfile.mkdtemp()

    def _write_py(self, name: str, content: str) -> Path:
        path = Path(self.tmpdir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_build_import_graph_empty_dir(self):
        count = self.analyzer.build_import_graph(self.tmpdir)
        assert count == 0

    def test_build_import_graph_counts_modules(self):
        self._write_py("a.py", "import os\n")
        self._write_py("b.py", "import sys\n")
        count = self.analyzer.build_import_graph(self.tmpdir)
        assert count == 2

    def test_internal_imports_tracked(self):
        self._write_py("a.py", "import pkg.b\n")
        self._write_py("b.py", "x = 1\n")
        self.analyzer.build_import_graph(self.tmpdir)
        graph = self.analyzer.import_graph
        assert "pkg.b" in graph.get("a", set())

    def test_external_imports_ignored(self):
        self._write_py("a.py", "import os\nimport json\n")
        self.analyzer.build_import_graph(self.tmpdir)
        graph = self.analyzer.import_graph
        assert graph.get("a", set()) == set()

    def test_detect_circular_imports(self):
        # Must use proper package structure so module names match imports
        (Path(self.tmpdir) / "pkg").mkdir()
        self._write_py("pkg/__init__.py", "")
        self._write_py("pkg/a.py", "import pkg.b\n")
        self._write_py("pkg/b.py", "import pkg.a\n")
        self.analyzer.build_import_graph(self.tmpdir)
        findings = self.analyzer.detect_circular_imports()
        assert len(findings) >= 1
        assert findings[0].finding_type == "circular_import"
        assert findings[0].severity == "error"

    def test_no_circular_imports(self):
        self._write_py("a.py", "import pkg.b\n")
        self._write_py("b.py", "x = 1\n")
        self.analyzer.build_import_graph(self.tmpdir)
        findings = self.analyzer.detect_circular_imports()
        assert len(findings) == 0

    def test_detect_layer_violation(self):
        # core (layer 2) importing gateway (layer 1) is a violation
        analyzer = ArchitectureAnalyzer(
            layer_config={"core": 2, "gateway": 1, "utils": 5},
            base_package="pkg",
        )
        (Path(self.tmpdir) / "core").mkdir()
        (Path(self.tmpdir) / "gateway").mkdir()
        self._write_py("core/__init__.py", "")
        self._write_py("gateway/__init__.py", "")
        self._write_py("core/service.py", "import pkg.gateway.handler\n")
        self._write_py("gateway/handler.py", "x = 1\n")
        analyzer.build_import_graph(self.tmpdir)
        findings = analyzer.detect_layer_violations()
        violations = [f for f in findings if f.finding_type == "layer_violation"]
        assert len(violations) >= 1

    def test_no_layer_violation_downward(self):
        # gateway (layer 1) importing core (layer 2) is fine
        analyzer = ArchitectureAnalyzer(
            layer_config={"core": 2, "gateway": 1, "utils": 5},
            base_package="pkg",
        )
        (Path(self.tmpdir) / "core").mkdir()
        (Path(self.tmpdir) / "gateway").mkdir()
        self._write_py("core/__init__.py", "")
        self._write_py("gateway/__init__.py", "")
        self._write_py("gateway/handler.py", "import pkg.core.service\n")
        self._write_py("core/service.py", "x = 1\n")
        analyzer.build_import_graph(self.tmpdir)
        findings = analyzer.detect_layer_violations()
        assert len(findings) == 0

    def test_utils_always_allowed(self):
        analyzer = ArchitectureAnalyzer(
            layer_config={"core": 2, "gateway": 1, "utils": 5},
            base_package="pkg",
        )
        (Path(self.tmpdir) / "core").mkdir()
        (Path(self.tmpdir) / "utils").mkdir()
        self._write_py("core/__init__.py", "")
        self._write_py("utils/__init__.py", "")
        self._write_py("core/service.py", "import pkg.utils.helpers\n")
        self._write_py("utils/helpers.py", "x = 1\n")
        analyzer.build_import_graph(self.tmpdir)
        findings = analyzer.detect_layer_violations()
        assert len(findings) == 0

    def test_dependency_metrics(self):
        self._write_py("a.py", "import pkg.b\nimport pkg.c\n")
        self._write_py("b.py", "import pkg.c\n")
        self._write_py("c.py", "x = 1\n")
        self.analyzer.build_import_graph(self.tmpdir)
        metrics = self.analyzer.get_dependency_metrics()
        # c is imported by a and b → afferent = 2
        assert metrics["pkg.c"]["afferent_coupling"] == 2
        # c imports nothing → efferent = 0
        assert metrics["pkg.c"]["efferent_coupling"] == 0
        # a imports b and c → efferent = 2
        assert metrics["a"]["efferent_coupling"] == 2

    def test_instability_metric(self):
        self._write_py("a.py", "import pkg.b\n")
        self._write_py("b.py", "x = 1\n")
        self.analyzer.build_import_graph(self.tmpdir)
        metrics = self.analyzer.get_dependency_metrics()
        # b: Ca=1, Ce=0 → instability = 0 (stable)
        assert metrics["pkg.b"]["instability"] == 0.0
        # a: Ca=0, Ce=1 → instability = 1 (unstable)
        assert metrics["a"]["instability"] == 1.0

    def test_module_count_property(self):
        self._write_py("a.py", "x = 1\n")
        self._write_py("b.py", "y = 2\n")
        self.analyzer.build_import_graph(self.tmpdir)
        assert self.analyzer.module_count == 2

    def test_syntax_error_skipped(self):
        self._write_py("bad.py", "def (:\n")
        self._write_py("good.py", "x = 1\n")
        count = self.analyzer.build_import_graph(self.tmpdir)
        # bad.py should be skipped but good.py counted
        assert count >= 1

    def test_pycache_ignored(self):
        cache_dir = Path(self.tmpdir) / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("x = 1\n")
        self._write_py("real.py", "x = 1\n")
        count = self.analyzer.build_import_graph(self.tmpdir)
        assert count == 1

    def test_init_file_module_name(self):
        pkg_dir = Path(self.tmpdir) / "mypkg"
        pkg_dir.mkdir()
        self._write_py("mypkg/__init__.py", "import pkg.mypkg.sub\n")
        self._write_py("mypkg/sub.py", "x = 1\n")
        self.analyzer.build_import_graph(self.tmpdir)
        graph = self.analyzer.import_graph
        # __init__.py should map to "mypkg" not "mypkg.__init__"
        assert "mypkg" in graph
