"""Tests fuer CodeSmellDetector."""

import pytest
import tempfile
from pathlib import Path

from jarvis.tools.code_analyzer import (
    CodeSmellDetector,
    _count_lines,
    _max_nesting,
    _jaccard_similarity,
)


class TestHelpers:
    def test_jaccard_identical(self):
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(sim - 0.5) < 0.01  # 2/4

    def test_jaccard_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0
        assert _jaccard_similarity(set(), {"a"}) == 0.0


class TestCodeSmellDetector:
    def setup_method(self):
        self.detector = CodeSmellDetector()
        self.tmpdir = tempfile.mkdtemp()

    def _write_py(self, name: str, content: str) -> Path:
        path = Path(self.tmpdir) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_detect_long_function(self):
        # 60-line function
        lines = ["    pass\n"] * 58
        code = "def long_func():\n" + "".join(lines)
        path = self._write_py("long.py", code)

        smells = self.detector.analyze_file(path)
        long_funcs = [s for s in smells if s.smell_type == "long_function"]
        assert len(long_funcs) == 1
        assert "long_func" in long_funcs[0].message

    def test_no_smell_short_function(self):
        code = "def short():\n    return 42\n"
        path = self._write_py("short.py", code)
        smells = self.detector.analyze_file(path)
        long_funcs = [s for s in smells if s.smell_type == "long_function"]
        assert len(long_funcs) == 0

    def test_detect_deep_nesting(self):
        code = (
            "def nested():\n"
            "    if True:\n"
            "        if True:\n"
            "            if True:\n"
            "                if True:\n"
            "                    if True:\n"
            "                        pass\n"
        )
        path = self._write_py("nested.py", code)
        smells = self.detector.analyze_file(path)
        deep = [s for s in smells if s.smell_type == "deep_nesting"]
        assert len(deep) == 1

    def test_detect_too_many_params(self):
        code = "def many(a, b, c, d, e, f, g, h):\n    pass\n"
        path = self._write_py("params.py", code)
        smells = self.detector.analyze_file(path)
        params = [s for s in smells if s.smell_type == "too_many_params"]
        assert len(params) == 1

    def test_self_not_counted(self):
        code = "class Foo:\n    def bar(self, a, b, c):\n        pass\n"
        path = self._write_py("cls.py", code)
        smells = self.detector.analyze_file(path)
        params = [s for s in smells if s.smell_type == "too_many_params"]
        assert len(params) == 0

    def test_detect_god_class(self):
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(20))
        code = f"class GodClass:\n{methods}\n"
        path = self._write_py("god.py", code)
        smells = self.detector.analyze_file(path)
        gods = [s for s in smells if s.smell_type == "god_class"]
        assert len(gods) == 1

    def test_syntax_error(self):
        path = self._write_py("bad.py", "def (:\n")
        smells = self.detector.analyze_file(path)
        assert any(s.smell_type == "syntax_error" for s in smells)

    def test_nonexistent_file(self):
        smells = self.detector.analyze_file("/nonexistent/file.py")
        assert smells == []

    def test_analyze_directory(self):
        self._write_py("a.py", "def short(): pass\n")
        self._write_py("b.py", "x = 1\n")
        smells = self.detector.analyze_directory(self.tmpdir)
        # No major smells in simple files
        assert isinstance(smells, list)

    def test_non_py_file_ignored(self):
        path = Path(self.tmpdir) / "readme.txt"
        path.write_text("hello")
        smells = self.detector.analyze_file(path)
        assert smells == []

    def test_custom_thresholds(self):
        detector = CodeSmellDetector(max_function_lines=5)
        code = "def f():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n    e = 5\n    f = 6\n"
        path = self._write_py("custom.py", code)
        smells = detector.analyze_file(path)
        assert any(s.smell_type == "long_function" for s in smells)
