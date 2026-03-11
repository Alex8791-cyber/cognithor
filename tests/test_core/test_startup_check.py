"""Tests for jarvis.core.startup_check -- StartupChecker & StartupReport."""

from __future__ import annotations

import json
import subprocess
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from jarvis.core.startup_check import (
    StartupChecker,
    StartupReport,
    OPTIONAL_GROUPS,
    _import_name,
    _can_import,
    _pip_install,
    _http_get_json,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def mock_config() -> MagicMock:
    """Minimal mock of JarvisConfig with models + ollama sub-configs."""
    config = MagicMock()
    config.jarvis_home = Path(tempfile.gettempdir()) / "test_jarvis_home"
    config.llm_backend_type = "ollama"
    config.ollama.base_url = "http://localhost:11434"

    # Models
    config.models.planner.name = "qwen3:32b"
    config.models.executor.name = "qwen3:8b"
    config.models.coder.name = "qwen3-coder:30b"
    config.models.embedding.name = "nomic-embed-text"

    return config


@pytest.fixture()
def checker(mock_config: MagicMock) -> StartupChecker:
    return StartupChecker(mock_config)


@pytest.fixture()
def checker_no_config() -> StartupChecker:
    return StartupChecker(config=None)


# ============================================================================
# StartupReport
# ============================================================================


class TestStartupReport:
    """Tests for the StartupReport dataclass."""

    def test_report_defaults_empty(self) -> None:
        report = StartupReport()
        assert report.checks_passed == []
        assert report.fixes_applied == []
        assert report.warnings == []
        assert report.errors == []

    def test_report_ok_when_no_errors(self) -> None:
        report = StartupReport(warnings=["something"], fixes_applied=["fix"])
        assert report.ok is True

    def test_report_not_ok_when_errors(self) -> None:
        report = StartupReport(errors=["boom"])
        assert report.ok is False

    def test_report_merge(self) -> None:
        r1 = StartupReport(
            checks_passed=["a"],
            fixes_applied=["f1"],
            warnings=["w1"],
            errors=["e1"],
        )
        r2 = StartupReport(
            checks_passed=["b"],
            fixes_applied=["f2"],
            warnings=["w2"],
            errors=["e2"],
        )
        r1.merge(r2)
        assert r1.checks_passed == ["a", "b"]
        assert r1.fixes_applied == ["f1", "f2"]
        assert r1.warnings == ["w1", "w2"]
        assert r1.errors == ["e1", "e2"]

    def test_report_merge_empty(self) -> None:
        r1 = StartupReport(checks_passed=["x"])
        r2 = StartupReport()
        r1.merge(r2)
        assert r1.checks_passed == ["x"]


# ============================================================================
# _import_name helper
# ============================================================================


class TestImportName:
    """Tests for the _import_name helper function."""

    def test_beautifulsoup4(self) -> None:
        assert _import_name("beautifulsoup4") == "bs4"

    def test_python_telegram_bot(self) -> None:
        assert _import_name("python-telegram-bot") == "telegram"

    def test_faster_whisper(self) -> None:
        assert _import_name("faster-whisper") == "faster_whisper"

    def test_fpdf2(self) -> None:
        assert _import_name("fpdf2") == "fpdf"

    def test_python_docx(self) -> None:
        assert _import_name("python-docx") == "docx"

    def test_ddgs(self) -> None:
        assert _import_name("ddgs") == "duckduckgo_search"

    def test_sqlite_vec(self) -> None:
        assert _import_name("sqlite-vec") == "sqlite_vec"

    def test_faiss_cpu(self) -> None:
        assert _import_name("faiss-cpu") == "faiss"

    def test_piper_tts(self) -> None:
        assert _import_name("piper-tts") == "piper"

    def test_unknown_package_hyphen(self) -> None:
        assert _import_name("some-package") == "some_package"

    def test_version_specifier_stripped(self) -> None:
        assert _import_name("mcp>=1.7") == "mcp"

    def test_extras_stripped(self) -> None:
        assert _import_name("uvicorn[standard]") == "uvicorn"


# ============================================================================
# _can_import helper
# ============================================================================


class TestCanImport:
    """Tests for the _can_import helper."""

    def test_importable_module(self) -> None:
        assert _can_import("os") is True
        assert _can_import("sys") is True

    def test_non_importable_module(self) -> None:
        assert _can_import("nonexistent_module_xyz_12345") is False

    @patch("jarvis.core.startup_check.importlib.import_module", side_effect=ImportError)
    def test_import_error(self, _mock: MagicMock) -> None:
        assert _can_import("anything") is False


# ============================================================================
# _pip_install helper
# ============================================================================


class TestPipInstall:
    """Tests for the _pip_install helper (mocked subprocess)."""

    @patch("jarvis.core.startup_check.subprocess.run")
    def test_successful_install(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        success, stderr = _pip_install(["numpy"])
        assert success is True
        assert stderr == ""
        # Verify the command structure
        args = mock_run.call_args[0][0]
        assert args[0] == str(subprocess.sys.executable)
        assert "-m" in args
        assert "pip" in args
        assert "install" in args
        assert "numpy" in args

    @patch("jarvis.core.startup_check.subprocess.run")
    def test_failed_install(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="No matching distribution")
        success, stderr = _pip_install(["bogus-package"])
        assert success is False
        assert "No matching distribution" in stderr

    @patch(
        "jarvis.core.startup_check.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=300),
    )
    def test_timeout(self, _mock: MagicMock) -> None:
        success, stderr = _pip_install(["slow-package"])
        assert success is False
        assert "timed out" in stderr

    @patch("jarvis.core.startup_check.subprocess.run", side_effect=OSError("boom"))
    def test_os_error(self, _mock: MagicMock) -> None:
        success, stderr = _pip_install(["pkg"])
        assert success is False
        assert "boom" in stderr


# ============================================================================
# _http_get_json helper
# ============================================================================


class TestHttpGetJson:
    """Tests for the _http_get_json helper."""

    @patch("jarvis.core.startup_check.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        body = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
        ctx.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = ctx
        result = _http_get_json("http://localhost:11434/api/tags")
        assert result is not None
        assert "models" in result

    @patch("jarvis.core.startup_check.urllib.request.urlopen", side_effect=Exception("refused"))
    def test_connection_error(self, _mock: MagicMock) -> None:
        result = _http_get_json("http://localhost:99999/api/tags")
        assert result is None


# ============================================================================
# check_python_packages
# ============================================================================


class TestCheckPythonPackages:
    """Tests for StartupChecker.check_python_packages()."""

    @patch("jarvis.core.startup_check._can_import", return_value=True)
    def test_all_packages_present(self, _mock: MagicMock, checker: StartupChecker) -> None:
        report = checker.check_python_packages()
        assert len(report.fixes_applied) == 0
        assert len(report.warnings) == 0
        assert len(report.checks_passed) > 0

    @patch("jarvis.core.startup_check._pip_install", return_value=(True, ""))
    @patch("jarvis.core.startup_check._can_import", return_value=False)
    def test_missing_packages_auto_installed(
        self, _mock_import: MagicMock, _mock_pip: MagicMock, mock_config: MagicMock
    ) -> None:
        # F-010: auto_install=True required for automatic installation
        auto_checker = StartupChecker(mock_config, auto_install=True)
        report = auto_checker.check_python_packages()
        assert len(report.fixes_applied) > 0
        # Should have attempted pip install for each group
        assert _mock_pip.call_count == len(OPTIONAL_GROUPS)

    @patch("jarvis.core.startup_check._can_import", return_value=False)
    def test_missing_packages_without_auto_install_warns(
        self, _mock_import: MagicMock, checker: StartupChecker
    ) -> None:
        """F-010: Without --auto-install, missing packages only produce warnings."""
        report = checker.check_python_packages()
        assert len(report.warnings) > 0
        assert len(report.fixes_applied) == 0

    @patch("jarvis.core.startup_check._pip_install", return_value=(False, "error"))
    @patch("jarvis.core.startup_check._can_import", return_value=False)
    def test_install_failure_creates_warning(
        self, _mock_import: MagicMock, _mock_pip: MagicMock, mock_config: MagicMock
    ) -> None:
        auto_checker = StartupChecker(mock_config, auto_install=True)
        report = auto_checker.check_python_packages()
        assert len(report.warnings) > 0
        assert len(report.fixes_applied) == 0

    @patch("jarvis.core.startup_check._can_import")
    def test_partial_missing(self, mock_import: MagicMock, checker: StartupChecker) -> None:
        """Only some packages in a group are missing."""
        # First call True (present), rest False (missing)
        call_count = 0

        def side_effect(name: str) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count % 2 == 0

        mock_import.side_effect = side_effect

        with patch("jarvis.core.startup_check._pip_install", return_value=(True, "")) as mock_pip:
            report = checker.check_python_packages()
            # Some groups had missing packages, some didn't
            assert len(report.checks_passed) > 0


# ============================================================================
# check_ollama
# ============================================================================


class TestCheckOllama:
    """Tests for StartupChecker.check_ollama()."""

    @patch.object(StartupChecker, "_ollama_is_running", return_value=True)
    def test_ollama_already_running(self, _mock: MagicMock, checker: StartupChecker) -> None:
        report = checker.check_ollama()
        assert "Ollama running" in report.checks_passed
        assert len(report.fixes_applied) == 0
        assert len(report.warnings) == 0

    @patch.object(StartupChecker, "_ollama_is_running", return_value=False)
    def test_ollama_not_running_without_auto_install(
        self, _mock_run: MagicMock, checker: StartupChecker
    ) -> None:
        """Without --auto-install, only warns."""
        report = checker.check_ollama()
        assert len(report.warnings) == 1
        assert "not running" in report.warnings[0].lower()

    @patch.object(StartupChecker, "_start_ollama", return_value=True)
    @patch.object(StartupChecker, "_find_ollama", return_value="/usr/local/bin/ollama")
    @patch.object(StartupChecker, "_ollama_is_running", return_value=False)
    def test_ollama_auto_started(
        self,
        _mock_run: MagicMock,
        _mock_find: MagicMock,
        _mock_start: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        report = checker.check_ollama()
        assert len(report.fixes_applied) == 1
        assert "auto-started" in report.fixes_applied[0]

    @patch.object(StartupChecker, "_find_ollama", return_value=None)
    @patch.object(StartupChecker, "_ollama_is_running", return_value=False)
    def test_ollama_not_found(
        self, _mock_run: MagicMock, _mock_find: MagicMock, mock_config: MagicMock
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        report = checker.check_ollama()
        assert len(report.warnings) == 1
        assert "not found" in report.warnings[0].lower()

    @patch.object(StartupChecker, "_start_ollama", return_value=False)
    @patch.object(StartupChecker, "_find_ollama", return_value="/usr/bin/ollama")
    @patch.object(StartupChecker, "_ollama_is_running", return_value=False)
    def test_ollama_start_failed(
        self,
        _mock_run: MagicMock,
        _mock_find: MagicMock,
        _mock_start: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        report = checker.check_ollama()
        assert len(report.warnings) == 1
        assert "could not be started" in report.warnings[0].lower()


# ============================================================================
# check_models
# ============================================================================


class TestCheckModels:
    """Tests for StartupChecker.check_models()."""

    @patch("jarvis.core.startup_check._http_get_json")
    def test_all_models_present(self, mock_http: MagicMock, checker: StartupChecker) -> None:
        mock_http.return_value = {
            "models": [
                {"name": "qwen3:32b"},
                {"name": "qwen3:8b"},
                {"name": "qwen3-coder:30b"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
        report = checker.check_models()
        assert len(report.warnings) == 0
        assert len(report.fixes_applied) == 0
        assert len(report.checks_passed) >= 4

    @patch.object(StartupChecker, "_pull_model", return_value=True)
    @patch.object(StartupChecker, "_find_ollama", return_value="/usr/bin/ollama")
    @patch("jarvis.core.startup_check._http_get_json")
    def test_missing_model_auto_pulled(
        self,
        mock_http: MagicMock,
        _mock_find: MagicMock,
        mock_pull: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        # Only qwen3:8b is installed
        mock_http.return_value = {"models": [{"name": "qwen3:8b"}]}
        report = checker.check_models()
        assert len(report.fixes_applied) > 0
        assert mock_pull.call_count > 0

    @patch.object(StartupChecker, "_pull_model", return_value=False)
    @patch.object(StartupChecker, "_find_ollama", return_value="/usr/bin/ollama")
    @patch("jarvis.core.startup_check._http_get_json")
    def test_model_pull_failed(
        self,
        mock_http: MagicMock,
        _mock_find: MagicMock,
        mock_pull: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        mock_http.return_value = {"models": []}
        report = checker.check_models()
        assert len(report.warnings) > 0
        assert any("Failed to pull" in w for w in report.warnings)

    @patch("jarvis.core.startup_check._http_get_json", return_value=None)
    def test_ollama_unreachable(self, _mock: MagicMock, checker: StartupChecker) -> None:
        report = checker.check_models()
        assert len(report.warnings) == 1
        assert "Cannot reach Ollama" in report.warnings[0]

    def test_no_config_skips(self, checker_no_config: StartupChecker) -> None:
        report = checker_no_config.check_models()
        assert len(report.warnings) == 1
        assert "No config" in report.warnings[0]

    @patch.object(StartupChecker, "_find_ollama", return_value=None)
    @patch("jarvis.core.startup_check._http_get_json")
    def test_missing_model_no_ollama_binary(
        self, mock_http: MagicMock, _mock_find: MagicMock, mock_config: MagicMock
    ) -> None:
        checker = StartupChecker(mock_config, auto_install=True)
        mock_http.return_value = {"models": []}
        report = checker.check_models()
        assert len(report.warnings) > 0
        assert any("binary not found" in w for w in report.warnings)


# ============================================================================
# check_directories
# ============================================================================


class TestCheckDirectories:
    """Tests for StartupChecker.check_directories()."""

    def test_creates_missing_dirs(self, checker: StartupChecker, tmp_path: Path) -> None:
        checker._config.jarvis_home = tmp_path / "jarvis_test"
        report = checker.check_directories()
        assert len(report.fixes_applied) > 0
        assert (tmp_path / "jarvis_test" / "memory").is_dir()
        assert (tmp_path / "jarvis_test" / "logs").is_dir()
        assert (tmp_path / "jarvis_test" / "cache" / "web_search").is_dir()

    def test_existing_dirs_pass(self, checker: StartupChecker, tmp_path: Path) -> None:
        home = tmp_path / "jarvis_test"
        for sub in [
            "memory",
            "memory/episodes",
            "memory/knowledge",
            "memory/procedures",
            "memory/sessions",
            "index",
            "logs",
            "cache",
            "cache/web_search",
            "vault",
            "policies",
            "skills",
        ]:
            (home / sub).mkdir(parents=True, exist_ok=True)
        checker._config.jarvis_home = home
        report = checker.check_directories()
        assert len(report.fixes_applied) == 0
        assert len(report.checks_passed) == 12

    def test_no_config_skips(self, checker_no_config: StartupChecker) -> None:
        report = checker_no_config.check_directories()
        assert len(report.warnings) == 1


# ============================================================================
# check_node_modules
# ============================================================================


class TestCheckNodeModules:
    """Tests for StartupChecker.check_node_modules()."""

    def test_no_package_json(self, checker: StartupChecker, tmp_path: Path) -> None:
        report = checker.check_node_modules(tmp_path)
        assert "No UI" in report.checks_passed[0]

    def test_node_modules_present(self, checker: StartupChecker, tmp_path: Path) -> None:
        ui = tmp_path / "ui"
        ui.mkdir()
        (ui / "package.json").write_text("{}")
        (ui / "node_modules").mkdir()
        report = checker.check_node_modules(tmp_path)
        assert "node_modules present" in report.checks_passed

    @patch("jarvis.core.startup_check.shutil.which", return_value=None)
    def test_npm_not_found(self, _mock: MagicMock, checker: StartupChecker, tmp_path: Path) -> None:
        ui = tmp_path / "ui"
        ui.mkdir()
        (ui / "package.json").write_text("{}")
        report = checker.check_node_modules(tmp_path)
        assert len(report.warnings) == 1
        assert "npm not found" in report.warnings[0]


# ============================================================================
# _model_installed
# ============================================================================


class TestModelInstalled:
    """Tests for StartupChecker._model_installed()."""

    def test_exact_match(self) -> None:
        assert StartupChecker._model_installed("qwen3:8b", ["qwen3:8b"]) is True

    def test_case_insensitive(self) -> None:
        assert StartupChecker._model_installed("Qwen3:8B", ["qwen3:8b"]) is True

    def test_no_match(self) -> None:
        assert StartupChecker._model_installed("llama3:70b", ["qwen3:8b"]) is False

    def test_base_name_without_tag(self) -> None:
        assert (
            StartupChecker._model_installed("nomic-embed-text", ["nomic-embed-text:latest"]) is True
        )

    def test_tag_prefix_match(self) -> None:
        assert StartupChecker._model_installed("qwen3:8b", ["qwen3:8b-q4_0"]) is True


# ============================================================================
# check_and_fix_all orchestration
# ============================================================================


class TestCheckAndFixAll:
    """Tests for the top-level check_and_fix_all orchestration."""

    @patch.object(
        StartupChecker, "check_node_modules", return_value=StartupReport(checks_passed=["node ok"])
    )
    @patch.object(StartupChecker, "_find_repo_root", return_value="/repo")
    @patch.object(
        StartupChecker, "check_directories", return_value=StartupReport(checks_passed=["dirs ok"])
    )
    @patch.object(
        StartupChecker, "check_models", return_value=StartupReport(checks_passed=["models ok"])
    )
    @patch.object(
        StartupChecker, "check_ollama", return_value=StartupReport(checks_passed=["ollama ok"])
    )
    @patch.object(
        StartupChecker,
        "check_python_packages",
        return_value=StartupReport(checks_passed=["pkgs ok"]),
    )
    def test_all_pass(
        self,
        _pkgs: MagicMock,
        _oll: MagicMock,
        _mod: MagicMock,
        _dirs: MagicMock,
        _root: MagicMock,
        _node: MagicMock,
        checker: StartupChecker,
    ) -> None:
        report = checker.check_and_fix_all()
        assert report.ok
        assert len(report.checks_passed) == 5

    @patch.object(StartupChecker, "_find_repo_root", return_value=None)
    @patch.object(StartupChecker, "check_directories", return_value=StartupReport())
    @patch.object(StartupChecker, "check_models", return_value=StartupReport())
    @patch.object(StartupChecker, "check_ollama", return_value=StartupReport())
    @patch.object(StartupChecker, "check_python_packages", side_effect=RuntimeError("boom"))
    def test_exception_in_check_creates_error(
        self,
        _pkgs: MagicMock,
        _oll: MagicMock,
        _mod: MagicMock,
        _dirs: MagicMock,
        _root: MagicMock,
        checker: StartupChecker,
    ) -> None:
        report = checker.check_and_fix_all()
        assert len(report.errors) >= 1
        assert "Package check failed" in report.errors[0]

    @patch.object(StartupChecker, "_find_repo_root", return_value=None)
    @patch.object(StartupChecker, "check_directories", return_value=StartupReport())
    @patch.object(StartupChecker, "check_models", return_value=StartupReport())
    @patch.object(StartupChecker, "check_ollama", side_effect=RuntimeError("network"))
    @patch.object(StartupChecker, "check_python_packages", return_value=StartupReport())
    def test_ollama_exception_handled(
        self,
        _pkgs: MagicMock,
        _oll: MagicMock,
        _mod: MagicMock,
        _dirs: MagicMock,
        _root: MagicMock,
        checker: StartupChecker,
    ) -> None:
        report = checker.check_and_fix_all()
        assert any("Ollama check failed" in e for e in report.errors)

    @patch.object(
        StartupChecker,
        "check_node_modules",
        return_value=StartupReport(fixes_applied=["npm install"]),
    )
    @patch.object(StartupChecker, "_find_repo_root", return_value="/repo")
    @patch.object(
        StartupChecker, "check_directories", return_value=StartupReport(fixes_applied=["mkdir"])
    )
    @patch.object(
        StartupChecker,
        "check_models",
        return_value=StartupReport(fixes_applied=["pulled qwen3:8b"]),
    )
    @patch.object(
        StartupChecker, "check_ollama", return_value=StartupReport(fixes_applied=["started"])
    )
    @patch.object(
        StartupChecker,
        "check_python_packages",
        return_value=StartupReport(fixes_applied=["installed numpy"]),
    )
    def test_fixes_aggregated(
        self,
        _pkgs: MagicMock,
        _oll: MagicMock,
        _mod: MagicMock,
        _dirs: MagicMock,
        _root: MagicMock,
        _node: MagicMock,
        checker: StartupChecker,
    ) -> None:
        report = checker.check_and_fix_all()
        assert len(report.fixes_applied) == 5


# ============================================================================
# _find_ollama
# ============================================================================


class TestFindOllama:
    """Tests for _find_ollama static method."""

    @patch("jarvis.core.startup_check.shutil.which", return_value="/usr/local/bin/ollama")
    def test_found_on_path(self, _mock: MagicMock) -> None:
        assert StartupChecker._find_ollama() == "/usr/local/bin/ollama"

    @patch("jarvis.core.startup_check.os.path.isfile", return_value=False)
    @patch("jarvis.core.startup_check.shutil.which", return_value=None)
    def test_not_found(self, _mock_which: MagicMock, _mock_isfile: MagicMock) -> None:
        with patch("jarvis.core.startup_check.platform.system", return_value="Linux"):
            result = StartupChecker._find_ollama()
            assert result is None


# ============================================================================
# _pull_model
# ============================================================================


class TestPullModel:
    """Tests for _pull_model static method."""

    @patch("jarvis.core.startup_check.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert StartupChecker._pull_model("qwen3:8b", "/usr/bin/ollama") is True

    @patch("jarvis.core.startup_check.subprocess.run")
    def test_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert StartupChecker._pull_model("qwen3:8b", "/usr/bin/ollama") is False

    @patch(
        "jarvis.core.startup_check.subprocess.run",
        side_effect=subprocess.TimeoutExpired("cmd", 1800),
    )
    def test_timeout(self, _mock: MagicMock) -> None:
        assert StartupChecker._pull_model("big-model:70b", "/usr/bin/ollama") is False
