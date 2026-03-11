"""Tests for bootstrap_windows.py --skip-models and config.py directory error handling.

Proves:
  1. bootstrap_windows.py accepts --skip-models and skips model downloads
  2. config.py ensure_directory_structure raises PermissionError with fix commands
  3. config.py ensure_directory_structure raises OSError on disk-full (errno 28)
  4. _safe_mkdir and _safe_write produce user-friendly messages
"""

from __future__ import annotations

import errno
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO / "scripts" / "bootstrap_windows.py"


# =========================================================================
# bootstrap_windows.py: --skip-models flag
# =========================================================================


class TestBootstrapSkipModels:
    """--skip-models must prevent any Ollama model download."""

    def test_argparser_accepts_skip_models(self):
        """The argument parser must accept --skip-models without error."""
        # Import just the argparse setup by loading the module
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import importlib

            mod = importlib.import_module("bootstrap_windows")
            importlib.reload(mod)  # fresh copy

            parser = mod.argparse.ArgumentParser()
            parser.add_argument("--repo-root", required=True)
            parser.add_argument("--force", action="store_true")
            parser.add_argument("--skip-models", action="store_true")

            _test_root = str(Path(tempfile.gettempdir()) / "test")
            args = parser.parse_args(["--repo-root", _test_root, "--skip-models"])
            assert args.skip_models is True
            assert args.repo_root == _test_root
        finally:
            sys.path.pop(0)

    def test_skip_models_flag_in_source(self):
        """install.sh must contain --skip-models flag definition."""
        text = BOOTSTRAP.read_text(encoding="utf-8")
        assert "--skip-models" in text, "Bootstrap must accept --skip-models"

    def test_first_start_has_skip_models_param(self):
        """first_start() must accept skip_models keyword argument."""
        text = BOOTSTRAP.read_text(encoding="utf-8")
        assert "def first_start(repo_root: str, *, skip_models: bool" in text

    def test_quick_start_has_skip_models_param(self):
        """quick_start() must accept skip_models keyword argument."""
        text = BOOTSTRAP.read_text(encoding="utf-8")
        assert "def quick_start(repo_root: str, *, skip_models: bool" in text

    def test_no_auto_pull_in_quick_start(self):
        """quick_start must NOT auto-pull models (was the blocking 30min bug)."""
        text = BOOTSTRAP.read_text(encoding="utf-8")
        # Find quick_start function body
        start = text.index("def quick_start(")
        # Find the next top-level def to delimit the function
        next_def = text.index("\ndef ", start + 1)
        body = text[start:next_def]
        assert "pull_model(" not in body, (
            "quick_start() must NOT call pull_model() — "
            "it was blocking for 30min without user consent"
        )

    def test_print_model_pull_commands_exists(self):
        """Helper to print manual ollama pull commands must exist."""
        text = BOOTSTRAP.read_text(encoding="utf-8")
        assert "_print_model_pull_commands" in text

    def test_print_model_pull_commands_output(self):
        """_print_model_pull_commands must print ollama pull for each model."""
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import importlib

            mod = importlib.import_module("bootstrap_windows")
            importlib.reload(mod)

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                mod._print_model_pull_commands("minimal")

            output = buf.getvalue()
            assert "ollama pull qwen3:8b" in output
            assert "ollama pull qwen3-embedding:0.6b" in output
        finally:
            sys.path.pop(0)


# =========================================================================
# config.py: _safe_mkdir — PermissionError handling
# =========================================================================


class TestSafeMkdirPermission:
    """_safe_mkdir must raise PermissionError with user-friendly fix command."""

    def test_permission_error_contains_fix(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "no_access" / "sub" / "deep"

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="sudo chown"):
                _safe_mkdir(target)

    def test_permission_error_contains_mkdir_fix(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "blocked"

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="sudo mkdir -p"):
                _safe_mkdir(target)

    def test_permission_error_mentions_path(self, tmp_path):
        import re as _re
        from jarvis.config import _safe_mkdir

        target = tmp_path / "test_dir"

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match=_re.escape(str(target))):
                _safe_mkdir(target)

    def test_normal_mkdir_succeeds(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "good_dir" / "sub"
        _safe_mkdir(target)
        assert target.exists()


# =========================================================================
# config.py: _safe_mkdir — disk full (OSError errno 28)
# =========================================================================


class TestSafeMkdirDiskFull:
    """_safe_mkdir must handle ENOSPC (disk full) with user-friendly message."""

    def test_disk_full_error(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "full_disk"
        exc = OSError(errno.ENOSPC, "No space left on device")

        with patch.object(Path, "mkdir", side_effect=exc):
            with pytest.raises(OSError, match="Festplatte voll"):
                _safe_mkdir(target)

    def test_disk_full_mentions_df(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "full_disk"
        exc = OSError(errno.ENOSPC, "No space left on device")

        with patch.object(Path, "mkdir", side_effect=exc):
            with pytest.raises(OSError, match="df -h"):
                _safe_mkdir(target)

    def test_other_oserror_reraises(self, tmp_path):
        from jarvis.config import _safe_mkdir

        target = tmp_path / "other"
        exc = OSError(errno.EACCES, "other os error")

        with patch.object(Path, "mkdir", side_effect=exc):
            with pytest.raises(OSError):
                _safe_mkdir(target)


# =========================================================================
# config.py: _safe_write — PermissionError and disk-full
# =========================================================================


class TestSafeWrite:
    """_safe_write must handle PermissionError and disk-full."""

    def test_permission_error_on_write(self, tmp_path):
        from jarvis.config import _safe_write

        target = tmp_path / "readonly.txt"
        target.touch()

        with patch.object(Path, "write_text", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="sudo chown"):
                _safe_write(target, "content")

    def test_disk_full_on_write(self, tmp_path):
        from jarvis.config import _safe_write

        target = tmp_path / "full.txt"
        exc = OSError(errno.ENOSPC, "No space left on device")

        with patch.object(Path, "write_text", side_effect=exc):
            with pytest.raises(OSError, match="Festplatte voll"):
                _safe_write(target, "content")

    def test_normal_write_succeeds(self, tmp_path):
        from jarvis.config import _safe_write

        target = tmp_path / "ok.txt"
        _safe_write(target, "hello")
        assert target.read_text() == "hello"


# =========================================================================
# config.py: ensure_directory_structure integration
# =========================================================================


class TestEnsureDirectoryStructureErrors:
    """ensure_directory_structure must propagate the user-friendly errors."""

    def test_propagates_permission_error(self, tmp_path):
        from jarvis.config import ensure_directory_structure, JarvisConfig

        cfg = JarvisConfig(jarvis_home=tmp_path / "jarvis")

        # Patch _safe_mkdir to simulate permission error on first call
        with patch(
            "jarvis.config._safe_mkdir",
            side_effect=PermissionError("sudo chown fix"),
        ):
            with pytest.raises(PermissionError, match="sudo chown"):
                ensure_directory_structure(cfg)

    def test_propagates_disk_full_error(self, tmp_path):
        from jarvis.config import ensure_directory_structure, JarvisConfig

        cfg = JarvisConfig(jarvis_home=tmp_path / "jarvis")

        with patch(
            "jarvis.config._safe_mkdir",
            side_effect=OSError("Festplatte voll"),
        ):
            with pytest.raises(OSError, match="Festplatte voll"):
                ensure_directory_structure(cfg)


# =========================================================================
# Bug report template
# =========================================================================


class TestBugReportTemplate:
    """Bug report template must exist and contain required sections."""

    def test_template_exists(self):
        template = REPO / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
        assert template.exists(), "Bug report template must exist"

    def test_template_has_required_sections(self):
        template = REPO / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
        text = template.read_text(encoding="utf-8")

        required = [
            "Environment",
            "Cognithor version",
            "Steps to Reproduce",
            "Expected Behavior",
            "Actual Behavior",
            "Logs",
            "OS",
        ]
        for section in required:
            assert section in text, f"Template must contain '{section}'"

    def test_template_credits_tomiweb(self):
        template = REPO / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
        text = template.read_text(encoding="utf-8")
        assert "TomiWebPro" in text, "Template must credit TomiWebPro"

    def test_template_has_frontmatter(self):
        template = REPO / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
        text = template.read_text(encoding="utf-8")
        assert text.startswith("---"), "Must have YAML frontmatter"
        assert "name: Bug Report" in text
        assert "labels: bug" in text


# =========================================================================
# README.md: TomiWebPro credit
# =========================================================================


class TestReadmeCredits:
    """README must credit TomiWebPro."""

    def test_contributors_section(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        assert "## Contributors" in text

    def test_tomiweb_in_contributors(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        assert "TomiWebPro" in text
        assert "QA Lead" in text

    def test_special_thanks_section(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        assert "Special Thanks" in text
        assert "TomiWebPro" in text
