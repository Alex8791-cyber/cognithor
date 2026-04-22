"""Tests for the model installer (Ollama tags + community GGUF imports)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cognithor.core.model_installer import (
    InstallResult,
    _ollama_has_tag,
    install_model,
    is_installed,
)


class TestOllamaPresenceCheck:
    def test_has_tag_hit(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": [{"name": "qwen3.6:35b"}]}
        with patch("httpx.get", return_value=resp):
            assert _ollama_has_tag("http://localhost:11434", "qwen3.6:35b") is True

    def test_has_tag_miss(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": [{"name": "qwen3:32b"}]}
        with patch("httpx.get", return_value=resp):
            assert _ollama_has_tag("http://localhost:11434", "qwen3.6:35b") is False

    def test_ollama_unreachable_returns_false(self):
        with patch("httpx.get", side_effect=ConnectionError("refused")):
            assert _ollama_has_tag("http://localhost:11434", "qwen3.6:35b") is False


class TestInstallOllamaTag:
    def test_already_present_short_circuits(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": [{"name": "qwen3.6:35b"}]}
        with patch("httpx.get", return_value=resp):
            r = install_model("qwen3.6:35b")
        assert r.status == "already_present"
        assert r.local_tag == "qwen3.6:35b"

    def test_missing_ollama_cli(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        with (
            patch("httpx.get", return_value=resp),
            patch("cognithor.core.model_installer.shutil.which", return_value=None),
        ):
            r = install_model("qwen3.6:35b")
        assert r.status == "failed"
        assert "ollama CLI not found" in r.message

    def test_pull_success_streams_progress(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}

        fake_proc = MagicMock()
        fake_proc.stdout = iter(["pulling manifest\n", "pulling layer\n", "success\n"])
        fake_proc.returncode = 0
        fake_proc.wait.return_value = None

        progress_lines: list[str] = []
        with (
            patch("httpx.get", return_value=resp),
            patch(
                "cognithor.core.model_installer.shutil.which",
                return_value="/usr/bin/ollama",
            ),
            patch(
                "cognithor.core.model_installer.subprocess.Popen",
                return_value=fake_proc,
            ),
        ):
            r = install_model("qwen3.6:35b", progress_cb=progress_lines.append)
        assert r.status == "installed"
        assert r.local_tag == "qwen3.6:35b"
        assert progress_lines == ["pulling manifest", "pulling layer", "success"]


class TestCommunityGGUFInstall:
    def test_routes_hf_repo_through_gguf_path(self):
        # When name matches the registry's community_gguf.entries, we must
        # NOT attempt an ollama pull — we go through the HF download path.
        with patch(
            "cognithor.core.model_installer._install_community_gguf",
            return_value=InstallResult(
                model_name="unsloth/Qwen3.6-27B-GGUF",
                status="installed",
                local_tag="qwen3.6:27b",
                message="ok",
            ),
        ) as mock_hf:
            r = install_model("unsloth/Qwen3.6-27B-GGUF")
        assert mock_hf.called
        assert r.local_tag == "qwen3.6:27b"

    def test_missing_huggingface_hub_degrades_gracefully(self):
        # Simulate huggingface_hub not being installed: the function must
        # return status=failed with a helpful message, not crash.
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            # Load registry naturally — only fakes out HF.
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"models": []}
            with patch("httpx.get", return_value=resp):
                r = install_model("unsloth/Qwen3.6-27B-GGUF")
        assert r.status == "failed"
        assert "huggingface_hub" in r.message


class TestIsInstalled:
    def test_community_gguf_checks_import_as(self):
        # For community GGUF entries, is_installed() should check
        # the import_as tag (qwen3.6:27b), not the raw HF repo id.
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": [{"name": "qwen3.6:27b"}]}
        with patch("httpx.get", return_value=resp):
            assert is_installed("unsloth/Qwen3.6-27B-GGUF") is True
