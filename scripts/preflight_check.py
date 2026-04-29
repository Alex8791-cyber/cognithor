#!/usr/bin/env python3
"""Cognithor · Pre-Flight Health Check.

Validates that all required and optional dependencies are available
before starting Cognithor.  Run this after installation to verify
your environment is correctly configured.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --verbose
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

MIN_PYTHON = (3, 12)
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Cognithor home: prefer COGNITHOR_HOME, fall back to legacy COGNITHOR_HOME
# (kept for backwards-compat with pre-rebrand installs), then default
# to ~/.cognithor/. The legacy ~/.jarvis/ path is no longer the default.
COGNITHOR_HOME = Path(
    os.environ.get("COGNITHOR_HOME")
    or os.environ.get("COGNITHOR_HOME")
    or (Path.home() / ".cognithor")
)

# Required Python packages (import_name, pip_name)
REQUIRED_PACKAGES = [
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
    ("yaml", "pyyaml"),
    ("structlog", "structlog"),
    ("rich", "rich"),
    ("prompt_toolkit", "prompt-toolkit"),
    ("anyio", "anyio"),
    ("cryptography", "cryptography"),
    ("dotenv", "python-dotenv"),
]

# Optional packages grouped by feature
OPTIONAL_GROUPS: dict[str, list[tuple[str, str]]] = {
    "memory": [("numpy", "numpy"), ("sqlite_vec", "sqlite-vec")],
    "web": [("fastapi", "fastapi"), ("uvicorn", "uvicorn")],
    "telegram": [("telegram", "python-telegram-bot")],
    "search": [("trafilatura", "trafilatura"), ("ddgs", "ddgs")],
    "voice": [
        ("faster_whisper", "faster-whisper"),
        ("piper", "piper-tts"),
        ("sounddevice", "sounddevice"),
    ],
    "documents": [("fpdf", "fpdf2"), ("docx", "python-docx")],
    "mcp": [("mcp", "mcp")],
    "discord": [("discord", "discord.py")],
    "slack": [("slack_sdk", "slack-sdk")],
    "browser": [("playwright", "playwright")],
    "cron": [("apscheduler", "apscheduler"), ("croniter", "croniter")],
}

# System binaries
SYSTEM_BINARIES: list[tuple[str, str, bool]] = [
    # (binary_name, purpose, required)
    ("ollama", "Local LLM inference", True),
    ("ffmpeg", "Audio conversion (voice mode)", False),
    ("node", "Control Center UI", False),
    ("npm", "Control Center UI", False),
    ("git", "Version control", False),
    ("docker", "Container sandbox (L2)", False),
]

# ── Output helpers ─────────────────────────────────────────────────────────

# Use ASCII symbols on Windows (cp1252 can't encode Unicode checkmarks)
if sys.platform == "win32":
    _PASS = "[OK]"
    _FAIL = "[FAIL]"
    _WARN = "[WARN]"
    _INFO = "[INFO]"
else:
    _PASS = "\033[92m\u2713\033[0m"
    _FAIL = "\033[91m\u2717\033[0m"
    _WARN = "\033[93m!\033[0m"
    _INFO = "\033[94mi\033[0m"

_pass_count = 0
_fail_count = 0
_warn_count = 0


def passed(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {_PASS} {msg}")


def failed(msg: str, hint: str = "") -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {_FAIL} {msg}")
    if hint:
        print(f"      ->{hint}")


def warned(msg: str, hint: str = "") -> None:
    global _warn_count
    _warn_count += 1
    print(f"  {_WARN} {msg}")
    if hint:
        print(f"      ->{hint}")


def info(msg: str) -> None:
    print(f"  {_INFO} {msg}")


def header(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ── Checks ─────────────────────────────────────────────────────────────────


def check_python_version() -> None:
    header("1. Python Version")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= MIN_PYTHON:
        passed(f"Python {version_str} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    else:
        failed(
            f"Python {version_str} (requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
            "Download from https://www.python.org/downloads/",
        )
    info(f"Executable: {sys.executable}")
    info(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")


def check_required_packages() -> None:
    header("2. Required Python Packages")
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
            passed(f"{pip_name} ({version})")
        except ImportError:
            failed(f"{pip_name} not installed", f"pip install {pip_name}")


def check_optional_packages(verbose: bool = False) -> None:
    header("3. Optional Python Packages")
    for group, packages in OPTIONAL_GROUPS.items():
        available = []
        missing = []
        for import_name, pip_name in packages:
            try:
                importlib.import_module(import_name)
                available.append(pip_name)
            except ImportError:
                missing.append(pip_name)

        if not missing:
            passed(f"[{group}] all installed ({', '.join(available)})")
        elif not available:
            warned(
                f"[{group}] not installed",
                f"pip install cognithor[{group}]",
            )
        else:
            warned(
                f"[{group}] partial ({', '.join(missing)} missing)",
                f"pip install cognithor[{group}]",
            )


def check_system_binaries() -> None:
    header("4. System Binaries")
    for binary, purpose, required in SYSTEM_BINARIES:
        # On Windows, also check .cmd/.exe variants
        path = shutil.which(binary)
        if path is None and sys.platform == "win32":
            path = shutil.which(f"{binary}.cmd") or shutil.which(f"{binary}.exe")

        if path:
            # Try to get version
            version = ""
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version = result.stdout.strip().split("\n")[0][:60] if result.stdout else ""
            except Exception:
                pass
            passed(f"{binary}: {path}" + (f" ({version})" if version else ""))
        elif required:
            failed(f"{binary} not found ({purpose})", "Install from the project PREREQUISITES.md")
        else:
            warned(f"{binary} not found ({purpose})", "Optional — install if needed")


def check_ollama_connection() -> None:
    header("5. Ollama Server")
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json

            data = json.loads(resp.read().decode())
            models = [m.get("name", "?") for m in data.get("models", [])]
            passed(f"Ollama running at {OLLAMA_URL}")
            if models:
                info(f"Models available: {', '.join(models[:10])}")
                # Check required models
                model_names = {m.split(":")[0] for m in models}
                for required in ["qwen3"]:
                    if any(required in m for m in model_names):
                        passed(f"Model '{required}' available")
                    else:
                        warned(f"Model '{required}' not found", f"ollama pull {required}:8b")
                if any("nomic-embed" in m or "embed" in m for m in models):
                    passed("Embedding model available")
                else:
                    warned("No embedding model found", "ollama pull qwen3-embedding:0.6b")
            else:
                warned(
                    "No models installed",
                    "ollama pull qwen3:8b && ollama pull qwen3-embedding:0.6b",
                )
    except urllib.error.URLError:
        warned(
            f"Ollama not reachable at {OLLAMA_URL}",
            "Start with: ollama serve (or install from https://ollama.com)",
        )
    except Exception as exc:
        warned(f"Ollama check failed: {exc}")


def check_directories() -> None:
    header("6. Directory Structure")
    home = COGNITHOR_HOME
    if home.exists():
        passed(f"Cognithor home: {home}")
    else:
        info(f"Cognithor home not created yet: {home} (will be created on first run)")

    # Check writable
    test_dir = Path(tempfile.gettempdir()) / "cognithor-preflight-test"
    try:
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        test_dir.rmdir()
        passed("Temp directory writable")
    except OSError as exc:
        failed(f"Temp directory not writable: {exc}")

    # Check home writable (if exists)
    if home.exists():
        try:
            test_file = home / ".preflight_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            passed("Jarvis home writable")
        except OSError as exc:
            failed(f"Jarvis home not writable: {exc}")


def check_env_vars() -> None:
    header("7. Environment Variables")
    # Check for API keys
    providers_found = []
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "CEREBRAS_API_KEY",
        "GITHUB_TOKEN",
        "HUGGINGFACE_API_KEY",
        "MOONSHOT_API_KEY",
    ]:
        if os.environ.get(key):
            providers_found.append(key.replace("_API_KEY", "").replace("_TOKEN", ""))

    if providers_found:
        passed(f"Cloud provider keys: {', '.join(providers_found)}")
    else:
        info("No cloud provider API keys set (using Ollama local inference)")

    # Check channel tokens
    channels_found = []
    for key in [
        "JARVIS_TELEGRAM_TOKEN",
        "JARVIS_DISCORD_TOKEN",
        "JARVIS_SLACK_TOKEN",
        "JARVIS_WHATSAPP_TOKEN",
        "JARVIS_SIGNAL_TOKEN",
        "JARVIS_MATRIX_TOKEN",
        "JARVIS_TEAMS_APP_ID",
    ]:
        if os.environ.get(key):
            channels_found.append(
                key.replace("JARVIS_", "").replace("_TOKEN", "").replace("_APP_ID", "")
            )

    if channels_found:
        passed(f"Channel tokens: {', '.join(channels_found)}")
    else:
        info("No channel tokens set (CLI mode only)")

    # Check .env file
    env_path = COGNITHOR_HOME / ".env"
    if env_path.exists():
        passed(f".env file found: {env_path}")
    else:
        info(f"No .env file at {env_path} (optional)")

    # API security
    if os.environ.get("COGNITHOR_API_TOKEN") or os.environ.get("JARVIS_API_TOKEN"):
        passed("API token set (Control Center secured)")
    else:
        warned(
            "COGNITHOR_API_TOKEN not set",
            "API is unprotected -- set for production use",
        )


def check_ports() -> None:
    header("8. Network Ports")
    import socket

    api_port = int(os.environ.get("JARVIS_API_PORT", "8741"))
    api_host = os.environ.get("JARVIS_API_HOST", "127.0.0.1")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((api_host, api_port))
        sock.close()
        if result == 0:
            warned(
                f"Port {api_port} already in use on {api_host}", "Another instance may be running"
            )
        else:
            passed(f"Port {api_port} available on {api_host}")
    except Exception as exc:
        warned(f"Port check failed: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("=" * 60)
    print("  COGNITHOR - Pre-Flight Health Check")
    print("=" * 60)

    check_python_version()
    check_required_packages()
    check_optional_packages(verbose)
    check_system_binaries()
    check_ollama_connection()
    check_directories()
    check_env_vars()
    check_ports()

    # Summary
    print(f"\n{'=' * 60}")
    total = _pass_count + _fail_count + _warn_count
    print(
        f"  Results: {_pass_count}/{total} passed, {_warn_count} warnings, {_fail_count} failures"
    )

    if _fail_count == 0:
        print("  All critical checks passed.")
    else:
        print(f"  {_fail_count} critical issue(s) found -- see above.")

    print("=" * 60)
    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
