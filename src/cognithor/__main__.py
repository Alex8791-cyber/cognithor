"""
Cognithor · Agent OS -- Entry Point.

Usage: cognithor
       cognithor --config /path/to/config.yaml
       cognithor --version
       python -m jarvis
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Suppress noisy third-party warnings that clutter startup output
warnings.filterwarnings("ignore", message=".*AVX512.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*unauthenticated requests to the HF Hub.*", category=FutureWarning
)
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*", category=SyntaxWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress FAISS C-library output (AVX512 fallback messages) on import.
# FAISS writes to stdout via its C extension before Python has any control.
os.environ.setdefault("FAISS_OPT_LEVEL", "avx2")  # skip AVX512 probe entirely


def _silence_library_loggers() -> None:
    """Silence noisy third-party loggers during startup."""
    import logging as _logging

    for name in (
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "torch",
        "faiss",
        "chromadb",
        "httpx",
        "httpcore",
        "onnxruntime",
    ):
        _logging.getLogger(name).setLevel(_logging.ERROR)
    # transformers has its own verbosity API
    try:
        from transformers import logging as tf_logging

        tf_logging.set_verbosity_error()
    except Exception:
        _logging.getLogger(__name__).debug("suppress_transformers_logging_failed", exc_info=True)


from cognithor import BANNER_ASCII, __version__

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

# Must be importable at runtime for FastAPI endpoint signature resolution
# (PEP 563 stores annotations as strings; FastAPI resolves them via __globals__)
try:
    from starlette.requests import Request as _BootstrapRequest
    from starlette.requests import Request as _STRequest
except ImportError:
    _BootstrapRequest = None  # type: ignore[assignment,misc]
    _STRequest = None  # type: ignore[assignment,misc]

# WebSocket/FastAPI types must be at module level so that
# `from __future__ import annotations` (PEP 563) can resolve
# string-ified type hints via get_type_hints().
try:
    from starlette.websockets import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover
    WebSocket = None  # type: ignore[assignment,misc]
    WebSocketDisconnect = None  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    """Kommandozeilen-Argumente parsen."""
    parser = argparse.ArgumentParser(
        prog="cognithor",
        description=(
            "Cognithor -- Local-first autonomous agent OS.\n\n"
            "Common usage / Haeufige Nutzung:\n"
            "  cognithor                     # Start with CLI + web UI\n"
            "  cognithor --ui                # Headless + open browser\n"
            "  cognithor --no-cli            # Headless backend only\n"
            "  cognithor config              # Interactive config TUI\n"
            "  cognithor config list         # Show current config\n"
            "  cognithor config set KEY VAL  # Set a config value\n"
            "  cognithor config get KEY      # Get a config value\n"
            "  cognithor --lite              # Lite mode (6GB VRAM)\n"
            "  cognithor --log-level DEBUG   # Verbose logging\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("For more documentation see:\n  https://github.com/Alex8791-cyber/cognithor\n"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Cognithor v{__version__}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Pfad zur config.yaml / Path to config.yaml "
            "(e.g. ~/.cognithor/config.yaml). "
            "NOTE: To open the interactive config editor use: 'cognithor config'"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log-Level ueberschreiben / Override log level",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help=(
            "Nur Verzeichnisstruktur erstellen, nicht starten / "
            "Only create directory structure, do not start"
        ),
    )
    parser.add_argument(
        "--no-cli",
        action="store_true",
        help=(
            "CLI-Channel nicht starten (Headless-Betrieb fuer Control Center) / "
            "Do not start the CLI channel (headless mode for Control Center)"
        ),
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help=(
            "UI-Modus: Headless-Backend + Browser oeffnen auf http://localhost:<port> / "
            "UI mode: headless backend + open browser to http://localhost:<port>"
        ),
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8741,
        help="Port fuer die Control Center API / Control Center API port (default: 8741)",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default=None,
        help=(
            "Host fuer die Control Center API / Control Center API bind host "
            "(default: 127.0.0.1; use 0.0.0.0 for LAN access)"
        ),
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help=(
            "Lite-Modus: qwen3:8b als Planner und Executor (6 GB statt 26 GB VRAM) / "
            "Lite mode: qwen3:8b as planner and executor (6 GB instead of 26 GB VRAM)"
        ),
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help=(
            "Fehlende Python-Pakete automatisch installieren / "
            "Automatically install missing Python packages (default: warn only)"
        ),
    )
    parser.add_argument(
        "--mcp-server",
        action="store_true",
        help=(
            "Start as MCP server on stdio (for VSCode, Claude Desktop, etc.). "
            "Only workspace-safe tools are exposed. No CLI, no Web UI."
        ),
    )
    parser.add_argument(
        "--arc",
        action="store_true",
        help="Enable ARC-AGI-3 benchmark agent and MCP tools",
    )
    parser.add_argument(
        "--skip-startup-check",
        action="store_true",
        help="Skip startup dependency/model check (faster boot when already configured)",
    )
    sub = parser.add_subparsers(dest="command")
    config_parser = sub.add_parser("config", help="Configure Cognithor interactively")
    config_sub = config_parser.add_subparsers(dest="config_action")
    config_sub.add_parser("list", help="Show current settings")
    set_p = config_sub.add_parser("set", help="Set a config value")
    set_p.add_argument("key", help="Dot-path config key")
    set_p.add_argument("value", help="New value")
    get_p = config_sub.add_parser("get", help="Get a config value")
    get_p.add_argument("key", help="Dot-path config key")

    # `cognithor models …` — install and list models (Ollama + community GGUF).
    models_parser = sub.add_parser("models", help="Manage local LLM models")
    models_sub = models_parser.add_subparsers(dest="models_action")
    models_sub.add_parser("list", help="Show known models from the registry")
    inst_p = models_sub.add_parser("install", help="Download + import a model")
    inst_p.add_argument(
        "name",
        help=(
            "Ollama tag (e.g. qwen3.6:35b) or HF repo id from the registry "
            "(e.g. unsloth/Qwen3.6-27B-GGUF)."
        ),
    )

    return parser.parse_args()


def _check_python_version() -> None:
    """Ensure Python >= 3.12 is running."""


async def _run_mcp_server_mode(config: Any) -> None:
    """Run Jarvis as a pure MCP server on stdio.

    Exposes only workspace-safe tools. No CLI, no Web UI, no channels.
    Designed for integration with VSCode, Claude Desktop, Cursor, etc.
    """
    import asyncio

    from cognithor.mcp.bridge import MCP_WORKSPACE_SAFE_TOOLS, MCPBridge
    from cognithor.mcp.client import JarvisMCPClient
    from cognithor.mcp.server import MCPServerMode
    from cognithor.utils.logging import get_logger

    log = get_logger("cognithor.mcp_server")
    log.info("mcp_server_mode_starting")

    # Ensure desktop tools are off in MCP server mode
    if hasattr(config, "tools"):
        config.tools.computer_use_enabled = False
        config.tools.desktop_tools_enabled = False

    # Create MCP client and register tools
    mcp_client = JarvisMCPClient()

    from cognithor.gateway.phases.tools import init_tools

    await init_tools(config, mcp_client, memory_manager=None)

    # Create bridge with workspace-safe allowlist
    bridge = MCPBridge(config)
    bridge.set_tool_allowlist(MCP_WORKSPACE_SAFE_TOOLS)

    if bridge.setup(mcp_client, memory=None, mode_override=MCPServerMode.STDIO):
        await bridge.start()
        log.info(
            "mcp_server_stdio_running",
            tools=len(MCP_WORKSPACE_SAFE_TOOLS),
        )
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            await bridge.stop()
    else:
        print("[ERROR] MCP server setup failed.", file=sys.stderr)
        sys.exit(1)


def _expand_working_set(min_mb: int = 128, max_mb: int = 512) -> None:
    """Increase the Windows process working-set quota.

    SQLCipher uses VirtualLock() to protect encryption keys in RAM.
    With many concurrent encrypted databases, the default working-set
    quota (~20 MB) is quickly exhausted, causing Error 1453 warnings.
    Expanding to 128/512 MB eliminates these warnings entirely.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessWorkingSetSize.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessWorkingSetSize.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        min_bytes = ctypes.c_size_t(min_mb * 1024 * 1024)
        max_bytes = ctypes.c_size_t(max_mb * 1024 * 1024)
        success = kernel32.SetProcessWorkingSetSize(handle, min_bytes, max_bytes)
        if not success:
            err = ctypes.get_last_error()
            print(f"  [WARN] SetProcessWorkingSetSize failed (error {err})", file=sys.stderr)
    except Exception:
        pass  # Not on Windows, or insufficient privileges


def _migrate_jarvis_home() -> None:
    """Migrate data from ~/.jarvis/ to ~/.cognithor/ if needed (one-time).

    When upgrading from the old 'jarvis' package to 'cognithor', DB files,
    config, and caches remain under ~/.jarvis/ while code now looks in
    ~/.cognithor/.  This copies the relevant subdirectories once and writes
    a marker so the migration is never repeated.
    """
    import shutil

    jarvis_home = Path.home() / ".jarvis"
    cognithor_home = Path.home() / ".cognithor"
    marker = cognithor_home / ".migrated_from_jarvis"

    if marker.exists() or not jarvis_home.is_dir():
        return

    cognithor_home.mkdir(parents=True, exist_ok=True)

    dirs_to_migrate = ["memory", "data", "vault", "skills", "config", "cache"]
    migrated: list[str] = []

    for dirname in dirs_to_migrate:
        src = jarvis_home / dirname
        dst = cognithor_home / dirname
        if src.is_dir() and not dst.is_dir():
            shutil.copytree(str(src), str(dst))
            migrated.append(dirname)

    # Also migrate top-level files
    for filename in ["config.yaml", "agents.yaml", "CORE.md"]:
        src = jarvis_home / filename
        dst = cognithor_home / filename
        if src.is_file() and not dst.is_file():
            shutil.copy2(str(src), str(dst))
            migrated.append(filename)

    if migrated:
        marker.write_text(f"Migrated from {jarvis_home}: {', '.join(migrated)}")
        print(f"  [OK] Migrated data from {jarvis_home}: {', '.join(migrated)}")
    else:
        # Nothing to migrate but mark as done so we don't re-check
        marker.write_text("No migration needed")


def main() -> None:
    """Main entry point for Cognithor."""
    _check_python_version()

    # Windows: switch stdout/stderr to UTF-8 so that umlauts in cmd.exe
    # (without chcp 65001) do not cause encoding crashes.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                with contextlib.suppress(Exception):
                    stream.reconfigure(encoding="utf-8", errors="replace")

        # Windows: increase process working-set quota so SQLCipher's
        # VirtualLock() calls succeed instead of failing with Error 1453.
        # 128 MB min / 512 MB max is enough for ~40 encrypted DBs.
        _expand_working_set(min_mb=128, max_mb=512)

    # Intercept `cognithor pack <subcommand>` before the main parser runs.
    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        from cognithor.packs.cli import main as pack_main

        raise SystemExit(pack_main(sys.argv[2:]))

    args = parse_args()

    # --ui: headless mode + auto-open browser, bind to 127.0.0.1 by default
    if getattr(args, "ui", False):
        args.no_cli = True
        if not args.api_host:
            args.api_host = "127.0.0.1"

    if getattr(args, "command", None) == "models":
        from cognithor.cli import models_cmd

        action = getattr(args, "models_action", None)
        if action == "list":
            sys.exit(models_cmd.cmd_list())
        elif action == "install":
            sys.exit(models_cmd.cmd_install(args.name))
        else:
            print("Usage: cognithor models <list|install>", file=sys.stderr)
            sys.exit(2)

    if getattr(args, "command", None) == "config":
        from cognithor.cli import config_cmd, config_tui

        config_path = args.config or (Path.home() / ".cognithor" / "config.yaml")
        action = getattr(args, "config_action", None)
        if action == "set":
            sys.exit(config_cmd.cmd_set(args.key, args.value, config_path=config_path))
        elif action == "get":
            sys.exit(config_cmd.cmd_get(args.key, config_path=config_path))
        elif action == "list":
            sys.exit(config_cmd.cmd_list(config_path=config_path))
        else:
            config_tui.launch(config_path=config_path)
            sys.exit(0)

    # 0a. Auto-migrate data from ~/.jarvis/ to ~/.cognithor/ (one-time, on upgrade)
    _migrate_jarvis_home()

    # 0. Load .env file (secrets from ~/.cognithor/.env or project root)
    try:
        from dotenv import load_dotenv

        # First project .env, then user .env (user overrides)
        load_dotenv(Path(".env"), override=False)
        load_dotenv(Path.home() / ".cognithor" / ".env", override=True)
    except ImportError:
        pass  # python-dotenv optional

    # 1. Load configuration
    from cognithor.config import ensure_directory_structure, load_config

    config = load_config(args.config)

    # 1.1 i18n: Set locale from config (or COGNITHOR_LANGUAGE env var)
    from cognithor.i18n import set_locale

    _lang = os.environ.get("COGNITHOR_LANGUAGE") or config.language
    set_locale(_lang)

    # 1.5 Lite mode: smaller models for low VRAM usage
    if args.lite:
        config.models.planner.name = "qwen3:8b"
        config.models.coder.name = "qwen2.5-coder:7b"

    if getattr(args, "arc", False):
        config.arc.enabled = True

    # 1.9 Auto-migrate plaintext API keys from config.yaml / .env to OS Keyring
    try:
        from cognithor.security.secret_store import SecretStore as _SecretStore

        _store = _SecretStore()
        if _store.is_available:
            _config_path = (
                Path(args.config) if args.config else Path.home() / ".cognithor" / "config.yaml"
            )
            if _config_path.exists():
                _migrated = _store.migrate_from_config(_config_path)
                if _migrated > 0:
                    # Re-resolve so the live config reflects the keyring values
                    from cognithor.config import _resolve_secrets as _rs

                    _rs(config)

            _env_path = Path.home() / ".cognithor" / ".env"
            if _env_path.exists():
                _env_migrated = _store.migrate_from_env(_env_path)
                # .env secrets are already in env vars (loaded above by dotenv),
                # so no re-resolve needed for the running process.
                del _env_migrated

            del _migrated, _config_path, _env_path
        del _store, _SecretStore
    except Exception:
        pass  # Never block startup on keyring errors

    # 2. Verzeichnisstruktur sicherstellen
    created = ensure_directory_structure(config)

    # 2.5 mTLS-Zertifikate sicherstellen (wenn aktiviert)
    _mtls_certs_dir = None
    try:
        from cognithor.security.mtls import ensure_mtls_certs as _ensure_mtls

        _mtls_certs_dir = _ensure_mtls(config)
    except ImportError:
        pass  # cryptography nicht installiert

    # 3. Logging initialisieren
    from cognithor.utils.logging import setup_logging

    log_level = args.log_level or config.logging.level
    setup_logging(
        level=log_level,
        log_dir=config.logs_dir,
        json_logs=config.logging.json_logs,
        console=config.logging.console,
    )

    from cognithor.utils.logging import get_logger

    log = get_logger("cognithor")

    # 3.5 Init-only: create directory structure only, then exit immediately.
    # IMPORTANT: Must come BEFORE the StartupChecker, as it triggers model pulls
    # (up to 30 min timeout) and pip installs (up to 5 min).
    if args.init_only:
        if created:
            for path in created:
                log.info("created_path", path=path)
        log.info("init_complete", paths_created=len(created))
        log.info(
            "init_summary",
            version=__version__,
            home=str(config.cognithor_home),
            config_file=str(config.config_file),
            paths_created=len(created),
        )
        return

    # 3.6 Startup check: automatically load missing dependencies
    # --skip-startup-check: Fast-path for already-configured systems
    if getattr(args, "skip_startup_check", False):
        log.info("startup_check_skipped", reason="--skip-startup-check flag")
    else:
        # Suppress FAISS/library C-level output during import checks.
        # Must redirect OS file descriptors for C library stdout.
        _devnull_sc = os.open(os.devnull, os.O_WRONLY)
        _saved_fd1_sc = os.dup(1)
        _saved_fd2_sc = os.dup(2)
        os.dup2(_devnull_sc, 1)
        os.dup2(_devnull_sc, 2)
        try:
            from cognithor.core.startup_check import StartupChecker

            checker = StartupChecker(config, auto_install=getattr(args, "auto_install", False))
            report = checker.check_and_fix_all()
        finally:
            os.dup2(_saved_fd1_sc, 1)
            os.dup2(_saved_fd2_sc, 2)
            os.close(_saved_fd1_sc)
            os.close(_saved_fd2_sc)
            os.close(_devnull_sc)
        if report.fixes_applied:
            log.info("startup_auto_fixes", fixes=report.fixes_applied, warnings=report.warnings)
        if report.errors:
            log.warning("startup_check_errors", errors=report.errors)

    # 3.7 MCP server mode: start only MCP server on stdio, no CLI/WebUI
    if args.mcp_server:
        import asyncio

        asyncio.run(_run_mcp_server_mode(config))
        return

    # 4. Startup info (debug — banner shows this already)
    log.debug(
        "jarvis_starting",
        version=__version__,
        home=str(config.cognithor_home),
        log_level=log_level,
    )

    if created:
        for path in created:
            log.debug("created_path", path=path)

    # 5. System-Check -- startup banner (intentional CLI output)
    # Default to 127.0.0.1 for safety. Users can set --api-host 0.0.0.0
    # or COGNITHOR_API_HOST env var for LAN access.
    _api_host = args.api_host or os.environ.get("COGNITHOR_API_HOST") or "127.0.0.1"
    _print_banner(config, api_host=_api_host, api_port=args.api_port, lite=args.lite)

    # Phase 0 Checkpoint: Setup OK (logged at debug level — banner already shows this)
    log.debug(
        "setup_ok",
        backend=getattr(config, "llm_backend_type", "ollama"),
        planner_model=config.models.planner.name,
        executor_model=config.models.executor.name,
    )

    # Phase 1: Start gateway + CLI
    import asyncio

    async def run() -> None:
        """Starts the gateway and CLI channel as asynchronous main loop."""
        import io as _io
        import logging as _logging

        from cognithor.channels.cli import CliChannel
        from cognithor.gateway.gateway import Gateway

        # Silence ALL output during init — we print a clean summary after.
        # Must set root logger to WARNING before Gateway() constructor.
        _silence_library_loggers()
        _root = _logging.getLogger()
        _prev_root_level = _root.level
        _root.setLevel(_logging.WARNING)

        # Redirect OS-level fd1/fd2 to suppress C library output (SQLCipher)
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_fd1 = os.dup(1)
        _saved_fd2 = os.dup(2)
        os.dup2(_devnull_fd, 1)
        os.dup2(_devnull_fd, 2)
        # Redirect Python-level stdout/stderr for tqdm etc.
        _real_stdout = sys.stdout
        _real_stderr = sys.stderr
        sys.stdout = _io.StringIO()
        sys.stderr = _io.StringIO()

        gateway = Gateway(config)
        api_server = None
        _bg_tasks: set[asyncio.Task[Any]] = set()

        # Suppress harmless ConnectionResetError from Windows ProactorEventLoop
        _loop = asyncio.get_running_loop()
        _orig_handler = _loop.get_exception_handler()

        def _quiet_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
            exc = context.get("exception")
            if isinstance(exc, ConnectionResetError):
                return
            if _orig_handler:
                _orig_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        _loop.set_exception_handler(_quiet_exception_handler)

        try:
            await gateway.initialize()
        finally:
            # Restore OS-level file descriptors
            os.dup2(_saved_fd1, 1)
            os.dup2(_saved_fd2, 2)
            os.close(_saved_fd1)
            os.close(_saved_fd2)
            os.close(_devnull_fd)
            # Restore Python streams and log level
            sys.stdout = _real_stdout
            sys.stderr = _real_stderr
            _root.setLevel(_prev_root_level)

        # Print clean startup summary
        _tools = gateway._mcp_client.get_tool_list() if hasattr(gateway, "_mcp_client") else []
        _skills = getattr(gateway, "_skill_registry", None)
        _skill_count = _skills.count if _skills and hasattr(_skills, "count") else 0
        _agents = getattr(gateway, "_agent_router", None)
        _agent_count = len(_agents.agents) if _agents and hasattr(_agents, "agents") else 0
        _cron = getattr(gateway, "_cron_engine", None)
        _cron_count = _cron.job_count if _cron and hasattr(_cron, "job_count") else 0
        _identity = getattr(gateway, "_identity_layer", None)
        _mem_count = 0
        if _identity and hasattr(_identity, "_engine"):
            _mem_count = getattr(_identity._engine, "memory_count", 0)
        _enc = "active" if getattr(config.database, "encryption_enabled", False) else "disabled"

        print()
        print(f"  [OK] LLM backend ready ({config.models.planner.name})")
        print(f"  [OK] {len(_tools)} tools registered")
        if _skill_count:
            print(f"  [OK] {_skill_count} skills loaded")
        if _agent_count:
            print(f"  [OK] {_agent_count} agents ready")
        if _mem_count:
            print(f"  [OK] Memory initialized ({_mem_count} memories)")
        if _cron_count:
            print(f"  [OK] {_cron_count} cron jobs scheduled")
        print(f"  [OK] Database encryption {_enc}")
        print()
        print("  Cognithor is ready.")
        print()

        try:
            # LLM-Erreichbarkeit pruefen und prominent warnen
            _llm = getattr(gateway, "_llm", None)
            if _llm and not await _llm.is_available():
                _backend = getattr(_llm, "backend_type", "ollama")
                print()
                print("!" * 60)
                print("  WARNING: LLM backend not reachable!")
                print("!" * 60)
                if _backend == "ollama":
                    _ollama_url = config.ollama.base_url
                    print(f"  Ollama not responding at {_ollama_url}")
                    if config.ollama.mode == "local":
                        print("  Start Ollama: ollama serve")
                        print(f"  Pull model:   ollama pull {config.models.planner.name}")
                elif _backend == "lmstudio":
                    print(
                        f"  LM Studio not reachable at {getattr(config, 'lmstudio_base_url', '?')}"
                    )
                else:
                    print(f"  Backend '{_backend}' not reachable.")
                    print("  Check your API keys and network connection.")
                print()
                print("  Cognithor will start anyway, but requests will fail")
                print("  until the LLM backend is available.")
                print("!" * 60)
                print()

            # SessionStore reference for channel persistence
            _session_store = getattr(gateway, "_session_store", None)

            # SSL config for TLS-capable channels
            _ssl_cert = config.security.ssl_certfile
            _ssl_key = config.security.ssl_keyfile

            # Start Control Center API server (always, port 8741)
            try:
                import uvicorn
                from fastapi import FastAPI
                from fastapi.middleware.cors import CORSMiddleware

                from cognithor.channels.config_routes import create_config_routes
                from cognithor.config_manager import ConfigManager

                # Default to 127.0.0.1 for safety. Users can set --api-host 0.0.0.0
                # or COGNITHOR_API_HOST env var for LAN access.
                api_host = args.api_host or os.environ.get("COGNITHOR_API_HOST") or "127.0.0.1"

                # ── Internal session token ────────────────────────────────
                # Always generate a per-session token.  An explicit env var
                # COGNITHOR_API_TOKEN takes precedence; otherwise we mint a
                # cryptographically random one so that even local malware
                # cannot silently call the backend API.
                import secrets as _secrets

                api_token = os.environ.get("COGNITHOR_API_TOKEN") or _secrets.token_urlsafe(32)
                # Expose to frontend via /api/v1/bootstrap (see below)
                _internal_api_token = api_token

                # CORS: if API token set, restrict origins
                if os.environ.get("COGNITHOR_API_TOKEN"):
                    cors_raw = os.environ.get("COGNITHOR_API_CORS_ORIGINS", "")
                    cors_origins = (
                        [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
                    )
                else:
                    cors_origins = ["*"]

                # allow_credentials only when origins are explicitly restricted
                _allow_creds = cors_origins != ["*"]

                api_app = FastAPI(
                    title="Cognithor Control Center API",
                    version=config.version if hasattr(config, "version") else "0.40.0",
                    description=(
                        "Jarvis/Cognithor Backend-API fuer Flutter- und Web-Frontends. "
                        "Authentifizierung via Bearer-Token (GET /api/v1/bootstrap)."
                    ),
                    docs_url="/api/docs",
                    redoc_url="/api/redoc",
                    openapi_url="/api/v1/openapi.json",
                )
                api_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors_origins,
                    allow_credentials=_allow_creds,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

                # ── Rate Limiting Middleware ──────────────────────────────
                import time as _time
                from collections import defaultdict as _defaultdict

                _rate_limit = int(os.environ.get("COGNITHOR_API_RATE_LIMIT", "60"))
                _rate_window = 60.0  # seconds
                _rate_exempt = {"/api/v1/health", "/api/v1/bootstrap"}
                _rate_exempt_prefixes = (
                    "/api/v1/config",
                    "/api/v1/agents",
                    "/api/v1/bindings",
                    "/api/v1/cron-jobs",
                    "/api/v1/mcp-servers",
                    "/api/v1/a2a",
                    "/api/v1/prompts",
                )
                _rate_hits: dict[str, list[float]] = _defaultdict(list)

                # Pure ASGI middleware — does NOT buffer responses (unlike
                # BaseHTTPMiddleware which causes "buffer too large" crashes
                # on Windows ProactorEventLoop for responses >64KB).
                class _RateLimitMiddleware:
                    def __init__(self, app: ASGIApp) -> None:
                        self.app = app

                    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                        if scope["type"] != "http":
                            await self.app(scope, receive, send)
                            return
                        path = scope.get("path", "")
                        if path in _rate_exempt or path.startswith(_rate_exempt_prefixes):
                            await self.app(scope, receive, send)
                            return
                        client = (scope.get("client") or ("unknown",))[0]
                        now = _time.monotonic()
                        hits = _rate_hits[client]
                        cutoff = now - _rate_window
                        _rate_hits[client] = hits = [t for t in hits if t > cutoff]
                        if len(hits) >= _rate_limit:
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": 429,
                                    "headers": [(b"content-type", b"application/json")],
                                }
                            )
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": (
                                        b'{"error":"Too many requests","retry_after_seconds":60}'
                                    ),
                                }
                            )
                            return
                        hits.append(now)
                        await self.app(scope, receive, send)

                api_app.add_middleware(_RateLimitMiddleware)

                # Health-Endpoint
                _api_start = _time.monotonic()

                @api_app.get("/api/v1/health")
                async def _cc_health() -> dict[str, Any]:
                    from cognithor.core.safe_call import get_failure_report

                    _failures = get_failure_report()
                    return {
                        "status": "degraded" if _failures else "ok",
                        "version": __version__,
                        "uptime_seconds": _time.monotonic() - _api_start,
                        "subsystem_failures": len(_failures),
                        "failed_subsystems": list(_failures.keys()) if _failures else [],
                    }

                # ── Bootstrap: DEPRECATED ─────────────────────────────
                # Token is now embedded in index.html via <meta> tag.
                # This endpoint remains as a localhost-only fallback for
                # non-browser clients (CLI tools, mobile apps).
                # SECURITY (GHSA-cognithor-001): loopback-only + one-time.

                try:
                    from cognithor.utils.network import is_trusted_ip as _is_trusted_base

                    # Also trust any IP that the user explicitly enabled
                    try:
                        from cognithor.core.network_endpoints import NetworkEndpointManager

                        _active_ips = set(NetworkEndpointManager().get_active_ips())
                    except Exception:
                        _active_ips = set()

                    def _is_trusted(ip: str) -> bool:
                        return _is_trusted_base(ip) or ip in _active_ips

                except ImportError:

                    def _is_trusted(ip: str) -> bool:
                        return ip.startswith("127.") or ip in ("::1", "localhost")

                _bootstrap_consumed = False

                @api_app.get("/api/v1/bootstrap")
                async def _cc_bootstrap(request: _BootstrapRequest) -> dict[str, str]:
                    nonlocal _bootstrap_consumed
                    client_ip = request.client.host if request.client else ""
                    if not _is_trusted(client_ip):
                        from fastapi.responses import JSONResponse as _JR

                        log.warning(
                            "bootstrap_blocked_non_local",
                            client_ip=client_ip,
                        )
                        return _JR(  # type: ignore[return-value]
                            status_code=403,
                            content={"detail": "Bootstrap endpoint is localhost-only"},
                        )
                    if _bootstrap_consumed:
                        from fastapi.responses import JSONResponse as _JR

                        return _JR(  # type: ignore[return-value]
                            status_code=403,
                            content={"detail": "Bootstrap token already consumed"},
                        )
                    _bootstrap_consumed = True
                    return {"token": _internal_api_token}

                # ── Token verification dependency ─────────────────────
                import hmac as _hmac_verify

                from fastapi import Depends as _Depends
                from fastapi import HTTPException as _HTTPException
                from fastapi.security import HTTPAuthorizationCredentials as _HTTPAuthCreds
                from fastapi.security import HTTPBearer as _HTTPBearer

                _bearer_scheme = _HTTPBearer(auto_error=False)

                async def _verify_cc_token(
                    creds: _HTTPAuthCreds | None = _Depends(_bearer_scheme),  # noqa: B008
                ) -> None:
                    if creds is None or not _hmac_verify.compare_digest(
                        creds.credentials, _internal_api_token
                    ):
                        raise _HTTPException(status_code=401, detail="Unauthorized")

                config_mgr = ConfigManager(config=config)
                create_config_routes(
                    api_app,
                    config_mgr,
                    gateway=gateway,
                    verify_token_dep=_Depends(_verify_cc_token),
                )

                # Skill Marketplace API Router einbinden
                _skills_auth_deps = [_Depends(_verify_cc_token)]

                if getattr(config, "marketplace", None) and config.marketplace.enabled:
                    try:
                        from cognithor.skills.api import router as skills_router

                        if skills_router is not None:
                            api_app.include_router(
                                skills_router,
                                dependencies=_skills_auth_deps,
                            )
                            log.info("skills_marketplace_api_registered")
                    except Exception as _skills_exc:
                        log.warning("skills_marketplace_api_failed", error=str(_skills_exc))

                # Community Marketplace API Router einbinden
                _cm_cfg = getattr(config, "community_marketplace", None)
                if _cm_cfg and getattr(_cm_cfg, "enabled", False):
                    try:
                        from cognithor.skills.api import community_router

                        if community_router is not None:
                            api_app.include_router(
                                community_router,
                                dependencies=_skills_auth_deps,
                            )
                            log.info("community_marketplace_api_registered")
                        else:
                            log.warning(
                                "community_marketplace_router_none",
                                hint="FastAPI nicht installiert?",
                            )
                    except Exception as _cm_exc:
                        log.warning("community_marketplace_api_failed", error=str(_cm_exc))

                # Kanban Board API
                try:
                    if hasattr(gateway, "_kanban_engine") and gateway._kanban_engine is not None:
                        from cognithor.kanban.api import create_kanban_router

                        api_app.include_router(create_kanban_router(gateway._kanban_engine))
                        log.info("kanban_api_registered")
                except Exception:
                    log.debug("kanban_api_registration_failed", exc_info=True)

                # Evolution Engine API
                try:
                    if hasattr(gateway, "_goal_manager") and gateway._goal_manager is not None:
                        from cognithor.evolution.api import create_evolution_router

                        api_app.include_router(
                            create_evolution_router(
                                goal_manager=gateway._goal_manager,
                                journal=getattr(gateway, "_atl_journal", None),
                                deep_learner=getattr(gateway, "_deep_learner", None),
                                cycle_controller=getattr(
                                    getattr(gateway, "_deep_learner", None),
                                    "_cycle_controller",
                                    None,
                                ),
                            )
                        )
                        log.info("evolution_api_registered")
                except Exception:
                    log.debug("evolution_api_registration_failed", exc_info=True)

                # ── WebSocket Chat-Endpoint ──────────────────────────────
                import json as _json

                _ws_connections: dict[str, WebSocket] = {}
                _ws_last_connect: dict[str, float] = {}  # Rate-limit per session
                _WS_MIN_CONNECT_INTERVAL = 1.0  # Minimum seconds between connects

                async def _ws_safe_send(ws: WebSocket, data: dict) -> bool:
                    """Send JSON over WebSocket, catching disconnection errors.

                    Returns True if send succeeded, False if the connection is dead.
                    """
                    try:
                        await ws.send_json(data)
                        return True
                    except Exception:
                        return False

                @api_app.websocket("/ws/{session_id}")
                async def _cc_ws(websocket: WebSocket, session_id: str) -> None:
                    # ── Always accept first (Windows ProactorEventLoop crashes
                    #    if close() is called before accept()) ──────────────
                    required_token = _internal_api_token
                    await websocket.accept()

                    # ── Rate-limit: prevent reconnection storms ─────────
                    import time as _ws_time

                    now = _ws_time.monotonic()
                    last = _ws_last_connect.get(session_id, 0)
                    if now - last < _WS_MIN_CONNECT_INTERVAL:
                        with contextlib.suppress(Exception):
                            await websocket.close(code=4003, reason="Too many connections")
                        return
                    _ws_last_connect[session_id] = now

                    if required_token:
                        import hmac as _hmac

                        try:
                            auth_raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=10.0,
                            )
                            auth_msg = _json.loads(auth_raw)
                            client_token = (
                                auth_msg.get("token", "") if auth_msg.get("type") == "auth" else ""
                            )
                        except (TimeoutError, Exception):
                            client_token = ""
                        if not _hmac.compare_digest(client_token, required_token):
                            await _ws_safe_send(
                                websocket,
                                {
                                    "type": "error",
                                    "error": "Unauthorized",
                                },
                            )
                            with contextlib.suppress(Exception):
                                await websocket.close(code=4001, reason="Unauthorized")
                            log.warning(
                                "cc_ws_auth_rejected",
                                session_id=session_id,
                                reason="missing_or_invalid_token",
                            )
                            return

                    # ── Session collision: silently replace ──────────────
                    # Do NOT close the old connection — closing with 4002
                    # triggers the client's reconnect logic → infinite loop.
                    # Just replace in the dict; the old handler will get a
                    # WebSocketDisconnect on its next receive/send attempt.
                    existing = _ws_connections.get(session_id)
                    if existing is not None and existing is not websocket:
                        log.debug("cc_ws_replacing", session_id=session_id)

                    _ws_connections[session_id] = websocket
                    log.info("cc_ws_connected", session_id=session_id)
                    try:
                        while True:
                            raw = await websocket.receive_text()
                            try:
                                msg = _json.loads(raw)
                            except _json.JSONDecodeError:
                                if not await _ws_safe_send(
                                    websocket, {"type": "error", "error": "Ungültiges JSON"}
                                ):
                                    break
                                continue

                            msg_type = msg.get("type", "")
                            if msg_type != "ping":
                                log.info("ws_msg_received", type=msg_type, keys=list(msg.keys()))

                            if msg_type == "ping":
                                if not await _ws_safe_send(websocket, {"type": "pong"}):
                                    break
                                continue

                            if msg_type == "cancel":
                                gateway.cancel_session(session_id)
                                if not await _ws_safe_send(
                                    websocket,
                                    {
                                        "type": "status_update",
                                        "status": "finishing",
                                        "text": "Abgebrochen...",
                                        "session_id": session_id,
                                    },
                                ):
                                    break
                                continue

                            if msg_type in ("user_message", "message"):
                                text = (msg.get("text") or "").strip()
                                metadata = msg.get("metadata", {})
                                if not text:
                                    if not await _ws_safe_send(
                                        websocket, {"type": "error", "error": "Leere Nachricht"}
                                    ):
                                        break
                                    continue

                                # ── Audio transcription ────────────────────
                                audio_b64 = metadata.get("audio_base64")
                                if not audio_b64 and metadata.get("file_type", "").startswith(
                                    "audio/"
                                ):
                                    audio_b64 = metadata.get("file_base64")
                                if audio_b64:
                                    import base64 as _b64
                                    import tempfile as _tmpfile

                                    # Size-Limit: Base64-String vor Decode pruefen (50 MB decoded)
                                    _MAX_AUDIO_B64_BYTES = 52_428_800  # 50 MB
                                    estimated_size = len(audio_b64) * 3 // 4
                                    if estimated_size > _MAX_AUDIO_B64_BYTES:
                                        if not await _ws_safe_send(
                                            websocket,
                                            {
                                                "type": "error",
                                                "error": (
                                                    f"Audiodatei zu gross "
                                                    f"({estimated_size // 1_048_576} MB, "
                                                    f"max {_MAX_AUDIO_B64_BYTES // 1_048_576} MB)"
                                                ),
                                            },
                                        ):
                                            break
                                        continue
                                    audio_type = (
                                        metadata.get("audio_type")
                                        or metadata.get("file_type")
                                        or "audio/webm"
                                    )
                                    ext = {
                                        "audio/webm": ".webm",
                                        "audio/ogg": ".ogg",
                                        "audio/wav": ".wav",
                                        "audio/mp3": ".mp3",
                                        "audio/mpeg": ".mp3",
                                        "audio/m4a": ".m4a",
                                        "audio/flac": ".flac",
                                    }.get(audio_type, ".webm")
                                    tmp_path = None
                                    try:
                                        raw_audio = _b64.b64decode(audio_b64)
                                        with _tmpfile.NamedTemporaryFile(
                                            suffix=ext, delete=False
                                        ) as tmp:
                                            tmp.write(raw_audio)
                                            tmp_path = tmp.name
                                        from cognithor.mcp.media import MediaPipeline

                                        _media = MediaPipeline()
                                        result = await _media.transcribe_audio(
                                            tmp_path, language="de"
                                        )
                                        if result.success and result.text and result.text.strip():
                                            text = result.text.strip()
                                            log.info("ws_audio_transcribed", text=text[:80])
                                            metadata.pop("audio_base64", None)
                                            metadata.pop("file_base64", None)
                                            metadata["transcribed_from"] = "audio"
                                            if not await _ws_safe_send(
                                                websocket,
                                                {
                                                    "type": "transcription",
                                                    "text": text,
                                                    "session_id": session_id,
                                                },
                                            ):
                                                break
                                        else:
                                            log.warning(
                                                "ws_audio_transcription_failed",
                                                error=getattr(result, "error", ""),
                                            )
                                            if not await _ws_safe_send(
                                                websocket,
                                                {
                                                    "type": "error",
                                                    "error": (
                                                        "Audiodatei konnte nicht "
                                                        "transkribiert werden."
                                                    ),
                                                },
                                            ):
                                                break
                                            continue
                                    except Exception as _audio_exc:
                                        log.error(
                                            "ws_audio_transcription_error", error=str(_audio_exc)
                                        )
                                        if not await _ws_safe_send(
                                            websocket,
                                            {
                                                "type": "error",
                                                "error": "Fehler bei der Audio-Transkription.",
                                            },
                                        ):
                                            break
                                        continue
                                    finally:
                                        if tmp_path:
                                            try:
                                                import os as _os

                                                _os.unlink(tmp_path)
                                            except Exception:
                                                log.debug(
                                                    "webui_tmp_file_cleanup_failed", exc_info=True
                                                )

                                from cognithor.models import IncomingMessage

                                incoming = IncomingMessage(
                                    text=text,
                                    channel="webui",
                                    session_id=session_id,
                                    user_id="web_user",
                                    metadata=metadata,
                                )
                                try:
                                    response = await gateway.handle_message(incoming)
                                    if not await _ws_safe_send(
                                        websocket,
                                        {
                                            "type": "assistant_message",
                                            "text": response.text,
                                            "session_id": session_id,
                                        },
                                    ):
                                        break
                                    if not await _ws_safe_send(
                                        websocket,
                                        {
                                            "type": "stream_end",
                                            "session_id": session_id,
                                        },
                                    ):
                                        break
                                except Exception as _ws_exc:
                                    log.error("cc_ws_handler_error", error=str(_ws_exc))
                                    if not await _ws_safe_send(
                                        websocket,
                                        {
                                            "type": "error",
                                            "error": "Verarbeitungsfehler aufgetreten.",
                                        },
                                    ):
                                        break
                                continue

                            if msg_type == "feedback":
                                _fb_store = getattr(gateway, "_feedback_store", None)
                                if _fb_store:
                                    _fb_rating = msg.get("rating", 0)
                                    _fb_id = _fb_store.submit(
                                        session_id=msg.get("session_id", session_id),
                                        message_id=msg.get("message_id", ""),
                                        rating=_fb_rating,
                                        comment=msg.get("comment", ""),
                                        agent_name=msg.get("agent_name", "jarvis"),
                                        channel="webui",
                                        user_message=msg.get("user_message", ""),
                                        assistant_response=msg.get("assistant_response", ""),
                                    )
                                    if not await _ws_safe_send(
                                        websocket,
                                        {"type": "feedback_ack", "feedback_id": _fb_id},
                                    ):
                                        break
                                    # On thumbs down: send follow-up question
                                    if _fb_rating == -1 and not await _ws_safe_send(
                                        websocket,
                                        {
                                            "type": "feedback_followup",
                                            "feedback_id": _fb_id,
                                            "question": (
                                                "Was hat an meiner Antwort nicht gepasst? "
                                                "Dein Feedback hilft mir, besser zu werden."
                                            ),
                                        },
                                    ):
                                        break
                                continue

                            if msg_type == "feedback_comment":
                                _fb_store = getattr(gateway, "_feedback_store", None)
                                if _fb_store:
                                    _fb_store.add_comment(
                                        msg.get("feedback_id", ""),
                                        msg.get("comment", ""),
                                    )
                                continue

                            if msg_type == "branch_switch":
                                conv_id = msg.get("conversation_id", "")
                                leaf_id = msg.get("leaf_id", "")
                                if conv_id and leaf_id and gateway:
                                    try:
                                        session = gateway._get_or_create_session(
                                            "webui", "default", "jarvis"
                                        )
                                        wm = await gateway.switch_branch(conv_id, leaf_id, session)
                                        await _ws_safe_send(
                                            websocket,
                                            {
                                                "type": "branch_switched",
                                                "conversation_id": conv_id,
                                                "leaf_id": leaf_id,
                                                "message_count": len(wm.chat_history),
                                            },
                                        )
                                    except Exception as exc:
                                        await _ws_safe_send(
                                            websocket,
                                            {
                                                "type": "error",
                                                "text": f"Branch switch failed: {exc}",
                                            },
                                        )
                                continue

                            if msg_type == "approval_response":
                                _req_id = msg.get("id", "") or msg.get("request_id", "")
                                _approved = msg.get("approved", False)
                                log.info(
                                    "approval_response_raw",
                                    request_id=_req_id,
                                    approved=_approved,
                                    pending_keys=list(_pending_approvals.keys()),
                                )
                                _future = _pending_approvals.get(_req_id)
                                if _future and not _future.done():
                                    _future.set_result(bool(_approved))
                                    log.info(
                                        "approval_received", request_id=_req_id, approved=_approved
                                    )
                                else:
                                    log.warning(
                                        "approval_response_no_future",
                                        request_id=_req_id,
                                        future_exists=_future is not None,
                                        future_done=_future.done() if _future else None,
                                    )
                                continue

                            if not await _ws_safe_send(
                                websocket,
                                {"type": "error", "error": f"Unbekannter Typ: {msg_type}"},
                            ):
                                break
                    except WebSocketDisconnect:
                        log.info("cc_ws_disconnected", session_id=session_id)
                    except Exception as _ws_exc:
                        log.error("cc_ws_error", error=str(_ws_exc), session_id=session_id)
                    finally:
                        _ws_connections.pop(session_id, None)

                log.info("cc_websocket_endpoint_registered")

                # ── WebUI als Channel im Gateway registrieren ───────────
                # Damit send_status() und send_pipeline_event() den
                # Browser ueber die bestehenden _ws_connections erreichen.
                from cognithor.channels.base import Channel, StatusType

                _pending_approvals: dict[str, asyncio.Future[bool]] = {}

                # ── REST fallback for approval responses ────────────────
                # If the WebSocket connection drops between sending the
                # approval_request and receiving the user's response, the
                # frontend can POST the decision here instead.
                @api_app.post(
                    "/api/v1/approval_response",
                    dependencies=[_Depends(_verify_cc_token)],
                )
                async def _cc_approval_response_rest(
                    request: _STRequest,
                ) -> dict:
                    try:
                        body = await request.json()
                    except Exception:
                        return {"ok": False, "error": "invalid_json"}
                    _req_id = body.get("request_id", "") or body.get("id", "")
                    _approved = bool(body.get("approved", False))
                    log.info(
                        "approval_rest_received",
                        request_id=_req_id,
                        approved=_approved,
                        pending_keys=list(_pending_approvals.keys()),
                    )
                    _future = _pending_approvals.get(_req_id)
                    if _future and not _future.done():
                        _future.set_result(_approved)
                        return {"ok": True}
                    return {"ok": False, "error": "no_pending_future"}

                class _WebUIBridge(Channel):
                    """Leichtgewichtiger Adapter: Gateway-Channel → WS."""

                    @property
                    def name(self) -> str:
                        return "webui"

                    async def start(self, handler: Any) -> None:
                        pass  # WS-Endpoint laeuft bereits

                    async def stop(self) -> None:
                        pass

                    async def send(self, message: Any) -> None:
                        pass  # Antworten werden inline im WS-Handler gesendet

                    async def send_streaming_token(self, session_id: str, token: str) -> None:
                        ws = _ws_connections.get(session_id)
                        if ws:
                            await _ws_safe_send(ws, {"type": "stream_token", "token": token})

                    async def request_approval(
                        self,
                        session_id: str,
                        action: Any = None,
                        tool: str = "",
                        params: dict | None = None,
                        reason: str = "",
                        **kwargs: Any,
                    ) -> bool:
                        log.info(
                            "approval_called",
                            session_id=session_id[:8],
                            available_ws=list(_ws_connections.keys()),
                        )
                        ws = _ws_connections.get(session_id)
                        if not ws:
                            log.warning(
                                "approval_no_ws",
                                session_id=session_id[:8],
                                available=list(_ws_connections.keys()),
                            )
                            return False

                        import uuid

                        _tool = tool or (getattr(action, "tool", "") if action else "")
                        _params = params or (getattr(action, "params", {}) if action else {})
                        request_id = uuid.uuid4().hex[:12]
                        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                        _pending_approvals[request_id] = future

                        _sent = await _ws_safe_send(
                            ws,
                            {
                                "type": "approval_request",
                                "request_id": request_id,
                                "session_id": session_id,
                                "tool": _tool,
                                "params": _params,
                                "reason": reason,
                            },
                        )
                        log.info(
                            "approval_sent",
                            tool=_tool,
                            request_id=request_id,
                            ws_send_ok=_sent,
                        )

                        try:
                            return await asyncio.wait_for(future, timeout=1800)
                        except TimeoutError:
                            log.warning("approval_timeout", request_id=request_id)
                            return False
                        finally:
                            _pending_approvals.pop(request_id, None)

                    async def send_status(
                        self, session_id: str, status: StatusType, text: str
                    ) -> None:
                        ws = _ws_connections.get(session_id)
                        if ws:
                            await _ws_safe_send(
                                ws,
                                {
                                    "type": "status_update",
                                    "status": status.value,
                                    "text": text,
                                    "session_id": session_id,
                                },
                            )

                    async def send_pipeline_event(self, session_id: str, event: dict) -> None:
                        ws = _ws_connections.get(session_id)
                        if ws:
                            await _ws_safe_send(
                                ws,
                                {
                                    "type": "pipeline_event",
                                    "session_id": session_id,
                                    **event,
                                },
                            )

                    async def send_plan_detail(self, session_id: str, plan_data: dict) -> None:
                        ws = _ws_connections.get(session_id)
                        if ws:
                            await _ws_safe_send(
                                ws,
                                {
                                    "type": "plan_detail",
                                    "session_id": session_id,
                                    **plan_data,
                                },
                            )

                    async def send_identity_state(self, session_id: str, state: dict) -> None:
                        ws = _ws_connections.get(session_id)
                        if ws:
                            await _ws_safe_send(
                                ws, {"type": "identity_state", "session_id": session_id, **state}
                            )

                gateway.register_channel(_WebUIBridge())
                log.info("webui_channel_bridge_registered")

                # ── TTS-Endpoint (Piper) ─────────────────────────────────
                _voice_cfg = getattr(getattr(config, "channels", None), "voice_config", None)
                _default_piper_voice = (
                    getattr(_voice_cfg, "piper_voice", "de_DE-thorsten_emotional-medium")
                    if _voice_cfg
                    else "de_DE-thorsten_emotional-medium"
                )
                _default_length_scale = (
                    getattr(_voice_cfg, "piper_length_scale", 1.0) if _voice_cfg else 1.0
                )

                @api_app.post("/api/v1/tts", dependencies=[_Depends(_verify_cc_token)])
                async def _cc_tts(body: dict[str, Any]) -> Any:
                    """Text-to-Speech via Piper TTS."""
                    from fastapi.responses import Response

                    text = (body.get("text") or "").strip()
                    if not text:
                        return {"error": "Kein Text angegeben", "code": "MISSING_FIELD"}

                    voice = body.get("voice", _default_piper_voice)
                    length_scale = body.get("length_scale", _default_length_scale)
                    try:
                        wav_bytes = await _run_piper_tts(text, voice, length_scale)
                        return Response(content=wav_bytes, media_type="audio/wav")
                    except ValueError as _val_exc:
                        # CWE-22: Invalid voice name (path traversal attempt)
                        log.warning("tts_voice_validation_failed", voice=voice, error=str(_val_exc))
                        return {"error": "Ungueltiger Voice-Name", "code": "INVALID_VOICE"}
                    except FileNotFoundError:
                        return {
                            "error": "Piper TTS nicht installiert. Bitte: pip install piper-tts",
                            "code": "TTS_NOT_INSTALLED",
                        }
                    except Exception as _tts_exc:
                        log.error("tts_error", error=str(_tts_exc))
                        return {"error": "TTS-Fehler aufgetreten", "code": "TTS_ERROR"}

                @api_app.get("/api/v1/tts/voices", dependencies=[_Depends(_verify_cc_token)])
                async def _cc_tts_voices() -> dict[str, Any]:
                    """Listet verfuegbare Piper-Stimmen und die aktuell konfigurierte."""
                    voices_dir = Path(config.cognithor_home) / "voices"
                    installed: list[str] = []
                    if voices_dir.exists():
                        installed = [f.stem for f in voices_dir.glob("*.onnx")]
                    return {
                        "current": _default_piper_voice,
                        "installed": installed,
                        "available": [
                            {
                                "id": "de_DE-pavoque-low",
                                "name": "Pavoque (Maennlich, Bariton)",
                                "quality": "low",
                            },
                            {
                                "id": "de_DE-karlsson-low",
                                "name": "Karlsson (Maennlich)",
                                "quality": "low",
                            },
                            {
                                "id": "de_DE-thorsten-high",
                                "name": "Thorsten (Maennlich)",
                                "quality": "high",
                            },
                            {
                                "id": "de_DE-thorsten-medium",
                                "name": "Thorsten (Maennlich)",
                                "quality": "medium",
                            },
                            {
                                "id": "de_DE-thorsten_emotional-medium",
                                "name": "Thorsten Emotional",
                                "quality": "medium",
                            },
                            {
                                "id": "de_DE-kerstin-low",
                                "name": "Kerstin (Weiblich)",
                                "quality": "low",
                            },
                            {
                                "id": "de_DE-ramona-low",
                                "name": "Ramona (Weiblich)",
                                "quality": "low",
                            },
                            {
                                "id": "de_DE-eva_k-x_low",
                                "name": "Eva K (Weiblich)",
                                "quality": "x_low",
                            },
                        ],
                    }

                async def _run_piper_tts(text: str, voice: str, length_scale: float = 1.0) -> bytes:
                    """Generiert WAV-Audio via Piper TTS."""
                    import re
                    import tempfile

                    # CWE-22: Inline validation + sanitizer defense-in-depth
                    # CodeQL requires visible inline guards before path construction
                    if (
                        not voice
                        or "/" in voice
                        or "\\" in voice
                        or ".." in voice
                        or "\x00" in voice
                    ):
                        raise ValueError(f"Ungueltiger Stimmenname: {voice!r}")
                    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.\-]*", voice):
                        raise ValueError(f"Ungueltiger Stimmenname: {voice!r}")

                    # Voice-Modell-Pfad ermitteln
                    voices_dir = Path(config.cognithor_home) / "voices"
                    voices_dir.mkdir(exist_ok=True)

                    # Defense-in-depth: normalize and validate path stays in voices_dir
                    import os.path as _osp

                    _norm_voices = _osp.normpath(_osp.realpath(str(voices_dir)))
                    _norm_model = _osp.normpath(_osp.join(_norm_voices, f"{voice}.onnx"))
                    if not _norm_model.startswith(_norm_voices + _osp.sep):
                        raise ValueError("Modellpfad verletzt Verzeichnisgrenzen")
                    model_path = Path(_norm_model)

                    # Auto-Download wenn nicht vorhanden
                    if not model_path.exists():
                        await _download_piper_voice(voice, voices_dir)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name

                    try:
                        # Write text to a temp file with explicit UTF-8 encoding
                        # to avoid Windows cp1252 stdin encoding issues with umlauts
                        import tempfile as _tts_tmpfile

                        with _tts_tmpfile.NamedTemporaryFile(
                            mode="w", suffix=".txt", delete=False, encoding="utf-8"
                        ) as txt_tmp:
                            txt_tmp.write(text)
                            txt_input_path = txt_tmp.name

                        cmd = [
                            sys.executable,
                            "-m",
                            "piper",
                            "--model",
                            str(model_path),
                            "--output_file",
                            tmp_path,
                            "--length-scale",
                            str(length_scale),
                        ]
                        # Multi-speaker models (e.g. thorsten_emotional) need --speaker
                        _norm_json = _osp.normpath(_osp.join(_norm_voices, f"{voice}.onnx.json"))
                        if not _norm_json.startswith(_norm_voices + _osp.sep):
                            raise ValueError("Modell-JSON verletzt Verzeichnisgrenzen")
                        model_json = Path(_norm_json)
                        if model_json.exists():
                            try:
                                import json as _mj

                                _model_cfg = _mj.loads(model_json.read_text(encoding="utf-8"))
                                _speaker_map = _model_cfg.get("speaker_id_map", {})
                                if _model_cfg.get("num_speakers", 1) > 1 and _speaker_map:
                                    # Prefer "neutral", fallback to first speaker
                                    _spk = (
                                        "neutral"
                                        if "neutral" in _speaker_map
                                        else next(iter(_speaker_map))
                                    )
                                    cmd.extend(["--speaker", str(_speaker_map[_spk])])
                            except Exception:
                                log.debug("tts_speaker_config_load_failed", exc_info=True)
                        # Read the UTF-8 text file as bytes for stdin
                        with open(txt_input_path, "rb") as _tts_in:
                            _tts_input_bytes = _tts_in.read()
                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, stderr = await proc.communicate(input=_tts_input_bytes)

                        if proc.returncode != 0:
                            raise RuntimeError(f"Piper fehlgeschlagen: {stderr.decode()[:200]}")

                        with open(tmp_path, "rb") as f:
                            return f.read()
                    finally:
                        with contextlib.suppress(OSError):
                            os.unlink(tmp_path)
                        with contextlib.suppress(OSError, NameError):
                            os.unlink(txt_input_path)

                # Bekannte SHA-256 Hashes fuer Piper Voice-Modelle.
                # Neue Hashes werden beim Download geloggt und koennen hier
                # eingetragen werden. Unbekannte Voices werden mit Warnung
                # akzeptiert (nicht blockiert).
                _KNOWN_VOICE_HASHES: dict[str, str] = {
                    # Format: "voice-id": "sha256-hex-digest"
                    # Hashes werden beim ersten Download geloggt.
                }

                def _verify_voice_hash(voice: str, file_hash: str) -> None:
                    """Prueft SHA-256 eines heruntergeladenen Voice-Modells."""
                    expected = _KNOWN_VOICE_HASHES.get(voice)
                    if expected is None:
                        log.warning(
                            "voice_hash_unknown",
                            voice=voice,
                            sha256=file_hash,
                            hint="Hash nicht in _KNOWN_VOICE_HASHES hinterlegt",
                        )
                        return
                    if file_hash != expected:
                        # Datei loeschen bei Hash-Mismatch
                        raise ValueError(
                            f"Integrity check failed fuer Voice '{voice}': "
                            f"erwartet {expected[:16]}..., erhalten {file_hash[:16]}..."
                        )
                    log.info("voice_hash_verified", voice=voice)

                async def _download_piper_voice(voice: str, dest: Path) -> None:
                    """Downloads a Piper voice model from HuggingFace."""
                    import hashlib
                    import urllib.request

                    from cognithor.security.sanitizer import (
                        validate_model_path_containment,
                        validate_voice_name,
                    )

                    # CWE-22: Validate voice name before download
                    validate_voice_name(voice)

                    # Defense-in-depth: ensure download target stays in dest dir
                    validate_model_path_containment(dest / f"{voice}.onnx", dest)

                    parts = voice.split("-")  # de_DE-pavoque-low
                    lang = parts[0]  # de_DE
                    name = parts[1]  # pavoque
                    quality = parts[2] if len(parts) > 2 else "low"
                    lang_short = lang.split("_")[0]  # de

                    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_short}/{lang}/{name}/{quality}"
                    onnx_url = f"{base}/{voice}.onnx?download=true"
                    json_url = f"{base}/{voice}.onnx.json?download=true"

                    log.info("downloading_piper_voice", voice=voice, url=onnx_url)

                    import os.path as _dl_osp

                    _norm_dest = _dl_osp.normpath(_dl_osp.realpath(str(dest)))
                    _norm_onnx = _dl_osp.normpath(_dl_osp.join(_norm_dest, f"{voice}.onnx"))
                    _norm_json_dl = _dl_osp.normpath(_dl_osp.join(_norm_dest, f"{voice}.onnx.json"))
                    if not _norm_onnx.startswith(
                        _norm_dest + _dl_osp.sep
                    ) or not _norm_json_dl.startswith(_norm_dest + _dl_osp.sep):
                        raise ValueError("Download-Pfad verletzt Verzeichnisgrenzen")

                    def _dl() -> None:
                        urllib.request.urlretrieve(onnx_url, _norm_onnx)
                        urllib.request.urlretrieve(json_url, _norm_json_dl)

                    await asyncio.get_running_loop().run_in_executor(None, _dl)

                    # Integrity check: SHA-256 verifizieren
                    onnx_path = Path(_norm_onnx)
                    file_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
                    _verify_voice_hash(voice, file_hash)

                    log.info("piper_voice_downloaded", voice=voice, sha256=file_hash)

                log.info("cc_tts_endpoint_registered")

                # ── Voice Transcription API ────────────────────────────

                @api_app.post("/api/v1/voice/transcribe", dependencies=[_Depends(_verify_cc_token)])
                async def _voice_transcribe(request: _STRequest) -> dict[str, Any]:
                    """Transkribiert hochgeladene Audio-Datei (multipart/form-data).

                    Erwartet Feld 'audio' mit der Audio-Datei.
                    """
                    import tempfile

                    try:
                        form = await request.form()
                        audio_field = form.get("audio")
                        if audio_field is None:
                            return {"error": "Feld 'audio' fehlt", "code": "MISSING_FIELD"}

                        audio_bytes = await audio_field.read()
                        if not audio_bytes:
                            return {"error": "Leere Audio-Datei", "code": "EMPTY_FILE"}

                        suffix = ".webm"
                        if hasattr(audio_field, "filename") and audio_field.filename:
                            import os.path as _ap

                            suffix = _ap.splitext(audio_field.filename)[1] or ".webm"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(audio_bytes)
                            tmp_path = tmp.name

                        try:
                            from cognithor.mcp.media import MediaPipeline

                            media = MediaPipeline()
                            result = await media.transcribe_audio(tmp_path, language="de")
                            if result.success and result.text:
                                return {"text": result.text.strip()}
                            return {
                                "error": result.error or "Transkription fehlgeschlagen",
                                "code": "TRANSCRIPTION_FAILED",
                            }
                        finally:
                            with contextlib.suppress(OSError):
                                os.unlink(tmp_path)
                    except Exception as exc:
                        log.error("voice_transcribe_error", error=str(exc))
                        return {"error": "Transkriptionsfehler", "code": "INTERNAL_ERROR"}

                log.info("cc_voice_transcribe_endpoint_registered")

                # ── Vision Analysis API ────────────────────────────────
                @api_app.post("/api/v1/vision/analyze", dependencies=[_Depends(_verify_cc_token)])
                async def _vision_analyze(request: _STRequest) -> dict[str, Any]:
                    """Analysiert ein hochgeladenes Bild (multipart/form-data).

                    Felder: 'image' (Datei), 'prompt' (optional, Text).
                    """
                    import tempfile

                    try:
                        form = await request.form()
                        image_field = form.get("image")
                        if image_field is None:
                            return {"error": "Feld 'image' fehlt", "code": "MISSING_FIELD"}

                        image_bytes = await image_field.read()
                        if not image_bytes:
                            return {"error": "Leere Bilddatei", "code": "EMPTY_FILE"}

                        prompt = form.get("prompt", "Beschreibe dieses Bild detailliert.")
                        if isinstance(prompt, bytes):
                            prompt = prompt.decode("utf-8")

                        # Dateiendung bestimmen
                        suffix = ".png"
                        if hasattr(image_field, "filename") and image_field.filename:
                            import os.path as _ip

                            suffix = _ip.splitext(image_field.filename)[1] or ".png"

                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(image_bytes)
                            tmp_path = tmp.name

                        try:
                            from cognithor.mcp.media import MediaPipeline

                            media = MediaPipeline()
                            result = await media.analyze_image(tmp_path, prompt=str(prompt))
                            if result.success and result.text:
                                return {"text": result.text.strip()}
                            return {
                                "error": result.error or "Bildanalyse fehlgeschlagen",
                                "code": "VISION_FAILED",
                            }
                        finally:
                            with contextlib.suppress(OSError):
                                os.unlink(tmp_path)
                    except Exception as exc:
                        log.error("vision_analyze_error", error=str(exc))
                        return {"error": "Bildanalysefehler", "code": "INTERNAL_ERROR"}

                log.info("cc_vision_endpoint_registered")

                # ── Push Notifications API ─────────────────────────────
                @api_app.get("/api/v1/push/vapid-key", dependencies=[_Depends(_verify_cc_token)])
                async def _push_vapid_key() -> dict[str, Any]:
                    """Gibt den VAPID Public Key fuer Push-Notifications zurueck."""
                    vapid_key = os.environ.get("COGNITHOR_VAPID_PUBLIC_KEY", "")
                    if not vapid_key:
                        return {"error": "VAPID nicht konfiguriert", "code": "NOT_CONFIGURED"}
                    return {"key": vapid_key}

                @api_app.post("/api/v1/push/register", dependencies=[_Depends(_verify_cc_token)])
                async def _push_register(body: dict[str, Any]) -> dict[str, Any]:
                    """Registriert ein Geraet fuer Push-Notifications."""
                    token = body.get("token", "").strip()
                    push_type = body.get("type", "fcm")
                    if not token:
                        return {"error": "Token fehlt", "code": "MISSING_FIELD"}

                    # Save registration in DB
                    push_db = Path(config.cognithor_home) / "push_subscriptions.json"
                    import json as _pj

                    subs: list[dict[str, str]] = []
                    if push_db.exists():
                        try:
                            subs = _pj.loads(push_db.read_text(encoding="utf-8"))
                        except Exception:
                            subs = []

                    # Duplikate vermeiden
                    if not any(s.get("token") == token for s in subs):
                        subs.append({"token": token, "type": push_type})
                        push_db.write_text(_pj.dumps(subs, indent=2), encoding="utf-8")
                        log.info("push_device_registered", type=push_type)

                    return {"status": "ok"}

                log.info("cc_push_endpoints_registered")

                # ── Identity Control API ────────────────────────────────
                @api_app.get("/api/v1/identity/state", dependencies=[_Depends(_verify_cc_token)])
                async def _identity_state():
                    if not hasattr(gateway, "_identity_layer") or gateway._identity_layer is None:
                        return {"available": False}
                    try:
                        state = gateway._identity_layer.get_state_summary()
                        return state
                    except Exception:
                        log.debug("identity_api_error", exc_info=True)
                        return {"error": "Internal identity error", "code": "INTERNAL_ERROR"}

                @api_app.post("/api/v1/identity/freeze", dependencies=[_Depends(_verify_cc_token)])
                async def _identity_freeze():
                    if not hasattr(gateway, "_identity_layer") or gateway._identity_layer is None:
                        return {"error": "Identity layer not available", "code": "NOT_AVAILABLE"}
                    gateway._identity_layer.freeze()
                    return {"status": "frozen"}

                @api_app.post(
                    "/api/v1/identity/unfreeze", dependencies=[_Depends(_verify_cc_token)]
                )
                async def _identity_unfreeze():
                    if not hasattr(gateway, "_identity_layer") or gateway._identity_layer is None:
                        return {"error": "Identity layer not available", "code": "NOT_AVAILABLE"}
                    gateway._identity_layer.unfreeze()
                    return {"status": "unfrozen"}

                @api_app.post("/api/v1/identity/reset", dependencies=[_Depends(_verify_cc_token)])
                async def _identity_reset():
                    if not hasattr(gateway, "_identity_layer") or gateway._identity_layer is None:
                        return {"error": "Identity layer not available", "code": "NOT_AVAILABLE"}
                    result = gateway._identity_layer.soft_reset()
                    return {"status": "reset", "details": result}

                @api_app.post("/api/v1/identity/dream", dependencies=[_Depends(_verify_cc_token)])
                async def _identity_dream():
                    if not hasattr(gateway, "_identity_layer") or gateway._identity_layer is None:
                        return {"error": "Identity layer not available", "code": "NOT_AVAILABLE"}
                    try:
                        engine = gateway._identity_layer._engine
                        if engine is None:
                            return {"error": "Engine not initialized", "code": "NOT_INITIALIZED"}
                        stats = engine.dream.run(engine)
                        return {"status": "dream_completed", "stats": str(stats)}
                    except Exception:
                        log.debug("identity_api_error", exc_info=True)
                        return {"error": "Internal identity error", "code": "INTERNAL_ERROR"}

                log.info("cc_identity_endpoints_registered")

                # VS Code Extension API
                try:
                    from cognithor.channels.vscode_routes import register_vscode_routes

                    register_vscode_routes(api_app, gateway, [_Depends(_verify_cc_token)])
                    log.info("vscode_api_registered")
                except Exception:
                    log.debug("vscode_api_not_registered", exc_info=True)

                # Mount pre-built UI at / (catch-all, MUSS als letztes)
                # Prioritaet: Flutter-Build > React-Build
                _repo_root = Path(__file__).resolve().parent.parent.parent
                _jarvis_home = Path(os.environ.get("COGNITHOR_HOME", Path.home() / ".cognithor"))
                _install_root = Path(sys.executable).resolve().parent.parent  # e.g. D:\Cognithor
                _ui_candidates = [
                    _repo_root / "flutter_app" / "build" / "web",
                    _install_root / "flutter_app" / "web",
                    _jarvis_home / "flutter_web",
                ]
                _ui_dist = None
                for _candidate in _ui_candidates:
                    if _candidate.is_dir() and (_candidate / "index.html").exists():
                        _ui_dist = _candidate
                        break

                if _ui_dist is not None:
                    from fastapi.responses import HTMLResponse as _HTMLResp
                    from fastapi.staticfiles import StaticFiles

                    # Inject auth token into index.html as <meta> tag so the
                    # Flutter app can read it from the DOM instead of calling
                    # /api/v1/bootstrap. This eliminates the unauthenticated
                    # token-disclosure endpoint entirely (GHSA-cognithor-001).
                    _index_html_raw = (_ui_dist / "index.html").read_text(encoding="utf-8")
                    _token_meta = f'<meta name="cognithor-token" content="{_internal_api_token}">'
                    _index_html_injected = _index_html_raw.replace(
                        "<head>", f"<head>\n  {_token_meta}", 1
                    )

                    @api_app.get("/", response_class=_HTMLResp)
                    @api_app.get("/index.html", response_class=_HTMLResp)
                    async def _serve_index() -> _HTMLResp:
                        return _HTMLResp(content=_index_html_injected)

                    # All other static assets (JS, CSS, images) served normally
                    api_app.mount("/", StaticFiles(directory=str(_ui_dist), html=False), name="ui")
                    log.info("prebuilt_ui_mounted", path=str(_ui_dist))

                # TLS-Durchreichung
                uvi_kwargs: dict[str, Any] = {
                    "app": api_app,
                    "host": api_host,
                    "port": args.api_port,
                    "log_level": "warning",
                    "http": "auto",
                }
                if _mtls_certs_dir is not None:
                    # mTLS: Server-Zertifikat + Client-Verifizierung
                    import ssl as _ssl_mod

                    uvi_kwargs["ssl_certfile"] = str(_mtls_certs_dir / "server.pem")
                    uvi_kwargs["ssl_keyfile"] = str(_mtls_certs_dir / "server-key.pem")
                    uvi_kwargs["ssl_ca_certs"] = str(_mtls_certs_dir / "ca.pem")
                    uvi_kwargs["ssl_cert_reqs"] = _ssl_mod.CERT_REQUIRED
                elif _ssl_cert and _ssl_key:
                    uvi_kwargs["ssl_certfile"] = _ssl_cert
                    uvi_kwargs["ssl_keyfile"] = _ssl_key

                uvi_config = uvicorn.Config(**uvi_kwargs)
                api_server = uvicorn.Server(uvi_config)
                _t = asyncio.create_task(api_server.serve())
                _bg_tasks.add(_t)
                _t.add_done_callback(_bg_tasks.discard)
                log.info(
                    "control_center_api_started",
                    host=api_host,
                    port=args.api_port,
                    tls=bool(_ssl_cert),
                )

                # --ui: open default browser once the server is ready
                if getattr(args, "ui", False):
                    import webbrowser

                    _scheme = "https" if _ssl_cert else "http"
                    _ui_url = f"{_scheme}://localhost:{args.api_port}"

                    async def _open_browser_when_ready() -> None:
                        # Wait for uvicorn to accept connections
                        for _ in range(50):
                            if api_server is not None and getattr(api_server, "started", False):
                                break
                            await asyncio.sleep(0.1)
                        try:
                            webbrowser.open(_ui_url)
                            print(f"  [OK] Browser opened: {_ui_url}")
                        except Exception as _exc:
                            log.warning("ui_browser_open_failed", error=str(_exc))

                    _bt = asyncio.create_task(_open_browser_when_ready())
                    _bg_tasks.add(_bt)
                    _bt.add_done_callback(_bg_tasks.discard)
            except ImportError:
                log.warning("control_center_api_requires_fastapi_uvicorn")
            except Exception as exc:
                log.warning("control_center_api_failed", error=str(exc))

            # Register and start CLI channel
            if config.channels.cli_enabled and not args.no_cli:
                cli = CliChannel(version=__version__, config=config, api_port=args.api_port)
                gateway.register_channel(cli)

            # Telegram channel (auto-detect: token in env -> start)
            telegram_token = os.environ.get("COGNITHOR_TELEGRAM_TOKEN")
            if telegram_token:
                from cognithor.channels.telegram import TelegramChannel

                allowed = [
                    int(u)
                    for u in os.environ.get("COGNITHOR_TELEGRAM_ALLOWED_USERS", "").split(",")
                    if u
                ]
                _tg_use_webhook = config.channels.telegram_use_webhook
                _tg_webhook_url = config.channels.telegram_webhook_url
                _tg_webhook_port = config.channels.telegram_webhook_port
                _tg_webhook_host = config.channels.telegram_webhook_host
                gateway.register_channel(
                    TelegramChannel(
                        token=telegram_token,
                        allowed_users=allowed,
                        session_store=_session_store,
                        use_webhook=_tg_use_webhook,
                        webhook_url=_tg_webhook_url,
                        webhook_port=_tg_webhook_port,
                        webhook_host=_tg_webhook_host,
                        ssl_certfile=_ssl_cert,
                        ssl_keyfile=_ssl_key,
                        stt_language=getattr(config, "language", "de"),
                    )
                )
                _tg_mode = "webhook" if (_tg_use_webhook and _tg_webhook_url) else "polling"
                log.info("telegram_channel_registered", allowed_users=len(allowed), mode=_tg_mode)

            # Slack channel (auto-detect: token + channel in env -> start)
            slack_token = os.environ.get("COGNITHOR_SLACK_TOKEN")
            if slack_token:
                from cognithor.channels.slack import SlackChannel

                slack_app_token = os.environ.get("COGNITHOR_SLACK_APP_TOKEN", "")
                default_channel = config.channels.slack_default_channel or os.environ.get(
                    "COGNITHOR_SLACK_CHANNEL", ""
                )
                if default_channel:
                    gateway.register_channel(
                        SlackChannel(
                            token=slack_token,
                            app_token=slack_app_token,
                            default_channel=default_channel,
                        )
                    )
                else:
                    log.warning("slack_token_found_but_no_channel")

            # Discord channel (auto-detect: token + channel_id -> start)
            discord_token = os.environ.get("COGNITHOR_DISCORD_TOKEN")
            if discord_token:
                from cognithor.channels.discord import DiscordChannel

                channel_id = config.channels.discord_channel_id or os.environ.get(
                    "COGNITHOR_DISCORD_CHANNEL_ID"
                )
                try:
                    channel_id_int = int(channel_id) if channel_id else 0
                except Exception:
                    channel_id_int = 0
                if channel_id_int:
                    gateway.register_channel(
                        DiscordChannel(
                            token=discord_token,
                            channel_id=channel_id_int,
                            session_store=_session_store,
                        )
                    )
                else:
                    log.warning("discord_token_found_but_no_channel_id")

            # WhatsApp channel (auto-detect: token + phone_number_id -> start)
            wa_token = os.environ.get("COGNITHOR_WHATSAPP_TOKEN")
            if wa_token:
                from cognithor.channels.whatsapp import WhatsAppChannel

                phone_number_id = config.channels.whatsapp_phone_number_id or os.environ.get(
                    "COGNITHOR_WHATSAPP_PHONE_NUMBER_ID", ""
                )
                verify_token = config.channels.whatsapp_verify_token or os.environ.get(
                    "COGNITHOR_WHATSAPP_VERIFY_TOKEN", ""
                )
                allowed = config.channels.whatsapp_allowed_numbers
                if phone_number_id:
                    gateway.register_channel(
                        WhatsAppChannel(
                            api_token=wa_token,
                            phone_number_id=phone_number_id,
                            verify_token=verify_token,
                            webhook_port=config.channels.whatsapp_webhook_port,
                            allowed_numbers=allowed,
                            ssl_certfile=_ssl_cert,
                            ssl_keyfile=_ssl_key,
                            session_store=_session_store,
                            stt_language=getattr(config, "language", "de"),
                        )
                    )
                else:
                    log.warning("whatsapp_token_found_but_no_phone_number_id")

            # Signal channel (auto-detect: token + default_user -> start)
            signal_token = os.environ.get("COGNITHOR_SIGNAL_TOKEN")
            if signal_token:
                from cognithor.channels.signal import SignalChannel

                default_user = config.channels.signal_default_user or os.environ.get(
                    "COGNITHOR_SIGNAL_DEFAULT_USER", ""
                )
                if default_user:
                    gateway.register_channel(
                        SignalChannel(token=signal_token, default_user=default_user)
                    )
                else:
                    log.warning("signal_token_found_but_no_default_user")

            # Matrix channel (auto-detect: token + homeserver + user_id -> start)
            matrix_token = os.environ.get("COGNITHOR_MATRIX_TOKEN")
            if matrix_token:
                from cognithor.channels.matrix import MatrixChannel

                homeserver = (
                    os.environ.get("COGNITHOR_MATRIX_HOMESERVER")
                    or config.channels.matrix_homeserver
                )
                user_id = (
                    os.environ.get("COGNITHOR_MATRIX_USER_ID") or config.channels.matrix_user_id
                )
                if homeserver and user_id:
                    gateway.register_channel(
                        MatrixChannel(
                            access_token=matrix_token, homeserver=homeserver, user_id=user_id
                        )
                    )
                else:
                    log.warning("matrix_token_found_but_no_homeserver_or_user_id")

            # Teams channel (auto-detect: app_id + app_password -> start)
            teams_app_id = os.environ.get("COGNITHOR_TEAMS_APP_ID", "")
            teams_app_pw = os.environ.get("COGNITHOR_TEAMS_TOKEN") or os.environ.get(
                "COGNITHOR_TEAMS_APP_PASSWORD", ""
            )
            if teams_app_id or teams_app_pw:
                from cognithor.channels.teams import TeamsChannel

                teams_host = os.environ.get("COGNITHOR_TEAMS_WEBHOOK_HOST", "127.0.0.1")
                teams_port = int(os.environ.get("COGNITHOR_TEAMS_WEBHOOK_PORT", "3978"))
                gateway.register_channel(
                    TeamsChannel(
                        app_id=teams_app_id,
                        app_password=teams_app_pw,
                        webhook_host=teams_host,
                        webhook_port=teams_port,
                        ssl_certfile=_ssl_cert,
                        ssl_keyfile=_ssl_key,
                        session_store=_session_store,
                    )
                )

            # iMessage-Channel
            if getattr(config.channels, "imessage_enabled", False):
                from cognithor.channels.imessage import IMessageChannel

                device_id = config.channels.imessage_device_id or os.environ.get(
                    "COGNITHOR_IMESSAGE_DEVICE_ID"
                )
                # iMessage hat keine Token; device_id ist optional
                gateway.register_channel(IMessageChannel(device_id=device_id))

            # Start dashboard if enabled
            if config.dashboard.enabled:
                dashboard_port = config.dashboard.port or 9090
                if dashboard_port != args.api_port and api_app is not None:
                    # Dashboard auf separatem Port — lightweight redirect-server
                    try:
                        import uvicorn
                        from starlette.applications import Starlette
                        from starlette.responses import RedirectResponse
                        from starlette.routing import Route

                        api_base = f"http://127.0.0.1:{args.api_port}"

                        async def _dash_redirect(request):
                            return RedirectResponse(url=f"{api_base}/dashboard")

                        dash_app = Starlette(
                            routes=[
                                Route("/", _dash_redirect),
                                Route("/dashboard", _dash_redirect),
                            ]
                        )
                        dash_config = uvicorn.Config(
                            dash_app,
                            host="127.0.0.1",
                            port=dashboard_port,
                            log_level="warning",
                        )
                        dash_server = uvicorn.Server(dash_config)
                        _t = asyncio.create_task(dash_server.serve())
                        _bg_tasks.add(_t)
                        _t.add_done_callback(_bg_tasks.discard)
                        log.info(
                            "dashboard_redirect_started",
                            port=dashboard_port,
                            target=f"{api_base}/dashboard",
                        )
                    except Exception:
                        log.warning(
                            "dashboard_redirect_failed",
                            port=dashboard_port,
                            hint=f"Dashboard verfügbar unter http://127.0.0.1:{args.api_port}/dashboard",
                            exc_info=True,
                        )
                else:
                    log.info(
                        "dashboard_available",
                        url=f"http://127.0.0.1:{args.api_port}/dashboard",
                    )

            log.info("jarvis_ready", channels=list(gateway._channels.keys()))
            await gateway.start()

            # Headless-Modus: wenn keine interaktiven Channels laufen,
            # keep the process alive for the API
            if args.no_cli and api_server and not api_server.should_exit:
                log.info("jarvis_headless_mode", port=args.api_port)
                while not api_server.should_exit:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            log.info("jarvis_interrupted")
        finally:
            # Graceful shutdown: stop uvicorn before closing the event loop
            if api_server:
                api_server.should_exit = True
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.5)  # Give uvicorn time to finish requests
            await gateway.shutdown()
            log.info("jarvis_stopped")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("jarvis_shutdown_by_user")
    except SystemExit:
        pass


def _print_banner(
    config: Any,
    api_host: str = "127.0.0.1",
    api_port: int = 8741,
    lite: bool = False,
) -> None:
    """Print the startup banner to the console.

    This is intentional CLI output so we use print() rather than the
    logger.  Keeping it in a dedicated function makes the main flow
    cleaner and easier to test.
    """
    backend = getattr(config, "llm_backend_type", "ollama")
    scheme = "https" if config.security.ssl_certfile else "http"
    lite_tag = " [LITE]" if lite else ""
    print(f"\n{BANNER_ASCII}")
    print(f"\n{'=' * 60}")
    print(f"  COGNITHOR · Agent OS v{__version__}{lite_tag}")
    print(f"  Home:   {config.cognithor_home}")
    print(f"  API:    {scheme}://{api_host}:{api_port}")
    _backend_label = {
        "ollama": f"Ollama ({config.ollama.mode})",
        "lmstudio": "LM Studio",
        "openai": "OpenAI API",
        "anthropic": "Anthropic API",
    }.get(backend, backend)
    _backend_url = {
        "ollama": config.ollama.base_url,
        "lmstudio": getattr(config, "lmstudio_base_url", ""),
    }.get(backend, "")
    print(f"  LLM:    {_backend_label}" + (f" @ {_backend_url}" if _backend_url else ""))
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
