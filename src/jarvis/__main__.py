"""
Cognithor · Agent OS -- Entry Point.

Usage: cognithor
       cognithor --config /path/to/config.yaml
       cognithor --version
       python -m jarvis
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

from jarvis import __version__


def parse_args() -> argparse.Namespace:
    """Kommandozeilen-Argumente parsen."""
    parser = argparse.ArgumentParser(
        prog="cognithor",
        description="Cognithor · Agent OS -- Local-first autonomous agent operating system",
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
        help="Pfad zur config.yaml (Default: ~/.jarvis/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log-Level überschreiben",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Nur Verzeichnisstruktur erstellen, nicht starten",
    )
    parser.add_argument(
        "--no-cli",
        action="store_true",
        help="CLI-Channel nicht starten (Headless-Betrieb für Control Center)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8741,
        help="Port für die Control Center API (Default: 8741)",
    )
    return parser.parse_args()


def main() -> None:
    """Haupteintrittspunkt für Jarvis."""
    args = parse_args()

    # 0. .env-Datei laden (Secrets aus ~/.jarvis/.env oder Projekt-Root)
    try:
        from dotenv import load_dotenv

        # Zuerst Projekt-.env, dann User-.env (User überschreibt)
        load_dotenv(Path(".env"), override=False)
        load_dotenv(Path.home() / ".jarvis" / ".env", override=True)
    except ImportError:
        pass  # python-dotenv optional

    # 1. Konfiguration laden
    from jarvis.config import ensure_directory_structure, load_config

    config = load_config(args.config)

    # 2. Verzeichnisstruktur sicherstellen
    created = ensure_directory_structure(config)

    # 3. Logging initialisieren
    from jarvis.utils.logging import setup_logging

    log_level = args.log_level or config.logging.level
    setup_logging(
        level=log_level,
        log_dir=config.logs_dir,
        json_logs=config.logging.json_logs,
        console=config.logging.console,
    )

    from jarvis.utils.logging import get_logger

    log = get_logger("jarvis")

    # 4. Startup-Info
    log.info(
        "jarvis_starting",
        version=__version__,
        home=str(config.jarvis_home),
        log_level=log_level,
    )

    if created:
        for path in created:
            log.info("created_path", path=path)

    if args.init_only:
        log.info("init_complete", paths_created=len(created))
        log.info(
            "init_summary",
            version=__version__,
            home=str(config.jarvis_home),
            config_file=str(config.config_file),
            paths_created=len(created),
        )
        return

    # 5. System-Check -- startup banner (intentional CLI output)
    _print_banner(config, api_port=args.api_port)

    # Phase 0 Checkpoint: Setup OK
    log.info(
        "setup_ok",
        ollama_url=config.ollama.base_url,
        planner_model=config.models.planner.name,
        executor_model=config.models.executor.name,
    )

    # Phase 1: Gateway + CLI starten
    import asyncio

    async def run() -> None:
        """Startet den Gateway und CLI-Channel als asynchrone Hauptschleife."""
        from jarvis.channels.cli import CliChannel
        from jarvis.gateway.gateway import Gateway

        gateway = Gateway(config)
        api_server = None

        try:
            # Alle Subsysteme initialisieren
            await gateway.initialize()

            # Control Center API-Server starten (immer, Port 8741)
            try:
                from fastapi import FastAPI
                from fastapi.middleware.cors import CORSMiddleware
                import uvicorn
                from jarvis.channels.config_routes import create_config_routes
                from jarvis.config_manager import ConfigManager

                api_app = FastAPI(title="Cognithor Control Center API")
                api_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
                config_mgr = ConfigManager(config=config)
                create_config_routes(api_app, config_mgr, gateway=gateway)

                uvi_config = uvicorn.Config(
                    api_app,
                    host="127.0.0.1",
                    port=args.api_port,
                    log_level="warning",
                )
                api_server = uvicorn.Server(uvi_config)
                asyncio.create_task(api_server.serve())
                log.info("control_center_api_started", port=args.api_port)
            except ImportError:
                log.warning("control_center_api_requires_fastapi_uvicorn")
            except Exception as exc:
                log.warning("control_center_api_failed", error=str(exc))

            # CLI-Channel registrieren und starten
            if config.channels.cli_enabled and not args.no_cli:
                cli = CliChannel(version=__version__)
                gateway.register_channel(cli)

            # Telegram-Channel (auto-detect: token in env → start)
            telegram_token = os.environ.get("JARVIS_TELEGRAM_TOKEN")
            if telegram_token:
                from jarvis.channels.telegram import TelegramChannel

                allowed = [int(u) for u in os.environ.get("JARVIS_TELEGRAM_ALLOWED_USERS", "").split(",") if u]
                gateway.register_channel(TelegramChannel(token=telegram_token, allowed_users=allowed))
                log.info("telegram_channel_registered", allowed_users=len(allowed))

            # Slack-Channel (auto-detect: token + channel in env → start)
            slack_token = os.environ.get("JARVIS_SLACK_TOKEN")
            if slack_token:
                from jarvis.channels.slack import SlackChannel

                slack_app_token = os.environ.get("JARVIS_SLACK_APP_TOKEN", "")
                default_channel = config.channels.slack_default_channel or os.environ.get(
                    "JARVIS_SLACK_CHANNEL", ""
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

            # Discord-Channel (auto-detect: token + channel_id → start)
            discord_token = os.environ.get("JARVIS_DISCORD_TOKEN")
            if discord_token:
                from jarvis.channels.discord import DiscordChannel

                channel_id = config.channels.discord_channel_id or os.environ.get("JARVIS_DISCORD_CHANNEL_ID")
                try:
                    channel_id_int = int(channel_id) if channel_id else 0
                except Exception:
                    channel_id_int = 0
                if channel_id_int:
                    gateway.register_channel(
                        DiscordChannel(token=discord_token, channel_id=channel_id_int)
                    )
                else:
                    log.warning("discord_token_found_but_no_channel_id")

            # WhatsApp-Channel (auto-detect: token + phone_number_id → start)
            wa_token = os.environ.get("JARVIS_WHATSAPP_TOKEN")
            if wa_token:
                from jarvis.channels.whatsapp import WhatsAppChannel

                phone_number_id = (
                    config.channels.whatsapp_phone_number_id
                    or os.environ.get("JARVIS_WHATSAPP_PHONE_NUMBER_ID", "")
                )
                verify_token = (
                    config.channels.whatsapp_verify_token
                    or os.environ.get("JARVIS_WHATSAPP_VERIFY_TOKEN", "")
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
                        )
                    )
                else:
                    log.warning("whatsapp_token_found_but_no_phone_number_id")

            # Signal-Channel (auto-detect: token + default_user → start)
            signal_token = os.environ.get("JARVIS_SIGNAL_TOKEN")
            if signal_token:
                from jarvis.channels.signal import SignalChannel

                default_user = config.channels.signal_default_user or os.environ.get(
                    "JARVIS_SIGNAL_DEFAULT_USER", ""
                )
                if default_user:
                    gateway.register_channel(
                        SignalChannel(token=signal_token, default_user=default_user)
                    )
                else:
                    log.warning("signal_token_found_but_no_default_user")

            # Matrix-Channel (auto-detect: token + homeserver + user_id → start)
            matrix_token = os.environ.get("JARVIS_MATRIX_TOKEN")
            if matrix_token:
                from jarvis.channels.matrix import MatrixChannel

                homeserver = (
                    os.environ.get("JARVIS_MATRIX_HOMESERVER")
                    or config.channels.matrix_homeserver
                )
                user_id = (
                    os.environ.get("JARVIS_MATRIX_USER_ID") or config.channels.matrix_user_id
                )
                if homeserver and user_id:
                    gateway.register_channel(
                        MatrixChannel(
                            token=matrix_token, homeserver=homeserver, user_id=user_id
                        )
                    )
                else:
                    log.warning("matrix_token_found_but_no_homeserver_or_user_id")

            # Teams-Channel (auto-detect: token + channel → start)
            teams_token = os.environ.get("JARVIS_TEAMS_TOKEN")
            if teams_token:
                from jarvis.channels.teams import TeamsChannel

                default_channel = config.channels.teams_default_channel or os.environ.get(
                    "JARVIS_TEAMS_DEFAULT_CHANNEL", ""
                )
                if default_channel:
                    gateway.register_channel(
                        TeamsChannel(token=teams_token, default_channel=default_channel)
                    )
                else:
                    log.warning("teams_token_found_but_no_channel")

            # iMessage-Channel
            if getattr(config.channels, "imessage_enabled", False):
                from jarvis.channels.imessage import IMessageChannel

                device_id = config.channels.imessage_device_id or os.environ.get(
                    "JARVIS_IMESSAGE_DEVICE_ID"
                )
                # iMessage hat keine Token; device_id ist optional
                gateway.register_channel(IMessageChannel(device_id=device_id))

            # Start Dashboard falls aktiviert
            if config.dashboard.enabled:
                try:
                    from jarvis.dashboard import Dashboard

                    dashboard = Dashboard(config, gateway)
                    await dashboard.start()
                except Exception:
                    log.warning("dashboard_failed_to_start")

            log.info("jarvis_ready", channels=list(gateway._channels.keys()))
            await gateway.start()

            # Headless-Modus: wenn keine interaktiven Channels laufen,
            # halte den Prozess am Leben für die API
            if args.no_cli and api_server and not api_server.should_exit:
                log.info("jarvis_headless_mode", port=args.api_port)
                while not api_server.should_exit:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            log.info("jarvis_interrupted")
        finally:
            if api_server:
                api_server.should_exit = True
            await gateway.shutdown()
            log.info("jarvis_stopped")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("jarvis_shutdown_by_user")


def _print_banner(config: Any, api_port: int = 8741) -> None:
    """Print the startup banner to the console.

    This is intentional CLI output so we use print() rather than the
    logger.  Keeping it in a dedicated function makes the main flow
    cleaner and easier to test.
    """
    print(f"\n{'=' * 60}")
    print(f"  COGNITHOR · Agent OS v{__version__}")
    print(f"  Home:   {config.jarvis_home}")
    print(f"  API:    http://127.0.0.1:{api_port}")
    print(f"  Ollama: {config.ollama.base_url}")
    print(f"  Planner: {config.models.planner.name}")
    print(f"  Executor: {config.models.executor.name}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
