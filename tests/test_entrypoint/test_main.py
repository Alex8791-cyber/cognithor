"""Tests für __main__.py – Entry-Point.

Testet:
  - parse_args(): --version, --config, --log-level, --init-only, Defaults
  - main() mit --init-only: Erstellt Verzeichnisse, startet Gateway NICHT
  - main() Startup-Flow: Config laden → Logging → Gateway (gemockt)
  - Fehler-Szenarien: Ungültiger Config-Pfad
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# parse_args()
# =============================================================================


class TestParseArgs:
    def test_no_args_defaults(self) -> None:
        """Ohne Argumente: alle Defaults."""
        with patch("sys.argv", ["jarvis"]):
            from jarvis.__main__ import parse_args

            args = parse_args()
        assert args.config is None
        assert args.log_level is None
        assert args.init_only is False

    def test_config_arg(self, tmp_path: Path) -> None:
        """--config setzt den Pfad."""
        cfg_path = str(tmp_path / "custom.yaml")
        with patch("sys.argv", ["jarvis", "--config", cfg_path]):
            from jarvis.__main__ import parse_args

            args = parse_args()
        assert args.config == Path(cfg_path)

    def test_log_level_debug(self) -> None:
        """--log-level DEBUG."""
        with patch("sys.argv", ["jarvis", "--log-level", "DEBUG"]):
            from jarvis.__main__ import parse_args

            args = parse_args()
        assert args.log_level == "DEBUG"

    def test_log_level_invalid_rejected(self) -> None:
        """Ungültiger Log-Level wird abgelehnt."""
        with patch("sys.argv", ["jarvis", "--log-level", "INVALID"]):
            with pytest.raises(SystemExit):
                from jarvis.__main__ import parse_args

                parse_args()

    def test_init_only_flag(self) -> None:
        """--init-only Flag."""
        with patch("sys.argv", ["jarvis", "--init-only"]):
            from jarvis.__main__ import parse_args

            args = parse_args()
        assert args.init_only is True

    def test_version_flag_exits(self) -> None:
        """--version zeigt Version und beendet."""
        with patch("sys.argv", ["jarvis", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                from jarvis.__main__ import parse_args

                parse_args()
            assert exc_info.value.code == 0

    def test_combined_args(self, tmp_path: Path) -> None:
        """Mehrere Argumente gleichzeitig."""
        cfg = str(tmp_path / "test.yaml")
        with patch(
            "sys.argv", ["jarvis", "--config", cfg, "--log-level", "WARNING", "--init-only"]
        ):
            from jarvis.__main__ import parse_args

            args = parse_args()
        assert args.config == Path(cfg)
        assert args.log_level == "WARNING"
        assert args.init_only is True


# =============================================================================
# main() mit --init-only
# =============================================================================


class TestMainInitOnly:
    def test_init_only_creates_dirs_and_exits(self, tmp_path: Path) -> None:
        """--init-only erstellt Verzeichnisse und beendet ohne Gateway-Start."""
        jarvis_home = tmp_path / ".jarvis_test"
        config_file = jarvis_home / "config.yaml"

        with patch("sys.argv", ["jarvis", "--init-only", "--config", str(config_file)]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch(
                    "jarvis.config.ensure_directory_structure",
                    return_value=[
                        str(jarvis_home / "logs"),
                        str(jarvis_home / "memory"),
                    ],
                ) as mock_ensure,
                patch("jarvis.utils.logging.setup_logging"),
            ):
                main()

            mock_ensure.assert_called_once_with(mock_config)

    def test_init_only_prints_summary(self, tmp_path: Path) -> None:
        """--init-only gibt Summary aus."""
        jarvis_home = tmp_path / ".jarvis_test"

        with patch("sys.argv", ["jarvis", "--init-only"]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)
            mock_logger = MagicMock()

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch("jarvis.config.ensure_directory_structure", return_value=["path1", "path2"]),
                patch("jarvis.utils.logging.setup_logging"),
                patch("jarvis.utils.logging.get_logger", return_value=mock_logger),
            ):
                main()

        # Verify init_summary or init_complete was logged (order-independent)
        logged_events = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "init_summary" in logged_events or "init_complete" in logged_events


# =============================================================================
# main() Startup-Flow
# =============================================================================


class TestMainStartup:
    def test_startup_prints_system_info(self, tmp_path: Path, capsys) -> None:
        """main() gibt System-Info aus bevor Gateway startet."""
        jarvis_home = tmp_path / ".jarvis_test"

        with patch("sys.argv", ["jarvis"]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch("jarvis.config.ensure_directory_structure", return_value=[]),
                patch("jarvis.utils.logging.setup_logging"),
                patch("asyncio.run"),
            ):
                main()

        captured = capsys.readouterr()
        assert "COGNITHOR" in captured.out
        assert "Agent OS" in captured.out
        assert str(jarvis_home) in captured.out

    def test_startup_calls_asyncio_run(self, tmp_path: Path) -> None:
        """main() ruft asyncio.run() auf wenn nicht --init-only."""
        jarvis_home = tmp_path / ".jarvis_test"

        with patch("sys.argv", ["jarvis"]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch("jarvis.config.ensure_directory_structure", return_value=[]),
                patch("jarvis.utils.logging.setup_logging"),
                patch("asyncio.run") as mock_asyncio_run,
            ):
                main()

            mock_asyncio_run.assert_called_once()

    def test_keyboard_interrupt_handled_gracefully(self, tmp_path: Path) -> None:
        """Ctrl+C wird sauber abgefangen."""
        jarvis_home = tmp_path / ".jarvis_test"

        with patch("sys.argv", ["jarvis"]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)
            mock_logger = MagicMock()

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch("jarvis.config.ensure_directory_structure", return_value=[]),
                patch("jarvis.utils.logging.setup_logging"),
                patch("jarvis.utils.logging.get_logger", return_value=mock_logger),
                patch("asyncio.run", side_effect=KeyboardInterrupt),
            ):
                main()  # Sollte NICHT crashen

        # Verify shutdown event was logged (order-independent of structlog config)
        logged_events = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "jarvis_shutdown_by_user" in logged_events

    def test_log_level_override(self, tmp_path: Path) -> None:
        """--log-level überschreibt den Config-Wert."""
        jarvis_home = tmp_path / ".jarvis_test"

        with patch("sys.argv", ["jarvis", "--log-level", "DEBUG", "--init-only"]):
            from jarvis.__main__ import main
            from jarvis.config import JarvisConfig

            mock_config = JarvisConfig(jarvis_home=jarvis_home)

            with (
                patch("jarvis.config.load_config", return_value=mock_config),
                patch("jarvis.config.ensure_directory_structure", return_value=[]),
                patch("jarvis.utils.logging.setup_logging") as mock_setup_log,
            ):
                main()

            # setup_logging muss mit DEBUG aufgerufen worden sein
            call_kwargs = mock_setup_log.call_args
            assert call_kwargs[1]["level"] == "DEBUG" or call_kwargs[0][0] == "DEBUG"


# =============================================================================
# Version
# =============================================================================


class TestVersion:
    def test_version_exists(self) -> None:
        from jarvis import __version__

        assert __version__
        assert isinstance(__version__, str)

    def test_version_format(self) -> None:
        """Version hat semver-Format."""
        from jarvis import __version__

        parts = __version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_matches_config(self) -> None:
        """Config.version stimmt mit __version__ überein."""
        from jarvis import __version__
        from jarvis.config import JarvisConfig

        cfg = JarvisConfig()
        assert cfg.version == __version__
