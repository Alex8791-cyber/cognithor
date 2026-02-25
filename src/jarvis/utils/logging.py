"""
Jarvis · Structured Logging Setup.

Zwei Renderer:
- Entwicklung: Farbige Konsole (Rich-kompatibel)
- Produktion: JSON-Lines in Log-Dateien

Verwendung in jedem Modul:
    from jarvis.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("event_name", key="value")
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

"""
Fallback logging utilities for Jarvis.

This module attempts to import and configure the `structlog` library for
structured logging. In environments where `structlog` is unavailable
(for example, when third‑party dependencies cannot be installed), the
functions in this module fall back to Python's built‑in `logging`
module. The public API (`get_logger`, `setup_logging`, `bind_context`,
`clear_context`) remains the same so that callers do not need to
distinguish between structured and basic logging.

When `structlog` is available, logging will behave exactly as
documented in the original implementation. If it is not, logging
messages will still be emitted but without structured context or JSON
rendering. File handlers are still supported via the standard
`logging` library to satisfy tests that check for log file creation.
"""

from importlib import import_module

try:
    # Attempt to import structlog. If this fails, we'll fall back to
    # Python's built‑in logging. It's important that this happens at
    # runtime so environments without structlog can still run the code.
    structlog = import_module("structlog")  # type: ignore[assignment]
except ModuleNotFoundError:
    structlog = None  # type: ignore[assignment]


# ============================================================================
# Lightweight structlog-compatible Wrapper
# ============================================================================


class _StructlogCompatLogger:
    """Akzeptiert structlog-Style Calls (event, **kwargs) ohne structlog.

    Der gesamte Jarvis-Codebase nutzt ``log.info("event", key=val)`` —
    der Standard-Logger wirft dabei TypeError. Dieser Wrapper formatiert
    die kwargs als ``key=val``-Paare im Log-Message.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, method: str, event: Any, *args: Any, **kwargs: Any) -> None:
        try:
            msg = str(event)
        except Exception:
            msg = repr(event)
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {' '.join(repr(a) for a in args)}"
        if kwargs:
            extras = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            msg = f"{msg} {extras}"
        getattr(self._logger, method)(msg)

    def info(self, event: Any, *args: Any, **kwargs: Any) -> None:
        self._log("info", event, *args, **kwargs)

    def warning(self, event: Any, *args: Any, **kwargs: Any) -> None:
        self._log("warning", event, *args, **kwargs)

    def error(self, event: Any, *args: Any, **kwargs: Any) -> None:
        self._log("error", event, *args, **kwargs)

    def debug(self, event: Any, *args: Any, **kwargs: Any) -> None:
        self._log("debug", event, *args, **kwargs)

    def exception(self, event: Any, *args: Any, **kwargs: Any) -> None:
        try:
            msg = str(event)
        except Exception:
            msg = repr(event)
        if kwargs:
            extras = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            msg = f"{msg} {extras}"
        self._logger.exception(msg)

    def bind(self, **kwargs: Any) -> "_StructlogCompatLogger":
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

if TYPE_CHECKING:
    from pathlib import Path


def get_logger(name: str | None = None):
    """
    Return a configured logger.

    If structlog is available, this returns a BoundLogger from the
    structlog stdlib wrapper. Otherwise it falls back to a standard
    `logging.Logger` instance. The return type annotation uses
    `structlog.stdlib.BoundLogger` for callers' type checking, but at
    runtime it may be a plain logger when structlog is missing.
    """
    if structlog is None:
        import logging

        return _StructlogCompatLogger(logging.getLogger(name))
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    json_logs: bool = False,
    console: bool = True,
) -> None:
    """Initialisiert das Logging-System. Muss einmal beim Start aufgerufen werden.

    Args:
        level: Log-Level als String (DEBUG, INFO, WARNING, ERROR).
        log_dir: Verzeichnis für JSONL-Log-Dateien. None = keine Datei-Logs.
        json_logs: True = JSON-Output auch auf Konsole (für Produktion).
        console: True = Log-Ausgabe auf stderr.
    """
    # Determine the desired log level from the provided string. Fall back to
    # logging.INFO if unknown. We always import the standard logging module
    # here because it may not have been imported at module load time.
    import logging

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Build a list of handlers for Python's logging module. Even when
    # structlog is present we need handlers so that Python's logging
    # messages (from third‑party libraries) are emitted.
    handler_list: list[logging.Handler] = []

    # Console handler: always attach if requested. We write to
    # stderr so that tests don't need to capture stdout.
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        handler_list.append(console_handler)

    # File handler: create log directory if necessary and always log at
    # DEBUG level into a file named jarvis.jsonl. Use a rotating handler
    # to prevent unbounded log growth. Even wenn wir kein JSON ausgeben,
    # wird die Datei erstellt. BackupCount begrenzt alte Dateien.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Wir verwenden RotatingFileHandler mit 5 MB Größe und 3 Backups
        try:
            from logging.handlers import RotatingFileHandler  # type: ignore
        except Exception:
            # Fallback auf normalen FileHandler, wenn Handler nicht verfügbar
            file_handler = logging.FileHandler(
                log_dir / "jarvis.jsonl",
                encoding="utf-8",
            )
        else:
            file_handler = RotatingFileHandler(
                log_dir / "jarvis.jsonl",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        # Always capture all logs in the file
        file_handler.setLevel(logging.DEBUG)
        handler_list.append(file_handler)

    # Configure the root logger with our handlers. The format is kept
    # simple: structlog will wrap this later when available. We force
    # reconfiguration so repeated setup calls overwrite previous state.
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handler_list,
        force=True,
    )

    # Silence noisy third‑party loggers. If structlog is unavailable,
    # nothing else will touch these loggers so this still applies.
    for noisy in ("httpx", "httpcore", "asyncio", "watchdog", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # If structlog is not available, no further configuration is possible.
    if structlog is None:
        return

    # Shared processors – werden in jeder Log-Nachricht durchlaufen
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Choose renderer based on json_logs flag. For JSON logs we omit
    # colours and ensure the output uses UTF‑8 characters. Otherwise
    # use structlog.dev.ConsoleRenderer for colourised console output.
    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer(
            ensure_ascii=False,
        )
    else:
        # Parameter was renamed between structlog versions:
        # <=25.4: pad_event, >=25.5: pad_event_to (pad_event deprecated)
        import inspect
        _cr_params = inspect.signature(structlog.dev.ConsoleRenderer).parameters
        _pad_kwarg = "pad_event_to" if "pad_event_to" in _cr_params else "pad_event"
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            **{_pad_kwarg: 40},
        )

    structlog.configure(
        processors=[
            *shared_processors,
            # Format-Prüfung: Verhindert fehlerhafte Log-Calls
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter für alle Handler setzen
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to subsequent log messages.

    When structlog is available, context variables are bound via
    structlog.contextvars.bind_contextvars. Otherwise, this function
    does nothing because the standard logging module has no notion of
    context variables.
    """
    if structlog is None:
        return
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all bound context variables.

    When structlog is available, this clears the contextvars store.
    Otherwise, it does nothing. This behaviour ensures callers can
    always call clear_context() without checking for structlog.
    """
    if structlog is None:
        return
    structlog.contextvars.clear_contextvars()
