"""
Cognithor · Structured Logging Setup.

Zwei Renderer:
- Entwicklung: Farbige Konsole (Rich-kompatibel)
- Produktion: JSON-Lines in Log-Dateien

Verwendung in jedem Modul:
    from cognithor.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("event_name", key="value")
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

"""
Fallback logging utilities for Cognithor.

This module attempts to import and configure the `structlog` library for
structured logging. In environments where `structlog` is unavailable
(for example, when third-party dependencies cannot be installed), the
functions in this module fall back to Python's built-in `logging`
module. The public API (`get_logger`, `setup_logging`, `bind_context`,
`clear_context`) remains the same so that callers do not need to
distinguish between structured and basic logging.

When `structlog` is available, logging will behave exactly as
documented in the original implementation. If it is not, logging
messages will still be emitted but without structured context or JSON
rendering. File handlers are still supported via the standard
`logging` library to satisfy tests that check for log file creation.
"""

try:
    # Attempt to import structlog. If this fails, we'll fall back to
    # Python's built-in logging. It's important that this happens at
    # runtime so environments without structlog can still run the code.
    structlog = import_module("structlog")
except ModuleNotFoundError:
    structlog = None  # type: ignore[assignment]


# ============================================================================
# Secret-Redaction (PASS-4 XC-3)
# ============================================================================

# Substrings (case-insensitive) that mark a key as sensitive. Anything that
# matches gets the value replaced with ``"***REDACTED***"`` before the log
# line ever leaves the process. Keys not in this set are kept verbatim.
_REDACT_KEY_SUBSTRINGS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "private_key",
    "privatekey",
    "bearer",
    "authorization",
    "cookie",
    "credential",
    "auth_header",
    "client_secret",
    "refresh_token",
    "access_token",
    "session_token",
)

# Value-level patterns that get redacted regardless of key name. Caught
# here so that an accidental ``log.info("got header", value="Bearer X")``
# also goes out scrubbed. Compiled lazily because ``re`` import cost is
# non-trivial on cold start of small CLIs.
import re as _re

_VALUE_REDACT_PATTERNS: tuple[_re.Pattern[str], ...] = (
    _re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]{8,}=*"),
    _re.compile(r"(?i)(oauth:)[A-Za-z0-9\-._~+/]{8,}"),
    _re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    _re.compile(r"\bghp_[A-Za-z0-9]{16,}\b"),
    _re.compile(r"\bgho_[A-Za-z0-9]{16,}\b"),
    _re.compile(r"\bsk-(?:ant|proj|live|test)?-?[A-Za-z0-9_\-]{20,}\b"),
    _re.compile(r"\bxoxb-[A-Za-z0-9\-]{10,}\b"),
    _re.compile(r"\bxoxp-[A-Za-z0-9\-]{10,}\b"),
)

_REDACTED = "***REDACTED***"


def _key_is_sensitive(key: str) -> bool:
    lo = key.lower()
    return any(needle in lo for needle in _REDACT_KEY_SUBSTRINGS)


def _redact_value(value: Any) -> Any:
    """Apply value-level regex scrubbers when *value* is a string."""
    if not isinstance(value, str):
        return value
    out = value
    for pat in _VALUE_REDACT_PATTERNS:
        out = pat.sub(lambda m: m.group(1) + _REDACTED if m.groups() else _REDACTED, out)
    return out


def _scrub_event_dict(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive keys/values in a structlog event dict.

    Mutates and returns the dict so downstream processors see the
    scrubbed payload. Nested dicts/lists are walked; scalars are
    matched against ``_VALUE_REDACT_PATTERNS``. Top-level ``event``
    field gets value-pattern scrubbing too.
    """
    for k, v in list(event_dict.items()):
        if _key_is_sensitive(k):
            event_dict[k] = _REDACTED
            continue
        if isinstance(v, dict):
            event_dict[k] = _scrub_event_dict(v)
        elif isinstance(v, list | tuple):
            scrubbed: list[Any] = []
            for item in v:
                if isinstance(item, dict):
                    scrubbed.append(_scrub_event_dict(dict(item)))
                else:
                    scrubbed.append(_redact_value(item))
            event_dict[k] = type(v)(scrubbed) if isinstance(v, tuple) else scrubbed
        else:
            event_dict[k] = _redact_value(v)
    return event_dict


def _structlog_redact_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor — scrubs sensitive keys/values from every log."""
    return _scrub_event_dict(event_dict)


# ============================================================================
# Lightweight structlog-compatible Wrapper
# ============================================================================


class _StructlogCompatLogger:
    """Akzeptiert structlog-Style Calls (event, **kwargs) ohne structlog.

    Der gesamte Jarvis-Codebase nutzt ``log.info("event", key=val)`` --
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
        msg = _redact_value(msg)
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {' '.join(repr(a) for a in args)}"
        if kwargs:
            scrubbed = _scrub_event_dict(dict(kwargs))
            extras = " ".join(f"{k}={v!r}" for k, v in scrubbed.items())
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
        msg = _redact_value(msg)
        if kwargs:
            scrubbed = _scrub_event_dict(dict(kwargs))
            extras = " ".join(f"{k}={v!r}" for k, v in scrubbed.items())
            msg = f"{msg} {extras}"
        self._logger.exception(msg)

    def bind(self, **kwargs: Any) -> _StructlogCompatLogger:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


if TYPE_CHECKING:
    from pathlib import Path


def get_logger(name: str | None = None) -> Any:
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
    return structlog.get_logger(name)


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
        log_dir: Verzeichnis fuer JSONL-Log-Dateien. None = keine Datei-Logs.
        json_logs: True = JSON-Output auch auf Konsole (fuer Produktion).
        console: True = Log-Ausgabe auf stderr.
    """
    # Determine the desired log level from the provided string. Fall back to
    # logging.INFO if unknown. We always import the standard logging module
    # here because it may not have been imported at module load time.
    import logging

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Build a list of handlers for Python's logging module. Even when
    # structlog is present we need handlers so that Python's logging
    # messages (from third-party libraries) are emitted.
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
        # Wir verwenden RotatingFileHandler mit 5 MB Groesse und 3 Backups
        try:
            from logging.handlers import RotatingFileHandler
        except Exception:
            # Fallback auf normalen FileHandler, wenn Handler nicht verfuegbar
            file_handler = logging.FileHandler(
                log_dir / "cognithor.jsonl",
                encoding="utf-8",
            )
        else:
            file_handler = RotatingFileHandler(
                log_dir / "cognithor.jsonl",
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

    # Silence noisy third-party loggers. If structlog is unavailable,
    # nothing else will touch these loggers so this still applies.
    for noisy in ("httpx", "httpcore", "asyncio", "watchdog", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # If structlog is not available, no further configuration is possible.
    if structlog is None:
        return

    # Shared processors -- werden in jeder Log-Nachricht durchlaufen.
    # ``_structlog_redact_processor`` scrubs sensitive keys/values from
    # every event_dict before any renderer or formatter sees it
    # (PASS-4 XC-3 — defence-in-depth, even if a caller logs a token
    # by accident).
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _structlog_redact_processor,
    ]

    # Choose renderer based on json_logs flag. For JSON logs we omit
    # colours and ensure the output uses UTF-8 characters. Otherwise
    # use structlog.dev.ConsoleRenderer for colourised console output.
    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer(
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

    # format_exc_info conflicts with ConsoleRenderer's pretty exceptions.
    # Only include it when using JSON output.
    exc_processors: list[Any] = [structlog.processors.format_exc_info] if json_logs else []

    structlog.configure(
        processors=[
            *shared_processors,
            *exc_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter fuer alle Handler setzen
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
