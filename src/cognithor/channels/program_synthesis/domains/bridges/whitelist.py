"""Sprint-26 bridge whitelist (Owner-Decision D5).

The 12 pairs below are the only cross-domain bridges sanctioned for
Sprint-26. Anything else is rejected by :class:`BridgeRegistry` at
registration time. Lernende Bridge-Discovery is deferred to Sprint-28.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

from cognithor.channels.program_synthesis.domains.bridges.registry import (
    BRIDGE_REGISTRY,
    BridgeOperator,
    BridgeRegistry,
)

# ---------------------------------------------------------------------------
# Canonical whitelist
# ---------------------------------------------------------------------------


SPRINT26_BRIDGE_WHITELIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("json", "datetime"),
        ("json", "number"),
        ("json", "string"),
        ("string", "datetime"),
        ("string", "number"),
        ("string", "json"),
        ("datetime", "sql_literal"),
        ("datetime", "string"),
        ("number", "sql_literal"),
        ("number", "string"),
        ("bytes", "string"),
        ("bytes", "number"),
    }
)


# ---------------------------------------------------------------------------
# Bridge operator implementations
# ---------------------------------------------------------------------------


def _json_to_datetime(value: Any) -> datetime:
    """Parse a JSON-extracted value into a tz-aware datetime."""
    if isinstance(value, str):
        return _parse_iso8601(value)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    msg = f"Cannot bridge json→datetime: {type(value).__name__}"
    raise TypeError(msg)


def _json_to_number(value: Any) -> float:
    if isinstance(value, bool):
        msg = "json→number: refusing bool (use string→number explicitly)"
        raise TypeError(msg)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    msg = f"Cannot bridge json→number: {type(value).__name__}"
    raise TypeError(msg)


def _json_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _string_to_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        msg = f"string→datetime expects str, got {type(value).__name__}"
        raise TypeError(msg)
    return _parse_iso8601(value)


def _string_to_number(value: str) -> float:
    if not isinstance(value, str):
        msg = f"string→number expects str, got {type(value).__name__}"
        raise TypeError(msg)
    return float(value)


def _string_to_json(value: str) -> Any:
    import json as _json

    if not isinstance(value, str):
        msg = f"string→json expects str, got {type(value).__name__}"
        raise TypeError(msg)
    return _json.loads(value)


def _datetime_to_sql_literal(value: datetime) -> str:
    if not isinstance(value, datetime):
        msg = f"datetime→sql_literal expects datetime, got {type(value).__name__}"
        raise TypeError(msg)
    # ANSI/duckdb-friendly TIMESTAMP literal. Escape single quotes.
    iso = value.isoformat(sep=" ").replace("'", "''")
    return f"TIMESTAMP '{iso}'"


def _datetime_to_string(value: datetime) -> str:
    if not isinstance(value, datetime):
        msg = f"datetime→string expects datetime, got {type(value).__name__}"
        raise TypeError(msg)
    return value.isoformat()


def _number_to_sql_literal(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"number→sql_literal expects int/float, got {type(value).__name__}"
        raise TypeError(msg)
    return str(value)


def _number_to_string(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"number→string expects int/float, got {type(value).__name__}"
        raise TypeError(msg)
    return str(value)


def _bytes_to_string(value: bytes) -> str:
    """Bytes → string. Tries UTF-8 first, then base64 fallback."""
    if not isinstance(value, bytes | bytearray):
        msg = f"bytes→string expects bytes, got {type(value).__name__}"
        raise TypeError(msg)
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        return base64.b64encode(bytes(value)).decode("ascii")


def _bytes_to_number(value: bytes) -> int:
    """Bytes → number. Big-endian unsigned integer (fits 1-8 byte payloads)."""
    if not isinstance(value, bytes | bytearray):
        msg = f"bytes→number expects bytes, got {type(value).__name__}"
        raise TypeError(msg)
    raw = bytes(value)
    if len(raw) == 0:
        return 0
    return int.from_bytes(raw, byteorder="big", signed=False)


# ---------------------------------------------------------------------------
# ISO-8601 parser shared by json→datetime and string→datetime
# ---------------------------------------------------------------------------


def _parse_iso8601(value: str) -> datetime:
    """Best-effort ISO-8601 parse, defaults to UTC for naive inputs."""
    cleaned = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def install_default_bridges(registry: BridgeRegistry | None = None) -> None:
    """Install all 12 whitelisted operators into ``registry``.

    Idempotent against a fresh registry; raises ``ValueError`` on the
    second call. The canonical ``BRIDGE_REGISTRY`` instance has the
    bridges installed once at module-import time below.
    """
    target = registry if registry is not None else BRIDGE_REGISTRY

    operators: list[BridgeOperator] = [
        BridgeOperator("json", "datetime", _json_to_datetime, "JSON value → tz-aware datetime"),
        BridgeOperator("json", "number", _json_to_number, "JSON value → float"),
        BridgeOperator("json", "string", _json_to_string, "JSON value → string (repr fallback)"),
        BridgeOperator("string", "datetime", _string_to_datetime, "ISO-8601 string → datetime"),
        BridgeOperator("string", "number", _string_to_number, "string → float"),
        BridgeOperator("string", "json", _string_to_json, "JSON-string → parsed value"),
        BridgeOperator(
            "datetime",
            "sql_literal",
            _datetime_to_sql_literal,
            "datetime → TIMESTAMP '...' literal",
        ),
        BridgeOperator("datetime", "string", _datetime_to_string, "datetime → ISO-8601 string"),
        BridgeOperator("number", "sql_literal", _number_to_sql_literal, "number → SQL literal"),
        BridgeOperator("number", "string", _number_to_string, "number → string"),
        BridgeOperator("bytes", "string", _bytes_to_string, "bytes → utf8 (base64 fallback)"),
        BridgeOperator("bytes", "number", _bytes_to_number, "bytes (BE unsigned) → int"),
    ]
    for op in operators:
        target.register(op)


# Install once on import so the canonical registry is ready for
# downstream synthesis-pipeline callers.
if not BRIDGE_REGISTRY.names():
    install_default_bridges()
