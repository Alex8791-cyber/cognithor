"""Jarvis i18n — Language Pack System.

Provides a lightweight, hash-verified internationalization layer.
Language packs are JSON files in the ``locales/`` directory.

Usage::

    from jarvis.i18n import t, set_locale, get_locale

    # Get a translated string (English is default)
    msg = t("error.timeout")

    # With interpolation
    msg = t("error.model_not_installed", model="qwen3:32b")

    # Switch language at runtime
    set_locale("de")

Architecture:
    - One JSON file per language (e.g., ``en.json``, ``de.json``, ``zh.json``)
    - Flat dot-notation keys (e.g., ``"error.timeout"``)
    - SHA-256 integrity hash per pack (optional, for community packs)
    - Thread-safe locale switching
    - Fallback chain: requested locale → English → raw key
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "t",
    "set_locale",
    "get_locale",
    "get_available_locales",
    "reload_locale",
    "verify_pack_integrity",
    "LOCALES_DIR",
]

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).parent / "locales"
_DEFAULT_LOCALE = "en"

# --- Internal state (thread-safe) -------------------------------------------

_lock = threading.Lock()
_current_locale: str = _DEFAULT_LOCALE
_packs: dict[str, dict[str, str]] = {}


# --- Public API --------------------------------------------------------------


def t(key: str, **kwargs: Any) -> str:
    """Translate *key* using the current locale.

    Falls back to English if the key is missing in the active locale,
    then falls back to the raw key if missing everywhere.

    Keyword arguments are interpolated via ``str.format_map()``.

    Examples::

        t("error.timeout")                         # "The operation timed out."
        t("error.rate_limited", service="Ollama")   # "Ollama is rate-limited."
    """
    _ensure_loaded(_current_locale)

    # Try current locale
    pack = _packs.get(_current_locale, {})
    value = pack.get(key)

    # Fallback to English
    if value is None and _current_locale != _DEFAULT_LOCALE:
        _ensure_loaded(_DEFAULT_LOCALE)
        value = _packs.get(_DEFAULT_LOCALE, {}).get(key)

    # Fallback to raw key
    if value is None:
        return key

    if kwargs:
        try:
            return value.format_map(kwargs)
        except (KeyError, ValueError):
            return value
    return value


def set_locale(locale: str) -> None:
    """Switch the active locale (e.g., ``"en"``, ``"de"``, ``"zh"``).

    The locale is applied globally. The language pack is loaded lazily
    on the next ``t()`` call.
    """
    global _current_locale
    with _lock:
        _current_locale = locale
    logger.debug("locale_changed locale=%s", locale)


def get_locale() -> str:
    """Return the active locale code."""
    return _current_locale


def get_available_locales() -> list[str]:
    """Return locale codes for all installed language packs."""
    if not LOCALES_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in LOCALES_DIR.glob("*.json") if p.stem != "_meta"
    )


def reload_locale(locale: str | None = None) -> None:
    """Force-reload a language pack from disk.

    If *locale* is ``None``, reloads the current locale.
    """
    locale = locale or _current_locale
    with _lock:
        _packs.pop(locale, None)
    _ensure_loaded(locale)


def verify_pack_integrity(locale: str) -> bool:
    """Verify SHA-256 hash of a language pack against its ``.sha256`` sidecar.

    Returns ``True`` if the hash matches or no sidecar exists (unsigned pack).
    Returns ``False`` only on mismatch (tampered pack).
    """
    pack_path = LOCALES_DIR / f"{locale}.json"
    hash_path = LOCALES_DIR / f"{locale}.sha256"

    if not pack_path.exists():
        return False
    if not hash_path.exists():
        return True  # No hash file → unsigned, trust by default

    expected = hash_path.read_text(encoding="utf-8").strip().lower()
    actual = _compute_hash(pack_path)
    if expected != actual:
        logger.warning(
            "i18n_integrity_mismatch locale=%s expected=%s actual=%s",
            locale,
            expected[:16],
            actual[:16],
        )
        return False
    return True


def generate_pack_hash(locale: str) -> str:
    """Compute and save SHA-256 hash for a language pack.

    Returns the hex digest. Writes ``<locale>.sha256`` sidecar file.
    """
    pack_path = LOCALES_DIR / f"{locale}.json"
    if not pack_path.exists():
        raise FileNotFoundError(f"Language pack not found: {pack_path}")

    digest = _compute_hash(pack_path)
    hash_path = LOCALES_DIR / f"{locale}.sha256"
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return digest


# --- Internal helpers --------------------------------------------------------


def _ensure_loaded(locale: str) -> None:
    """Load a language pack if not already cached."""
    if locale in _packs:
        return

    pack_path = LOCALES_DIR / f"{locale}.json"
    if not pack_path.exists():
        if locale != _DEFAULT_LOCALE:
            logger.debug("i18n_pack_not_found locale=%s", locale)
        with _lock:
            _packs[locale] = {}
        return

    try:
        raw = pack_path.read_text(encoding="utf-8")
        data = json.loads(raw)

        # Flatten nested dicts: {"error": {"timeout": "..."}} → {"error.timeout": "..."}
        flat = _flatten(data)

        with _lock:
            _packs[locale] = flat

        logger.debug(
            "i18n_pack_loaded locale=%s keys=%d", locale, len(flat)
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("i18n_pack_load_failed locale=%s error=%s", locale, exc)
        with _lock:
            _packs[locale] = {}


def _flatten(
    data: dict[str, Any], prefix: str = ""
) -> dict[str, str]:
    """Flatten nested dict into dot-notation keys."""
    result: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            result.update(_flatten(value, full_key))
        else:
            result[full_key] = str(value)
    return result


def _compute_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
