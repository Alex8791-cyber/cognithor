"""TTLDict — Generisches Dict mit Time-To-Live und LRU-Eviction.

Fuer Channel-Dicts die sonst unbegrenzt wachsen (Session-Maps, Typing-Tasks, etc.).
Sync-only — kein asyncio.Lock noetig, da dict-ops nicht yielden.

Referenz: Stabilitaets-Verbesserung §5 (Memory-Begrenzung)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["TTLDict"]

KT = TypeVar("KT")
VT = TypeVar("VT")


@dataclass(slots=True)
class _Entry[VT]:
    """Interner Eintrag mit Ablauf-Zeitstempel."""

    value: VT
    expires_at: float
    last_access: float = field(default_factory=time.monotonic)


class TTLDict[KT, VT]:
    """Dict mit TTL-Ablauf und LRU-Eviction.

    Args:
        max_size: Maximale Anzahl Eintraege (LRU-Eviction bei Ueberschreitung).
        ttl_seconds: Standard-TTL in Sekunden.
        cleanup_interval: Sekunden zwischen periodischen Sweeps (Default: 60).
    """

    def __init__(
        self,
        *,
        max_size: int = 10_000,
        ttl_seconds: float = 86_400,
        cleanup_interval: float = 60.0,
    ) -> None:
        self._data: OrderedDict[KT, _Entry[VT]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.monotonic()
        self._eviction_count = 0
        self._expired_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, key: KT, value: VT, *, ttl: float | None = None) -> None:
        """Setzt einen Eintrag mit optionalem Custom-TTL."""
        now = time.monotonic()
        effective_ttl = ttl if ttl is not None else self._ttl_seconds
        expires_at = now + effective_ttl

        if key in self._data:
            self._data[key] = _Entry(value=value, expires_at=expires_at, last_access=now)
            self._data.move_to_end(key)
        else:
            self._data[key] = _Entry(value=value, expires_at=expires_at, last_access=now)
            # LRU-Eviction wenn voll
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
                self._eviction_count += 1

        self._maybe_cleanup(now)

    @overload
    def get(self, key: KT) -> VT | None: ...
    @overload
    def get(self, key: KT, default: VT) -> VT: ...
    @overload
    def get(self, key: KT, default: None) -> VT | None: ...
    def get(self, key: KT, default: VT | None = None) -> VT | None:
        """Gibt den Wert zurueck oder default wenn abgelaufen/nicht vorhanden."""
        now = time.monotonic()
        self._maybe_cleanup(now)

        entry = self._data.get(key)
        if entry is None:
            return default

        if now >= entry.expires_at:
            del self._data[key]
            self._expired_count += 1
            return default

        entry.last_access = now
        self._data.move_to_end(key)
        return entry.value

    _MISSING: Any = object()

    @overload
    def pop(self, key: KT) -> VT: ...
    @overload
    def pop(self, key: KT, default: VT) -> VT: ...
    @overload
    def pop(self, key: KT, default: None) -> VT | None: ...
    def pop(self, key: KT, default: Any = _MISSING) -> Any:
        """Entfernt und gibt den Wert zurueck. Wie dict.pop()."""
        entry = self._data.pop(key, None)
        if entry is None:
            if default is TTLDict._MISSING:
                raise KeyError(key)
            return default
        return entry.value

    @overload
    def setdefault(self, key: KT) -> VT | None: ...
    @overload
    def setdefault(self, key: KT, default: VT) -> VT: ...
    @overload
    def setdefault(self, key: KT, default: None) -> VT | None: ...
    def setdefault(self, key: KT, default: VT | None = None) -> VT | None:
        """Gibt den Wert zurueck wenn vorhanden, sonst setzt default und gibt ihn zurueck.

        Verhaelt sich wie dict.setdefault(): fehlender Key mit default=None
        speichert und gibt None zurueck (kein KeyError).
        """
        now = time.monotonic()
        self._maybe_cleanup(now)
        entry = self._data.get(key)
        if entry is not None and now < entry.expires_at:
            entry.last_access = now
            self._data.move_to_end(key)
            return entry.value
        # Key fehlt oder abgelaufen — default setzen
        if entry is not None:
            # Abgelaufen: entfernen
            del self._data[key]
            self._expired_count += 1
        if default is not None:
            self.set(key, default)
        return default

    def clear(self) -> None:
        """Entfernt alle Eintraege."""
        self._data.clear()

    def keys(self) -> list[KT]:
        """Gibt nicht-abgelaufene Keys zurueck."""
        self._purge_expired()
        return list(self._data.keys())

    def values(self) -> list[VT]:
        """Gibt nicht-abgelaufene Values zurueck."""
        self._purge_expired()
        return [e.value for e in self._data.values()]

    def items(self) -> list[tuple[KT, VT]]:
        """Gibt nicht-abgelaufene (key, value)-Paare zurueck."""
        self._purge_expired()
        return [(k, e.value) for k, e in self._data.items()]

    @property
    def stats(self) -> dict[str, int]:
        """Statistiken: size, max_size, eviction_count, expired_count."""
        return {
            "size": len(self._data),
            "max_size": self._max_size,
            "eviction_count": self._eviction_count,
            "expired_count": self._expired_count,
        }

    # ------------------------------------------------------------------
    # Dunder-Methoden (dict-Kompatibilitaet)
    # ------------------------------------------------------------------

    def __setitem__(self, key: KT, value: VT) -> None:
        self.set(key, value)

    def __getitem__(self, key: KT) -> VT:
        now = time.monotonic()
        entry = self._data.get(key)
        if entry is None:
            raise KeyError(key)
        if now >= entry.expires_at:
            del self._data[key]
            self._expired_count += 1
            raise KeyError(key)
        entry.last_access = now
        self._data.move_to_end(key)
        return entry.value

    def __contains__(self, key: object) -> bool:
        entry = self._data.get(key)  # type: ignore[arg-type]
        if entry is None:
            return False
        if time.monotonic() >= entry.expires_at:
            del self._data[key]  # type: ignore[arg-type]
            self._expired_count += 1
            return False
        return True

    def __delitem__(self, key: KT) -> None:
        del self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            self._purge_expired()
            return {k: e.value for k, e in self._data.items()} == other
        if isinstance(other, TTLDict):
            self._purge_expired()
            other._purge_expired()
            return {k: e.value for k, e in self._data.items()} == {
                k: e.value for k, e in other._data.items()
            }
        return NotImplemented

    def __iter__(self) -> Iterator[KT]:
        self._purge_expired()
        return iter(list(self._data.keys()))

    def __repr__(self) -> str:
        return (
            f"TTLDict(size={len(self._data)}, max_size={self._max_size}, ttl={self._ttl_seconds}s)"
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _maybe_cleanup(self, now: float) -> None:
        """Periodischer Sweep abgelaufener Eintraege."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        self._purge_expired()

    def purge_expired(self) -> int:
        """Entfernt alle abgelaufenen Eintraege (public API).

        Returns:
            Anzahl entfernter Eintraege.
        """
        return self._purge_expired()

    def _purge_expired(self) -> int:
        """Entfernt alle abgelaufenen Eintraege.

        Returns:
            Anzahl entfernter Eintraege.
        """
        now = time.monotonic()
        expired_keys = [k for k, e in self._data.items() if now >= e.expires_at]
        for k in expired_keys:
            del self._data[k]
            self._expired_count += 1
        return len(expired_keys)
