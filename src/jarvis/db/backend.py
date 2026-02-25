"""Database Backend Protocol."""
from __future__ import annotations
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class DatabaseBackend(Protocol):
    """Abstraktes Interface fuer Datenbank-Backends."""

    async def execute(self, query: str, params: Sequence[Any] = ()) -> Any:
        """Fuehrt ein SQL-Statement aus."""
        ...

    async def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        """Fuehrt ein SQL-Statement mit mehreren Parametersaetzen aus."""
        ...

    async def executescript(self, script: str) -> None:
        """Fuehrt ein SQL-Skript aus (mehrere Statements)."""
        ...

    async def fetchone(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        """Fuehrt Query aus und gibt eine Zeile als Dict zurueck."""
        ...

    async def fetchall(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Fuehrt Query aus und gibt alle Zeilen als Dicts zurueck."""
        ...

    async def commit(self) -> None:
        """Committed die aktuelle Transaktion."""
        ...

    async def close(self) -> None:
        """Schliesst die Datenbankverbindung."""
        ...

    @property
    def placeholder(self) -> str:
        """SQL-Platzhalter: '?' fuer SQLite, '%s' fuer PostgreSQL."""
        ...

    @property
    def backend_type(self) -> str:
        """Name des Backends: 'sqlite' oder 'postgresql'."""
        ...
