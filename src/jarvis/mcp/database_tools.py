"""Datenbank-Tools fuer Jarvis -- SQL-Zugriff als MCP-Tools.

Tools:
  - db_query: SELECT-Abfragen ausfuehren (Read-Only)
  - db_schema: Datenbank-Schema anzeigen
  - db_execute: Schreibende SQL-Abfragen (INSERT/UPDATE/DELETE/CREATE)
  - db_connect: Datenbankverbindung testen und Infos anzeigen

Factory: register_database_tools(mcp_client, config) -> DatabaseTools

Bibel-Referenz: $5.3 (MCP-Tools)
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*$")

_DEFAULT_ROW_LIMIT = 500
_MAX_ROW_LIMIT = 5000
_CONN_TIMEOUT = 10
_QUERY_TIMEOUT_S = 30
_MAX_CELL_WIDTH = 200
_MAX_DATA_POINTS = 10_000

# Patterns that indicate obvious SQL injection when no params are provided
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r";\s*DROP\s", re.IGNORECASE),
    re.compile(r";\s*DELETE\s", re.IGNORECASE),
    re.compile(r";\s*UPDATE\s", re.IGNORECASE),
    re.compile(r";\s*INSERT\s", re.IGNORECASE),
    re.compile(r";\s*ALTER\s", re.IGNORECASE),
    re.compile(r";\s*CREATE\s", re.IGNORECASE),
    re.compile(r"UNION\s+SELECT", re.IGNORECASE),
    re.compile(r";\s*EXEC\s", re.IGNORECASE),
    re.compile(r"--\s*$", re.MULTILINE),
]

__all__ = [
    "DatabaseError",
    "DatabaseTools",
    "register_database_tools",
]


class DatabaseError(Exception):
    """Fehler bei Datenbank-Operationen."""


def _truncate(value: str, max_len: int = _MAX_CELL_WIDTH) -> str:
    """Truncate a cell value to *max_len* characters."""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _format_table(columns: list[str], rows: list[tuple[Any, ...]], row_count: int) -> str:
    """Format query results as an ASCII table with headers."""
    if not columns:
        return f"({row_count} rows affected, no result set)"

    # Convert every cell to string and truncate
    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append([_truncate(str(v) if v is not None else "NULL") for v in row])

    # Compute column widths
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build header
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    separator = "-+-".join("-" * w for w in widths)

    lines = [header, separator]
    for row in str_rows:
        line = " | ".join(
            (row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(columns))
        )
        lines.append(line)

    lines.append(f"\n({row_count} row{'s' if row_count != 1 else ''})")
    return "\n".join(lines)


def _check_injection(sql: str, params: list[Any] | None) -> None:
    """Raise if the raw SQL contains obvious injection patterns when no params."""
    if params:
        return  # parameterised queries are safe
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sql):
            raise DatabaseError(
                f"Potential SQL injection detected (pattern: {pattern.pattern}). "
                "Use parameterized queries instead."
            )


def _is_pg_connection_string(database: str) -> bool:
    """Heuristic: is *database* a PostgreSQL connection string?"""
    lower = database.lower().strip()
    return lower.startswith("postgres://") or lower.startswith("postgresql://")


class DatabaseTools:
    """SQL-Datenbank-Zugriff mit Sandbox-Validierung.

    Unterstuetzt SQLite (stdlib) und PostgreSQL (asyncpg/psycopg2, optional).
    """

    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._workspace: Path = config.workspace_dir
        self._jarvis_home: Path = config.jarvis_home
        self._allowed_roots: list[Path] = [
            Path(p).expanduser().resolve() for p in config.security.allowed_paths
        ]
        # Also allow workspace and jarvis_home explicitly
        for extra in (self._workspace, self._jarvis_home):
            resolved = extra.expanduser().resolve()
            if resolved not in self._allowed_roots:
                self._allowed_roots.append(resolved)

    # ------------------------------------------------------------------ #
    # Path validation (SQLite files)
    # ------------------------------------------------------------------ #

    def _validate_sqlite_path(self, path_str: str) -> Path:
        """Validate that a SQLite path is inside allowed directories."""
        try:
            path = Path(path_str).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise DatabaseError(f"Ungueltiger Pfad: {path_str}") from exc

        for root in self._allowed_roots:
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue

        raise DatabaseError(
            f"Zugriff verweigert: {path_str} liegt ausserhalb erlaubter Verzeichnisse "
            f"({', '.join(str(r) for r in self._allowed_roots)})"
        )

    # ------------------------------------------------------------------ #
    # SQLite helpers (sync, run via executor)
    # ------------------------------------------------------------------ #

    def _sqlite_connect(self, db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
        """Open a SQLite connection with timeout and optional read-only pragma."""
        if not db_path.exists():
            raise DatabaseError(f"Datenbank nicht gefunden: {db_path}")
        conn = encrypted_connect(str(db_path), timeout=_CONN_TIMEOUT)
        if read_only:
            conn.execute("PRAGMA query_only = ON")
        return conn

    def _sqlite_query(
        self,
        db_path: Path,
        sql: str,
        params: list[Any] | None,
        limit: int,
    ) -> str:
        """Execute a read-only query on SQLite (sync)."""
        _check_injection(sql, params)
        conn = self._sqlite_connect(db_path, read_only=True)
        try:
            # Progress handler for timeout enforcement
            start = time.monotonic()

            def _progress() -> int:
                if time.monotonic() - start > _QUERY_TIMEOUT_S:
                    return 1  # abort
                return 0

            conn.set_progress_handler(_progress, 1000)

            cursor = conn.execute(sql, params or [])
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit)
            # Check if there are more rows
            extra = cursor.fetchone()
            if extra is not None:
                row_count_note = f" (showing first {limit}, more rows available)"
            else:
                row_count_note = ""

            table = _format_table(columns, rows, len(rows))
            if row_count_note:
                table += row_count_note
            return table
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise DatabaseError(f"Query-Timeout nach {_QUERY_TIMEOUT_S}s") from exc
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            conn.close()

    def _sqlite_schema(self, db_path: Path, table: str | None) -> str:
        """Return schema info for a SQLite database (sync)."""
        conn = self._sqlite_connect(db_path, read_only=True)
        try:
            if table:
                if not _SAFE_IDENTIFIER_RE.match(table):
                    raise DatabaseError(f"Ungueltiger Tabellenname: {table!r}")
                safe_table = f"[{table}]"
                # Detailed column info for a specific table
                cursor = conn.execute(f"PRAGMA table_info({safe_table})")
                cols = cursor.fetchall()
                if not cols:
                    raise DatabaseError(f"Tabelle nicht gefunden: {table}")

                lines = [f"Schema fuer Tabelle: {table}", ""]
                lines.append(
                    f"{'Name':<30} {'Type':<15} {'Nullable':<10} {'PK':<5} {'Default':<20}"
                )
                lines.append("-" * 80)
                for col in cols:
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    cid, name, ctype, notnull, dflt, pk = col
                    nullable = "NO" if notnull else "YES"
                    pk_str = "YES" if pk else ""
                    dflt_str = str(dflt) if dflt is not None else ""
                    lines.append(
                        f"{name:<30} {ctype:<15} {nullable:<10} {pk_str:<5} {dflt_str:<20}"
                    )

                # Also show indexes
                idx_cursor = conn.execute(f"PRAGMA index_list({safe_table})")
                indexes = idx_cursor.fetchall()
                if indexes:
                    lines.append("")
                    lines.append("Indexes:")
                    for idx in indexes:
                        idx_name = idx[1]
                        unique = "UNIQUE" if idx[2] else ""
                        lines.append(f"  - {idx_name} {unique}")

                return "\n".join(lines)
            else:
                # List all tables
                cursor = conn.execute(
                    "SELECT name, type FROM sqlite_master "
                    "WHERE type IN ('table', 'view') ORDER BY type, name"
                )
                objects = cursor.fetchall()
                if not objects:
                    return "Keine Tabellen oder Views gefunden."

                lines = ["Datenbank-Schema:", ""]
                current_type = ""
                for name, obj_type in objects:
                    if obj_type != current_type:
                        if current_type:
                            lines.append("")
                        lines.append(f"## {obj_type.upper()}S:")
                        current_type = obj_type
                    # Get row count
                    try:
                        _cnt_row = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()
                        cnt = _cnt_row[0] if _cnt_row else "?"
                    except Exception:
                        cnt = "?"
                    lines.append(f"  - {name} ({cnt} rows)")

                return "\n".join(lines)
        finally:
            conn.close()

    def _sqlite_execute(
        self,
        db_path: Path,
        sql: str,
        params: list[Any] | None,
    ) -> str:
        """Execute a write query on SQLite (sync)."""
        _check_injection(sql, params)

        # Block DROP statements at this level too (defence in depth)
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("DROP "):
            raise DatabaseError("DROP-Anweisungen sind blockiert. Bitte manuell ausfuehren.")

        conn = self._sqlite_connect(db_path, read_only=False)
        try:
            start = time.monotonic()

            def _progress() -> int:
                if time.monotonic() - start > _QUERY_TIMEOUT_S:
                    return 1
                return 0

            conn.set_progress_handler(_progress, 1000)

            cursor = conn.execute(sql, params or [])
            conn.commit()
            rows_affected = cursor.rowcount
            return f"Erfolgreich. {rows_affected} Zeile(n) betroffen."
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise DatabaseError(f"Query-Timeout nach {_QUERY_TIMEOUT_S}s") from exc
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            conn.close()

    def _sqlite_connect_info(self, db_path: Path) -> str:
        """Return connection info for a SQLite database (sync)."""
        if not db_path.exists():
            raise DatabaseError(f"Datenbank nicht gefunden: {db_path}")

        conn = self._sqlite_connect(db_path, read_only=True)
        try:
            row = conn.execute("SELECT sqlite_version()").fetchone()
            version = row[0] if row else "unknown"
            row = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()
            table_count = row[0] if row else "unknown"
            row = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'").fetchone()
            view_count = row[0] if row else "unknown"
        finally:
            conn.close()

        size_bytes = db_path.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1_048_576:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / 1_048_576:.1f} MB"

        lines = [
            "Datenbankverbindung erfolgreich.",
            "",
            "  Typ:       SQLite",
            f"  Version:   {version}",
            f"  Datei:     {db_path}",
            f"  Groesse:   {size_str}",
            f"  Tabellen:  {table_count}",
            f"  Views:     {view_count}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # PostgreSQL helpers (async)
    # ------------------------------------------------------------------ #

    async def _pg_query(
        self,
        connstr: str,
        sql: str,
        params: list[Any] | None,
        limit: int,
    ) -> str:
        """Execute a read-only query on PostgreSQL via asyncpg."""
        _check_injection(sql, params)
        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError:
            return self._pg_psycopg2_query(connstr, sql, params, limit)

        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(connstr),
                timeout=_CONN_TIMEOUT,
            )
        except TimeoutError as exc:
            raise DatabaseError(f"Verbindungs-Timeout nach {_CONN_TIMEOUT}s") from exc
        except Exception as exc:
            raise DatabaseError(f"Verbindungsfehler: {exc}") from exc

        try:
            await conn.execute(f"SET statement_timeout = {_QUERY_TIMEOUT_S * 1000}")
            await conn.execute("SET default_transaction_read_only = ON")

            if params:
                # asyncpg uses $1, $2 notation; convert ? placeholders
                pg_sql = sql
                for i in range(len(params)):
                    pg_sql = pg_sql.replace("?", f"${i + 1}", 1)
                records = await conn.fetch(pg_sql, *params)
            else:
                records = await conn.fetch(sql)

            if not records:
                return "(0 rows)"

            columns = list(records[0].keys())
            limited = records[:limit]
            rows = [tuple(r.values()) for r in limited]
            table = _format_table(columns, rows, len(rows))

            if len(records) > limit:
                table += f" (showing first {limit}, more rows available)"
            return table
        except Exception as exc:
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            await conn.close()

    def _pg_psycopg2_query(
        self,
        connstr: str,
        sql: str,
        params: list[Any] | None,
        limit: int,
    ) -> str:
        """Fallback: PostgreSQL via psycopg2 (sync)."""
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DatabaseError(
                "PostgreSQL nicht verfuegbar. Installiere: pip install asyncpg "
                "oder pip install psycopg2-binary"
            ) from exc

        try:
            conn = psycopg2.connect(connstr, connect_timeout=_CONN_TIMEOUT)
        except Exception as exc:
            raise DatabaseError(f"Verbindungsfehler: {exc}") from exc

        try:
            conn.set_session(readonly=True)
            cursor = conn.cursor()
            cursor.execute(f"SET statement_timeout = '{_QUERY_TIMEOUT_S * 1000}'")
            cursor.execute(sql, params or None)

            if cursor.description is None:
                return "(No result set)"

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(limit)
            table = _format_table(columns, rows, len(rows))
            return table
        except Exception as exc:
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            conn.close()

    async def _pg_schema(self, connstr: str, table: str | None) -> str:
        """Return schema info for a PostgreSQL database."""
        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError:
            try:
                import psycopg2  # type: ignore[import-untyped]  # noqa: F401

                return self._pg_psycopg2_schema(connstr, table)
            except ImportError as exc:
                raise DatabaseError(
                    "PostgreSQL nicht verfuegbar. Installiere: pip install asyncpg"
                ) from exc

        try:
            conn = await asyncio.wait_for(asyncpg.connect(connstr), timeout=_CONN_TIMEOUT)
        except TimeoutError as exc:
            raise DatabaseError(f"Verbindungs-Timeout nach {_CONN_TIMEOUT}s") from exc
        except Exception as exc:
            raise DatabaseError(f"Verbindungsfehler: {exc}") from exc

        try:
            if table:
                records = await conn.fetch(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = $1 ORDER BY ordinal_position",
                    table,
                )
                if not records:
                    raise DatabaseError(f"Tabelle nicht gefunden: {table}")

                # Get primary key columns
                pk_records = await conn.fetch(
                    "SELECT a.attname FROM pg_index i "
                    "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                    "WHERE i.indrelid = $1::regclass AND i.indisprimary",
                    table,
                )
                pk_cols = {r["attname"] for r in pk_records}

                lines = [f"Schema fuer Tabelle: {table}", ""]
                lines.append(
                    f"{'Name':<30} {'Type':<20} {'Nullable':<10} {'PK':<5} {'Default':<30}"
                )
                lines.append("-" * 95)
                for rec in records:
                    name = rec["column_name"]
                    pk_str = "YES" if name in pk_cols else ""
                    dflt = str(rec["column_default"]) if rec["column_default"] else ""
                    lines.append(
                        f"{name:<30} {rec['data_type']:<20} {rec['is_nullable']:<10} "
                        f"{pk_str:<5} {dflt:<30}"
                    )
                return "\n".join(lines)
            else:
                records = await conn.fetch(
                    "SELECT table_name, table_type FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_type, table_name"
                )
                if not records:
                    return "Keine Tabellen oder Views gefunden."

                lines = ["Datenbank-Schema:", ""]
                for rec in records:
                    lines.append(f"  - {rec['table_name']} ({rec['table_type']})")
                return "\n".join(lines)
        finally:
            await conn.close()

    def _pg_psycopg2_schema(self, connstr: str, table: str | None) -> str:
        """Fallback schema info via psycopg2."""
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(connstr, connect_timeout=_CONN_TIMEOUT)
        try:
            cursor = conn.cursor()
            if table:
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = %s ORDER BY ordinal_position",
                    (table,),
                )
                rows = cursor.fetchall()
                if not rows:
                    raise DatabaseError(f"Tabelle nicht gefunden: {table}")

                lines = [f"Schema fuer Tabelle: {table}", ""]
                lines.append(f"{'Name':<30} {'Type':<20} {'Nullable':<10} {'Default':<30}")
                lines.append("-" * 90)
                for name, dtype, nullable, default in rows:
                    dflt = str(default) if default else ""
                    lines.append(f"{name:<30} {dtype:<20} {nullable:<10} {dflt:<30}")
                return "\n".join(lines)
            else:
                cursor.execute(
                    "SELECT table_name, table_type FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_type, table_name"
                )
                rows = cursor.fetchall()
                if not rows:
                    return "Keine Tabellen oder Views gefunden."
                lines = ["Datenbank-Schema:", ""]
                for name, ttype in rows:
                    lines.append(f"  - {name} ({ttype})")
                return "\n".join(lines)
        finally:
            conn.close()

    async def _pg_execute(
        self,
        connstr: str,
        sql: str,
        params: list[Any] | None,
    ) -> str:
        """Execute a write query on PostgreSQL."""
        _check_injection(sql, params)

        sql_upper = sql.strip().upper()
        if sql_upper.startswith("DROP "):
            raise DatabaseError("DROP-Anweisungen sind blockiert. Bitte manuell ausfuehren.")

        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError:
            return self._pg_psycopg2_execute(connstr, sql, params)

        try:
            conn = await asyncio.wait_for(asyncpg.connect(connstr), timeout=_CONN_TIMEOUT)
        except TimeoutError as exc:
            raise DatabaseError(f"Verbindungs-Timeout nach {_CONN_TIMEOUT}s") from exc
        except Exception as exc:
            raise DatabaseError(f"Verbindungsfehler: {exc}") from exc

        try:
            await conn.execute(f"SET statement_timeout = {_QUERY_TIMEOUT_S * 1000}")
            if params:
                pg_sql = sql
                for i in range(len(params)):
                    pg_sql = pg_sql.replace("?", f"${i + 1}", 1)
                result = await conn.execute(pg_sql, *params)
            else:
                result = await conn.execute(sql)
            return f"Erfolgreich. {result}"
        except Exception as exc:
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            await conn.close()

    def _pg_psycopg2_execute(
        self,
        connstr: str,
        sql: str,
        params: list[Any] | None,
    ) -> str:
        """Fallback write via psycopg2."""
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DatabaseError(
                "PostgreSQL nicht verfuegbar. Installiere: pip install asyncpg"
            ) from exc

        conn = psycopg2.connect(connstr, connect_timeout=_CONN_TIMEOUT)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params or None)
            conn.commit()
            rows_affected = cursor.rowcount
            return f"Erfolgreich. {rows_affected} Zeile(n) betroffen."
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"SQL-Fehler: {exc}") from exc
        finally:
            conn.close()

    async def _pg_connect_info(self, connstr: str) -> str:
        """Return connection info for PostgreSQL."""
        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError:
            return self._pg_psycopg2_connect_info(connstr)

        try:
            conn = await asyncio.wait_for(asyncpg.connect(connstr), timeout=_CONN_TIMEOUT)
        except TimeoutError as exc:
            raise DatabaseError(f"Verbindungs-Timeout nach {_CONN_TIMEOUT}s") from exc
        except Exception as exc:
            raise DatabaseError(f"Verbindungsfehler: {exc}") from exc

        try:
            version = await conn.fetchval("SELECT version()")
            table_count = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            db_name = await conn.fetchval("SELECT current_database()")
            db_size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            lines = [
                "Datenbankverbindung erfolgreich.",
                "",
                "  Typ:       PostgreSQL",
                f"  Version:   {version}",
                f"  Datenbank: {db_name}",
                f"  Groesse:   {db_size}",
                f"  Tabellen:  {table_count}",
            ]
            return "\n".join(lines)
        finally:
            await conn.close()

    def _pg_psycopg2_connect_info(self, connstr: str) -> str:
        """Fallback connection info via psycopg2."""
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DatabaseError(
                "PostgreSQL nicht verfuegbar. Installiere: pip install asyncpg"
            ) from exc

        conn = psycopg2.connect(connstr, connect_timeout=_CONN_TIMEOUT)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            row = cursor.fetchone()
            version = row[0] if row else "unknown"
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            row = cursor.fetchone()
            table_count = row[0] if row else "unknown"
            cursor.execute("SELECT current_database()")
            row = cursor.fetchone()
            db_name = row[0] if row else "unknown"
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            row = cursor.fetchone()
            db_size = row[0] if row else "unknown"
            lines = [
                "Datenbankverbindung erfolgreich.",
                "",
                "  Typ:       PostgreSQL",
                f"  Version:   {version}",
                f"  Datenbank: {db_name}",
                f"  Groesse:   {db_size}",
                f"  Tabellen:  {table_count}",
            ]
            return "\n".join(lines)
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Public async API (MCP tool handlers)
    # ------------------------------------------------------------------ #

    async def db_query(
        self,
        database: str,
        sql: str,
        params: list[Any] | None = None,
        limit: int = _DEFAULT_ROW_LIMIT,
    ) -> str:
        """Execute a SELECT query (read-only).

        Args:
            database: SQLite file path or PostgreSQL connection string.
            sql: SQL query to execute.
            params: Parameters for parameterized query.
            limit: Maximum rows to return (default 500, max 5000).

        Returns:
            Formatted ASCII table with results.
        """
        if not sql.strip():
            return "Fehler: Kein SQL angegeben."

        limit = max(1, min(limit, _MAX_ROW_LIMIT))

        if _is_pg_connection_string(database):
            return await self._pg_query(database, sql, params, limit)

        # SQLite
        db_path = self._validate_sqlite_path(database)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sqlite_query, db_path, sql, params, limit)

    async def db_schema(
        self,
        database: str,
        table: str | None = None,
    ) -> str:
        """Show database schema.

        Args:
            database: SQLite file path or PostgreSQL connection string.
            table: Optional specific table name.

        Returns:
            Schema information as formatted text.
        """
        if _is_pg_connection_string(database):
            return await self._pg_schema(database, table)

        db_path = self._validate_sqlite_path(database)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sqlite_schema, db_path, table)

    async def db_execute(
        self,
        database: str,
        sql: str,
        params: list[Any] | None = None,
    ) -> str:
        """Execute a write query (INSERT/UPDATE/DELETE/CREATE).

        Requires Gatekeeper approval (ORANGE risk level).

        Args:
            database: SQLite file path or PostgreSQL connection string.
            sql: SQL statement to execute.
            params: Parameters for parameterized query.

        Returns:
            Confirmation with rows affected.
        """
        if not sql.strip():
            return "Fehler: Kein SQL angegeben."

        if _is_pg_connection_string(database):
            return await self._pg_execute(database, sql, params)

        db_path = self._validate_sqlite_path(database)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sqlite_execute, db_path, sql, params)

    async def db_connect(
        self,
        database: str,
    ) -> str:
        """Test database connection and show info.

        Args:
            database: SQLite file path or PostgreSQL connection string.

        Returns:
            Database type, version, size, table count.
        """
        if _is_pg_connection_string(database):
            return await self._pg_connect_info(database)

        db_path = self._validate_sqlite_path(database)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sqlite_connect_info, db_path)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def register_database_tools(
    mcp_client: Any,
    config: JarvisConfig,
) -> DatabaseTools:
    """Registriert Datenbank-Tools beim MCP-Client.

    Returns:
        DatabaseTools-Instanz.
    """
    tools = DatabaseTools(config)

    mcp_client.register_builtin_handler(
        "db_query",
        tools.db_query,
        description=(
            "Fuehrt eine SELECT-Abfrage auf einer Datenbank aus (read-only). "
            "Unterstuetzt SQLite (Dateipfad) und PostgreSQL (Connection-String). "
            "Ergebnis: Formatierte Tabelle mit Spalten und Zeilen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": (
                        "Pfad zur SQLite-Datei oder PostgreSQL Connection-String "
                        "(z.B. postgresql://user:pass@host/db)"
                    ),
                },
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT-Abfrage",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Parameter fuer parametrisierte Abfragen (? als Platzhalter)",
                    "default": [],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximale Anzahl Zeilen (Default: 500, Max: 5000)",
                    "default": _DEFAULT_ROW_LIMIT,
                },
            },
            "required": ["database", "sql"],
        },
    )

    mcp_client.register_builtin_handler(
        "db_schema",
        tools.db_schema,
        description=(
            "Zeigt das Schema einer Datenbank an. "
            "Ohne table-Parameter: Liste aller Tabellen. "
            "Mit table-Parameter: Detaillierte Spalteninformationen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Pfad zur SQLite-Datei oder PostgreSQL Connection-String",
                },
                "table": {
                    "type": "string",
                    "description": "Optionaler Tabellenname fuer Detail-Ansicht",
                    "default": None,
                },
            },
            "required": ["database"],
        },
    )

    mcp_client.register_builtin_handler(
        "db_execute",
        tools.db_execute,
        description=(
            "Fuehrt eine schreibende SQL-Anweisung aus (INSERT/UPDATE/DELETE/CREATE). "
            "Erfordert Gatekeeper-Genehmigung. DROP-Anweisungen sind blockiert."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Pfad zur SQLite-Datei oder PostgreSQL Connection-String",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL-Anweisung (INSERT, UPDATE, DELETE, CREATE TABLE, etc.)",
                },
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Parameter fuer parametrisierte Abfragen",
                    "default": [],
                },
            },
            "required": ["database", "sql"],
        },
    )

    mcp_client.register_builtin_handler(
        "db_connect",
        tools.db_connect,
        description=(
            "Testet eine Datenbankverbindung und zeigt Informationen an "
            "(Typ, Version, Groesse, Anzahl Tabellen)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Pfad zur SQLite-Datei oder PostgreSQL Connection-String",
                },
            },
            "required": ["database"],
        },
    )

    log.info(
        "database_tools_registered",
        tools=["db_query", "db_schema", "db_execute", "db_connect"],
    )
    return tools
