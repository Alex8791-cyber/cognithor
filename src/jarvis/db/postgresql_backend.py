"""PostgreSQL Database Backend mit Connection Pooling.

Nutzt psycopg v3 mit asyncio Connection Pool.
Unterstuetzt pgvector fuer native Vector-Suche.

Besonderheiten:
  - pgvector Extension fuer HNSW Vector-Index
  - tsvector + GIN Index statt FTS5 fuer BM25
  - BYTEA statt BLOB
  - SERIAL statt AUTOINCREMENT
  - %s Platzhalter statt ?
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger("jarvis.db.postgresql")


class PostgreSQLBackend:
    """PostgreSQL-Backend mit psycopg v3 Connection Pool."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "jarvis",
        user: str = "jarvis",
        password: str = "",
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        # Use keyword-based conninfo construction to prevent parameter injection
        # psycopg.conninfo.make_conninfo escapes special chars in values
        try:
            from psycopg.conninfo import make_conninfo
            self._conninfo = make_conninfo(host=host, port=port, dbname=dbname, user=user, password=password)
        except ImportError:
            # Fallback: manually escape values (single quotes in libpq conninfo)
            def _esc(v: str) -> str:
                return v.replace("\\", "\\\\").replace("'", "\\'")
            self._conninfo = (
                f"host='{_esc(host)}' port={int(port)} dbname='{_esc(dbname)}' "
                f"user='{_esc(user)}' password='{_esc(password)}'"
            )
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any | None = None

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            try:
                from psycopg_pool import AsyncConnectionPool
                self._pool = AsyncConnectionPool(
                    conninfo=self._conninfo,
                    min_size=self._pool_min,
                    max_size=self._pool_max,
                    open=False,
                )
                await self._pool.open()
                logger.info("PostgreSQL Connection Pool gestartet (%d-%d)", self._pool_min, self._pool_max)

                # pgvector Extension aktivieren
                async with self._pool.connection() as conn:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.commit()
                    logger.info("pgvector Extension aktiviert")
            except ImportError:
                raise ImportError(
                    "psycopg[binary] und psycopg-pool nicht installiert. "
                    "Installation: pip install 'psycopg[binary]' psycopg-pool pgvector"
                ) from None
        return self._pool

    async def execute(self, query: str, params: Sequence[Any] = ()) -> Any:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(query, params)
            await conn.commit()
            return cursor

    async def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, params_seq)
            await conn.commit()

    async def executescript(self, script: str) -> None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            for statement in script.split(";"):
                stmt = statement.strip()
                if stmt:
                    await conn.execute(stmt)
            await conn.commit()

    async def fetchone(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                row = await cur.fetchone()
                if row is None:
                    return None
                columns = [desc[0] for desc in cur.description or []]
                return dict(zip(columns, row))

    async def fetchall(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
                columns = [desc[0] for desc in cur.description or []]
                return [dict(zip(columns, row)) for row in rows]

    async def commit(self) -> None:
        pass  # Auto-commit in pool connections

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL Connection Pool geschlossen")

    @property
    def placeholder(self) -> str:
        return "%s"

    @property
    def backend_type(self) -> str:
        return "postgresql"
