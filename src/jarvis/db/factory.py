"""Database Backend Factory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.db.factory")


def create_backend(config: Any) -> Any:
    """Erstellt das passende Database-Backend basierend auf der Konfiguration.

    Args:
        config: JarvisConfig oder DatabaseConfig Objekt.
              Prueft config.database.backend (wenn vorhanden).

    Returns:
        SQLiteBackend oder PostgreSQLBackend Instanz.
    """
    # DatabaseConfig extrahieren
    db_config = getattr(config, "database", None)

    if db_config is None or getattr(db_config, "backend", "sqlite") == "sqlite":
        from jarvis.db.sqlite_backend import SQLiteBackend
        db_path = getattr(config, "db_path", Path.home() / ".jarvis" / "index" / "memory.db")
        backend = SQLiteBackend(db_path)
        logger.info("Database-Backend: SQLite (%s)", db_path)
        return backend

    if getattr(db_config, "backend", "") == "postgresql":
        from jarvis.db.postgresql_backend import PostgreSQLBackend
        backend = PostgreSQLBackend(
            host=db_config.pg_host,
            port=db_config.pg_port,
            dbname=db_config.pg_dbname,
            user=db_config.pg_user,
            password=db_config.pg_password,
            pool_min=db_config.pg_pool_min,
            pool_max=db_config.pg_pool_max,
        )
        logger.info("Database-Backend: PostgreSQL (%s:%d/%s)", db_config.pg_host, db_config.pg_port, db_config.pg_dbname)
        return backend

    raise ValueError(f"Unbekanntes Database-Backend: {db_config.backend}")
