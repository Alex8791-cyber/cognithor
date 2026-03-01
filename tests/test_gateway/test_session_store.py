"""Tests für SessionStore -- inkl. Channel-Mapping-Persistenz."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.gateway.session_store import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    """Frischer SessionStore mit temporärer DB."""
    return SessionStore(tmp_path / "test_sessions.db")


# ============================================================================
# Channel-Mappings
# ============================================================================


class TestChannelMappings:
    """Tests für die Channel-Mapping-Persistenz."""

    def test_save_and_load_mapping(self, store: SessionStore) -> None:
        """Einzelnes Mapping speichern und laden."""
        store.save_channel_mapping("telegram_session", "sess_123", "456789")
        result = store.load_channel_mapping("telegram_session", "sess_123")
        assert result == "456789"

    def test_load_nonexistent_mapping(self, store: SessionStore) -> None:
        """Nicht existierendes Mapping gibt None zurück."""
        result = store.load_channel_mapping("telegram_session", "nonexistent")
        assert result is None

    def test_overwrite_mapping(self, store: SessionStore) -> None:
        """Überschreiben eines bestehenden Mappings."""
        store.save_channel_mapping("discord_user", "user_1", "100")
        store.save_channel_mapping("discord_user", "user_1", "200")
        result = store.load_channel_mapping("discord_user", "user_1")
        assert result == "200"

    def test_load_all_mappings(self, store: SessionStore) -> None:
        """Alle Mappings eines Channels laden."""
        store.save_channel_mapping("telegram_session", "s1", "100")
        store.save_channel_mapping("telegram_session", "s2", "200")
        store.save_channel_mapping("telegram_session", "s3", "300")

        mappings = store.load_all_channel_mappings("telegram_session")
        assert mappings == {"s1": "100", "s2": "200", "s3": "300"}

    def test_load_all_empty(self, store: SessionStore) -> None:
        """Leere Ergebnismenge bei unbekanntem Channel."""
        mappings = store.load_all_channel_mappings("unknown_channel")
        assert mappings == {}

    def test_channel_isolation(self, store: SessionStore) -> None:
        """Verschiedene Channel-Namespaces sind isoliert."""
        store.save_channel_mapping("telegram_session", "key", "tg_value")
        store.save_channel_mapping("discord_session", "key", "dc_value")

        assert store.load_channel_mapping("telegram_session", "key") == "tg_value"
        assert store.load_channel_mapping("discord_session", "key") == "dc_value"

    def test_cleanup_channel_mappings(self, store: SessionStore) -> None:
        """Alte Mappings werden bereinigt."""
        # Mapping speichern
        store.save_channel_mapping("test_ch", "old_key", "old_value")

        # updated_at manuell auf 60 Tage zurücksetzen
        cutoff = time.time() - (60 * 86400)
        store.conn.execute(
            "UPDATE channel_mappings SET updated_at = ? WHERE mapping_key = ?",
            (cutoff, "old_key"),
        )
        store.conn.commit()

        # Frisches Mapping
        store.save_channel_mapping("test_ch", "new_key", "new_value")

        # Cleanup mit 30 Tage
        deleted = store.cleanup_channel_mappings(max_age_days=30)
        assert deleted == 1

        # Altes Mapping weg, neues noch da
        assert store.load_channel_mapping("test_ch", "old_key") is None
        assert store.load_channel_mapping("test_ch", "new_key") == "new_value"

    def test_cleanup_no_old_mappings(self, store: SessionStore) -> None:
        """Cleanup ohne alte Mappings gibt 0 zurück."""
        store.save_channel_mapping("test", "k", "v")
        deleted = store.cleanup_channel_mappings(max_age_days=30)
        assert deleted == 0

    def test_mapping_table_created(self, store: SessionStore) -> None:
        """channel_mappings Tabelle existiert nach Initialisierung."""
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='channel_mappings'"
        ).fetchall()
        assert len(tables) == 1
