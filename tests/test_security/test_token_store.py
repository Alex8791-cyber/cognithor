"""Tests für SecureTokenStore und TLS-Helper."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


class TestSecureTokenStore:
    """Tests für den SecureTokenStore."""

    def test_store_and_retrieve(self) -> None:
        """Roundtrip: store → retrieve gibt Klartext zurück."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("my_token", "super-secret-123")
        assert store.retrieve("my_token") == "super-secret-123"

    def test_retrieve_unknown_key(self) -> None:
        """Unbekannter Key wirft KeyError."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        with pytest.raises(KeyError):
            store.retrieve("nonexistent")

    def test_clear_removes_all(self) -> None:
        """clear() entfernt alle Tokens."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("a", "1")
        store.store("b", "2")
        store.clear()
        with pytest.raises(KeyError):
            store.retrieve("a")
        with pytest.raises(KeyError):
            store.retrieve("b")

    def test_different_tokens_isolated(self) -> None:
        """Verschiedene Tokens sind voneinander isoliert."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("token_a", "value_a")
        store.store("token_b", "value_b")
        assert store.retrieve("token_a") == "value_a"
        assert store.retrieve("token_b") == "value_b"

    def test_stored_value_is_encrypted(self) -> None:
        """Gespeicherter Wert ist NICHT der Klartext."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        plaintext = "my-secret-token-12345"
        store.store("test", plaintext)
        # Direkt auf die interne Struktur zugreifen
        raw_bytes = store._tokens["test"]
        assert raw_bytes != plaintext.encode("utf-8")

    def test_overwrite_existing_token(self) -> None:
        """Überschreiben eines bestehenden Tokens funktioniert."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("tok", "old_value")
        store.store("tok", "new_value")
        assert store.retrieve("tok") == "new_value"

    def test_contains_check(self) -> None:
        """__contains__ prüft ob ein Token existiert."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("exists", "val")
        assert "exists" in store
        assert "missing" not in store

    def test_unicode_token(self) -> None:
        """Unicode-Tokens werden korrekt gespeichert und abgerufen."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("unicode", "tökën-wïth-ümlauts-🔑")
        assert store.retrieve("unicode") == "tökën-wïth-ümlauts-🔑"

    def test_empty_token(self) -> None:
        """Leerer String als Token ist erlaubt."""
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        store.store("empty", "")
        assert store.retrieve("empty") == ""


class TestSecureTokenStoreFallback:
    """Tests für den Base64-Fallback ohne cryptography."""

    def test_fallback_without_cryptography(self) -> None:
        """Ohne cryptography-Paket wird Base64-Fallback genutzt."""
        import jarvis.security.token_store as mod

        original_has_crypto = mod._HAS_CRYPTO
        try:
            mod._HAS_CRYPTO = False
            store = mod.SecureTokenStore()
            assert store._fernet is None
            store.store("fallback_test", "secret-123")
            assert store.retrieve("fallback_test") == "secret-123"
            # Verify stored value is not plaintext
            raw = store._tokens["fallback_test"]
            assert raw != b"secret-123"
        finally:
            mod._HAS_CRYPTO = original_has_crypto


class TestGetTokenStore:
    """Tests für die Singleton-Funktion."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_token_store() gibt immer dieselbe Instanz zurück."""
        from jarvis.security.token_store import get_token_store

        store1 = get_token_store()
        store2 = get_token_store()
        assert store1 is store2
