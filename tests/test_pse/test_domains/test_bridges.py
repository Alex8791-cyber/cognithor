"""Tests for the cross-domain bridge layer (Sprint-26.2, D5)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cognithor.channels.program_synthesis.domains.bridges import (
    BRIDGE_REGISTRY,
    SPRINT26_BRIDGE_WHITELIST,
    BridgeNotWhitelistedError,
    BridgeOperator,
    BridgeRegistry,
    install_default_bridges,
)


class TestWhitelist:
    def test_has_exactly_12_pairs(self) -> None:
        assert len(SPRINT26_BRIDGE_WHITELIST) == 12

    def test_pairs_are_distinct(self) -> None:
        pairs = list(SPRINT26_BRIDGE_WHITELIST)
        assert len(pairs) == len(set(pairs))

    def test_no_self_loops(self) -> None:
        for f, t in SPRINT26_BRIDGE_WHITELIST:
            assert f != t

    def test_canonical_registry_is_populated(self) -> None:
        # Module-import-time install means the canonical registry has
        # all 12 bridges available without manual setup.
        assert len(BRIDGE_REGISTRY) == 12
        for pair in SPRINT26_BRIDGE_WHITELIST:
            assert pair in BRIDGE_REGISTRY


class TestRegistry:
    def test_register_outside_whitelist_raises(self) -> None:
        reg = BridgeRegistry()
        with pytest.raises(BridgeNotWhitelistedError, match="not in the Sprint-26"):
            reg.register(BridgeOperator(from_type="image", to_type="json", fn=lambda x: x))

    def test_double_register_raises(self) -> None:
        reg = BridgeRegistry()
        op = BridgeOperator("json", "string", str)
        reg.register(op)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(op)

    def test_get_unknown_pair(self) -> None:
        reg = BridgeRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("json", "datetime")

    def test_has_works(self) -> None:
        reg = BridgeRegistry()
        assert not reg.has("json", "datetime")
        reg.register(BridgeOperator("json", "string", str))
        assert reg.has("json", "string")

    def test_install_default_bridges_into_fresh_registry(self) -> None:
        reg = BridgeRegistry()
        install_default_bridges(reg)
        assert len(reg) == 12

    def test_invalid_operator_construction(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BridgeOperator("", "string", str)
        with pytest.raises(ValueError, match="must differ"):
            BridgeOperator("string", "string", str)


class TestOperators:
    def test_json_to_datetime_iso(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "datetime")
        result = op.fn("2026-05-04T12:00:00Z")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_json_to_datetime_epoch(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "datetime")
        result = op.fn(0)
        assert result == datetime(1970, 1, 1, tzinfo=UTC)

    def test_json_to_datetime_rejects_bool_unsupported(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "datetime")
        with pytest.raises(TypeError):
            op.fn([1, 2])  # arrays aren't bridgeable

    def test_json_to_number_rejects_bool(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "number")
        with pytest.raises(TypeError, match="bool"):
            op.fn(True)

    def test_json_to_number_int(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "number")
        assert op.fn(42) == 42.0

    def test_json_to_string_passthrough(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "string")
        assert op.fn("hello") == "hello"

    def test_json_to_string_repr_for_non_str(self) -> None:
        op = BRIDGE_REGISTRY.get("json", "string")
        # Non-str → repr fallback
        out = op.fn({"k": 1})
        assert "k" in out

    def test_string_to_datetime(self) -> None:
        op = BRIDGE_REGISTRY.get("string", "datetime")
        result = op.fn("2026-05-04T12:00:00")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None  # naive coerced to UTC

    def test_string_to_number(self) -> None:
        op = BRIDGE_REGISTRY.get("string", "number")
        assert op.fn("3.14") == pytest.approx(3.14)

    def test_string_to_json(self) -> None:
        op = BRIDGE_REGISTRY.get("string", "json")
        assert op.fn('{"a": 1}') == {"a": 1}

    def test_datetime_to_sql_literal(self) -> None:
        op = BRIDGE_REGISTRY.get("datetime", "sql_literal")
        out = op.fn(datetime(2026, 5, 4, 12, 0, tzinfo=UTC))
        assert out.startswith("TIMESTAMP '")
        assert "2026-05-04" in out

    def test_datetime_to_string(self) -> None:
        op = BRIDGE_REGISTRY.get("datetime", "string")
        out = op.fn(datetime(2026, 5, 4, 12, 0, tzinfo=UTC))
        assert "2026-05-04" in out and "+00:00" in out

    def test_number_to_sql_literal(self) -> None:
        op = BRIDGE_REGISTRY.get("number", "sql_literal")
        assert op.fn(42) == "42"
        assert op.fn(3.14) == "3.14"

    def test_number_to_string(self) -> None:
        op = BRIDGE_REGISTRY.get("number", "string")
        assert op.fn(42) == "42"

    def test_number_to_string_rejects_bool(self) -> None:
        op = BRIDGE_REGISTRY.get("number", "string")
        with pytest.raises(TypeError, match="bool"):
            op.fn(True)

    def test_bytes_to_string_utf8(self) -> None:
        op = BRIDGE_REGISTRY.get("bytes", "string")
        assert op.fn(b"hello") == "hello"

    def test_bytes_to_string_base64_fallback(self) -> None:
        op = BRIDGE_REGISTRY.get("bytes", "string")
        out = op.fn(b"\xff\xfe\xfd")
        assert isinstance(out, str)
        # base64 uses [A-Za-z0-9+/=]
        assert all(c.isalnum() or c in "+/=" for c in out)

    def test_bytes_to_number_be(self) -> None:
        op = BRIDGE_REGISTRY.get("bytes", "number")
        assert op.fn(b"\x00\x01") == 1
        assert op.fn(b"\x01\x00") == 256

    def test_bytes_to_number_empty(self) -> None:
        op = BRIDGE_REGISTRY.get("bytes", "number")
        assert op.fn(b"") == 0
