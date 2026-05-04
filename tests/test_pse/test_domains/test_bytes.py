"""Tests for the BinaryData domain (Sprint-26.4)."""

from __future__ import annotations

import struct

import pytest

from cognithor.channels.program_synthesis.domains.bytes_dsl import (
    BYTES_PRIMITIVE_NAMES,
    BytesCatalog,
    BytesDomain,
    BytesPrimitive,
    BytesVerifierError,
    build_bytes_catalog,
    register_bytes_domain,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestBytesCatalog:
    def test_builds(self) -> None:
        cat = build_bytes_catalog()
        assert isinstance(cat, BytesCatalog)
        assert len(cat) == len(BYTES_PRIMITIVE_NAMES)

    def test_at_least_25_primitives(self) -> None:
        assert len(BYTES_PRIMITIVE_NAMES) >= 25

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid Bytes"):
            BytesPrimitive(name="bad-!", fn=lambda: b"", cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            BytesPrimitive(name="p", fn=lambda: b"", cost=-1.0)


class TestReadPrimitives:
    def test_read_u8(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("read_u8").fn(b"\x42") == 0x42

    def test_read_u16le_vs_be(self) -> None:
        cat = build_bytes_catalog()
        data = b"\x01\x02"
        assert cat.get("read_u16le").fn(data) == 0x0201
        assert cat.get("read_u16be").fn(data) == 0x0102

    def test_read_u32(self) -> None:
        cat = build_bytes_catalog()
        data = b"\x00\x00\x00\x01"
        assert cat.get("read_u32be").fn(data) == 1
        assert cat.get("read_u32le").fn(data) == 16777216

    def test_read_f32(self) -> None:
        cat = build_bytes_catalog()
        data = struct.pack("<f", 3.14)
        out = cat.get("read_f32").fn(data)
        assert abs(out - 3.14) < 1e-6

    def test_read_f64(self) -> None:
        cat = build_bytes_catalog()
        data = struct.pack("<d", 3.141592653589793)
        out = cat.get("read_f64").fn(data)
        assert abs(out - 3.141592653589793) < 1e-12

    def test_read_bytes(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("read_bytes").fn(b"hello world", 6, 5) == b"world"

    def test_read_until(self) -> None:
        cat = build_bytes_catalog()
        out = cat.get("read_until").fn(b"hello\x00world", 0)
        assert out == b"hello"

    def test_read_varint(self) -> None:
        cat = build_bytes_catalog()
        # 300 = 0xAC 0x02 in LEB128
        assert cat.get("read_varint").fn(b"\xac\x02") == 300

    def test_read_u8_rejects_non_bytes(self) -> None:
        cat = build_bytes_catalog()
        with pytest.raises(TypeError):
            cat.get("read_u8").fn("not bytes")


class TestWritePrimitives:
    def test_write_u8(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("write_u8").fn(0x42) == b"\x42"

    def test_write_u8_out_of_range(self) -> None:
        cat = build_bytes_catalog()
        with pytest.raises(ValueError, match="out of range"):
            cat.get("write_u8").fn(300)

    def test_write_u16le(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("write_u16le").fn(0x0201) == b"\x01\x02"

    def test_write_u32be(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("write_u32be").fn(1) == b"\x00\x00\x00\x01"


class TestEncodingRoundtrip:
    def test_base64(self) -> None:
        cat = build_bytes_catalog()
        encoded = cat.get("encode_base64").fn(b"hello")
        decoded = cat.get("decode_base64").fn(encoded)
        assert decoded == b"hello"

    def test_hex(self) -> None:
        cat = build_bytes_catalog()
        encoded = cat.get("encode_hex").fn(b"\x42\xab")
        decoded = cat.get("decode_hex").fn(encoded)
        assert decoded == b"\x42\xab"

    def test_gzip(self) -> None:
        cat = build_bytes_catalog()
        compressed = cat.get("compress_gzip").fn(b"hello world" * 100)
        decompressed = cat.get("decompress_gzip").fn(compressed)
        assert decompressed == b"hello world" * 100

    def test_decode_base64_rejects_non_str(self) -> None:
        cat = build_bytes_catalog()
        with pytest.raises(TypeError):
            cat.get("decode_base64").fn(b"raw bytes")


class TestHashes:
    def test_sha256_deterministic(self) -> None:
        cat = build_bytes_catalog()
        h1 = cat.get("sha256_hex").fn(b"hello")
        h2 = cat.get("sha256_hex").fn(b"hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_blake2b_deterministic(self) -> None:
        cat = build_bytes_catalog()
        h = cat.get("blake2b_hex").fn(b"hello")
        assert len(h) == 64

    def test_crc32(self) -> None:
        cat = build_bytes_catalog()
        c1 = cat.get("crc32_int").fn(b"hello")
        c2 = cat.get("crc32_int").fn(b"hello")
        assert c1 == c2
        assert 0 <= c1 <= 0xFFFFFFFF


class TestBitfield:
    def test_bit_get(self) -> None:
        cat = build_bytes_catalog()
        # 0x05 = 0b00000101 → bit 0 = 1, bit 2 = 1, bit 1 = 0
        assert cat.get("bit_get").fn(b"\x05", 0) == 1
        assert cat.get("bit_get").fn(b"\x05", 1) == 0
        assert cat.get("bit_get").fn(b"\x05", 2) == 1

    def test_bit_set(self) -> None:
        cat = build_bytes_catalog()
        out = cat.get("bit_set").fn(b"\x00", 3, 1)
        assert out == b"\x08"  # bit 3 set = 0b00001000

    def test_bit_set_clear(self) -> None:
        cat = build_bytes_catalog()
        out = cat.get("bit_set").fn(b"\xff", 0, 0)
        assert out == b"\xfe"

    def test_byte_swap(self) -> None:
        cat = build_bytes_catalog()
        assert cat.get("byte_swap").fn(b"\x01\x02\x03\x04") == b"\x04\x03\x02\x01"


class TestBytesDomain:
    def test_metadata(self) -> None:
        d = BytesDomain()
        m = d.metadata
        assert m.name == "bytes"
        assert m.benchmark_target == 0.70

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_bytes_domain(reg)
        assert isinstance(reg.get("bytes"), BytesDomain)

    def test_verify_pipeline(self) -> None:
        d = BytesDomain()
        ok = d.verify(
            [
                {"primitive": "encode_base64", "args": {}},
            ],
            [{"input": b"hello", "output": "aGVsbG8="}],
        )
        assert ok

    def test_verify_mismatch_raises(self) -> None:
        d = BytesDomain()
        with pytest.raises(BytesVerifierError, match="!= expected"):
            d.verify(
                [{"primitive": "sha256_hex", "args": {}}],
                [{"input": b"hello", "output": "wrong"}],
            )

    def test_verify_unknown_primitive(self) -> None:
        d = BytesDomain()
        with pytest.raises(BytesVerifierError, match="Unknown Bytes"):
            d.verify(
                [{"primitive": "no_such_thing", "args": {}}],
                [{"input": b"x", "output": "x"}],
            )

    def test_program_must_be_list_or_dict(self) -> None:
        d = BytesDomain()
        with pytest.raises(BytesVerifierError, match="must be"):
            d.verify("nope", [])

    def test_step_not_a_mapping(self) -> None:
        d = BytesDomain()
        with pytest.raises(BytesVerifierError, match="must be a mapping"):
            d.verify(["not-a-step"], [])
