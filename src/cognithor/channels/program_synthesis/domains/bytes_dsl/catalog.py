"""BinaryData primitive catalog (Sprint-26.4).

25 primitives covering byte-level read/write/encoding/hash/bitfield
operations. Endianness is always explicit (``read_u16le`` vs
``read_u16be``). Roundtrip property — ``decode(encode(x)) == x`` —
must hold for every encode/decode pair we ship.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BytesPrimitive:
    name: str
    fn: Callable[..., Any]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            msg = f"Invalid Bytes primitive name: {self.name!r}"
            raise ValueError(msg)
        if self.cost < 0:
            msg = f"Bytes primitive cost must be >= 0, got {self.cost}"
            raise ValueError(msg)


class BytesCatalog:
    def __init__(self) -> None:
        self._entries: dict[str, BytesPrimitive] = {}

    def add(self, primitive: BytesPrimitive) -> None:
        if primitive.name in self._entries:
            msg = f"Bytes primitive {primitive.name!r} already registered"
            raise ValueError(msg)
        self._entries[primitive.name] = primitive

    def get(self, name: str) -> BytesPrimitive:
        if name not in self._entries:
            msg = f"Unknown Bytes primitive {name!r}"
            raise KeyError(msg)
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


BYTES_PRIMITIVE_NAMES: tuple[str, ...] = (
    # Read primitives — endianness explicit
    "read_u8",
    "read_u16le",
    "read_u16be",
    "read_u32le",
    "read_u32be",
    "read_f32",
    "read_f64",
    "read_bytes",
    "read_until",
    "read_varint",
    # Write primitives
    "write_u8",
    "write_u16le",
    "write_u32be",
    # Encoding round-trip pairs
    "encode_base64",
    "decode_base64",
    "encode_hex",
    "decode_hex",
    "compress_gzip",
    "decompress_gzip",
    # Hash
    "sha256_hex",
    "blake2b_hex",
    "crc32_int",
    # Bitfield
    "bit_get",
    "bit_set",
    "byte_swap",
)


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _coerce_bytes(value: Any, kind: str) -> bytes:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    msg = f"{kind} expects bytes, got {type(value).__name__}"
    raise TypeError(msg)


def _read_u8(data: bytes, offset: int = 0) -> int:
    return _coerce_bytes(data, "read_u8")[offset]


def _read_u16le(data: bytes, offset: int = 0) -> int:
    raw = _coerce_bytes(data, "read_u16le")
    return int.from_bytes(raw[offset : offset + 2], "little", signed=False)


def _read_u16be(data: bytes, offset: int = 0) -> int:
    raw = _coerce_bytes(data, "read_u16be")
    return int.from_bytes(raw[offset : offset + 2], "big", signed=False)


def _read_u32le(data: bytes, offset: int = 0) -> int:
    raw = _coerce_bytes(data, "read_u32le")
    return int.from_bytes(raw[offset : offset + 4], "little", signed=False)


def _read_u32be(data: bytes, offset: int = 0) -> int:
    raw = _coerce_bytes(data, "read_u32be")
    return int.from_bytes(raw[offset : offset + 4], "big", signed=False)


def _read_f32(data: bytes, offset: int = 0, *, big_endian: bool = False) -> float:
    raw = _coerce_bytes(data, "read_f32")
    fmt = ">f" if big_endian else "<f"
    return float(struct.unpack(fmt, raw[offset : offset + 4])[0])


def _read_f64(data: bytes, offset: int = 0, *, big_endian: bool = False) -> float:
    raw = _coerce_bytes(data, "read_f64")
    fmt = ">d" if big_endian else "<d"
    return float(struct.unpack(fmt, raw[offset : offset + 8])[0])


def _read_bytes(data: bytes, offset: int, length: int) -> bytes:
    return _coerce_bytes(data, "read_bytes")[offset : offset + length]


def _read_until(data: bytes, delim: int, offset: int = 0) -> bytes:
    raw = _coerce_bytes(data, "read_until")
    end = raw.find(bytes([delim]), offset)
    if end < 0:
        return raw[offset:]
    return raw[offset:end]


def _read_varint(data: bytes, offset: int = 0) -> int:
    """LEB128-style varint (used by protobuf)."""
    raw = _coerce_bytes(data, "read_varint")
    result = 0
    shift = 0
    pos = offset
    while pos < len(raw):
        byte = raw[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result
        shift += 7
    msg = "read_varint: data ended mid-varint"
    raise ValueError(msg)


def _write_u8(value: int) -> bytes:
    if not 0 <= int(value) <= 0xFF:
        msg = f"write_u8: value {value} out of range"
        raise ValueError(msg)
    return bytes([int(value)])


def _write_u16le(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=False)


def _write_u32be(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


def _encode_base64(data: bytes) -> str:
    return base64.b64encode(_coerce_bytes(data, "encode_base64")).decode("ascii")


def _decode_base64(text: str) -> bytes:
    if not isinstance(text, str):
        msg = f"decode_base64 expects str, got {type(text).__name__}"
        raise TypeError(msg)
    return base64.b64decode(text)


def _encode_hex(data: bytes) -> str:
    return _coerce_bytes(data, "encode_hex").hex()


def _decode_hex(text: str) -> bytes:
    if not isinstance(text, str):
        msg = f"decode_hex expects str, got {type(text).__name__}"
        raise TypeError(msg)
    return bytes.fromhex(text)


def _compress_gzip(data: bytes) -> bytes:
    return gzip.compress(_coerce_bytes(data, "compress_gzip"))


def _decompress_gzip(data: bytes) -> bytes:
    return gzip.decompress(_coerce_bytes(data, "decompress_gzip"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(_coerce_bytes(data, "sha256_hex")).hexdigest()


def _blake2b_hex(data: bytes, *, digest_size: int = 32) -> str:
    return hashlib.blake2b(
        _coerce_bytes(data, "blake2b_hex"),
        digest_size=digest_size,
    ).hexdigest()


def _crc32_int(data: bytes) -> int:
    return zlib.crc32(_coerce_bytes(data, "crc32_int")) & 0xFFFFFFFF


def _bit_get(data: bytes, bit_index: int) -> int:
    raw = _coerce_bytes(data, "bit_get")
    byte_idx = bit_index // 8
    bit_in_byte = bit_index % 8
    return (raw[byte_idx] >> bit_in_byte) & 1


def _bit_set(data: bytes, bit_index: int, value: int) -> bytes:
    raw = bytearray(_coerce_bytes(data, "bit_set"))
    byte_idx = bit_index // 8
    bit_in_byte = bit_index % 8
    if value:
        raw[byte_idx] |= 1 << bit_in_byte
    else:
        raw[byte_idx] &= ~(1 << bit_in_byte)
    return bytes(raw)


def _byte_swap(data: bytes) -> bytes:
    return bytes(reversed(_coerce_bytes(data, "byte_swap")))


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_bytes_catalog() -> BytesCatalog:
    cat = BytesCatalog()

    def add(name: str, fn: Callable[..., Any], cost: float, desc: str) -> None:
        cat.add(BytesPrimitive(name=name, fn=fn, cost=cost, description=desc))

    add("read_u8", _read_u8, 0.1, "Read 1 unsigned byte")
    add("read_u16le", _read_u16le, 0.2, "Read u16 little-endian")
    add("read_u16be", _read_u16be, 0.2, "Read u16 big-endian")
    add("read_u32le", _read_u32le, 0.2, "Read u32 little-endian")
    add("read_u32be", _read_u32be, 0.2, "Read u32 big-endian")
    add("read_f32", _read_f32, 0.3, "Read IEEE 754 float32")
    add("read_f64", _read_f64, 0.3, "Read IEEE 754 float64")
    add("read_bytes", _read_bytes, 0.2, "Read N bytes from offset")
    add("read_until", _read_until, 0.3, "Read up to delim byte")
    add("read_varint", _read_varint, 0.4, "LEB128 protobuf-style varint")
    add("write_u8", _write_u8, 0.1, "Encode u8 to 1 byte")
    add("write_u16le", _write_u16le, 0.2, "Encode u16 little-endian")
    add("write_u32be", _write_u32be, 0.2, "Encode u32 big-endian")
    add("encode_base64", _encode_base64, 0.2, "Bytes → base64 string")
    add("decode_base64", _decode_base64, 0.2, "base64 string → bytes")
    add("encode_hex", _encode_hex, 0.2, "Bytes → hex string")
    add("decode_hex", _decode_hex, 0.2, "hex string → bytes")
    add("compress_gzip", _compress_gzip, 0.5, "gzip compress")
    add("decompress_gzip", _decompress_gzip, 0.5, "gzip decompress")
    add("sha256_hex", _sha256_hex, 0.4, "SHA-256 hex digest")
    add("blake2b_hex", _blake2b_hex, 0.4, "BLAKE2b hex digest")
    add("crc32_int", _crc32_int, 0.3, "CRC32 unsigned integer")
    add("bit_get", _bit_get, 0.2, "Read bit at index")
    add("bit_set", _bit_set, 0.3, "Set bit at index")
    add("byte_swap", _byte_swap, 0.2, "Reverse byte order")

    # binascii is used only via the hex/base64 helpers above; the import
    # stays load-bearing because some platforms keep ``hex`` as a thin
    # binascii wrapper.
    _ = binascii.hexlify

    return cat
