from __future__ import annotations

import gzip
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class NbtError(ValueError):
    pass


@dataclass(frozen=True)
class NbtList:
    item_type: int
    items: list[Any]


@dataclass(frozen=True)
class NbtByteArray:
    data: bytes


@dataclass(frozen=True)
class NbtIntArray:
    data: list[int]


@dataclass(frozen=True)
class NbtLongArray:
    data: list[int]


@dataclass(frozen=True)
class NbtRoot:
    name: str
    value: dict[str, Any]


class NbtReader:
    def __init__(self, data: bytes) -> None:
        self.file = io.BytesIO(data)

    def read(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise NbtError("Unexpected end of NBT data")
        return data

    def read_u8(self) -> int:
        return self.read(1)[0]

    def read_i8(self) -> int:
        return struct.unpack(">b", self.read(1))[0]

    def read_i16(self) -> int:
        return struct.unpack(">h", self.read(2))[0]

    def read_u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def read_i32(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def read_i64(self) -> int:
        return struct.unpack(">q", self.read(8))[0]

    def read_f32(self) -> float:
        return struct.unpack(">f", self.read(4))[0]

    def read_f64(self) -> float:
        return struct.unpack(">d", self.read(8))[0]

    def read_string(self) -> str:
        length = self.read_u16()
        return self.read(length).decode("utf-8", errors="replace")

    def read_root(self) -> NbtRoot:
        tag_type = self.read_u8()
        if tag_type != TAG_COMPOUND:
            raise NbtError(f"NBT root must be TAG_Compound, got {tag_type}")
        name = self.read_string()
        return NbtRoot(name=name, value=self.read_compound_payload())

    def read_payload(self, tag_type: int) -> Any:
        if tag_type == TAG_BYTE:
            return self.read_i8()
        if tag_type == TAG_SHORT:
            return self.read_i16()
        if tag_type == TAG_INT:
            return self.read_i32()
        if tag_type == TAG_LONG:
            return self.read_i64()
        if tag_type == TAG_FLOAT:
            return self.read_f32()
        if tag_type == TAG_DOUBLE:
            return self.read_f64()
        if tag_type == TAG_BYTE_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise NbtError("Negative byte array length")
            return NbtByteArray(self.read(length))
        if tag_type == TAG_STRING:
            return self.read_string()
        if tag_type == TAG_LIST:
            item_type = self.read_u8()
            length = self.read_i32()
            if length < 0:
                raise NbtError("Negative list length")
            return NbtList(item_type=item_type, items=[self.read_payload(item_type) for _ in range(length)])
        if tag_type == TAG_COMPOUND:
            return self.read_compound_payload()
        if tag_type == TAG_INT_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise NbtError("Negative int array length")
            return NbtIntArray([self.read_i32() for _ in range(length)])
        if tag_type == TAG_LONG_ARRAY:
            length = self.read_i32()
            if length < 0:
                raise NbtError("Negative long array length")
            return NbtLongArray([self.read_i64() for _ in range(length)])
        raise NbtError(f"Unsupported NBT tag type: {tag_type}")

    def read_compound_payload(self) -> dict[str, Any]:
        value: dict[str, Any] = {}
        while True:
            tag_type = self.read_u8()
            if tag_type == TAG_END:
                return value
            name = self.read_string()
            value[name] = self.read_payload(tag_type)


class NbtWriter:
    def __init__(self) -> None:
        self.file = io.BytesIO()

    def write(self, data: bytes) -> None:
        self.file.write(data)

    def write_u8(self, value: int) -> None:
        self.write(struct.pack(">B", value))

    def write_i8(self, value: int) -> None:
        self.write(struct.pack(">b", value))

    def write_i16(self, value: int) -> None:
        self.write(struct.pack(">h", value))

    def write_i32(self, value: int) -> None:
        self.write(struct.pack(">i", value))

    def write_i64(self, value: int) -> None:
        self.write(struct.pack(">q", value))

    def write_string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        if len(encoded) > 65535:
            raise NbtError("NBT string is too long")
        self.write(struct.pack(">H", len(encoded)))
        self.write(encoded)

    def write_named(self, tag_type: int, name: str, payload: Any) -> None:
        self.write_u8(tag_type)
        self.write_string(name)
        self.write_payload(tag_type, payload)

    def write_payload(self, tag_type: int, payload: Any) -> None:
        if tag_type == TAG_BYTE:
            self.write_i8(int(payload))
        elif tag_type == TAG_SHORT:
            self.write_i16(int(payload))
        elif tag_type == TAG_INT:
            self.write_i32(int(payload))
        elif tag_type == TAG_LONG:
            self.write_i64(int(payload))
        elif tag_type == TAG_BYTE_ARRAY:
            data = payload.data if isinstance(payload, NbtByteArray) else bytes(payload)
            self.write_i32(len(data))
            self.write(data)
        elif tag_type == TAG_STRING:
            self.write_string(str(payload))
        elif tag_type == TAG_LIST:
            if not isinstance(payload, NbtList):
                raise NbtError("TAG_List payload must be NbtList")
            self.write_u8(payload.item_type)
            self.write_i32(len(payload.items))
            for item in payload.items:
                self.write_payload(payload.item_type, item)
        elif tag_type == TAG_COMPOUND:
            for child_name, child_type, child_payload in payload:
                self.write_named(child_type, child_name, child_payload)
            self.write_u8(TAG_END)
        elif tag_type == TAG_INT_ARRAY:
            data = payload.data if isinstance(payload, NbtIntArray) else list(payload)
            self.write_i32(len(data))
            for item in data:
                self.write_i32(int(item))
        elif tag_type == TAG_LONG_ARRAY:
            data = payload.data if isinstance(payload, NbtLongArray) else list(payload)
            self.write_i32(len(data))
            for item in data:
                self.write_i64(int(item))
        else:
            raise NbtError(f"Unsupported NBT writer tag type: {tag_type}")

    def write_root(self, name: str, compound_items: list[tuple[str, int, Any]]) -> bytes:
        self.write_u8(TAG_COMPOUND)
        self.write_string(name)
        self.write_payload(TAG_COMPOUND, compound_items)
        return self.file.getvalue()


def read_nbt(path: Path) -> NbtRoot:
    raw = path.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    return NbtReader(raw).read_root()


def write_gzip_nbt(path: Path, root_name: str, compound_items: list[tuple[str, int, Any]]) -> None:
    writer = NbtWriter()
    raw = writer.write_root(root_name, compound_items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw))


def encode_varints(values: list[int]) -> bytes:
    out = bytearray()
    for value in values:
        if value < 0:
            raise NbtError("VarInt value cannot be negative")
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
    return bytes(out)


def decode_varints(data: bytes, expected_count: int) -> list[int]:
    values: list[int] = []
    value = 0
    shift = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 35:
                raise NbtError("Schematic VarInt is too long")
            continue
        values.append(value)
        if len(values) == expected_count:
            return values
        value = 0
        shift = 0
    if len(values) != expected_count:
        raise NbtError(f"Expected {expected_count} VarInts, got {len(values)}")
    return values
