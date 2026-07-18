"""Minimal GGUF metadata reader.

Only parses the key-value metadata header (not tensor data) to pull out
`general.architecture` and detect an embedded `tokenizer.chat_template`.
Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FORMATS = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<B",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, value_type: int) -> None:
    """Read past a value without materializing it (arrays included)."""
    if value_type == _STRING:
        _read_string(f)
    elif value_type == _ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (length,) = struct.unpack("<Q", f.read(8))
        for _ in range(length):
            _skip_value(f, elem_type)
    else:
        fmt = _SCALAR_FORMATS[value_type]
        f.read(struct.calcsize(fmt))


def read_gguf_metadata(path: Path, wanted_keys: set[str]) -> dict[str, str]:
    """Return {key: value} for any of `wanted_keys` found (string values only)."""
    found: dict[str, str] = {}
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return found
        f.read(4)  # version, unused
        f.read(8)  # tensor_count, unused
        (kv_count,) = struct.unpack("<Q", f.read(8))
        for _ in range(kv_count):
            key = _read_string(f)
            (value_type,) = struct.unpack("<I", f.read(4))
            if key in wanted_keys and value_type == _STRING:
                found[key] = _read_string(f)
            else:
                _skip_value(f, value_type)
            if len(found) == len(wanted_keys):
                break
    return found
