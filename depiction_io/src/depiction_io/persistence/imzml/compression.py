from __future__ import annotations

from enum import Enum


class Compression(str, Enum):
    Uncompressed = "uncompressed"
    Zlib = "zlib"
