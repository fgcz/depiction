from __future__ import annotations

from enum import Enum


class Compression(Enum):
    Uncompressed = "uncompressed"
    Zlib = "zlib"
