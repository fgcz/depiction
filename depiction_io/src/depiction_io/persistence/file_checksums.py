from __future__ import annotations

import hashlib
from functools import cached_property
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pathlib import Path


class FileChecksums:
    """Calculates the checksums of a file.
    :param file_path: the path to the file
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    @property
    def file_path(self) -> Path:
        """The path to the file."""
        return self._file_path

    @cached_property
    def checksum_md5(self) -> str:
        """The MD5 checksum of the file."""
        return self._compute_checksum(hashlib_method=hashlib.md5)

    @cached_property
    def checksum_sha1(self) -> str:
        """The SHA-1 checksum of the file."""
        return self._compute_checksum(hashlib_method=hashlib.sha1)

    @cached_property
    def checksum_sha256(self) -> str:
        """The SHA-256 checksum of the file."""
        return self._compute_checksum(hashlib_method=hashlib.sha256)

    def _compute_checksum(self, hashlib_method: Callable[[], hashlib._Hash]) -> str:
        """Returns the checksum of the file using the native tool, or falls back to hashlib if the
        native tool is not available.
        :param hashlib_method: the hashlib method to use, e.g. `hashlib.md5`
        """
        hasher = hashlib_method()
        with self._file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(16384), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
