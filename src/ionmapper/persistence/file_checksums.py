import hashlib
import shutil
import subprocess
from functools import cached_property
from typing import Any, Optional


class FileChecksums:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    @property
    def file_path(self) -> str:
        return self._file_path

    @cached_property
    def checksum_md5(self) -> str:
        """The MD5 checksum of the file."""
        return self._compute_checksum(native_tool="md5sum", hashlib_method=hashlib.md5)

    @cached_property
    def checksum_sha1(self) -> str:
        """The SHA-1 checksum of the file."""
        return self._compute_checksum(native_tool="sha1sum", hashlib_method=hashlib.sha1)

    @cached_property
    def checksum_sha256(self) -> str:
        """The SHA-256 checksum of the file."""
        return self._compute_checksum(native_tool="sha256sum", hashlib_method=hashlib.sha256)

    def _compute_checksum(self, native_tool: str, hashlib_method: Any) -> str:
        """Returns the checksum of the file using the native tool, or falls back to hashlib if the
        native tool is not available.
        :param native_tool: the name of the binary tool to use, e.g. `md5sum`
        :param hashlib_method: the hashlib method to use, e.g. `hashlib.md5`
        """
        # default to the native unix tool since these are usually much faster than python's hashlib
        checksum = self._compute_checksum_native_tool(binary_name=native_tool, file=self.file_path)
        if checksum is not None:
            return checksum

        # fallback to the hashlib method
        with open(self.file_path, "rb") as f:
            return hashlib_method(f.read()).hexdigest()

    def _compute_checksum_native_tool(self, binary_name: str, file: str) -> Optional[str]:
        """Returns the checksum of the file using the native tool, or None if the tool is not available.
        The checksum is returned as a string in lower case.
        :param binary_name: the name of the binary tool to use
        :param file: the file to compute the checksum for
        """
        binary_path = shutil.which(binary_name)
        if binary_path is None:
            return None
        else:
            result = subprocess.run(
                [binary_path, file],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            return result.stdout.split()[0].lower()
