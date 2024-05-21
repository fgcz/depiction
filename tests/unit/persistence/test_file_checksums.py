import hashlib
import unittest
from functools import cached_property
from pathlib import Path
from unittest.mock import MagicMock, patch

from ionplotter.persistence.file_checksums import FileChecksums


class TestFileChecksums(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_file_path = MagicMock(name="file_path", spec=Path)

    @cached_property
    def mock_checksums(self) -> FileChecksums:
        return FileChecksums(file_path=self.mock_file_path)

    def test_file_path(self) -> None:
        self.assertEqual(self.mock_file_path, self.mock_checksums.file_path)

    @patch.object(FileChecksums, "_compute_checksum")
    def test_checksum_md5(self, method_compute_checksum) -> None:
        self.assertEqual(method_compute_checksum.return_value, self.mock_checksums.checksum_md5)
        method_compute_checksum.assert_called_once_with(native_tool="md5sum", hashlib_method=hashlib.md5)

    @patch.object(FileChecksums, "_compute_checksum")
    def test_checksum_sha1(self, method_compute_checksum) -> None:
        self.assertEqual(method_compute_checksum.return_value, self.mock_checksums.checksum_sha1)
        method_compute_checksum.assert_called_once_with(native_tool="sha1sum", hashlib_method=hashlib.sha1)

    @patch.object(FileChecksums, "_compute_checksum")
    def test_checksum_sha256(self, method_compute_checksum) -> None:
        self.assertEqual(method_compute_checksum.return_value, self.mock_checksums.checksum_sha256)
        method_compute_checksum.assert_called_once_with(native_tool="sha256sum", hashlib_method=hashlib.sha256)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_checksum_when_native_tool_available(self, mock_subprocess_run, mock_shutil_which) -> None:
        mock_shutil_which.return_value = "some/path"
        mock_subprocess_run.return_value.stdout = "checksum"
        mock_hashlib_method = MagicMock(name="mock_hashlib_method")
        self.mock_file_path = Path("/dev/null/hello")
        self.assertEqual(
            "checksum", self.mock_checksums._compute_checksum(native_tool="tool", hashlib_method=mock_hashlib_method)
        )
        mock_subprocess_run.assert_called_once_with(
            ["some/path", "/dev/null/hello"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        mock_shutil_which.assert_called_once_with("tool")

    @patch("shutil.which")
    def test_checksum_when_native_tool_not_available(self, mock_shutil_which) -> None:
        mock_shutil_which.return_value = None
        self.mock_file_path.read_bytes.return_value = b"content"
        mock_hashlib_method = MagicMock(name="mock_hashlib_method")

        self.assertEqual(
            mock_hashlib_method.return_value.hexdigest.return_value,
            self.mock_checksums._compute_checksum(native_tool="tool", hashlib_method=mock_hashlib_method),
        )

        mock_shutil_which.assert_called_once_with("tool")
        self.mock_file_path.read_bytes.assert_called_once_with()
        mock_hashlib_method.assert_called_once_with(b"content")
        mock_hashlib_method.return_value.hexdigest.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()