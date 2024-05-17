import hashlib
import unittest
from functools import cached_property
from unittest.mock import MagicMock, patch

from ionmapper.persistence.file_checksums import FileChecksums


class TestFileChecksums(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_file_path = MagicMock(name="file_path", spec=[])

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
        self.assertEqual(
            "checksum", self.mock_checksums._compute_checksum(native_tool="tool", hashlib_method=mock_hashlib_method)
        )
        mock_subprocess_run.assert_called_once_with(
            ["some/path", self.mock_file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        mock_shutil_which.assert_called_once_with("tool")

    @patch("builtins.open")
    @patch("shutil.which")
    def test_checksum_when_native_tool_not_available(self, mock_shutil_which, mock_open) -> None:
        mock_shutil_which.return_value = None
        mock_open.return_value.__enter__.return_value.read.return_value = b"content"
        mock_hashlib_method = MagicMock(name="mock_hashlib_method")

        self.assertEqual(
            mock_hashlib_method.return_value.hexdigest.return_value,
            self.mock_checksums._compute_checksum(native_tool="tool", hashlib_method=mock_hashlib_method),
        )

        mock_shutil_which.assert_called_once_with("tool")
        mock_open.assert_called_once_with(self.mock_file_path, "rb")
        mock_hashlib_method.assert_called_once_with(b"content")
        mock_hashlib_method.return_value.hexdigest.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
