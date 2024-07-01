import hashlib
import io
from pathlib import Path

import pytest
from pytest_mock import MockFixture

from depiction.persistence.file_checksums import FileChecksums


@pytest.fixture
def mock_file_path(mocker: MockFixture):
    return mocker.MagicMock(name="file_path", spec=Path)


@pytest.fixture
def mock_checksums(mock_file_path) -> FileChecksums:
    return FileChecksums(file_path=mock_file_path)


def test_file_path(mock_file_path, mock_checksums) -> None:
    assert mock_checksums.file_path == mock_file_path


def test_checksum_md5(mocker: MockFixture, mock_checksums: FileChecksums) -> None:
    mock_compute_checksum = mocker.patch.object(FileChecksums, "_compute_checksum")
    assert mock_checksums.checksum_md5 == mock_compute_checksum.return_value
    mock_compute_checksum.assert_called_once_with(hashlib_method=hashlib.md5)


def test_compute_checksum_sha1(mocker: MockFixture, mock_checksums: FileChecksums) -> None:
    mock_compute_checksum = mocker.patch.object(FileChecksums, "_compute_checksum")
    assert mock_checksums.checksum_sha1 == mock_compute_checksum.return_value
    mock_compute_checksum.assert_called_once_with(hashlib_method=hashlib.sha1)


def test_compute_checksum_sha256(mocker: MockFixture, mock_checksums: FileChecksums) -> None:
    mock_compute_checksum = mocker.patch.object(FileChecksums, "_compute_checksum")
    assert mock_checksums.checksum_sha256 == mock_compute_checksum.return_value
    mock_compute_checksum.assert_called_once_with(hashlib_method=hashlib.sha256)


def test_compute_checksum(mocker: MockFixture) -> None:
    mock_file_path = mocker.MagicMock(name="file_path", spec=Path)
    mock_file_path.open.return_value.__enter__.return_value = io.BytesIO(b"content")
    mock_hashlib_method = mocker.MagicMock(name="mock_hashlib_method")
    mock_checksums = FileChecksums(file_path=mock_file_path)

    checksum = mock_checksums._compute_checksum(hashlib_method=mock_hashlib_method)
    assert checksum == mock_hashlib_method.return_value.hexdigest.return_value

    mock_file_path.open.assert_called_once_with("rb")
    mock_hashlib_method.assert_called_once_with()
    mock_hashlib_method.return_value.update.assert_called_once_with(b"content")


if __name__ == "__main__":
    pytest.main()
