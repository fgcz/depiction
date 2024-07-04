import unittest
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from depiction.persistence import ImzmlReadFile, ImzmlModeEnum


@pytest.fixture()
def mock_path() -> Path:
    return Path("/dev/null/mock_path.imzML")


@pytest.fixture()
def mock_read_file(mock_path) -> ImzmlReadFile:
    return ImzmlReadFile(path=mock_path)


def test_imzml_file(mock_read_file, mock_path) -> None:
    assert mock_read_file.imzml_file == mock_path


def test_imzml_file_when_case_insensitive() -> None:
    mock_path = "/dev/null/mock_path.IMZML"
    mock_read_file = ImzmlReadFile(path=mock_path)
    assert mock_read_file.imzml_file == Path(mock_path)


def test_imzml_file_when_invalid_path() -> None:
    mock_path = "/dev/null/mock_path.txt"
    read_file = ImzmlReadFile(path=mock_path)
    with pytest.raises(ValueError) as error:
        _ = read_file.imzml_file
    assert "Expected .imzML file, got" in str(error)


def test_ibd_file(mock_read_file) -> None:
    assert mock_read_file.ibd_file == Path("/dev/null/mock_path.ibd")


def test_ibd_file_when_case_insensitive() -> None:
    mock_path = "/dev/null/mock_path.IMZML"
    mock_read_file = ImzmlReadFile(path=mock_path)
    assert mock_read_file.ibd_file == Path("/dev/null/mock_path.ibd")


def test_ibd_file_when_invalid_path() -> None:
    mock_path = "/dev/null/mock_path.txt"
    read_file = ImzmlReadFile(path=mock_path)
    with pytest.raises(ValueError) as error:
        _ = read_file.ibd_file
    assert "Expected .imzML file, got" in str(error)


def test_reader_when_normal(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_reader = mocker.MagicMock(name="mock_reader")
    method_get_reader = mocker.patch.object(ImzmlReadFile, "get_reader", return_value=mock_reader)
    with mock_read_file.reader() as reader:
        method_get_reader.assert_called_once_with()
        assert mock_reader == reader
    mock_reader.close.assert_called_once_with()


def test_reader_when_exception(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> NoReturn:
    mock_reader = mocker.MagicMock(name="mock_reader")
    mocker.patch.object(ImzmlReadFile, "get_reader", return_value=mock_reader)
    with pytest.raises(ValueError) as error, mock_read_file.reader() as reader:
        assert mock_reader == reader
        raise ValueError("mock_error")
    assert "mock_error" in str(error)
    mock_reader.close.assert_called_once_with()


def test_get_reader(mocker: MockerFixture, mock_path: Path, mock_read_file: ImzmlReadFile) -> None:
    mock_reader = mocker.MagicMock(name="mock_reader", mzPrecision="d", intensityPrecision="i")
    mock_imzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_imzml_parser.return_value.__enter__.return_value.portable_spectrum_reader.return_value = mock_reader
    reader = mock_read_file.get_reader()
    mock_imzml_parser.assert_called_once_with(mock_path)
    assert reader.imzml_path == Path(mock_read_file._path)
    assert reader._mz_bytes == 8
    assert reader._int_bytes == 4


def test_n_spectra(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "_cached_properties", {"n_spectra": 123})
    assert mock_read_file.n_spectra == 123


def test_imzml_mode(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "_cached_properties", {"imzml_mode": "mock_mode"})
    assert mock_read_file.imzml_mode == "mock_mode"


def test_coordinates(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "_cached_properties", {"coordinates": "mock_coordinates"})
    assert mock_read_file.coordinates == "mock_coordinates"


def test_coordinates_2d(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "_cached_properties", {"coordinates": np.array([[1, 2, 3], [4, 5, 6]])})
    np.testing.assert_array_equal(np.array([[1, 2], [4, 5]]), mock_read_file.coordinates_2d)


def test_compact_metadata(mock_read_file: ImzmlReadFile) -> None:
    mock_coordinates = np.array([[1, 2, 3], [3, 3, 3], [1 + 3, 2 + 4, 3 + 5]])
    with (
        patch.object(ImzmlReadFile, "n_spectra", 15),
        patch.object(ImzmlReadFile, "imzml_mode", ImzmlModeEnum.PROCESSED),
        patch.object(ImzmlReadFile, "coordinates", mock_coordinates),
        patch.object(ImzmlReadFile, "imzml_file", "mock_imzml_file_path"),
        patch.object(ImzmlReadFile, "ibd_file", "mock_ibd_file_path"),
    ):
        assert mock_read_file.compact_metadata == {
            "n_spectra": 15,
            "imzml_mode": "PROCESSED",
            "coordinate_extent": [4, 5, 6],
            "imzml_file": "mock_imzml_file_path",
            "ibd_file": "mock_ibd_file_path",
        }


def test_metadata_checksums_when_none(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_pyimzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {}
    assert mock_read_file.metadata_checksums == {}


def test_metadata_checksums_when_md5_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_pyimzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {"ibd MD5": "ABCD"}
    assert mock_read_file.metadata_checksums == {"md5": "abcd"}


def test_metadata_checksums_when_sha1_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_pyimzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {"ibd SHA-1": "ABCD"}
    assert mock_read_file.metadata_checksums == {"sha1": "abcd"}


def test_metadata_checksums_when_sha256_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_pyimzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {"ibd SHA-256": "ABCD"}
    assert mock_read_file.metadata_checksums == {"sha256": "abcd"}


def test_metadata_checksums_when_both_checksums_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_pyimzml_parser = mocker.patch("pyimzml.ImzMLParser.ImzMLParser")
    mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {
        "ibd MD5": "ABCD",
        "ibd SHA-1": "EF00",
    }
    assert mock_read_file.metadata_checksums == {"md5": "abcd", "sha1": "ef00"}


def test_ibd_checksums(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_file_checksums = mocker.patch("depiction.persistence.imzml_read_file.FileChecksums")
    assert mock_file_checksums.return_value == mock_read_file.ibd_checksums
    mock_file_checksums.assert_called_once_with(file_path=mock_read_file.ibd_file)


def test_is_checksum_valid_when_no_checksums_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={})
    assert mock_read_file.is_checksum_valid is None


def test_is_checksum_valid_when_md5_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={"md5": "1234"})
    mock_checksums = mocker.patch.object(ImzmlReadFile, "ibd_checksums")
    mock_checksums.checksum_md5 = "1234"
    assert mock_read_file.is_checksum_valid


def test_is_checksum_valid_when_sha1_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={"sha1": "1234"})
    mock_checksums = mocker.patch.object(ImzmlReadFile, "ibd_checksums")
    mock_checksums.checksum_sha1 = "1234"
    assert mock_read_file.is_checksum_valid


def test_is_checksum_valid_when_sha256_available(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={"sha256": "1234"})
    mock_checksums = mocker.patch.object(ImzmlReadFile, "ibd_checksums")
    mock_checksums.checksum_sha256 = "1234"
    assert mock_read_file.is_checksum_valid


def test_is_checksum_valid_when_multiple_checksums_available(
    mocker: MockerFixture, mock_read_file: ImzmlReadFile
) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={"md5": "1234", "sha1": "5678"})
    mock_checksums = mocker.patch.object(ImzmlReadFile, "ibd_checksums")
    mock_checksums.checksum_sha1 = "5678"
    mock_checksums.checksum_md5 = "1234"
    assert mock_read_file.is_checksum_valid


def test_is_checksum_valid_when_invalid_metadata_checksums(
    mocker: MockerFixture, mock_read_file: ImzmlReadFile
) -> None:
    mocker.patch.object(ImzmlReadFile, "metadata_checksums", new={"abc": "1234"})
    with pytest.raises(ValueError) as error:
        _ = mock_read_file.is_checksum_valid
    assert "Invalid metadata_checksums" in str(error)


def test_file_sizes_bytes(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_imzml_file = mocker.patch.object(ImzmlReadFile, "imzml_file")
    mock_ibd_file = mocker.patch.object(ImzmlReadFile, "ibd_file")
    mock_imzml_file.stat.return_value.st_size = 1234
    mock_ibd_file.stat.return_value.st_size = 5678
    assert mock_read_file.file_sizes_bytes == {"imzml": 1234, "ibd": 5678}


def test_get_file_sizes_mb(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_imzml_file = mocker.patch.object(ImzmlReadFile, "imzml_file")
    mock_ibd_file = mocker.patch.object(ImzmlReadFile, "ibd_file")
    mock_imzml_file.stat.return_value.st_size = round(2.5 * 1024**2)
    mock_ibd_file.stat.return_value.st_size = round(3.5 * 1024**2)
    assert mock_read_file.file_sizes_mb["imzml"] == pytest.approx(2.5, 1e-8)
    assert mock_read_file.file_sizes_mb["ibd"] == pytest.approx(3.5, 1e-8)


def test_summary_when_processed(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "imzml_mode", ImzmlModeEnum.PROCESSED)
    mocker.patch.object(ImzmlReadFile, "n_spectra", 15)
    mocker.patch.object(ImzmlReadFile, "is_checksum_valid", True)
    mocker.patch.object(ImzmlReadFile, "file_sizes_mb", {"imzml": 1.5, "ibd": 2.5})
    assert mock_read_file.summary() == (
        "imzML file: /dev/null/mock_path.imzML (1.50 MB)\n"
        "ibd file: /dev/null/mock_path.ibd (2.50 MB)\n"
        "imzML mode: PROCESSED\n"
        "n_spectra: 15\n"
        "is_checksum_valid: True\n"
    )


def test_summary_when_continuous(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mocker.patch.object(ImzmlReadFile, "imzml_mode", ImzmlModeEnum.CONTINUOUS)
    mocker.patch.object(ImzmlReadFile, "n_spectra", 15)
    mocker.patch.object(ImzmlReadFile, "is_checksum_valid", True)
    mocker.patch.object(ImzmlReadFile, "file_sizes_mb", {"imzml": 1.5, "ibd": 2.5})
    mock_reader = mocker.MagicMock(name="mock_reader")
    mock_reader.get_spectrum_mz.return_value = np.array([1, 2, 3])
    mocker.patch.object(ImzmlReadFile, "reader").return_value.__enter__.return_value = mock_reader
    assert mock_read_file.summary() == (
        "imzML file: /dev/null/mock_path.imzML (1.50 MB)\n"
        "ibd file: /dev/null/mock_path.ibd (2.50 MB)\n"
        "imzML mode: CONTINUOUS\n"
        "n_spectra: 15\n"
        "is_checksum_valid: True\n"
        "m/z range: 1.00 - 3.00 (3 bins)\n"
    )


def test_print_summary(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_print = mocker.patch("builtins.print")
    mock_summary = mocker.patch.object(ImzmlReadFile, "summary")
    mock_read_file.print_summary()
    mock_summary.assert_called_once_with(checksums=True)
    mock_print.assert_called_once_with(mock_summary.return_value, file=None)


def test_copy_to(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_output_file = mocker.MagicMock(name="mock_output_file")
    mock_copy = mocker.patch("shutil.copy")
    mock_read_file.copy_to(mock_output_file)
    assert mock_copy.mock_calls == [
        mocker.call(mock_read_file.imzml_file, mock_output_file),
        mocker.call(mock_read_file.ibd_file, mock_output_file.with_suffix.return_value),
    ]
    mock_output_file.with_suffix.assert_called_once_with(".ibd")


def test_cached_properties(mocker: MockerFixture, mock_read_file: ImzmlReadFile) -> None:
    mock_reader = MagicMock(name="mock_reader")
    mocker.patch.object(ImzmlReadFile, "reader").return_value.__enter__.return_value = mock_reader
    assert mock_read_file._cached_properties == {
        "n_spectra": mock_reader.n_spectra,
        "imzml_mode": mock_reader.imzml_mode,
        "coordinates": mock_reader.coordinates,
    }


def test_repr(mock_path: Path, mock_read_file: ImzmlReadFile) -> None:
    assert repr(mock_read_file) == f"ImzmlReadFile('{mock_path}')"


def test_str(mock_path: Path, mock_read_file: ImzmlReadFile) -> None:
    assert str(mock_read_file) == f"ImzmlReadFile('{mock_path}')"


if __name__ == "__main__":
    unittest.main()
