from __future__ import annotations

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from depiction.persistence import ImzmlModeEnum, ImzmlWriteFile, ImzmlWriter

mock_path = Path("/dev/null/test.imzML")
mock_imzml_mode = ImzmlModeEnum.CONTINUOUS


@pytest.fixture
def mock_write_file() -> ImzmlWriteFile:
    return ImzmlWriteFile(path=mock_path, imzml_mode=mock_imzml_mode)


def test_imzml_file(mock_write_file: ImzmlWriteFile) -> None:
    assert mock_write_file.imzml_file == mock_path


def test_imzml_file_when_invalid() -> None:
    file = ImzmlWriteFile(path="/dev/null/test.txt", imzml_mode=mock_imzml_mode)
    with pytest.raises(ValueError):
        _ = file.imzml_file


def test_ibd_file(mock_write_file: ImzmlWriteFile) -> None:
    assert mock_write_file.ibd_file == Path("/dev/null/test.ibd")


def test_imzml_mode(mock_write_file: ImzmlWriteFile) -> None:
    assert mock_write_file.imzml_mode == mock_imzml_mode


def test_writer_when_success(mocker: MockerFixture, mock_write_file: ImzmlWriteFile) -> None:
    mock_imzml_file = mocker.MagicMock(name="mock_imzml_file", spec=Path)
    mock_imzml_file.exists.return_value = False
    mocker.patch.object(Path, "exists", return_value=False)
    mock_open = mocker.patch.object(ImzmlWriter, "open")
    with mock_write_file.writer() as writer:
        assert writer == mock_open.return_value
    mock_open.assert_called_once_with(path=mock_write_file.imzml_file, imzml_mode=mock_imzml_mode)
    mock_open.return_value.close.assert_called_once_with()


def test_writer_when_mode_x_file_exists(mocker: MockerFixture, mock_write_file: ImzmlWriteFile) -> None:
    mocker.patch.object(Path, "exists", return_value=True)
    with pytest.raises(ValueError):
        with mock_write_file.writer():
            pass


def test_writer_when_mode_w_file_exists(mocker: MockerFixture) -> None:
    mock_write_file = ImzmlWriteFile(path=mock_path, imzml_mode=mock_imzml_mode, write_mode="w")
    mocker.patch.object(Path, "exists", return_value=True)
    mock_unlink = mocker.patch.object(Path, "unlink")
    mock_open = mocker.patch.object(ImzmlWriter, "open")
    with mock_write_file.writer() as writer:
        assert writer == mock_open.return_value
    assert mock_unlink.mock_calls == [mocker.call(), mocker.call()]
    mock_open.assert_called_once_with(path=mock_write_file.imzml_file, imzml_mode=mock_imzml_mode)
    mock_open.return_value.close.assert_called_once_with()


def test_repr(mock_write_file: ImzmlWriteFile) -> None:
    assert (
        repr(mock_write_file) == f"ImzmlWriteFile(path={mock_path!r}, imzml_mode={mock_imzml_mode!r}, write_mode='x')"
    )


if __name__ == "__main__":
    pytest.main()
