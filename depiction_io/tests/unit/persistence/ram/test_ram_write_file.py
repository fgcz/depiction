from typing import NoReturn

import pytest

from depiction_io.persistence.ram.ram_write_file import RamWriteFile


@pytest.fixture
def mock_imzml_mode(mocker):
    return mocker.MagicMock(name="mock_imzml_mode")


@pytest.fixture
def mock_write_file(mock_imzml_mode):
    return RamWriteFile(imzml_mode=mock_imzml_mode)


def test_imzml_mode(mock_write_file, mock_imzml_mode):
    assert mock_imzml_mode == mock_write_file.imzml_mode


def test_add_spectrum(mock_write_file, mocker):
    mock_mz_arr = mocker.MagicMock(name="mock_mz_arr")
    mock_int_arr = mocker.MagicMock(name="mock_int_arr")
    mock_coordinates = mocker.MagicMock(name="mock_coordinates")

    with mock_write_file.writer() as writer:
        writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)

    assert [mock_mz_arr] == mock_write_file._mz_arr_list
    assert [mock_int_arr] == mock_write_file._int_arr_list
    assert [mock_coordinates] == mock_write_file._coordinates_list


@pytest.mark.skip(reason="Not implemented")
def test_copy_spectra(self) -> NoReturn:
    raise NotImplementedError


def test_to_read_file(mock_write_file, mocker):
    mock_mz_arr_list = mocker.MagicMock(name="mock_mz_arr_list", **{"copy.return_value": "x"})
    mock_int_arr_list = mocker.MagicMock(name="mock_int_arr_list", **{"copy.return_value": "y"})
    mock_coordinates_list = mocker.MagicMock(name="mock_coordinates_list", **{"copy.return_value": "z"})

    mock_write_file._mz_arr_list = mock_mz_arr_list
    mock_write_file._int_arr_list = mock_int_arr_list
    mock_write_file._coordinates_list = mock_coordinates_list

    read_file = mock_write_file.to_read_file()

    assert "x" == read_file._mz_arr_list
    assert "y" == read_file._int_arr_list
    assert "z" == read_file._coordinates
