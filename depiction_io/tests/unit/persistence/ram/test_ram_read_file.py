import numpy as np
import pytest

from depiction_io.persistence import ImzmlModeEnum
from depiction_io.persistence.ram.ram_read_file import RamReadFile


@pytest.fixture
def mock_mz_arr_list():
    return np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


@pytest.fixture
def mock_int_arr_list():
    return np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]])


@pytest.fixture
def mock_coordinates():
    return np.array([[0, 1, 2], [3, 4, 5]])


@pytest.fixture
def mock_read_file(mock_mz_arr_list, mock_int_arr_list, mock_coordinates):
    return RamReadFile(
        mz_arr_list=mock_mz_arr_list,
        int_arr_list=mock_int_arr_list,
        coordinates=mock_coordinates,
    )


def test_reader(mocker, mock_read_file):
    mock_reader = mocker.MagicMock(name="mock_reader")
    mocker.patch.object(mock_read_file, "get_reader", return_value=mock_reader)

    with mock_read_file.reader() as reader:
        assert mock_reader == reader
        mock_reader.close.assert_not_called()

    mock_reader.close.assert_called_once_with()
    mock_read_file.get_reader.assert_called_once_with()


def test_get_reader(mocker, mock_read_file, mock_mz_arr_list, mock_int_arr_list, mock_coordinates):
    ram_reader_mock = mocker.patch("depiction_io.persistence.ram.ram_read_file.RamReader")

    reader = mock_read_file.get_reader()

    ram_reader_mock.assert_called_once_with(
        mz_arr_list=mock_mz_arr_list,
        int_arr_list=mock_int_arr_list,
        coordinates=mock_coordinates,
    )
    assert ram_reader_mock.return_value == reader


def test_n_spectra(mock_read_file):
    assert 2 == mock_read_file.n_spectra


def test_imzml_mode_when_continuous(mock_read_file):
    assert ImzmlModeEnum.CONTINUOUS == mock_read_file.imzml_mode


def test_imzml_mode_when_processed(mock_mz_arr_list, mock_int_arr_list, mock_coordinates):
    # Create a new array with different mz values to make it 'processed' mode
    processed_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])

    read_file = RamReadFile(
        mz_arr_list=processed_mz_arr_list,
        int_arr_list=mock_int_arr_list,
        coordinates=mock_coordinates,
    )

    assert ImzmlModeEnum.PROCESSED == read_file.imzml_mode


def test_coordinates(mock_read_file, mock_coordinates):
    np.testing.assert_array_equal(mock_coordinates, mock_read_file.coordinates)


def test_coordinates_2d(mock_read_file):
    expected_coordinates_2d = np.array([[0, 1], [3, 4]])
    np.testing.assert_array_equal(expected_coordinates_2d, mock_read_file.coordinates_2d)
