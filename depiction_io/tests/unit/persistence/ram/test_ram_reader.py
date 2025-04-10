import pytest
import numpy as np

from depiction_io.persistence import ImzmlModeEnum
from depiction_io.persistence.ram.ram_reader import RamReader


@pytest.fixture
def mock_data():
    """Fixture to create test data for RamReader tests"""
    return {
        "mz_arr_list": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "int_arr_list": np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]]),
        "coordinates": np.array([[0, 1, 2], [3, 4, 5]]),
    }


@pytest.fixture
def ram_reader(mock_data):
    """Fixture to create a RamReader instance for testing"""
    reader = RamReader(
        mz_arr_list=mock_data["mz_arr_list"],
        int_arr_list=mock_data["int_arr_list"],
        coordinates=mock_data["coordinates"],
    )
    yield reader
    # This handles the teardown (equivalent to the close method in the original)
    reader.close()


def test_enter(ram_reader):
    """Test the __enter__ method"""
    reader = ram_reader.__enter__()
    assert ram_reader == reader


def test_close(ram_reader):
    """Test the close method"""
    ram_reader.close()


def test_imzml_mode_when_continuous(ram_reader):
    """Test imzml_mode property when data is continuous"""
    assert ImzmlModeEnum.CONTINUOUS == ram_reader.imzml_mode


def test_imzml_mode_when_processed(mock_data):
    """Test imzml_mode property when data is processed"""
    # Create a different mz_arr_list with non-continuous values
    processed_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])
    reader = RamReader(
        mz_arr_list=processed_mz_arr_list,
        int_arr_list=mock_data["int_arr_list"],
        coordinates=mock_data["coordinates"],
    )
    assert ImzmlModeEnum.PROCESSED == reader.imzml_mode


def test_n_spectra(ram_reader, mock_data):
    """Test n_spectra property"""
    assert 2 == ram_reader.n_spectra
    assert len(mock_data["mz_arr_list"]) == ram_reader.n_spectra


def test_coordinates(ram_reader, mock_data):
    """Test coordinates property"""
    np.testing.assert_array_equal(mock_data["coordinates"], ram_reader.coordinates)


def test_coordinates_2d(ram_reader, mock_data):
    """Test coordinates_2d property"""
    np.testing.assert_array_equal(mock_data["coordinates"][:, :2], ram_reader.coordinates_2d)


def test_get_spectrum(ram_reader, mock_data):
    """Test get_spectrum method"""
    mz_arr, int_arr = ram_reader.get_spectrum(1)
    np.testing.assert_array_equal(mock_data["mz_arr_list"][1], mz_arr)
    np.testing.assert_array_equal(mock_data["int_arr_list"][1], int_arr)


def test_get_spectra_when_continuous(ram_reader, mock_data):
    """Test get_spectra method with continuous data"""
    mz_arr_list, int_arr_list = ram_reader.get_spectra([0, 1])
    np.testing.assert_array_equal(mock_data["mz_arr_list"], mz_arr_list)
    np.testing.assert_array_equal(mock_data["int_arr_list"], int_arr_list)


def test_get_spectra_when_processed(mock_data):
    """Test get_spectra method with processed data"""
    processed_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])
    reader = RamReader(
        mz_arr_list=processed_mz_arr_list,
        int_arr_list=mock_data["int_arr_list"],
        coordinates=mock_data["coordinates"],
    )
    mz_arr_list, int_arr_list = reader.get_spectra([0, 1])
    np.testing.assert_array_equal(processed_mz_arr_list, mz_arr_list)
    np.testing.assert_array_equal(mock_data["int_arr_list"], int_arr_list)


def test_get_spectrum_mz(ram_reader, mock_data):
    """Test get_spectrum_mz method"""
    np.testing.assert_array_equal(mock_data["mz_arr_list"][1], ram_reader.get_spectrum_mz(1))


def test_get_spectrum_int(ram_reader, mock_data):
    """Test get_spectrum_int method"""
    np.testing.assert_array_equal(mock_data["int_arr_list"][1], ram_reader.get_spectrum_int(1))


def test_get_spectrum_n_points(ram_reader):
    """Test get_spectrum_n_points method"""
    assert 3 == ram_reader.get_spectrum_n_points(1)


def test_get_spectrum_metadata(ram_reader, mock_data):
    """Test get_spectrum_metadata method"""
    metadata = ram_reader.get_spectrum_metadata(1)

    assert "i_spectrum" in metadata
    assert "coordinates" in metadata
    assert 1 == metadata["i_spectrum"]
    np.testing.assert_array_equal(mock_data["coordinates"][1], metadata["coordinates"])


def test_get_spectra_metadata(ram_reader, mock_data):
    """Test get_spectra_metadata method"""
    metadata = ram_reader.get_spectra_metadata([0, 1])

    assert 2 == len(metadata)
    # Check individual fields instead of comparing whole dictionaries
    assert 0 == metadata[0]["i_spectrum"]
    assert 1 == metadata[1]["i_spectrum"]
    np.testing.assert_array_equal(mock_data["coordinates"][0], metadata[0]["coordinates"])
    np.testing.assert_array_equal(mock_data["coordinates"][1], metadata[1]["coordinates"])


def test_get_spectra_mz_range(ram_reader):
    """Test get_spectra_mz_range method"""
    mz_range = ram_reader.get_spectra_mz_range([0, 1])
    assert (1.0, 3.0) == mz_range
