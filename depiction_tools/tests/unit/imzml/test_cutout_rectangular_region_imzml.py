import numpy as np
import pytest
from unittest.mock import call, ANY

from depiction_tools.imzml.cutout_rectangular_region import CutoutRectangularRegionImzml


@pytest.fixture
def mock_read_file(mocker):
    return mocker.MagicMock(name="mock_read_file")


def test_from_relative_ranges(mock_read_file, mocker):
    # Patch the from_absolute_ranges method
    method_from_absolute_ranges = mocker.patch.object(CutoutRectangularRegionImzml, "from_absolute_ranges")

    mock_read_file.coordinates_2d = np.array([[x, y] for x in range(100) for y in range(3, 100)])
    mock_x_range_rel = (0.2, 0.8)
    mock_y_range_rel = (0.3, 0.7)

    instance = CutoutRectangularRegionImzml.from_relative_ranges(
        read_file=mock_read_file,
        x_range_rel=mock_x_range_rel,
        y_range_rel=mock_y_range_rel,
    )

    method_from_absolute_ranges.assert_called_once_with(
        read_file=mock_read_file,
        x_range_abs=(20, 80),
        y_range_abs=(32, 71),
        verbose=True,
    )
    assert method_from_absolute_ranges.return_value == instance


def test_from_absolute_ranges(mock_read_file):
    mock_x_range_abs = (0, 10)
    mock_y_range_abs = (0, 10)

    instance = CutoutRectangularRegionImzml.from_absolute_ranges(
        read_file=mock_read_file,
        x_range_abs=mock_x_range_abs,
        y_range_abs=mock_y_range_abs,
    )

    assert isinstance(instance, CutoutRectangularRegionImzml)


def test_write_imzml(mock_read_file, mocker):
    mock_write_file = mocker.MagicMock(name="mock_write_file")
    mock_writer = mocker.MagicMock(name="mock_writer")
    mock_write_file.writer.return_value.__enter__.return_value = mock_writer
    mock_reader = mocker.MagicMock(name="mock_reader")
    mock_reader.coordinates_2d = np.array([[x, y] for x in range(10) for y in range(3, 10)])
    mock_read_file.reader.return_value.__enter__.return_value = mock_reader

    mock_x_range_abs = (0, 2)
    mock_y_range_abs = (5, 7)

    cutout = CutoutRectangularRegionImzml.from_absolute_ranges(
        read_file=mock_read_file,
        x_range_abs=mock_x_range_abs,
        y_range_abs=mock_y_range_abs,
    )
    cutout.write_imzml(write_file=mock_write_file)

    assert mock_writer.mock_calls == [call.copy_spectra(reader=mock_reader, spectra_indices=ANY)]

    # Extract the actual spectra_indices passed to copy_spectra
    actual_indices = mock_writer.mock_calls[0][2]["spectra_indices"]
    expected_indices = np.array([2, 3, 4, 9, 10, 11, 16, 17, 18])

    np.testing.assert_array_equal(expected_indices, actual_indices)
