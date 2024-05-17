import unittest
from unittest.mock import patch, MagicMock, call, ANY

import numpy as np

from ionmapper.tools.cutout_rectangular_region_imzml import (
    CutoutRectangularRegionImzml,
)


class TestCutoutRectangularRegionImzml(unittest.TestCase):
    def setUp(self):
        self.mock_read_file = MagicMock(name="mock_read_file")

    @patch.object(CutoutRectangularRegionImzml, "from_absolute_ranges")
    def test_from_relative_ranges(self, method_from_absolute_ranges):
        self.mock_read_file.coordinates_2d = np.array([[x, y] for x in range(100) for y in range(3, 100)])
        mock_x_range_rel = (0.2, 0.8)
        mock_y_range_rel = (0.3, 0.7)

        instance = CutoutRectangularRegionImzml.from_relative_ranges(
            read_file=self.mock_read_file,
            x_range_rel=mock_x_range_rel,
            y_range_rel=mock_y_range_rel,
        )

        method_from_absolute_ranges.assert_called_once_with(
            read_file=self.mock_read_file,
            x_range_abs=(20, 80),
            y_range_abs=(32, 71),
            verbose=True,
        )
        self.assertEqual(method_from_absolute_ranges.return_value, instance)

    def test_from_absolute_ranges(self):
        mock_x_range_abs = MagicMock(name="mock_x_range_abs")
        mock_y_range_abs = MagicMock(name="mock_y_range_abs")

        instance = CutoutRectangularRegionImzml.from_absolute_ranges(
            read_file=self.mock_read_file,
            x_range_abs=mock_x_range_abs,
            y_range_abs=mock_y_range_abs,
        )

        self.assertIsInstance(instance, CutoutRectangularRegionImzml)

    def test_write_imzml(self):
        mock_write_file = MagicMock(name="mock_write_file")
        mock_writer = MagicMock(name="mock_writer")
        mock_write_file.writer.return_value.__enter__.return_value = mock_writer
        mock_reader = MagicMock(name="mock_reader")
        mock_reader.coordinates_2d = np.array([[x, y] for x in range(10) for y in range(3, 10)])
        self.mock_read_file.reader.return_value.__enter__.return_value = mock_reader

        mock_x_range_abs = (0, 2)
        mock_y_range_abs = (5, 7)

        cutout = CutoutRectangularRegionImzml.from_absolute_ranges(
            read_file=self.mock_read_file,
            x_range_abs=mock_x_range_abs,
            y_range_abs=mock_y_range_abs,
        )
        cutout.write_imzml(write_file=mock_write_file)

        self.assertListEqual(
            [call.copy_spectra(reader=mock_reader, spectra_indices=ANY)],
            mock_writer.mock_calls,
        )

        np.testing.assert_array_equal([2, 3, 4, 9, 10, 11, 16, 17, 18], mock_writer.mock_calls[0][2]["spectra_indices"])


if __name__ == "__main__":
    unittest.main()
