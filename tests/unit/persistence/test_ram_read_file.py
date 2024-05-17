import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np

from ionmapper.persistence import ImzmlModeEnum
from ionmapper.persistence.ram_read_file import RamReadFile


class TestRamReadFile(unittest.TestCase):
    def setUp(self):
        self.mock_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        self.mock_int_arr_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]])
        self.mock_coordinates = np.array([[0, 1, 2], [3, 4, 5]])

    @cached_property
    def mock_read_file(self) -> RamReadFile:
        return RamReadFile(
            mz_arr_list=self.mock_mz_arr_list,
            int_arr_list=self.mock_int_arr_list,
            coordinates=self.mock_coordinates,
        )

    @patch.object(RamReadFile, "get_reader")
    def test_reader(self, method_get_reader):
        mock_reader = MagicMock(name="mock_reader")
        method_get_reader.return_value = mock_reader

        with self.mock_read_file.reader() as reader:
            self.assertEqual(mock_reader, reader)
            mock_reader.close.assert_not_called()

        mock_reader.close.assert_called_once_with()
        method_get_reader.assert_called_once_with()

    @patch("ionmapper.persistence.ram_read_file.RamReader")
    def test_get_reader(self, construct_ram_reader):
        reader = self.mock_read_file.get_reader()
        construct_ram_reader.assert_called_once_with(
            mz_arr_list=self.mock_mz_arr_list,
            int_arr_list=self.mock_int_arr_list,
            coordinates=self.mock_coordinates,
        )
        self.assertEqual(construct_ram_reader.return_value, reader)

    def test_n_spectra(self):
        self.assertEqual(2, self.mock_read_file.n_spectra)

    def test_imzml_mode_when_continuous(self):
        self.assertEqual(ImzmlModeEnum.CONTINUOUS, self.mock_read_file.imzml_mode)

    def test_imzml_mode_when_processed(self):
        self.mock_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])
        self.assertEqual(ImzmlModeEnum.PROCESSED, self.mock_read_file.imzml_mode)

    def test_coordinates(self):
        np.testing.assert_array_equal(self.mock_coordinates, self.mock_read_file.coordinates)

    def test_coordinates_2d(self):
        np.testing.assert_array_equal(np.array([[0, 1], [3, 4]]), self.mock_read_file.coordinates_2d)


if __name__ == "__main__":
    unittest.main()
