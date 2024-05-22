import unittest
from functools import cached_property
from unittest.mock import ANY

import numpy as np

from depiction.persistence import ImzmlModeEnum
from depiction.persistence.ram_reader import RamReader


class TestRamReader(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        self.mock_int_arr_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]])
        self.mock_coordinates = np.array([[0, 1, 2], [3, 4, 5]])

    @cached_property
    def mock_ram_reader(self) -> RamReader:
        return RamReader(
            mz_arr_list=self.mock_mz_arr_list,
            int_arr_list=self.mock_int_arr_list,
            coordinates=self.mock_coordinates,
        )

    def test_enter(self) -> None:
        reader = self.mock_ram_reader.__enter__()
        self.assertEqual(self.mock_ram_reader, reader)

    def test_close(self) -> None:
        self.mock_ram_reader.close()

    def test_imzml_mode_when_continuous(self) -> None:
        self.assertEqual(ImzmlModeEnum.CONTINUOUS, self.mock_ram_reader.imzml_mode)

    def test_imzml_mode_when_processed(self) -> None:
        self.mock_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])
        self.assertEqual(ImzmlModeEnum.PROCESSED, self.mock_ram_reader.imzml_mode)

    def test_n_spectra(self) -> None:
        self.assertEqual(2, self.mock_ram_reader.n_spectra)

    def test_coordinates(self) -> None:
        np.testing.assert_array_equal(self.mock_coordinates, self.mock_ram_reader.coordinates)

    def test_coordinates_2d(self) -> None:
        np.testing.assert_array_equal(self.mock_coordinates[:, :2], self.mock_ram_reader.coordinates_2d)

    def test_get_spectrum(self) -> None:
        mz_arr, int_arr = self.mock_ram_reader.get_spectrum(1)
        np.testing.assert_array_equal(self.mock_mz_arr_list[1], mz_arr)
        np.testing.assert_array_equal(self.mock_int_arr_list[1], int_arr)

    def test_get_spectra_when_continuous(self) -> None:
        mz_arr_list, int_arr_list = self.mock_ram_reader.get_spectra([0, 1])
        np.testing.assert_array_equal(self.mock_mz_arr_list, mz_arr_list)
        np.testing.assert_array_equal(self.mock_int_arr_list, int_arr_list)

    def test_get_spectra_when_processed(self) -> None:
        self.mock_mz_arr_list = np.array([[1.0, 2.0, 3.0], [1.2, 2.5, 3.0]])
        mz_arr_list, int_arr_list = self.mock_ram_reader.get_spectra([0, 1])
        np.testing.assert_array_equal(self.mock_mz_arr_list, mz_arr_list)
        np.testing.assert_array_equal(self.mock_int_arr_list, int_arr_list)

    def test_get_spectrum_mz(self) -> None:
        np.testing.assert_array_equal(self.mock_mz_arr_list[1], self.mock_ram_reader.get_spectrum_mz(1))

    def test_get_spectrum_int(self) -> None:
        np.testing.assert_array_equal(self.mock_int_arr_list[1], self.mock_ram_reader.get_spectrum_int(1))

    def test_get_spectrum_n_points(self) -> None:
        self.assertEqual(3, self.mock_ram_reader.get_spectrum_n_points(1))

    def test_get_spectrum_metadata(self) -> None:
        metadata = self.mock_ram_reader.get_spectrum_metadata(1)
        self.assertDictEqual({"i_spectrum": 1, "coordinates": ANY}, metadata)
        np.testing.assert_array_equal(self.mock_coordinates[1], metadata["coordinates"])

    def test_get_spectra_metadata(self) -> None:
        metadata = self.mock_ram_reader.get_spectra_metadata([0, 1])
        self.assertEqual(2, len(metadata))
        self.assertDictEqual({"i_spectrum": 0, "coordinates": ANY}, metadata[0])
        self.assertDictEqual({"i_spectrum": 1, "coordinates": ANY}, metadata[1])
        np.testing.assert_array_equal(self.mock_coordinates[0], metadata[0]["coordinates"])
        np.testing.assert_array_equal(self.mock_coordinates[1], metadata[1]["coordinates"])

    def test_get_spectra_mz_range(self) -> None:
        mz_range = self.mock_ram_reader.get_spectra_mz_range([0, 1])
        self.assertEqual((1.0, 3.0), mz_range)


if __name__ == "__main__":
    unittest.main()
