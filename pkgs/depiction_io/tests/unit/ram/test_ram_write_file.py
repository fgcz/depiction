import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from depiction_io import ImzmlModeEnum
from depiction_io.ram.ram_write_file import RamWriteFile
from typing import NoReturn


class TestRamWriteFile(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imzml_mode = MagicMock(name="mock_imzml_mode")

    @cached_property
    def mock_write_file(self) -> RamWriteFile:
        return RamWriteFile(imzml_mode=self.mock_imzml_mode)

    def test_imzml_mode(self) -> None:
        self.assertEqual(self.mock_imzml_mode, self.mock_write_file.imzml_mode)

    def test_add_spectrum(self) -> None:
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        mock_int_arr = MagicMock(name="mock_int_arr")
        mock_coordinates = MagicMock(name="mock_coordinates")
        with self.mock_write_file.writer() as writer:
            writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
        self.assertListEqual([mock_mz_arr], self.mock_write_file._mz_arr_list)
        self.assertListEqual([mock_int_arr], self.mock_write_file._int_arr_list)
        self.assertListEqual([mock_coordinates], self.mock_write_file._coordinates_list)

    @unittest.skip
    def test_copy_spectra(self) -> NoReturn:
        raise NotImplementedError

    def test_to_read_file(self) -> None:
        """Round trip through real arrays.

        This asserted against `MagicMock(copy=lambda: "x")` sentinels, which pinned that
        `to_read_file` copies its lists but said nothing about the result being usable. It was
        not: `add_spectrum` collects one coordinate array per spectrum, and `RamReadFile` slices
        `coordinates[:, :2]`, which a list of arrays raises `TypeError` on.
        """
        write_file = RamWriteFile(imzml_mode=ImzmlModeEnum.PROCESSED)
        with write_file.writer() as writer:
            writer.add_spectrum(np.array([100.0, 200.0]), np.array([1.0, 2.0]), (1, 4, 1))
            writer.add_spectrum(np.array([150.0]), np.array([3.0]), (2, 4, 1))

        read_file = write_file.to_read_file()

        self.assertEqual(2, read_file.n_spectra)
        np.testing.assert_array_equal(np.array([[1, 4], [2, 4]]), read_file.coordinates_2d)
        with read_file.reader() as reader:
            np.testing.assert_array_equal(np.array([150.0]), reader.get_spectrum_mz(1))
            np.testing.assert_array_equal(np.array([3.0]), reader.get_spectrum_int(1))

    def test_to_read_file_is_a_snapshot(self) -> None:
        """The `.copy()` calls the mocked version pinned, asserted through behaviour instead."""
        write_file = RamWriteFile(imzml_mode=ImzmlModeEnum.PROCESSED)
        with write_file.writer() as writer:
            writer.add_spectrum(np.array([100.0]), np.array([1.0]), (1, 1, 1))
        read_file = write_file.to_read_file()

        with write_file.writer() as writer:
            writer.add_spectrum(np.array([200.0]), np.array([2.0]), (2, 1, 1))

        self.assertEqual(1, read_file.n_spectra)
        self.assertEqual(2, write_file.to_read_file().n_spectra)


if __name__ == "__main__":
    unittest.main()
