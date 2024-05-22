import unittest
from functools import cached_property
from unittest.mock import MagicMock

from depiction.persistence.ram_write_file import RamWriteFile
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
        mock_mz_arr_list = MagicMock(name="mock_mz_arr_list", copy=lambda: "x")
        mock_int_arr_list = MagicMock(name="mock_int_arr_list", copy=lambda: "y")
        mock_coordinates_list = MagicMock(name="mock_coordinates_list", copy=lambda: "z")
        self.mock_write_file._mz_arr_list = mock_mz_arr_list
        self.mock_write_file._int_arr_list = mock_int_arr_list
        self.mock_write_file._coordinates_list = mock_coordinates_list
        read_file = self.mock_write_file.to_read_file()
        self.assertEqual("x", read_file._mz_arr_list)
        self.assertEqual("y", read_file._int_arr_list)
        self.assertEqual("z", read_file._coordinates)


if __name__ == "__main__":
    unittest.main()
