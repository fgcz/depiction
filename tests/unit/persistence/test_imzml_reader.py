import pickle
import unittest
from functools import cached_property
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import numpy as np

from depiction.persistence import ImzmlReader, ImzmlModeEnum


class TestImzmlReader(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_mz_arr_list = [[100, 200], [100, 150, 200], [101, 201]]
        self.mock_int_arr_list = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]]
        self.mock_mz_arr_offsets, self.mock_mz_arr_lengths = self._generate_offsets_lengths(self.mock_mz_arr_list,
                                                                                            n_bytes=4)
        self.mock_int_arr_offsets, self.mock_int_arr_lengths = self._generate_offsets_lengths(self.mock_int_arr_list,
                                                                                              n_bytes=4)
        self.mock_int_arr_offsets = [offset + 100 for offset in self.mock_int_arr_offsets]
        self.mock_int_arr_dtype = "f"
        self.mock_mz_arr_dtype = "i"
        self.mock_coordinates = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)

        self.mock_imzml_path = Path("/dev/null/test.imzML")

    @cached_property
    def mock_reader(self) -> ImzmlReader:
        return ImzmlReader(mz_arr_offsets=self.mock_mz_arr_offsets, mz_arr_lengths=self.mock_mz_arr_lengths,
                           mz_arr_dtype=self.mock_mz_arr_dtype, int_arr_offsets=self.mock_int_arr_offsets,
                           int_arr_lengths=self.mock_int_arr_lengths, int_arr_dtype=self.mock_int_arr_dtype,
                           coordinates=self.mock_coordinates, imzml_path=self.mock_imzml_path, )

    def _generate_offsets_lengths(self, arr_list, n_bytes: int):
        offsets = [0]
        lengths = []
        for arr in arr_list:
            lengths.append(len(arr))
            offsets.append(offsets[-1] * n_bytes + len(arr))
        return offsets[:-1], lengths

    def test_pickle(self) -> None:
        """Checks getstate and setstate are implemented correctly."""
        obj_original = self.mock_reader
        obj_reread = pickle.loads(pickle.dumps(obj_original))
        self.assertEqual(obj_original.imzml_path, obj_reread.imzml_path)
        self.assertEqual(obj_original.ibd_path, obj_reread.ibd_path)
        np.testing.assert_array_equal(self.mock_mz_arr_offsets, obj_reread._mz_arr_offsets)
        np.testing.assert_array_equal(self.mock_mz_arr_lengths, obj_reread._mz_arr_lengths)
        np.testing.assert_array_equal(self.mock_int_arr_offsets, obj_reread._int_arr_offsets)
        np.testing.assert_array_equal(self.mock_int_arr_lengths, obj_reread._int_arr_lengths)
        np.testing.assert_array_equal(self.mock_coordinates, obj_reread._coordinates)
        self.assertEqual(self.mock_mz_arr_dtype, obj_reread._mz_arr_dtype)
        self.assertEqual(self.mock_int_arr_dtype, obj_reread._int_arr_dtype)
        self.assertEqual(self.mock_reader._mz_bytes, obj_reread._mz_bytes)
        self.assertEqual(self.mock_reader._int_bytes, obj_reread._int_bytes)

    def test_imzml_path(self) -> None:
        self.assertEqual(Path("/dev/null/test.imzML"), self.mock_reader.imzml_path)

    def test_ibd_path(self) -> None:
        self.assertEqual(Path("/dev/null/test.ibd"), self.mock_reader.ibd_path)

    # TODO ibd_mmap, open_ibd_mmap

    def test_enter(self) -> None:
        self.assertEqual(self.mock_reader, self.mock_reader.__enter__())

    @patch.object(ImzmlReader, "close")
    def test_exit(self, mock_close) -> None:
        self.mock_reader.__exit__(None, None, None)
        mock_close.assert_called_once_with()

    def test_close(self) -> None:
        with (
            patch.object(self.mock_reader, "_ibd_mmap") as mock_ibd_mmap, patch.object(self.mock_reader,
                                                                                       "_ibd_file") as mock_ibd_file, ):
            self.mock_reader.close()
        mock_ibd_mmap.close.assert_called_once_with()
        mock_ibd_file.close.assert_called_once_with()

    def test_close_when_none(self) -> None:
        with patch.object(self.mock_reader, "_ibd_mmap", None), patch.object(self.mock_reader, "_ibd_file", None):
            # main check is that this does not raise
            self.mock_reader.close()

    def test_imzml_mode_when_continuous(self):
        self.mock_mz_arr_offsets = [7, 7, 7]
        self.assertEqual(ImzmlModeEnum.CONTINUOUS, self.mock_reader.imzml_mode)

    def test_imzml_mode_when_processed(self):
        self.assertEqual(ImzmlModeEnum.PROCESSED, self.mock_reader.imzml_mode)

    def test_n_spectra(self) -> None:
        self.assertEqual(3, self.mock_reader.n_spectra)

    def test_coordinates(self):
        self.mock_coordinates = np.array([(1, 2, 3), (4, 5, 6)])
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        np.testing.assert_array_equal(expected, self.mock_reader.coordinates)

    def test_coordinates_2d(self):
        self.mock_coordinates = np.array([(1, 2, 3), (4, 5, 6)])
        expected = np.array([[1, 2], [4, 5]], dtype=int)
        np.testing.assert_array_equal(expected, self.mock_reader.coordinates_2d)

    @patch.object(ImzmlReader, "get_spectrum_mz")
    @patch.object(ImzmlReader, "get_spectrum_int")
    def test_get_spectrum(self, mock_get_spectrum_int, mock_get_spectrum_mz):
        mock_i_spectrum = MagicMock(name="mock_i_spectrum", spec=[])
        mz_arr, int_arr = self.mock_reader.get_spectrum(i_spectrum=mock_i_spectrum)
        self.assertEqual(mock_get_spectrum_mz.return_value, mz_arr)
        self.assertEqual(mock_get_spectrum_int.return_value, int_arr)
        mock_get_spectrum_mz.assert_called_once_with(i_spectrum=mock_i_spectrum)
        mock_get_spectrum_int.assert_called_once_with(i_spectrum=mock_i_spectrum)

    @patch.object(ImzmlReader, "get_spectrum_mz")
    @patch.object(ImzmlReader, "get_spectrum_int")
    @patch.object(ImzmlReader, "get_spectrum_coordinates")
    def test_get_spectrum_with_coords(self, mock_get_spectrum_coordinates, mock_get_spectrum_int, mock_get_spectrum_mz):
        mock_i_spectrum = MagicMock(name="mock_i_spectrum", spec=[])
        mz_arr, int_arr, coords = self.mock_reader.get_spectrum_with_coords(i_spectrum=mock_i_spectrum)
        self.assertEqual(mock_get_spectrum_mz.return_value, mz_arr)
        self.assertEqual(mock_get_spectrum_int.return_value, int_arr)
        self.assertEqual(mock_get_spectrum_coordinates.return_value, coords)
        mock_get_spectrum_mz.assert_called_once_with(i_spectrum=mock_i_spectrum)
        mock_get_spectrum_int.assert_called_once_with(i_spectrum=mock_i_spectrum)
        mock_get_spectrum_coordinates.assert_called_once_with(i_spectrum=mock_i_spectrum)

    @patch.object(ImzmlReader, "get_spectrum_mz")
    @patch.object(ImzmlReader, "get_spectrum_int")
    @patch.object(ImzmlReader, "imzml_mode", new_callable=PropertyMock)
    def test_get_spectra_when_continuous(self, prop_imzml_mode, mock_get_spectrum_int, mock_get_spectrum_mz):
        prop_imzml_mode.return_value = ImzmlModeEnum.CONTINUOUS
        i_spectra = [2, 4, 9]
        mock_get_spectrum_mz.return_value = np.array([1., 2])
        mock_get_spectrum_int.side_effect = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        mz_arr_list, int_arr_list = self.mock_reader.get_spectra(i_spectra=i_spectra)
        np.testing.assert_array_equal(np.array([[1., 2], [1, 2], [1, 2]]), mz_arr_list)
        mock_get_spectrum_mz.assert_called_once_with(i_spectrum=2)
        expected_int_arr_list = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(expected_int_arr_list, int_arr_list)
        self.assertListEqual([call(i_spectrum=2), call(i_spectrum=4), call(i_spectrum=9)],
                             mock_get_spectrum_int.mock_calls)

    @patch.object(ImzmlReader, "get_spectrum")
    @patch.object(ImzmlReader, "imzml_mode", new_callable=PropertyMock)
    def test_get_spectra_when_processed(self, prop_imzml_mode, mock_get_spectrum):
        prop_imzml_mode.return_value = ImzmlModeEnum.PROCESSED
        i_spectra = [2, 4, 9]
        mock_get_spectrum.side_effect = [(self.mock_mz_arr_list[0], self.mock_int_arr_list[0]),
                                         (self.mock_mz_arr_list[1], self.mock_int_arr_list[1]),
                                         (self.mock_mz_arr_list[2], self.mock_int_arr_list[2])]
        mz_arr_list, int_arr_list = self.mock_reader.get_spectra(i_spectra=i_spectra)
        expected_mz_arr_list = [np.array([100, 200]), np.array([100, 150, 200]), np.array([101, 201])]
        expected_int_arr_list = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([6.0, 7.0])]
        np.testing.assert_equal(expected_mz_arr_list, mz_arr_list)
        np.testing.assert_equal(expected_int_arr_list, int_arr_list)
        self.assertListEqual([call(i_spectrum=2), call(i_spectrum=4), call(i_spectrum=9)],
                             mock_get_spectrum.mock_calls)

    @patch.object(ImzmlReader, "ibd_mmap", new_callable=PropertyMock)
    def test_get_spectrum_mz(self, mock_ibd_mmap):
        self.mock_mz_arr_dtype = "i"
        mock_ibd_mmap.return_value.read.return_value = b"\x01\x00\x00\x00\x02\x00\x00\x00"
        mz_arr = self.mock_reader.get_spectrum_mz(1)
        np.testing.assert_array_equal([1, 2], mz_arr)
        self.assertListEqual(
            [call.seek(self.mock_mz_arr_offsets[1]),
             call.read(self.mock_mz_arr_lengths[1] * 4)],
            mock_ibd_mmap.return_value.mock_calls,
        )

    @patch.object(ImzmlReader, "ibd_mmap", new_callable=PropertyMock)
    def test_get_spectrum_int(self, mock_ibd_mmap):
        self.mock_int_arr_dtype = "i"
        mock_ibd_mmap.return_value.read.return_value = b"\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"
        mz_arr = self.mock_reader.get_spectrum_int(1)
        np.testing.assert_array_equal([2, 3, 4], mz_arr)
        self.assertListEqual(
            [call.seek(self.mock_int_arr_offsets[1]),
             call.read(self.mock_int_arr_lengths[1] * 4)],
            mock_ibd_mmap.return_value.mock_calls,
        )

    def test_get_spectrum_coordinates(self):
        self.mock_coordinates = np.array([(1, 2, 3), (4, 5, 6)])
        np.testing.assert_array_equal([1, 2, 3], self.mock_reader.get_spectrum_coordinates(0))
        np.testing.assert_array_equal([4, 5, 6], self.mock_reader.get_spectrum_coordinates(1))

    def test_get_spectrum_n_points(self):
        self.assertEqual(2, self.mock_reader.get_spectrum_n_points(0))
        self.assertEqual(3, self.mock_reader.get_spectrum_n_points(1))
        self.assertEqual(2, self.mock_reader.get_spectrum_n_points(2))

    @patch.object(ImzmlReader, "get_spectrum_mz")
    def test_get_spectra_mz_range_when_specified(self, mock_get_spectrum_mz):
        mock_i_spectra = [1, 3, 5]
        mock_get_spectrum_mz.side_effect = [np.array([1, 2]), np.array([0.5, 4]), np.array([5, 6])]
        mz_range = self.mock_reader.get_spectra_mz_range(i_spectra=mock_i_spectra)
        self.assertTupleEqual((0.5, 6), mz_range)
        self.assertListEqual([call(1), call(3), call(5)], mock_get_spectrum_mz.mock_calls)

    @patch.object(ImzmlReader, "get_spectrum_mz")
    def test_get_spectra_mz_range_when_none(self, mock_get_spectrum_mz) -> None:
        mock_get_spectrum_mz.side_effect = [np.array([1, 2]), np.array([0.5, 4]), np.array([5, 6])]
        with patch.object(ImzmlReader, "n_spectra", new_callable=PropertyMock) as mock_n_spectra:
            mock_n_spectra.return_value = 3
            mz_range = self.mock_reader.get_spectra_mz_range(i_spectra=None)
        self.assertTupleEqual((0.5, 6), mz_range)
        self.assertListEqual([call(0), call(1), call(2)], mock_get_spectrum_mz.mock_calls)

    def test_str(self) -> None:
        self.assertEqual("ImzmlReader[/dev/null/test.imzML, n_spectra=3, int_arr_dtype=f, mz_arr_dtype=i]",
                         str(self.mock_reader))


if __name__ == "__main__":
    unittest.main()
