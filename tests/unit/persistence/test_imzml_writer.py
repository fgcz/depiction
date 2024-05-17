import os
import unittest
from functools import cached_property
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch, call

from ionmapper.persistence import ImzmlWriter, ImzmlModeEnum


class TestImzmlWriter(unittest.TestCase):
    def setUp(self):
        # test setup
        self.mock_wrapped_imzml_writer = MagicMock(name="mock_wrapped_imzml_writer")
        self.mock_imzml_alignment_tracker = MagicMock(name="mock_imzml_alignment_tracker")

        # common variables
        self.mock_mz_arr = MagicMock(name="mock_mz_arr")
        self.mock_int_arr = MagicMock(name="mock_int_arr")
        self.mock_coordinates = MagicMock(name="mock_coordinates")

    @cached_property
    def mock_imzml_writer(self) -> ImzmlWriter:
        return ImzmlWriter(
            wrapped_imzml_writer=self.mock_wrapped_imzml_writer,
            imzml_alignment_tracker=self.mock_imzml_alignment_tracker,
        )

    def test_open_when_continuous(self):
        with TemporaryDirectory() as tmpdir:
            mock_path = os.path.join(tmpdir, "test.imzML")
            writer = ImzmlWriter.open(path=mock_path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
            self.assertEqual(mock_path, writer.imzml_path)
            self.assertEqual(ImzmlModeEnum.CONTINUOUS, writer.imzml_mode)

    def test_open_when_processed(self):
        with TemporaryDirectory() as tmpdir:
            mock_path = os.path.join(tmpdir, "test.imzML")
            writer = ImzmlWriter.open(path=mock_path, imzml_mode=ImzmlModeEnum.PROCESSED)
            self.assertEqual(mock_path, writer.imzml_path)
            self.assertEqual(ImzmlModeEnum.PROCESSED, writer.imzml_mode)

    def test_close(self):
        self.mock_imzml_writer.close()
        self.mock_wrapped_imzml_writer.close.assert_called_once_with()

    def test_deactivate_alignment_tracker(self):
        self.mock_imzml_writer.deactivate_alignment_tracker()
        self.assertIsNone(self.mock_imzml_writer._imzml_alignment_tracker)

    @patch.object(ImzmlModeEnum, "from_pyimzml_str")
    def test_imzml_mode(self, mock_from_pyimzml_str):
        mode = self.mock_imzml_writer.imzml_mode
        mock_from_pyimzml_str.assert_called_once_with(self.mock_wrapped_imzml_writer.mode)
        self.assertEqual(mock_from_pyimzml_str.return_value, mode)

    def test_imzml_path(self):
        self.assertEqual(self.mock_wrapped_imzml_writer.filename, self.mock_imzml_writer.imzml_path)

    def test_ibd_path(self):
        self.assertEqual(self.mock_wrapped_imzml_writer.ibd_filename, self.mock_imzml_writer.ibd_path)

    def test_is_aligned(self):
        self.assertEqual(self.mock_imzml_alignment_tracker.is_aligned, self.mock_imzml_writer.is_aligned)

    @patch.object(ImzmlWriter, "imzml_mode", new=ImzmlModeEnum.CONTINUOUS)
    def test_add_spectrum_when_continuous_when_no_tracker(self):
        self.mock_imzml_alignment_tracker = None
        self.mock_imzml_writer.add_spectrum(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr, coordinates=self.mock_coordinates
        )
        self.mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(
            self.mock_mz_arr, self.mock_int_arr, self.mock_coordinates
        )

    @patch.object(ImzmlWriter, "imzml_mode", new=ImzmlModeEnum.CONTINUOUS)
    def test_add_spectrum_when_continuous_when_with_tracker(self):
        self.mock_mz_arr = MagicMock(name="self.mock_mz_arr")
        self.mock_imzml_writer.add_spectrum(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr, coordinates=self.mock_coordinates
        )
        self.mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(
            self.mock_mz_arr, self.mock_int_arr, self.mock_coordinates
        )
        self.mock_imzml_alignment_tracker.track_mz_array.assert_called_once_with(self.mock_mz_arr)

    @patch.object(ImzmlWriter, "imzml_mode", new=ImzmlModeEnum.PROCESSED)
    def test_add_spectrum_when_processed_when_no_tracker(self):
        self.mock_imzml_alignment_tracker = None
        self.mock_imzml_writer.add_spectrum(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr, coordinates=self.mock_coordinates
        )
        self.mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(
            self.mock_mz_arr, self.mock_int_arr, self.mock_coordinates
        )

    @patch.object(ImzmlWriter, "imzml_mode", new=ImzmlModeEnum.PROCESSED)
    def test_add_spectrum_when_processed_when_with_tracker(self):
        self.mock_imzml_writer.add_spectrum(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr, coordinates=self.mock_coordinates
        )
        self.mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(
            self.mock_mz_arr, self.mock_int_arr, self.mock_coordinates
        )
        self.mock_imzml_alignment_tracker.track_mz_array.assert_called_once_with(self.mock_mz_arr)

    @patch.object(ImzmlWriter, "imzml_mode", new=ImzmlModeEnum.CONTINUOUS)
    def test_add_spectrum_when_alignment_not_satisfied(self):
        self.mock_imzml_alignment_tracker.is_aligned = False
        with self.assertRaises(ValueError) as error:
            self.mock_imzml_writer.add_spectrum(
                mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr, coordinates=self.mock_coordinates
            )
        self.assertIn(
            "The m/z array of the first spectrum must be identical to the m/z array of all other spectra!",
            str(error.exception),
        )

    @patch.object(ImzmlWriter, "add_spectrum")
    def test_copy_spectra(self, mock_add_spectrum):
        mock_reader = MagicMock(name="mock_reader", spec=["get_spectrum_with_coords"])
        mock_reader.get_spectrum_with_coords.side_effect = [("a", "b", "C1"), ("c", "d", "C2")]
        self.mock_imzml_writer.copy_spectra(reader=mock_reader, spectra_indices=[10, 20])
        self.assertListEqual([call("a", "b", "C1"), call("c", "d", "C2")], mock_add_spectrum.mock_calls)


if __name__ == "__main__":
    unittest.main()
