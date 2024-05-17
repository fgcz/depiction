import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np

from ionmapper.persistence import ImzmlReadFile, ImzmlModeEnum


class TestImzmlReadFile(unittest.TestCase):
    def setUp(self):
        self.mock_path = "/dev/null/mock_path.imzML"

    @cached_property
    def mock_read_file(self) -> ImzmlReadFile:
        return ImzmlReadFile(path=self.mock_path)

    def test_imzml_file(self):
        self.assertEqual(self.mock_path, self.mock_read_file.imzml_file)

    def test_imzml_file_when_case_insensitive(self):
        self.mock_path = "/dev/null/mock_path.IMZML"
        self.assertEqual(self.mock_path, self.mock_read_file.imzml_file)

    def test_imzml_file_when_invalid_path(self):
        self.mock_path = "/dev/null/mock_path.txt"
        with self.assertRaises(ValueError) as error:
            _ = self.mock_read_file.imzml_file
        self.assertIn("Expected .imzML file, got", str(error.exception))

    def test_ibd_file(self):
        self.assertEqual("/dev/null/mock_path.ibd", self.mock_read_file.ibd_file)

    def test_ibd_file_when_case_insensitive(self):
        self.mock_path = "/dev/null/mock_path.IMZML"
        self.assertEqual("/dev/null/mock_path.ibd", self.mock_read_file.ibd_file)

    def test_ibd_file_when_invalid_path(self):
        self.mock_path = "/dev/null/mock_path.txt"
        with self.assertRaises(ValueError) as error:
            _ = self.mock_read_file.ibd_file
        self.assertIn("Expected .imzML file, got", str(error.exception))

    @patch.object(ImzmlReadFile, "get_reader")
    def test_reader_when_normal(self, method_get_reader):
        mock_reader = MagicMock(name="mock_reader")
        method_get_reader.return_value = mock_reader
        with self.mock_read_file.reader() as reader:
            method_get_reader.assert_called_once_with()
            self.assertEqual(mock_reader, reader)
        mock_reader.close.assert_called_once_with()

    @patch.object(ImzmlReadFile, "get_reader")
    def test_get_reader_when_exception(self, method_get_reader):
        mock_reader = MagicMock(name="mock_reader")
        method_get_reader.return_value = mock_reader
        with self.assertRaises(ValueError) as error:
            with self.mock_read_file.reader() as reader:
                method_get_reader.assert_called_once_with()
                self.assertEqual(mock_reader, reader)
                raise ValueError("mock_error")
        self.assertEqual("mock_error", str(error.exception))
        mock_reader.close.assert_called_once_with()

    @patch("pyimzml.ImzMLParser.ImzMLParser")
    def test_get_reader(self, mock_imzml_parser):
        mock_reader = MagicMock(name="mock_reader", mzPrecision="d", intensityPrecision="i")
        mock_imzml_parser.return_value.__enter__.return_value.portable_spectrum_reader.return_value = mock_reader
        reader = self.mock_read_file.get_reader()
        mock_imzml_parser.assert_called_once_with(self.mock_path)
        self.assertEqual(mock_reader, reader._portable_reader)
        self.assertEqual(self.mock_path, reader._imzml_path)
        self.assertEqual(8, reader._mz_bytes)
        self.assertEqual(4, reader._int_bytes)

    @patch.object(ImzmlReadFile, "_cached_properties", {"n_spectra": 123})
    def test_n_spectra(self):
        self.assertEqual(123, self.mock_read_file.n_spectra)

    @patch.object(ImzmlReadFile, "_cached_properties", {"imzml_mode": "mock_mode"})
    def test_imzml_mode(self):
        self.assertEqual("mock_mode", self.mock_read_file.imzml_mode)

    @patch.object(ImzmlReadFile, "_cached_properties", {"coordinates": "mock_coordinates"})
    def test_coordinates(self):
        self.assertEqual("mock_coordinates", self.mock_read_file.coordinates)

    @patch.object(ImzmlReadFile, "_cached_properties", {"coordinates": np.array([[1, 2, 3], [4, 5, 6]])})
    def test_coordinates_2d(self):
        np.testing.assert_array_equal(np.array([[1, 2], [4, 5]]), self.mock_read_file.coordinates_2d)

    def test_compact_metadata(self):
        mock_coordinates = np.array([[1, 2, 3], [3, 3, 3], [1 + 3, 2 + 4, 3 + 5]])
        with (
            patch.object(ImzmlReadFile, "n_spectra", 15),
            patch.object(ImzmlReadFile, "imzml_mode", ImzmlModeEnum.PROCESSED),
            patch.object(ImzmlReadFile, "coordinates", mock_coordinates),
            patch.object(ImzmlReadFile, "imzml_file", "mock_imzml_file_path"),
            patch.object(ImzmlReadFile, "ibd_file", "mock_ibd_file_path"),
        ):
            self.assertDictEqual(
                {
                    "n_spectra": 15,
                    "imzml_mode": "PROCESSED",
                    "coordinate_extent": [4, 5, 6],
                    "imzml_file": "mock_imzml_file_path",
                    "ibd_file": "mock_ibd_file_path",
                },
                self.mock_read_file.compact_metadata,
            )

    @patch("pyimzml.ImzMLParser.ImzMLParser")
    def test_metadata_checksums_when_none(self, mock_pyimzml_parser):
        mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {}
        self.assertEqual({}, self.mock_read_file.metadata_checksums)

    @patch("pyimzml.ImzMLParser.ImzMLParser")
    def test_metadata_checksums_when_md5_available(self, mock_pyimzml_parser):
        mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {"ibd MD5": "ABCD"}
        self.assertEqual({"md5": "abcd"}, self.mock_read_file.metadata_checksums)

    @patch("pyimzml.ImzMLParser.ImzMLParser")
    def test_metadata_checksums_when_sha1_available(self, mock_pyimzml_parser):
        mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {"ibd SHA-1": "ABCD"}
        self.assertEqual({"sha1": "abcd"}, self.mock_read_file.metadata_checksums)

    @patch("pyimzml.ImzMLParser.ImzMLParser")
    def test_metadata_checksums_when_both_checksums_available(self, mock_pyimzml_parser):
        mock_pyimzml_parser.return_value.__enter__.return_value.metadata.file_description = {
            "ibd MD5": "ABCD",
            "ibd SHA-1": "EF00",
        }
        self.assertEqual({"md5": "abcd", "sha1": "ef00"}, self.mock_read_file.metadata_checksums)

    @patch.object(ImzmlReadFile, "metadata_checksums", new={})
    def test_is_checksum_valid_when_no_checksums_available(self):
        self.assertIsNone(self.mock_read_file.is_checksum_valid)

    @patch.object(ImzmlReadFile, "metadata_checksums", new={"md5": "1234"})
    @patch.object(ImzmlReadFile, "ibd_checksums")
    def test_is_checksum_valid_when_md5_available(self, mock_checksums):
        mock_checksums.checksum_md5 = "1234"
        self.assertTrue(self.mock_read_file.is_checksum_valid)

    @patch.object(ImzmlReadFile, "metadata_checksums", new={"sha1": "1234"})
    @patch.object(ImzmlReadFile, "ibd_checksums")
    def test_is_checksum_valid_when_sha1_available(self, mock_checksums):
        mock_checksums.checksum_sha1 = "1234"
        self.assertTrue(self.mock_read_file.is_checksum_valid)

    @patch.object(ImzmlReadFile, "metadata_checksums", new={"sha256": "1234"})
    @patch.object(ImzmlReadFile, "ibd_checksums")
    def test_is_checksum_valid_when_sha256_available(self, mock_checksums):
        mock_checksums.checksum_sha256 = "1234"
        self.assertTrue(self.mock_read_file.is_checksum_valid)

    @patch.object(ImzmlReadFile, "metadata_checksums", new={"md5": "1234", "sha1": "5678"})
    @patch.object(ImzmlReadFile, "ibd_checksums")
    def test_is_checksum_valid_when_multiple_checksums_available(self, mock_ibd_checksums):
        mock_ibd_checksums.checksum_sha1 = "5678"
        mock_ibd_checksums.checksum_md5 = "1234"
        self.assertTrue(self.mock_read_file.is_checksum_valid)

    @patch("os.path.getsize")
    def test_file_sizes_bytes(self, mock_get_size):
        mock_get_size.side_effect = {
            "/dev/null/mock_path.imzML": 1234,
            "/dev/null/mock_path.ibd": 5678,
        }.__getitem__
        self.assertDictEqual({"imzml": 1234, "ibd": 5678}, self.mock_read_file.file_sizes_bytes)

    @patch("os.path.getsize")
    def test_get_file_sizes_mb(self, mock_get_size):
        mock_get_size.side_effect = {
            "/dev/null/mock_path.imzML": round(2.5 * 1024**2),
            "/dev/null/mock_path.ibd": round(3.5 * 1024**2),
        }.__getitem__
        self.assertEqual({"imzml", "ibd"}, self.mock_read_file.file_sizes_mb.keys())
        self.assertAlmostEqual(2.5, self.mock_read_file.file_sizes_mb["imzml"], places=8)
        self.assertAlmostEqual(3.5, self.mock_read_file.file_sizes_mb["ibd"], places=8)

    @patch.object(ImzmlReadFile, "imzml_mode", new=ImzmlModeEnum.PROCESSED)
    @patch.object(ImzmlReadFile, "n_spectra", new=42)
    @patch.object(ImzmlReadFile, "is_checksum_valid", new=True)
    @patch.object(ImzmlReadFile, "file_sizes_mb", new={"imzml": 2.5, "ibd": 3.5})
    def test_summary_when_processed(self):
        summary = self.mock_read_file.summary()
        self.assertIn("imzML file: /dev/null/mock_path.imzML (2.50 MB)", summary)
        self.assertIn("ibd file: /dev/null/mock_path.ibd (3.50 MB)", summary)
        self.assertIn("imzML mode: PROCESSED", summary)
        self.assertIn("n_spectra: 42", summary)
        self.assertIn("is_checksum_valid: True", summary)

    @patch.object(ImzmlReadFile, "imzml_mode", new=ImzmlModeEnum.CONTINUOUS)
    @patch.object(ImzmlReadFile, "n_spectra", new=42)
    @patch.object(ImzmlReadFile, "is_checksum_valid", new=True)
    @patch.object(ImzmlReadFile, "file_sizes_mb", new={"imzml": 2.5, "ibd": 3.5})
    @patch.object(ImzmlReadFile, "reader")
    def test_summary_when_continuous(self, method_reader):
        mock_reader = MagicMock(name="mock_reader")
        mock_reader.get_spectrum_mz.return_value = np.array([1, 2, 3])
        method_reader.return_value.__enter__.return_value = mock_reader
        summary = self.mock_read_file.summary()
        self.assertIn("imzML file: /dev/null/mock_path.imzML (2.50 MB)", summary)
        self.assertIn("ibd file: /dev/null/mock_path.ibd (3.50 MB)", summary)
        self.assertIn("imzML mode: CONTINUOUS", summary)
        self.assertIn("n_spectra: 42", summary)
        self.assertIn("is_checksum_valid: True", summary)
        self.assertIn("m/z range: 1.00 - 3.00 (3 bins)", summary)
        method_reader.assert_called_once_with()

    @patch("builtins.print")
    @patch.object(ImzmlReadFile, "summary")
    def test_print_summary(self, mock_summary, mock_print):
        self.mock_read_file.print_summary()
        mock_summary.assert_called_once_with(checksums=True)
        mock_print.assert_called_once_with(mock_summary.return_value)

    @patch.object(ImzmlReadFile, "reader")
    def test_cached_properties(self, method_reader):
        mock_reader = MagicMock(name="mock_reader")
        method_reader.return_value.__enter__.return_value = mock_reader
        self.assertDictEqual(
            {
                "n_spectra": mock_reader.n_spectra,
                "imzml_mode": mock_reader.imzml_mode,
                "coordinates": mock_reader.coordinates,
            },
            self.mock_read_file._cached_properties,
        )

    def test_repr(self):
        self.assertEqual(f"ImzmlReadFile('{self.mock_path}')", repr(self.mock_read_file))

    def test_str(self):
        self.assertEqual(f"ImzmlReadFile('{self.mock_path}')", str(self.mock_read_file))


if __name__ == "__main__":
    unittest.main()
