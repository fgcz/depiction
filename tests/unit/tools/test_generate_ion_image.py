import unittest
from unittest.mock import MagicMock, patch, ANY

import numpy as np

from ionmapper.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from ionmapper.tools.generate_ion_image import GenerateIonImage


class TestGenerateIonImage(unittest.TestCase):
    def setUp(self):
        self.mock_parallel_config = MagicMock(name="mock_parallel_config")
        self.mock_generate = GenerateIonImage(parallel_config=self.mock_parallel_config)

    @patch.object(GenerateIonImage, "_compute_channels_chunk")
    @patch.object(ReadSpectraParallel, "from_config")
    @patch("ionmapper.tools.generate_ion_image.SparseImage2d")
    def test_generate_ion_images_for_file(self, mock_sparse_image, mock_from_config, method_compute_channels):
        mock_input_file = MagicMock(name="input_file", spec=["coordinates_2d"])
        mock_mz_values = MagicMock(name="mz_values", spec=[])
        mock_tol = MagicMock(name="tol", spec=[])

        result = self.mock_generate.generate_ion_images_for_file(
            input_file=mock_input_file,
            mz_values=mock_mz_values,
            tol=mock_tol,
        )
        mock_from_config.assert_called_once_with(self.mock_parallel_config)
        mock_parallelize = mock_from_config.return_value
        mock_parallelize.map_chunked.assert_called_once_with(
            read_file=mock_input_file,
            operation=method_compute_channels,
            bind_args=dict(mz_values=mock_mz_values, tol_values=mock_tol),
            reduce_fn=ANY,
        )
        # check that the reduce_fn is behaving as expected
        reduce_fn = mock_parallelize.map_chunked.call_args[1]["reduce_fn"]
        reduced = reduce_fn([np.array([[1], [2]]), np.array([[3], [4]])])
        np.testing.assert_array_equal(np.array([[1], [2], [3], [4]]), reduced)

        mock_sparse_image.assert_called_once_with(
            values=mock_parallelize.map_chunked.return_value,
            coordinates=mock_input_file.coordinates_2d,
            channel_names=None,
        )
        self.assertEqual(mock_sparse_image.return_value, result)

    @patch.object(GenerateIonImage, "_compute_for_mz_ranges")
    @patch.object(ReadSpectraParallel, "from_config")
    @patch("ionmapper.tools.generate_ion_image.SparseImage2d")
    def test_generate_range_images_for_file(self, mock_sparse_image, mock_from_config, method_compute_for_mz_ranges):
        mock_input_file = MagicMock(name="input_file", spec=["coordinates_2d"])
        mock_mz_ranges = MagicMock(name="mz_ranges", spec=[])

        result = self.mock_generate.generate_range_images_for_file(
            input_file=mock_input_file,
            mz_ranges=mock_mz_ranges,
            channel_names=None,
        )
        mock_from_config.assert_called_once_with(self.mock_parallel_config)
        mock_parallelize = mock_from_config.return_value
        mock_parallelize.map_chunked.assert_called_once_with(
            read_file=mock_input_file,
            operation=method_compute_for_mz_ranges,
            bind_args=dict(mz_ranges=mock_mz_ranges),
            reduce_fn=ANY,
        )
        # check that the reduce_fn is behaving as expected
        reduce_fn = mock_parallelize.map_chunked.call_args[1]["reduce_fn"]
        reduced = reduce_fn([np.array([[1], [2]]), np.array([[3], [4]])])
        np.testing.assert_array_equal(np.array([[1], [2], [3], [4]]), reduced)

        mock_sparse_image.assert_called_once_with(
            values=mock_parallelize.map_chunked.return_value,
            coordinates=mock_input_file.coordinates_2d,
            channel_names=None,
        )
        self.assertEqual(mock_sparse_image.return_value, result)

    def test_compute_for_mz_ranges(self):
        mock_reader = MagicMock(name="reader")
        mock_reader.get_spectrum.side_effect = {
            7: (np.array([1, 2, 3, 4, 4.2]), np.array([0, 0, 1, 1, 1])),
            11: (np.array([1, 2, 3, 4, 4.2]), np.array([1, 1, 1, 0, 0])),
        }.__getitem__
        mz_ranges = [(1, 3), (3, 4)]

        result = GenerateIonImage._compute_for_mz_ranges(reader=mock_reader, mz_ranges=mz_ranges, spectra_ids=[7, 11])

        self.assertTupleEqual((2, 2), result.shape)
        np.testing.assert_array_equal([1.0, 2.0], result[0])
        np.testing.assert_array_equal([3.0, 1.0], result[1])

    def test_compute_channels_chunk(self):
        mock_reader = MagicMock(name="reader")
        mock_reader.get_spectrum.side_effect = {
            7: (np.array([1, 2, 3, 4, 4.2]), np.array([0, 0, 1, 1, 1])),
            11: (np.array([1, 2, 3, 4, 4.2]), np.array([1, 1, 1, 0, 0])),
        }.__getitem__
        mz_values = [1, 3, 4]
        tol = [0.5, 0.5, 0.5]
        spectra_ids = [7, 11]

        result = GenerateIonImage._compute_channels_chunk(
            reader=mock_reader, spectra_ids=spectra_ids, mz_values=mz_values, tol_values=tol
        )

        self.assertTupleEqual((2, 3), result.shape)
        np.testing.assert_array_equal([0, 1, 2], result[0])
        np.testing.assert_array_equal([1, 1, 0], result[1])


if __name__ == "__main__":
    unittest.main()
