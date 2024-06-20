import unittest
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import xarray
from xarray import DataArray

from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.tools.generate_ion_image import GenerateIonImage


@pytest.fixture
def mock_parallel_config() -> MagicMock:
    return MagicMock(name="mock_parallel_config", spec=[])


@pytest.fixture
def mock_generate(mock_parallel_config: MagicMock) -> GenerateIonImage:
    return GenerateIonImage(parallel_config=mock_parallel_config)


def test_generate_ion_images_for_file(mocker, mock_generate: GenerateIonImage) -> None:
    mock_generate_channel_values = mocker.patch.object(GenerateIonImage, "_generate_channel_values")
    mock_generate_channel_values.return_value = DataArray(
        [[1, 2], [3, 4], [5, 6]], dims=("i", "c"), attrs={"bg_value": np.nan}
    )

    mock_input_file = MagicMock(name="mock_input_file", coordinates_2d=np.array([[0, 0], [0, 1], [1, 0]]))
    mock_mz_values = MagicMock(name="mock_mz_values", spec=[])
    mock_tol = MagicMock(name="mock_tol", spec=[])

    image = mock_generate.generate_ion_images_for_file(
        input_file=mock_input_file, mz_values=mock_mz_values, tol=mock_tol, channel_names=["channel A", "channel B"]
    )

    assert image.channel_names == ["channel A", "channel B"]
    xarray.testing.assert_equal(
        image.bg_mask, DataArray([[False, False], [False, True]], dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    )
    img_array = image.data_spatial.transpose("x", "y", "c").values
    assert img_array[0, 0, 0] == 1
    assert img_array[0, 0, 1] == 2
    assert img_array[1, 0, 0] == 5
    assert img_array[1, 0, 1] == 6

    mock_generate_channel_values.assert_called_once_with(
        input_file=mock_input_file, mz_values=mock_mz_values, tol=mock_tol
    )


def test_generate_channel_values(mocker, mock_generate: GenerateIonImage, mock_parallel_config) -> None:
    mock_read_parallel_from = mocker.patch("depiction.tools.generate_ion_image.ReadSpectraParallel.from_config")
    mock_read_parallel_from.return_value.map_chunked.return_value = np.array([[1, 2], [3, 4]])
    mock_input_file = MagicMock(name="mock_input_file", spec=[])
    mock_mz_values = MagicMock(name="mock_mz_values")
    tol = [0.25, 0.5, 0.25]

    values = mock_generate._generate_channel_values(input_file=mock_input_file, mz_values=mock_mz_values, tol=tol)
    xarray.testing.assert_identical(
        values, DataArray(np.array([[1.0, 2], [3, 4]]), dims=("i", "c"), attrs={"bg_value": np.nan})
    )
    mock_read_parallel_from.assert_called_once_with(mock_parallel_config)
    mock_read_parallel_from.return_value.map_chunked.assert_called_once_with(
        read_file=mock_input_file,
        operation=GenerateIonImage._compute_channels_chunk,
        bind_args=dict(mz_values=mock_mz_values, tol_values=tol),
        reduce_fn=ANY,
    )


class TestGenerateIonImage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_parallel_config = MagicMock(name="mock_parallel_config")
        self.mock_generate = GenerateIonImage(parallel_config=self.mock_parallel_config)

    @patch.object(GenerateIonImage, "_compute_for_mz_ranges")
    @patch.object(ReadSpectraParallel, "from_config")
    @patch("depiction.tools.generate_ion_image.MultiChannelImage")
    def test_generate_range_images_for_file(
        self, mock_multi_channel_image, mock_from_config, method_compute_for_mz_ranges
    ) -> None:
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

        mock_multi_channel_image.from_numpy_sparse.assert_called_once_with(
            values=mock_parallelize.map_chunked.return_value,
            coordinates=mock_input_file.coordinates_2d,
            channel_names=None,
            bg_value=np.nan,
        )
        self.assertEqual(mock_multi_channel_image.from_numpy_sparse.return_value, result)

    def test_compute_for_mz_ranges(self) -> None:
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

    def test_compute_channels_chunk(self) -> None:
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
