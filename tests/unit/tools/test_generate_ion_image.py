from unittest.mock import MagicMock, ANY

import numpy as np
import pytest

from depiction.image import MultiChannelImage
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.tools.generate_ion_image import GenerateIonImage


@pytest.fixture
def mock_parallel_config() -> MagicMock:
    return MagicMock(name="mock_parallel_config", spec=[])


@pytest.fixture
def mock_generate(mock_parallel_config: MagicMock) -> GenerateIonImage:
    return GenerateIonImage(parallel_config=mock_parallel_config)


def test_generate_ion_images_for_file(mocker, mock_parallel_config, mock_generate):
    # TODO This test could be improved in the future
    mock_input_file = mocker.MagicMock(name="mock_input_file", spec=["coordinates_array_2d"])
    mock_channel_names = mocker.MagicMock(name="mock_channel_names", spec=[])
    mock_parallelize = mocker.patch.object(ReadSpectraParallel, "from_config").return_value
    mock_parallelize.map_chunked.return_value = np.array([[1, 2], [3, 4]])
    mock_from_flat = mocker.patch.object(MultiChannelImage, "from_flat")

    result = mock_generate.generate_ion_images_for_file(
        input_file=mock_input_file,
        mz_values=np.array([1.0, 2.0, 3.0]),
        tol=0.5,
        channel_names=mock_channel_names,
    )

    assert result == mock_from_flat.return_value
    mock_from_flat.assert_called_once_with(
        values=ANY, coordinates=mock_input_file.coordinates_array_2d, channel_names=mock_channel_names
    )
    mock_parallelize.map_chunked.assert_called_once_with(
        read_file=mock_input_file,
        operation=GenerateIonImage._compute_channels_chunk,
        bind_args=dict(mz_values=ANY, tol_values=ANY),
        reduce_fn=ANY,
    )


def test_compute_for_mz_ranges() -> None:
    mock_reader = MagicMock(name="reader")
    mock_reader.get_spectrum.side_effect = {
        7: (np.array([1, 2, 3, 4, 4.2]), np.array([0, 0, 1, 1, 1])),
        11: (np.array([1, 2, 3, 4, 4.2]), np.array([1, 1, 1, 0, 0])),
    }.__getitem__
    mz_ranges = [(1, 3), (3, 4)]

    result = GenerateIonImage._compute_for_mz_ranges(reader=mock_reader, mz_ranges=mz_ranges, spectra_ids=[7, 11])

    assert result.shape == (2, 2)
    np.testing.assert_array_equal([1.0, 2.0], result[0])
    np.testing.assert_array_equal([3.0, 1.0], result[1])


def test_compute_channels_chunk() -> None:
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

    assert result.shape == (2, 3)
    np.testing.assert_array_equal([0, 1, 2], result[0])
    np.testing.assert_array_equal([1, 1, 0], result[1])


def test_generate_range_images_for_file(mocker, mock_generate: GenerateIonImage, mock_parallel_config) -> None:
    mock_from_config = mocker.patch.object(ReadSpectraParallel, "from_config")
    method_compute_for_mz_ranges = mocker.patch.object(GenerateIonImage, "_compute_for_mz_ranges")
    mock_multi_channel_image = mocker.patch("depiction.tools.generate_ion_image.MultiChannelImage")

    mock_input_file = MagicMock(name="input_file", spec=["coordinates_array_2d"])
    mock_mz_ranges = MagicMock(name="mz_ranges", spec=[])

    result = mock_generate.generate_range_images_for_file(
        input_file=mock_input_file,
        mz_ranges=mock_mz_ranges,
        channel_names=None,
    )
    mock_from_config.assert_called_once_with(mock_parallel_config)
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

    mock_multi_channel_image.from_flat.assert_called_once_with(
        values=mock_parallelize.map_chunked.return_value,
        coordinates=mock_input_file.coordinates_array_2d,
        channel_names=None,
    )
    assert result == mock_multi_channel_image.from_flat.return_value


if __name__ == "__main__":
    pytest.main()
