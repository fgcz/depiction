from __future__ import annotations

import numpy as np
import pytest
import xarray
from pytest_mock import MockerFixture
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.multi_channel_image_concatenation import MultiChannelImageConcatenation


@pytest.fixture()
def channel_names() -> list[str]:
    return ["Channel A", "Channel B"]


def _construct_single_image(data: np.ndarray, channel_names: list[str]) -> MultiChannelImage:
    return MultiChannelImage(
        data=DataArray(
            data=data,
            dims=("y", "x", "c"),
            coords={"c": channel_names},
            attrs={"bg_value": np.nan},
        )
    )


@pytest.fixture()
def image_0(channel_names: list[str]) -> MultiChannelImage:
    """shape (y=2, x=3, c=2)"""
    data = np.arange(12).reshape(2, 3, 2).astype(float)
    return _construct_single_image(data=data, channel_names=channel_names)


@pytest.fixture()
def image_1(channel_names: list[str]) -> MultiChannelImage:
    """shape (y=3, x=4, c=2)"""
    data = np.arange(24).reshape(3, 4, 2).astype(float) * 3.0
    return _construct_single_image(data=data, channel_names=channel_names)


@pytest.fixture()
def concat_image(image_0: MultiChannelImage, image_1: MultiChannelImage) -> MultiChannelImageConcatenation:
    # TODO not sure if this is the nicest way for testing, in general it would probably be nicer if the fixture would
    #      not use the method for construction but rather do it directly (maybe tbd later)
    return MultiChannelImageConcatenation.concat_images([image_0, image_1])


def test_concat_images(concat_image: MultiChannelImageConcatenation) -> None:
    assert isinstance(concat_image, MultiChannelImageConcatenation)


def test_n_individual_images(concat_image: MultiChannelImageConcatenation) -> None:
    assert concat_image.n_individual_images == 2


def test_get_combined_image(
    concat_image: MultiChannelImageConcatenation, image_0: MultiChannelImage, image_1: MultiChannelImage
) -> None:
    combined_image = concat_image.get_combined_image()
    assert combined_image.data_spatial.shape == (3, 7, 2)
    assert combined_image.channel_names == image_0.channel_names == image_1.channel_names
    assert np.isnan(combined_image.bg_value)
    # image 0
    assert combined_image.data_spatial[0, 0, 1] == 1.0
    # image 1
    assert combined_image.data_spatial[0, 3, 1] == 3.0
    # check nan (because of shape differences)
    assert np.isnan(combined_image.data_spatial[2, 0, 0])


def test_get_combined_image_index(concat_image: MultiChannelImageConcatenation) -> None:
    image_index = concat_image.get_combined_image_index()
    assert image_index.data_spatial.shape == (3, 7, 1)
    assert image_index.channel_names == ["image_index"]
    expected_indices = np.zeros((3, 7, 1), dtype=int)
    expected_indices[:, 3:, :] = 1
    np.testing.assert_array_equal(image_index.data_spatial, expected_indices)


@pytest.mark.parametrize(["image_index", "image_fixture"], [(0, "image_0"), (1, "image_1")])
def test_get_single_image(request, concat_image: MultiChannelImageConcatenation, image_index: int, image_fixture: str):
    expected_image = request.getfixturevalue(image_fixture)
    result_image = concat_image.get_single_image(index=image_index)
    assert result_image.dimensions == expected_image.dimensions
    xarray.testing.assert_equal(result_image.coordinates_flat, expected_image.coordinates_flat)
    xarray.testing.assert_equal(result_image.data_flat, expected_image.data_flat)


def test_get_single_images(concat_image: MultiChannelImageConcatenation, mocker: MockerFixture) -> None:
    mock_get_single_image = mocker.patch.object(
        MultiChannelImageConcatenation, "get_single_image", side_effect=["img1", "img2"]
    )
    result = concat_image.get_single_images()
    assert result == ["img1", "img2"]
    assert mock_get_single_image.mock_calls == [mocker.call(index=0), mocker.call(index=1)]


def test_relabel_combined_image(
    concat_image: MultiChannelImageConcatenation,
) -> None:
    new_data = np.ones((3, 7, 4))
    new_combined_image = MultiChannelImage(
        data=xarray.DataArray(
            data=new_data,
            dims=("y", "x", "c"),
            coords={"c": ["A", "B", "C", "D"]},
            attrs={"bg_value": np.nan},
        )
    )
    relabeled_image = concat_image.relabel_combined_image(new_combined_image)
    assert relabeled_image.get_combined_image().channel_names == ["A", "B", "C", "D"]
    assert relabeled_image.n_individual_images == 2


def test_relabel_combined_image_different_shape(
    concat_image: MultiChannelImageConcatenation, channel_names: list[str]
) -> None:
    new_data = np.ones((4, 8, len(channel_names))) * 5.0
    new_combined_image = MultiChannelImage(
        data=xarray.DataArray(
            data=new_data,
            dims=("y", "x", "c"),
            coords={"c": channel_names},
            attrs={"bg_value": np.nan},
        )
    )

    with pytest.raises(ValueError, match="The new image must have the same shape as the original combined image"):
        concat_image.relabel_combined_image(new_combined_image)


def test_read_hdf5(mocker: MockerFixture) -> None:
    mock_read_hdf5 = mocker.patch.object(MultiChannelImage, "read_hdf5")
    mock_path = mocker.Mock(name="path", spec=[])
    # TODO nicer assertion
    assert MultiChannelImageConcatenation.read_hdf5(path=mock_path)._data == mock_read_hdf5.return_value
    mock_read_hdf5.assert_called_once_with(path=mock_path)


def test_write_hdf5(mocker: MockerFixture, concat_image: MultiChannelImageConcatenation) -> None:
    mock_write_hdf5 = mocker.patch.object(MultiChannelImage, "write_hdf5")
    mock_path = mocker.Mock(name="path", spec=[])
    concat_image.write_hdf5(path=mock_path)
    mock_write_hdf5.assert_called_once_with(path=mock_path)


if __name__ == "__main__":
    pytest.main()
