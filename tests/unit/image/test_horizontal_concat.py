import copy

import numpy as np
import pytest
import xarray as xr
from depiction.image.horizontal_concat import horizontal_concat
from depiction.image.multi_channel_image import MultiChannelImage


@pytest.fixture
def sample_image():
    data = xr.DataArray(
        np.random.rand(2, 3, 4),
        dims=["c", "y", "x"],
        coords={"c": ["red", "green"], "y": [0, 1, 2], "x": [0, 1, 2, 3]},
        attrs={"bg_value": 0.0},
    )
    return MultiChannelImage(data)


def test_horizontal_concat_success(sample_image):
    image1 = sample_image
    image2 = copy.deepcopy(sample_image)

    result = horizontal_concat([image1, image2])

    assert isinstance(result, MultiChannelImage)
    assert result.data_spatial.shape == (3, 8, 2)  # y, x, c
    assert list(result.data_spatial.x.values) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_horizontal_concat_when_add_index(sample_image):
    image1 = sample_image
    image2 = copy.deepcopy(sample_image)

    result = horizontal_concat([image1, image2], add_index=True)

    assert isinstance(result, MultiChannelImage)
    assert result.data_spatial.shape == (3, 8, 3)  # y, x, c
    assert list(result.data_spatial.x.values) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(result.data_spatial.c.values) == ["red", "green", "image_index"]

    assert result.data_spatial.sel(c="image_index").values.tolist() == [
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]


def test_horizontal_concat_single_image(sample_image):
    with pytest.raises(ValueError, match="At least two images are required for concatenation."):
        horizontal_concat([sample_image])


def test_horizontal_concat_x_coordinate_shift(sample_image):
    image1 = sample_image
    image2 = copy.deepcopy(sample_image)

    result = horizontal_concat([image1, image2])

    assert list(result.data_spatial.x.values) == [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("num_images", [2, 3, 5])
def test_horizontal_concat_multiple_images(sample_image, num_images):
    images = [copy.deepcopy(sample_image) for _ in range(num_images)]

    result = horizontal_concat(images)

    assert isinstance(result, MultiChannelImage)
    assert result.data_spatial.shape == (3, 4 * num_images, 2)  # y, x, c
    assert list(result.data_spatial.x.values) == list(range(4 * num_images))


def test_horizontal_concat_data_order(sample_image):
    image1 = sample_image
    image2 = copy.deepcopy(sample_image)

    result = horizontal_concat([image1, image2])

    assert result.data_spatial.dims == ("y", "x", "c")
    assert result.data_spatial.shape == (3, 8, 2)  # y, x, c


if __name__ == "__main__":
    pytest.main()
