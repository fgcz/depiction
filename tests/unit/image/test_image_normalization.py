import numpy as np
import pytest
import xarray as xr

from depiction.image.image_normalization import ImageNormalizationVariant, ImageNormalization
from depiction.image.multi_channel_image import MultiChannelImage


@pytest.fixture
def image_normalizer():
    return ImageNormalization()


@pytest.fixture
def single_image():
    return xr.DataArray(
        data=[[[2, 0], [0, 2]], [[1, 1], [4, 1]], [[0, 0], [0, 0]]], dims=["y", "x", "c"], attrs={"bg_value": 0}
    )


@pytest.fixture
def multiple_images():
    return xr.DataArray(data=[[[[2, 0]]], [[[0, 3]]]], dims=["whatever", "y", "x", "c"], attrs={"bg_value": 0})


def test_normalize_xarray_single_vec_norm(image_normalizer, single_image):
    norm_vec = image_normalizer.normalize_xarray(single_image, variant=ImageNormalizationVariant.VEC_NORM)
    expected = xr.DataArray(
        data=[[[1, 0], [0, 1]], [[0.707107, 0.707107], [0.970143, 0.242536]], [[0, 0], [0, 0]]],
        dims=["y", "x", "c"],
        attrs={"bg_value": 0},
    )
    xr.testing.assert_allclose(expected, norm_vec)
    assert norm_vec.attrs["bg_value"] == single_image.attrs["bg_value"]


def test_normalize_xarray_single_vec_norm_with_nans(image_normalizer):
    image_with_nans = xr.DataArray(
        data=[[[2, np.nan], [0, 2]], [[1, 1], [4, np.nan]], [[np.nan, 0], [0, 0]]],
        dims=["y", "x", "c"],
        attrs={"bg_value": np.nan},
    )
    norm_vec = image_normalizer.normalize_xarray(image_with_nans, variant=ImageNormalizationVariant.VEC_NORM)
    expected = xr.DataArray(
        data=[[[1, np.nan], [0, 1]], [[0.707107, 0.707107], [1, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]]],
        dims=["y", "x", "c"],
        attrs={"bg_value": np.nan},
    )
    xr.testing.assert_allclose(expected, norm_vec)
    assert np.isnan(norm_vec.attrs["bg_value"])


def test_normalize_xarray_single_std(image_normalizer, single_image):
    norm_std = image_normalizer.normalize_xarray(single_image, variant=ImageNormalizationVariant.STD)
    expected = xr.DataArray(
        data=[[[1.0, -1.0], [-1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        dims=["y", "x", "c"],
        attrs={"bg_value": 0},
    )
    xr.testing.assert_allclose(norm_std, expected, rtol=1e-5)
    assert norm_std.attrs["bg_value"] == single_image.attrs["bg_value"]


def test_normalize_xarray_multiple_vec_norm(image_normalizer, multiple_images):
    norm_vec = image_normalizer.normalize_xarray(multiple_images, variant=ImageNormalizationVariant.VEC_NORM)
    expected = xr.DataArray(
        data=[[[[1, 0]]], [[[0, 1]]]],
        dims=["whatever", "y", "x", "c"],
        coords={"whatever": [0, 1]},
        attrs={"bg_value": 0},
    )
    xr.testing.assert_allclose(expected, norm_vec)
    assert norm_vec.attrs["bg_value"] == multiple_images.attrs["bg_value"]


def test_normalize_xarray_multiple_std(image_normalizer, multiple_images):
    norm_std = image_normalizer.normalize_xarray(multiple_images, variant=ImageNormalizationVariant.STD)
    expected = xr.DataArray(
        data=[[[[1, -1]]], [[[-1, 1]]]],
        dims=["whatever", "y", "x", "c"],
        coords={"whatever": [0, 1]},
        attrs={"bg_value": 0},
    )
    xr.testing.assert_allclose(expected, norm_std)
    assert norm_std.attrs["bg_value"] == multiple_images.attrs["bg_value"]


def test_normalize_image(image_normalizer, single_image):
    multi_channel_image = MultiChannelImage(single_image)
    normalized_image = image_normalizer.normalize_image(multi_channel_image, variant=ImageNormalizationVariant.VEC_NORM)
    assert isinstance(normalized_image, MultiChannelImage)
    xr.testing.assert_allclose(
        normalized_image.data_spatial,
        image_normalizer.normalize_xarray(single_image, variant=ImageNormalizationVariant.VEC_NORM),
    )
    assert normalized_image.data_spatial.attrs["bg_value"] == single_image.attrs["bg_value"]


def test_missing_dimensions(image_normalizer):
    invalid_image = xr.DataArray(data=[[2, 0], [0, 2]], dims=["y", "x"])
    with pytest.raises(ValueError, match="Missing required dimensions: {'c'}"):
        image_normalizer.normalize_xarray(invalid_image, variant=ImageNormalizationVariant.VEC_NORM)


def test_multiple_index_dimensions(image_normalizer):
    invalid_image = xr.DataArray(data=[[[[[2, 0]]], [[[0, 3]]]]], dims=["dim1", "dim2", "y", "x", "c"])
    with pytest.raises(NotImplementedError, match="Multiple index columns are not supported yet."):
        image_normalizer.normalize_xarray(invalid_image, variant=ImageNormalizationVariant.VEC_NORM)


def test_unknown_variant(image_normalizer, single_image):
    with pytest.raises(NotImplementedError, match="Unknown variant: unknown"):
        image_normalizer.normalize_xarray(single_image, variant="unknown")


if __name__ == "__main__":
    pytest.main()
