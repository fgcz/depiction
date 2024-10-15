from datetime import timedelta

import hypothesis
import numpy as np
import pytest
import xarray.testing
from hypothesis import given, strategies
from sparse import GCXS
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction.image.smoothing.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware
from depiction.image.xarray_helper import XarrayHelper


@pytest.fixture(autouse=True)
def _setup_test(treat_warnings_as_error):
    pass


@pytest.fixture
def mock_kernel_size():
    return 5


@pytest.fixture
def mock_kernel_std():
    return 1.0


@pytest.fixture
def mock_use_interpolation():
    return False


@pytest.fixture
def mock_smooth(mock_kernel_size, mock_kernel_std, mock_use_interpolation):
    return SpatialSmoothingSparseAware(
        kernel_size=mock_kernel_size,
        kernel_std=mock_kernel_std,
        use_interpolation=mock_use_interpolation,
    )


def _convert_array(arr: DataArray, variant: str) -> DataArray:
    if variant == "dense":
        return arr
    elif variant == "sparse":
        values = GCXS.from_numpy(arr)
        return DataArray(values, dims=arr.dims, coords=arr.coords, attrs=arr.attrs, name=arr.name)


@pytest.fixture()
def dense_data():
    return DataArray(
        np.concatenate([np.ones((5, 5, 1)), np.zeros((5, 5, 1))], axis=0),
        dims=("y", "x", "c"),
        coords={"c": ["channel"]},
    )


@pytest.fixture(params=["dense", "sparse"])
def image(request, dense_data):
    # TODO the whole idea of the old test of using nan and zero is not so relevant anymore, maybe it can be simplified
    data = _convert_array(dense_data, request.param)
    is_fg = data.isel(c=0) != 0
    return MultiChannelImage(data=data, is_foreground=is_fg)


def test_smooth_image_when_unchanged(mock_smooth, image):
    smoothed = mock_smooth.smooth_image(image)
    xarray.testing.assert_equal(smoothed.fg_mask, image.fg_mask)
    np.testing.assert_array_almost_equal(1.0, smoothed.data_spatial[:5, :, 0], decimal=8)
    np.testing.assert_array_almost_equal(0.0, smoothed.data_spatial[-5:, :, 0], decimal=8)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
@given(fill_value=strategies.floats(min_value=0, max_value=1e12, allow_subnormal=False))
@hypothesis.settings(
    deadline=timedelta(seconds=1), suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_smooth_preserves_values(mock_smooth, fill_value, variant):
    dense_data = DataArray(np.full((2, 5, 1), fill_value=fill_value), dims=("y", "x", "c"), coords={"c": ["channel"]})
    is_foreground = DataArray(np.full((2, 5), fill_value=True), dims=("y", "x"))
    image = MultiChannelImage(data=_convert_array(dense_data, variant), is_foreground=is_foreground)
    smoothed = mock_smooth.smooth_image(image)
    xarray.testing.assert_equal(smoothed.fg_mask, image.fg_mask)
    xarray.testing.assert_allclose(smoothed.data_spatial, dense_data, rtol=1e-8)


# @pytest.mark.parametrize("variant", ["dense", "sparse"])
# def test_smooth_when_bg_nan(mock_smooth, variant):
#    dense_image = DataArray(
#        np.concatenate([np.full((5, 5, 1), np.nan), np.ones((5, 5, 1)), np.zeros((5, 5, 1))]),
#        dims=("y", "x", "c"),
#    )
#    image = _convert_array(dense_image, variant)
#    smoothed = mock_smooth.smooth(image, bg_value=np.nan)
#    np.testing.assert_array_equal(np.nan, smoothed.values[:5, 0])
#    np.testing.assert_array_almost_equal(1.0, smoothed.values[5:8, :, 0], decimal=8)
#    np.testing.assert_array_almost_equal(0.0, smoothed.values[-3:, :, 0], decimal=8)
#    smoothed_part = smoothed.values[8:12, :, 0]
#    for i_value, value in enumerate([0.94551132, 0.70130997, 0.29869003, 0.05448868]):
#        np.testing.assert_array_almost_equal(value, smoothed_part[i_value, :], decimal=8)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_smooth_casts_when_integer(mock_smooth, variant):
    data_full = DataArray(np.full((2, 5, 1), fill_value=10, dtype=int), dims=("y", "x", "c"), coords={"c": ["channel"]})
    is_foreground = DataArray(np.full((2, 5), fill_value=True), dims=("y", "x"))
    image = MultiChannelImage(data=_convert_array(data_full, variant), is_foreground=is_foreground)
    res_values = mock_smooth.smooth_image(image=image)
    assert res_values.dtype == np.float64
    if variant == "dense":
        xarray.testing.assert_allclose(res_values.data_spatial, image.data_spatial)
    else:
        np.testing.assert_allclose(
            XarrayHelper.ensure_dense(res_values.data_spatial).values,
            XarrayHelper.ensure_dense(image.data_spatial).values,
        )


@pytest.mark.parametrize("mock_use_interpolation", [True])
def test_smooth_dense_when_use_interpolation(mock_smooth):
    mock_data = DataArray(np.full((9, 5, 1), fill_value=3.0), dims=("y", "x", "c"), coords={"c": ["channel"]})
    mock_data[4, 2, 0] = np.nan
    is_foreground = ~mock_data.isel(c=0).isnull()
    mock_image = MultiChannelImage(data=mock_data, is_foreground=is_foreground)
    smoothed = mock_smooth.smooth_image(mock_image)
    assert np.sum(smoothed.bg_mask) == 0
    np.testing.assert_almost_equal(smoothed.data_spatial[4, 2], 3, decimal=6)


@pytest.mark.parametrize("mock_kernel_size, mock_kernel_std", [(3, 1.0)])
def test_gaussian_kernel(mock_smooth):
    expected_arr = np.array(
        [
            [0.075114, 0.123841, 0.075114],
            [0.123841, 0.20418, 0.123841],
            [0.075114, 0.123841, 0.075114],
        ]
    )
    np.testing.assert_allclose(expected_arr, mock_smooth.gaussian_kernel, atol=1e-6)
