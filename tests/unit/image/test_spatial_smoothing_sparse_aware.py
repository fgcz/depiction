from datetime import timedelta

import hypothesis
import numpy as np
import pytest
from hypothesis import given, strategies
from sparse import GCXS
from xarray import DataArray

from depiction.image.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware


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


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_smooth_when_unchanged(mock_smooth, variant):
    image = DataArray(np.concatenate([np.ones((5, 5, 1)), np.zeros((5, 5, 1))], axis=0), dims=("y", "x", "c"))
    image = _convert_array(image, variant)
    smoothed = mock_smooth.smooth(image, bg_value=0)
    np.testing.assert_array_almost_equal(1.0, smoothed.values[:5, :, 0], decimal=8)
    np.testing.assert_array_almost_equal(0.0, smoothed.values[-5:, :, 0], decimal=8)
    assert smoothed.dims == ("y", "x", "c")


@pytest.mark.parametrize("variant", ["dense", "sparse"])
@given(fill_value=strategies.floats(min_value=0, max_value=1e12, allow_subnormal=False))
@hypothesis.settings(
    deadline=timedelta(seconds=1), suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_smooth_preserves_values(mock_smooth, fill_value, variant):
    dense_image = DataArray(np.full((2, 5, 1), fill_value=fill_value), dims=("y", "x", "c"))
    image = _convert_array(dense_image, variant)
    smoothed = mock_smooth.smooth(image, bg_value=0)
    np.testing.assert_allclose(dense_image.values, smoothed.values, rtol=1e-8)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_smooth_when_bg_nan(mock_smooth, variant):
    dense_image = DataArray(
        np.concatenate([np.full((5, 5, 1), np.nan), np.ones((5, 5, 1)), np.zeros((5, 5, 1))]),
        dims=("y", "x", "c"),
    )
    image = _convert_array(dense_image, variant)
    smoothed = mock_smooth.smooth(image, bg_value=np.nan)
    np.testing.assert_array_equal(np.nan, smoothed.values[:5, 0])
    np.testing.assert_array_almost_equal(1.0, smoothed.values[5:8, :, 0], decimal=8)
    np.testing.assert_array_almost_equal(0.0, smoothed.values[-3:, :, 0], decimal=8)
    smoothed_part = smoothed.values[8:12, :, 0]
    for i_value, value in enumerate([0.94551132, 0.70130997, 0.29869003, 0.05448868]):
        np.testing.assert_array_almost_equal(value, smoothed_part[i_value, :], decimal=8)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_smooth_casts_when_integer(mock_smooth, variant):
    image_dense = DataArray(np.full((2, 5, 1), fill_value=10, dtype=int), dims=("y", "x", "c"))
    image = _convert_array(image_dense, variant)
    res_values = mock_smooth.smooth(image=image)
    assert res_values.dtype == np.float64
    np.testing.assert_allclose(image_dense.values, res_values.values, rtol=1e-8)


@pytest.mark.parametrize("mock_use_interpolation", [True])
def test_smooth_dense_when_use_interpolation(mock_smooth):
    mock_image = np.full((9, 5), fill_value=3.0)
    mock_image[4, 2] = np.nan
    smoothed = mock_smooth._smooth_dense(image_2d=mock_image, bg_value=np.nan)
    assert np.sum(np.isnan(smoothed)) == 0
    np.testing.assert_almost_equal(smoothed[4, 2], 3, decimal=6)


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
