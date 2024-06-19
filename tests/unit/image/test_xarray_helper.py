import numpy as np
import pytest
import sparse
import xarray.testing
from xarray import DataArray

from depiction.image.xarray_helper import XarrayHelper


@pytest.fixture
def dense_dataarray() -> DataArray:
    data = np.array([[1, 2, 3], [4, 5, 6]])
    return DataArray(data, dims=["x", "y"])


@pytest.fixture
def sparse_dataarray() -> DataArray:
    data = sparse.COO.from_numpy(np.array([[1, 0, 0], [0, 0, 6]]))
    return DataArray(data, dims=["x", "y"])


def test_is_sparse_dense(dense_dataarray) -> None:
    assert not XarrayHelper.is_sparse(dense_dataarray)


def test_is_sparse_sparse(sparse_dataarray) -> None:
    assert XarrayHelper.is_sparse(sparse_dataarray)


def test_ensure_dense_dense(dense_dataarray) -> None:
    result = XarrayHelper.ensure_dense(dense_dataarray)
    assert isinstance(result.data, np.ndarray)
    assert np.array_equal(result.data, dense_dataarray.data)


def test_ensure_dense_sparse(sparse_dataarray) -> None:
    result = XarrayHelper.ensure_dense(sparse_dataarray)
    assert isinstance(result.data, np.ndarray)
    assert np.array_equal(result.data, sparse_dataarray.data.todense())


def test_ensure_dense_copy(dense_dataarray) -> None:
    result = XarrayHelper.ensure_dense(dense_dataarray, copy=True)
    assert isinstance(result.data, np.ndarray)
    assert np.array_equal(result.data, dense_dataarray.data)
    assert result is not dense_dataarray
    assert np.shares_memory(result.data, dense_dataarray.data) == False


def test_ensure_dense_no_copy(dense_dataarray) -> None:
    result = XarrayHelper.ensure_dense(dense_dataarray, copy=False)
    assert result is dense_dataarray


def dummy_function(dataarray: DataArray) -> DataArray:
    # A simple function for testing that doubles the values
    return dataarray * 2


@pytest.fixture
def array_spatial() -> DataArray:
    return DataArray(
        [[[1, 2], [np.nan, np.nan]], [[np.nan, 6], [7, 8]]],
        dims=["y", "x", "c"],
        coords={"y": [0, 1], "x": [0, 1], "c": ["a", "b"]},
    )


@pytest.fixture
def array_flat(array_spatial) -> DataArray:
    return array_spatial.stack(i=("x", "y")).dropna("i", how="all")


@pytest.fixture
def array_flat_transposed(array_spatial) -> DataArray:
    return array_spatial.stack(i=("y", "x")).dropna("i", how="all")


def test_apply_on_spatial_view_array_spatial(array_spatial) -> None:
    result = XarrayHelper.apply_on_spatial_view(array_spatial, dummy_function)
    xarray.testing.assert_equal(result, array_spatial * 2)


def test_apply_on_spatial_view_array_flat_no_nan(array_flat) -> None:
    array_flat = array_flat.fillna(0)
    result = XarrayHelper.apply_on_spatial_view(array_flat, dummy_function)
    xarray.testing.assert_equal(result, array_flat * 2)


def test_apply_on_spatial_view_array_flat_with_nan(array_flat) -> None:
    result = XarrayHelper.apply_on_spatial_view(array_flat, dummy_function)
    xarray.testing.assert_equal(result, array_flat * 2)


def test_apply_on_spatial_view_array_flat_no_nan_transposed(array_flat_transposed) -> None:
    array_flat_transposed = array_flat_transposed.fillna(0)
    result = XarrayHelper.apply_on_spatial_view(array_flat_transposed, dummy_function)
    xarray.testing.assert_equal(result, array_flat_transposed * 2)


def test_apply_on_spatial_view_array_flat_with_nan_transposed(array_flat_transposed) -> None:
    result = XarrayHelper.apply_on_spatial_view(array_flat_transposed, dummy_function)
    xarray.testing.assert_equal(result, array_flat_transposed * 2)


if __name__ == "__main__":
    pytest.main()
