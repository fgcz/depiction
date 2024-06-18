import numpy as np
import pytest
import sparse
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


if __name__ == "__main__":
    pytest.main()
