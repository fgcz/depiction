import unittest

import numpy as np
import xarray.testing
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class TestSparseRepresentation(unittest.TestCase):
    def test_flat_to_grid_when_real_bg(self):
        coordinates = DataArray(np.array([[0, 0], [1, 1], [1, 0]]), dims=["i", "d"])
        sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=0)
        expected_array = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        xarray.testing.assert_equal(expected_array, dense_values)

    def test_flat_to_grid_when_nan_bg(self):
        coordinates = DataArray(np.array([[0, 0], [1, 1], [1, 0]]), dims=["i", "d"])
        sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=np.nan)
        expected_array = DataArray([[[2], [np.nan]], [[4], [3]]], dims=["y", "x", "c"])
        xarray.testing.assert_equal(expected_array, dense_values)

    def test_grid_to_flat_when_real_bg(self):
        dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        sparse_values, coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=0)
        expected_sparse_values = DataArray([[2], [4], [3]], dims=["i", "c"])
        expected_coordinates = DataArray([[0, 0], [1, 0], [1, 1]], dims=["i", "d"])
        xarray.testing.assert_equal(expected_sparse_values, sparse_values)
        xarray.testing.assert_equal(expected_coordinates, coordinates)

    def test_grid_to_flat_when_nan_bg(self):
        dense_values = DataArray([[[2], [np.nan]], [[4], [3]]], dims=["y", "x", "c"])
        sparse_values, coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=np.nan)
        expected_sparse_values = DataArray([[2], [4], [3]], dims=["i", "c"])
        expected_coordinates = DataArray([[0, 0], [1, 0], [1, 1]], dims=["i", "d"])
        xarray.testing.assert_equal(expected_sparse_values, sparse_values)
        xarray.testing.assert_equal(expected_coordinates, coordinates)

    def test_grid_to_flat_when_none_bg(self):
        dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        sparse_values, coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=None)
        expected_sparse_values = DataArray([[2], [0], [4], [3]], dims=["i", "c"])
        expected_coordinates = DataArray([[0, 0], [0, 1], [1, 0], [1, 1]], dims=["i", "d"])
        xarray.testing.assert_equal(expected_sparse_values, sparse_values)
        xarray.testing.assert_equal(expected_coordinates, coordinates)


if __name__ == "__main__":
    unittest.main()
