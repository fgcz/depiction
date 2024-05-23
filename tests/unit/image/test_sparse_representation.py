import unittest

import numpy as np
import xarray.testing
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class TestSparseRepresentation(unittest.TestCase):
    def test_flat_to_grid_when_real_bg(self):
        coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
        sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=0)
        expected_array = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        xarray.testing.assert_equal(expected_array, dense_values)

    def test_flat_to_grid_when_nan_bg(self):
        coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
        sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=np.nan)
        expected_array = DataArray([[[2], [np.nan]], [[4], [3]]], dims=["y", "x", "c"])
        xarray.testing.assert_equal(expected_array, dense_values)

    def test_flat_to_grid_when_multi_channel(self):
        coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
        sparse_values = DataArray([[1, 2], [3, 4], [5, 6]], dims=["i", "c"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=0)
        expected_array = DataArray([[[1, 2], [0, 0]], [[5, 6], [3, 4]]], dims=["y", "x", "c"])
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

    def test_round_trip(self):
        # NOTE: This is not really a unit test (TODO keep it?)
        sparse_values = DataArray([[2, 3], [4, 5], [6, 7]], dims=["i", "c"])
        coordinates = DataArray([[0, 0], [1, 0], [2, 1]], dims=["i", "d"])

        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=0)
        new_sparse_values, new_coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=0)

        xarray.testing.assert_equal(sparse_values, new_sparse_values)
        xarray.testing.assert_equal(coordinates, new_coordinates)


if __name__ == "__main__":
    unittest.main()
