import unittest

import numpy as np
import xarray.testing
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class TestSparseRepresentation(unittest.TestCase):
    def setUp(self) -> None:
        self.define_samples()

    def define_samples(self) -> None:
        """The samples are all images, and commented for clarity in the following coordinate system
        ^ +y
        + -> +x
        """
        # Simple example:
        # +----+----+
        # | 5  | NA |
        # +----+----+
        # | 4  | 6  |
        # +----+----+
        self.dense_1_simple = DataArray([[[4.0], [6]], [[5], [np.nan]]], dims=["y", "x", "c"])
        self.sparse_1_simple = (
            DataArray([[4.0], [5], [6]], dims=["i", "c"]),
            DataArray([[0, 0], [0, 1], [1, 0]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )

        # Multi-channel example
        # +-------+--------+
        # | 5, 15 | NA, NA |
        # +-------+--------+
        # | 4, 14 | 6, 16  |
        # +-------+--------+
        self.dense_2_multi = DataArray([[[4, 14], [6, 16]], [[5.0, 15], [np.nan, np.nan]]], dims=["y", "x", "c"])
        self.sparse_2_multi = (
            DataArray([[4.0, 14], [5, 15], [6, 16]], dims=["i", "c"]),
            DataArray([[0, 0], [0, 1], [1, 0]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )

        # Offset example
        # +----+----+
        # | NA | 5  |
        # +----+----+
        # | NA | NA |
        # +----+----+
        # | NA | NA |
        # +----+----+
        self.dense_3_offset = DataArray(
            [[[np.nan], [5]], [[np.nan], [np.nan]], [[np.nan], [np.nan]]], dims=["y", "x", "c"]
        )
        self.sparse_3_offset = (
            DataArray([[5]], dims=["i", "c"]),
            DataArray([[1, 2]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )

    def test_dense_to_sparse_when_simple(self) -> None:
        values, coords = SparseRepresentation.dense_to_sparse(grid_values=self.dense_1_simple, bg_value=np.nan)
        xarray.testing.assert_equal(self.sparse_1_simple[1], coords)
        xarray.testing.assert_equal(self.sparse_1_simple[0], values)

    def test_dense_to_sparse_when_multi_channel(self) -> None:
        values, coords = SparseRepresentation.dense_to_sparse(grid_values=self.dense_2_multi, bg_value=np.nan)
        xarray.testing.assert_equal(self.sparse_2_multi[1], coords)
        xarray.testing.assert_equal(self.sparse_2_multi[0], values)

    #def test_dense_to_sparse_when_offset(self) -> None:
    #    values, coords = SparseRepresentation.dense_to_sparse(grid_values=self.dense_3_offset, bg_value=np.nan)
    #    xarray.testing.assert_equal(self.sparse_3_offset[1], coords)
    #    xarray.testing.assert_equal(self.sparse_3_offset[0], values)

    # def test_sparse_to_dense_when_real_bg(self):
    #    coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
    #    sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
    #    dense_values = SparseRepresentation.sparse_to_dense(sparse_values, coordinates, background_value=0)
    #    expected_array = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
    #    xarray.testing.assert_equal(expected_array, dense_values)

    # def test_sparse_to_dense_when_nan_bg(self):
    #    coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
    #    sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
    #    dense_values = SparseRepresentation.sparse_to_dense(sparse_values, coordinates, background_value=np.nan)
    #    expected_array = DataArray([[[2], [np.nan]], [[4], [3]]], dims=["y", "x", "c"])
    #    xarray.testing.assert_equal(expected_array, dense_values)

    # def test_sparse_to_dense_when_multi_channel(self):
    #    coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
    #    sparse_values = DataArray([[1, 2], [3, 4], [5, 6]], dims=["i", "c"])
    #    dense_values = SparseRepresentation.sparse_to_dense(sparse_values, coordinates, background_value=0)
    #    expected_array = DataArray([[[1, 2], [0, 0]], [[5, 6], [3, 4]]], dims=["y", "x", "c"])
    #    xarray.testing.assert_equal(expected_array, dense_values)

    # def test_dense_to_sparse_when_real_bg(self):
    #    dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
    #    sparse_values, coordinates = SparseRepresentation.dense_to_sparse(dense_values, bg_value=0)
    #    expected_sparse_values = DataArray([[2], [4], [3]], dims=["i", "c"])
    #    expected_coordinates = DataArray([[0, 0], [1, 0], [1, 1]], dims=["i", "d"])
    #    xarray.testing.assert_equal(expected_sparse_values, sparse_values)
    #    xarray.testing.assert_equal(expected_coordinates, coordinates)

    # def test_dense_to_sparse_when_nan_bg(self):
    #    dense_values = DataArray([[[2], [np.nan]], [[4], [3]]], dims=["y", "x", "c"])
    #    sparse_values, coordinates = SparseRepresentation.dense_to_sparse(dense_values, bg_value=np.nan)
    #    expected_sparse_values = DataArray([[2], [4], [3]], dims=["i", "c"])
    #    expected_coordinates = DataArray([[0, 0], [1, 0], [1, 1]], dims=["i", "d"])
    #    xarray.testing.assert_equal(expected_sparse_values, sparse_values)
    #    xarray.testing.assert_equal(expected_coordinates, coordinates)

    # def test_dense_to_sparse_when_none_bg(self):
    #    dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
    #    sparse_values, coordinates = SparseRepresentation.dense_to_sparse(dense_values, bg_value=None)
    #    expected_sparse_values = DataArray([[2], [0], [4], [3]], dims=["i", "c"])
    #    expected_coordinates = DataArray([[0, 0], [0, 1], [1, 0], [1, 1]], dims=["i", "d"])
    #    xarray.testing.assert_equal(expected_sparse_values, sparse_values)
    #    xarray.testing.assert_equal(expected_coordinates, coordinates)

    # def test_dense_to_sparse_coords(self):
    #    dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
    #    coordinates = DataArray([[0, 0], [0, 1]])
    #    # sparse_values =

    # def test_round_trip(self):
    #    # NOTE: This is not really a unit test (TODO keep it?)
    #    sparse_values = DataArray([[2, 3], [4, 5], [6, 7]], dims=["i", "c"])
    #    coordinates = DataArray([[0, 0], [1, 0], [2, 1]], dims=["i", "d"])

    #    dense_values = SparseRepresentation.sparse_to_dense(sparse_values, coordinates, background_value=0)
    #    new_sparse_values, new_coordinates = SparseRepresentation.dense_to_sparse(dense_values, bg_value=0)

    #    xarray.testing.assert_equal(sparse_values, new_sparse_values)
    #    xarray.testing.assert_equal(coordinates, new_coordinates)


if __name__ == "__main__":
    unittest.main()
