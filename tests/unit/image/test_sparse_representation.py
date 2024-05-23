import unittest

import numpy as np
import xarray.testing
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class TestSparseRepresentation(unittest.TestCase):
    def test_flat_to_grid(self):
        coordinates = DataArray(np.array([[0, 0], [1, 1], [1, 0]]), dims=["i", "d"])
        sparse_values = DataArray([[2, 3, 4]], dims=["c", "i"])
        dense_values = SparseRepresentation.flat_to_grid(sparse_values, coordinates, background_value=0)
        expected_array = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        xarray.testing.assert_equal(expected_array, dense_values)

    def test_grid_to_flat_when_real_bg(self):
        dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        sparse_values, coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=0)
        expected_sparse_values = DataArray([[2, 3, 4]], dims=["i", "c"])
        expected_coordinates = DataArray([[0, 0], [1, 1], [1, 0]], dims=["i", "d"])
        xarray.testing.assert_equal(expected_sparse_values, sparse_values)
        xarray.testing.assert_equal(expected_coordinates, coordinates)

    def test_grid_to_flat_when_nan_bg(self):
        pass

    def test_grid_to_flat_when_none_bg(self):
        dense_values = DataArray([[[2], [0]], [[4], [3]]], dims=["y", "x", "c"])
        sparse_values, coordinates = SparseRepresentation.grid_to_flat(dense_values, bg_value=None)
        expected_sparse_values = DataArray([[2, 0, 4, 3]], dims=["i", "c"])
        expected_coordinates = DataArray([[0, 0], [0, 1], [1, 0], [1, 1]], dims=["i", "d"])
        xarray.testing.assert_equal(expected_sparse_values, sparse_values)
        xarray.testing.assert_equal(expected_coordinates, coordinates)


if __name__ == "__main__":
    unittest.main()

# class SparseRepresentation:
#    """Utility to convert to and from sparse representation. Even when memory usage is not a concern, some operations
#    can be simpler or require the sparse format, so this utility can be useful.
#
#    The basic idea is that an image is either stored in a dense/grid format or a sparse/flat format.
#    - Dense format: (y, x, c) for a 2D image with c channels.
#    - Sparse format: (spec_id, c) for a 2D image with c channels, plus (spec_id, 2) for the coordinates.
#
#    At the moment the functionality does not support z-axis, however it's built in a way that this functionality could
#    easily be added on top by using DataArray which annotates the dimensions.
#    """
#
#    @classmethod
#    def flat_to_grid(cls, sparse_values: DataArray, coordinates: DataArray, background_value: float) -> DataArray:
#        """Converts the sparse image representation into a dense image representation.
#        :param sparse_values: DataArray with "i" (index of value) and "c" (channel) dimensions
#        :param coordinates: DataArray with "i" (index of value) and "d" (dimension) dimensions
#        :param background_value: the value to use for the background
#        :return: DataArray with "y", "x", and "c" dimensions
#        """
#        n_channels = sparse_values.sizes["c"]
#        print(f"{n_channels=}")
#        sparse_values = sparse_values.transpose("i", "c").values
#        coordinates = coordinates.transpose("i", "d").values
#
#        coordinates_extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
#        coordinates_shifted = coordinates - coordinates.min(axis=0)
#
#        dtype = np.promote_types(sparse_values.dtype, np.obj2sctype(type(background_value)))
#        values_grid = np.full((coordinates_extent[0], coordinates_extent[1], n_channels), fill_value=background_value,
#                              dtype=dtype)
#        print(values_grid.shape)
#        for i_channel in range(n_channels):
#            values_grid[tuple(coordinates_shifted.T) + (i_channel,)] = sparse_values[:, i_channel]
#
#        return DataArray(values_grid, dims=("y", "x", "c"))
#
#    @classmethod
#    def grid_to_flat(cls, grid_values: DataArray, bg_value: float | None) -> tuple[DataArray, DataArray]:
#        """Converts the dense image representation into a sparse image representation.
#        :param grid_values: DataArray with "y", "x", and "c" dimensions
#        :param bg_value: the value to use for the background, or None to use all values
#            (i.e. the result won't be actually sparse)
#        :return: a tuple with two DataArrays, the first with "i" (index of value) and "c" (channel) dimensions,
#            and the second with "i" (index of value) and "d" (dimension) dimensions
#        """
#        grid_values = grid_values.transpose("y", "x", "c").values
#        if bg_value is None:
#            # use all values
#            sparse_values = grid_values.reshape(-1, grid_values.shape[-1])
#            coordinates = np.array(list(np.ndindex(grid_values.shape[:-1])))
#        else:
#            # select the background
#            if np.isnan(bg_value):
#                mask = np.all(np.isnan(grid_values), axis=-1)
#            else:
#                mask = np.all(grid_values == bg_value, axis=-1)
#            sparse_values = grid_values[~mask]
#            coordinates = np.array(list(np.ndindex(grid_values.shape[:-1])))[~mask]
#
#        return DataArray(sparse_values, dims=("i", "c")), DataArray(coordinates, dims=("i", "d"))
