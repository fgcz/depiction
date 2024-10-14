from enum import Enum

import numpy as np
import pytest
import xarray.testing
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class Variant(Enum):
    """The samples are all images, and commented for clarity in the following coordinate system
    ^ +y
    + -> +x
    """

    Simple = "1_simple"
    MultiChannel = "2_multi"
    Offset = "3_offset"


@pytest.fixture()
def sample(request):
    if request.param == Variant.Simple:
        # +----+----+
        # | 5  | NA |
        # +----+----+
        # | 4  | 6  |
        # +----+----+
        return (
            DataArray([[[4.0], [6]], [[5], [np.nan]]], dims=["y", "x", "c"], coords={"x": [0, 1], "y": [0, 1]}),
            DataArray([[4.0], [5], [6]], dims=["i", "c"]),
            DataArray([[0, 0], [0, 1], [1, 0]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )
    elif request.param == Variant.MultiChannel:
        # +-------+--------+
        # | 5, 15 | NA, NA |
        # +-------+--------+
        # | 4, 14 | 6, 16  |
        # +-------+--------+
        return (
            DataArray(
                [[[4, 14], [6, 16]], [[5.0, 15], [np.nan, np.nan]]],
                dims=["y", "x", "c"],
                coords={"x": [0, 1], "y": [0, 1]},
            ),
            DataArray([[4.0, 14], [5, 15], [6, 16]], dims=["i", "c"]),
            DataArray([[0, 0], [0, 1], [1, 0]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )
    elif request.param == Variant.Offset:
        # +----+----+
        # | NA | 5  |
        # +----+----+
        # | NA | NA |
        # +----+----+
        # | NA | NA |
        # +----+----+
        return (
            DataArray(
                [[[np.nan], [5]], [[np.nan], [np.nan]], [[np.nan], [np.nan]]],
                dims=["y", "x", "c"],
                coords={"x": [0, 1], "y": [0, 1, 2]},
            ),
            DataArray([[5]], dims=["i", "c"]),
            DataArray([[1, 2]], dims=["i", "d"], coords={"d": ["x", "y"]}),
        )


@pytest.fixture()
def spatial_array(sample) -> DataArray:
    return sample[0]


@pytest.fixture()
def flat_data(sample) -> DataArray:
    return sample[1]


@pytest.fixture()
def flat_coords(sample) -> DataArray:
    return sample[2]


@pytest.mark.parametrize("sample", [Variant.Simple, Variant.MultiChannel], indirect=True)
def test_flat_to_spatial(flat_data, flat_coords, spatial_array):
    result_values, result_is_fg = SparseRepresentation.flat_to_spatial(flat_data, flat_coords, bg_value=np.nan)
    xarray.testing.assert_equal(result_values, spatial_array)
    xarray.testing.assert_equal(result_is_fg, spatial_array.isel(c=0).notnull())


@pytest.mark.parametrize("sample", [Variant.Offset], indirect=True)
def test_flat_to_spatial_when_offset(flat_data, flat_coords, spatial_array):
    result_values, result_is_fg = SparseRepresentation.flat_to_spatial(flat_data, flat_coords, bg_value=np.nan)
    xarray.testing.assert_equal(
        result_values, xarray.DataArray([[[5]]], dims=["y", "x", "c"], coords={"x": [1], "y": [2]})
    )
    xarray.testing.assert_equal(result_is_fg, xarray.DataArray([[True]], dims=["y", "x"], coords={"x": [1], "y": [2]}))


# def test_dense_to_sparse_when_offset(self) -> None:
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
