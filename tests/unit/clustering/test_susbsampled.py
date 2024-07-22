import numpy as np
import pandas as pd
import pytest
from depiction.clustering.subsampled import StratifiedGrid
from xarray import DataArray


@pytest.fixture()
def stratified_grid() -> StratifiedGrid:
    return StratifiedGrid(cells_x=2, cells_y=4)


def test_edges_x(stratified_grid):
    np.testing.assert_almost_equal(stratified_grid.edges_x, np.array([0, 0.5, 1]))


def test_edges_y(stratified_grid):
    np.testing.assert_almost_equal(stratified_grid.edges_y, np.array([0, 0.25, 0.5, 0.75, 1]))


def test_grid_limits_unit(stratified_grid):
    np.testing.assert_almost_equal(stratified_grid.grid_limits_unit(0), (0, 0, 0.5, 0.25))
    np.testing.assert_almost_equal(stratified_grid.grid_limits_unit(1), (0.5, 0, 1, 0.25))
    np.testing.assert_almost_equal(stratified_grid.grid_limits_unit(3), (0.5, 0.25, 1, 0.5))
    np.testing.assert_almost_equal(stratified_grid.grid_limits_unit(5), (0.5, 0.5, 1, 0.75))


def test_grid_limits_scaled(stratified_grid):
    array = DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"y": [-10, 10], "x": [-10, 10]})
    min_x, min_y, max_x, max_y = stratified_grid.grid_limits_scaled(0, array)
    assert min_x == pytest.approx(-10.0)
    assert min_y == pytest.approx(-10.0)
    assert max_x == pytest.approx(0.0)
    assert max_y == pytest.approx(-5.0)


def test_assign_points_basic(stratified_grid):
    coords = [(0, 0), (0.4, 0.2), (0.6, 0.6), (1, 0.8)]
    array = DataArray(
        np.ones(4),
        dims="i",
        coords={"i": pd.MultiIndex.from_tuples(coords, names=["x", "y"])},
    )

    assignments = stratified_grid.assign_points(array)
    print(assignments)

    assert len(assignments) == 8  # 2x4 grid
    assert set(assignments.keys()) == set(range(8))
    assert sum(len(v) for v in assignments.values()) == 4

    np.testing.assert_array_equal(assignments[0], [0])
    np.testing.assert_array_equal(assignments[1], [])
    np.testing.assert_array_equal(assignments[2], [1])
    np.testing.assert_array_equal(assignments[3], [])
    np.testing.assert_array_equal(assignments[4], [])
    np.testing.assert_array_equal(assignments[5], [2])
    np.testing.assert_array_equal(assignments[6], [])
    np.testing.assert_array_equal(assignments[7], [3])


def test_assign_points_multiple_in_cell(stratified_grid):
    data = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.15], [0.8, 0.8]])
    array = DataArray(data, dims=("i", "c"), coords={"i": range(4), "x": ("i", data[:, 0]), "y": ("i", data[:, 1])})

    assignments = stratified_grid.assign_points(array)

    assert len(assignments) == 8
    np.testing.assert_array_equal(assignments[0], [0, 1, 2])
    np.testing.assert_array_equal(assignments[7], [3])
    assert all(len(assignments[i]) == 0 for i in [1, 2, 3, 4, 5, 6])


def test_assign_points_missing_dimension(stratified_grid):
    data = np.array([[0, 0], [1, 1]])
    array = DataArray(
        data, dims=("wrong_dim", "coord"), coords={"x": ("wrong_dim", data[:, 0]), "y": ("wrong_dim", data[:, 1])}
    )

    with pytest.raises(ValueError, match="DataArray must have a dimension 'i'."):
        stratified_grid.assign_points(array)


def test_assign_points_missing_coordinates(stratified_grid):
    data = np.array([[0, 0], [1, 1]])
    array = DataArray(data, dims=("i", "coord"), coords={"i": range(2)})

    with pytest.raises(ValueError, match="DataArray must have coordinates 'x' and 'y'."):
        stratified_grid.assign_points(array)


if __name__ == "__main__":
    pytest.main()
